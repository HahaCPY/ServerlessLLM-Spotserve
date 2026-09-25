"""Verify actual expert ownership across a TP2/EP2 to TP4/EP4 replan."""

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import ray


def post_json(endpoint, path, payload, timeout):
    request = Request(
        f"{endpoint.rstrip('/')}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_ready_instance(model_name, exclude, timeout):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            for instance_id, state in latest.items():
                if (
                    instance_id not in exclude
                    and state.get("pool") == "ready"
                    and state.get("state") == "ready"
                ):
                    return instance_id, state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"No ready instance for {model_name}; states={latest}")


def runtime_profile(instance_id, state):
    actor = ray.get_actor(instance_id, namespace="sllm")
    metadata = ray.get(
        actor.get_runtime_metadata.remote(
            instance_id=instance_id,
            node_id=str(state.get("node_id") or ""),
        ),
        timeout=90,
    )
    return metadata["model_resource_profile"]


def expert_owners(profile):
    shards = profile.get("runtime_expert_placement_shards", {})
    if not profile.get("runtime_expert_placement_available") or not shards:
        raise RuntimeError("vLLM did not return actual expert placement")
    owners = {}
    for expert_key, rows in shards.items():
        ranks = sorted(
            int(row["ep_rank"])
            for row in rows
            if isinstance(row, dict) and row.get("ep_rank") is not None
        )
        if not ranks or len(ranks) != len(rows):
            raise RuntimeError(f"Missing EP rank for {expert_key}: {rows}")
        owners[expert_key] = ranks
    return owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/spotserve/config-vllm-reparallelization-applied-performance.json"
        ),
    )
    parser.add_argument("--model-path", default="/models/Qwen2-MoE-Tiny")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8343")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--event-timeout", type=float, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/spotserve_ep_transition_validation/report.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = f"vllm-ep-transition-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["num_gpus"] = 2
    backend = config["backend_config"]
    backend.update(
        pretrained_model_name_or_path=args.model_path,
        load_format="auto",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
    )
    planner = config["router_config"]["reparallelization_config"]
    planner.update(
        target_replica_gpus=4,
        min_tensor_parallel_size=4,
        max_tensor_parallel_size=4,
        max_pipeline_parallel_size=1,
        allow_stop_before_recreate=True,
        allow_preempting_target_recreate=True,
    )
    planner["synthetic_worker_nodes"]["0"].update(
        free_gpu=4, total_gpu=4
    )
    config["router_config"]["metrics_path"] = str(
        args.output.with_suffix(".jsonl").resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto", namespace="sllm")
    if ray.available_resources().get("GPU", 0) < 4:
        raise RuntimeError("Four free Ray GPUs are required for this test")

    registered = False
    try:
        registered = True
        post_json(args.endpoint, "register", config, args.ready_timeout)
        source_id, source_state = wait_for_ready_instance(
            model_name, set(), args.ready_timeout
        )
        before = runtime_profile(source_id, source_state)
        before_owners = expert_owners(before)
        if (
            before.get("tensor_parallel_size") != 2
            or before.get("runtime_expert_placement_worker_count") != 2
        ):
            raise RuntimeError("Initial runtime is not TP2/EP2")

        event = post_json(
            args.endpoint,
            "spot/event",
            {
                "event": "preempt",
                "instance_id": source_id,
                "model_name": model_name,
            },
            args.event_timeout,
        )
        target_id, target_state = wait_for_ready_instance(
            model_name, {source_id}, args.ready_timeout
        )
        after = runtime_profile(target_id, target_state)
        after_owners = expert_owners(after)
        if (
            after.get("tensor_parallel_size") != 4
            or after.get("runtime_expert_placement_worker_count") != 4
            or not after.get("expert_placement_runtime_verified_placement")
        ):
            raise RuntimeError("Target runtime is not verified TP4/EP4")
        if before_owners.keys() != after_owners.keys():
            raise RuntimeError("Expert coverage changed across the replan")
        changed = {
            key: {"before_ep_ranks": before_owners[key], "after_ep_ranks": after_owners[key]}
            for key in before_owners
            if before_owners[key] != after_owners[key]
        }
        if not changed:
            raise RuntimeError("No expert changed EP rank")
        replan = event.get("result", {}).get(model_name, {}).get(
            "reparallelization", {}
        )
        if replan.get("expert_placement_plan_movement_source") != (
            "runtime_expert_placement_shards"
        ):
            raise RuntimeError("Planner did not use observed runtime placement")
        if replan.get("expert_placement_plan_moved_experts") != len(changed):
            raise RuntimeError(
                "Planner movement count disagrees with runtime snapshots: "
                f"planner={replan.get('expert_placement_plan_moved_experts')}, "
                f"runtime={len(changed)}"
            )

        report = {
            "model": model_name,
            "source_instance_id": source_id,
            "target_instance_id": target_id,
            "spot_event": event,
            "before_worker_count": before["runtime_expert_placement_worker_count"],
            "after_worker_count": after["runtime_expert_placement_worker_count"],
            "before_expert_owners": before_owners,
            "after_expert_owners": after_owners,
            "changed_experts": changed,
            "planner_movement_source": replan[
                "expert_placement_plan_movement_source"
            ],
            "planner_moved_experts": replan[
                "expert_placement_plan_moved_experts"
            ],
            "planner_moved_weight_bytes": replan.get(
                "expert_placement_plan_moved_weight_bytes"
            ),
            "planner_movement_cost_estimate_ms": replan.get(
                "expert_placement_plan_estimated_weight_movement_cost_ms"
            ),
            "runtime_verified_placement": True,
            "physical_weight_migration": bool(
                after.get("expert_placement_physical_weight_migration")
            ),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"TP2/EP2 -> TP4/EP4 verified: {len(changed)}/{len(after_owners)} "
            f"expert shards changed EP rank; report={args.output}"
        )
    finally:
        if registered:
            try:
                post_json(args.endpoint, "delete", {"model": model_name}, 120)
            except Exception as exc:
                print(f"Cleanup warning for {model_name}: {exc}")


if __name__ == "__main__":
    main()
