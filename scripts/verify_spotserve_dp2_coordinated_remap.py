"""Verify a globally preflighted physical expert remap across two DP engines."""

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


def wait_for_instance(model_name, timeout, *, require_idle=False):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            for instance_id, state in latest.items():
                ready = (
                    state.get("pool") == "ready"
                    and state.get("state") == "ready"
                )
                idle = int(state.get("concurrency", 0) or 0) == 0
                if ready and (idle or not require_idle):
                    return instance_id, state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(0.5)
    qualifier = "idle ready" if require_idle else "ready"
    raise TimeoutError(
        f"No {qualifier} instance for {model_name}; states={latest}"
    )


def runtime_profile(instance_id, state):
    actor = ray.get_actor(instance_id, namespace="sllm")
    metadata = ray.get(
        actor.get_runtime_metadata.remote(
            instance_id=instance_id,
            node_id=str(state.get("node_id") or ""),
        ),
        timeout=120,
    )
    return metadata["model_resource_profile"]


def expert_owners(profile):
    shards = profile.get("runtime_expert_placement_shards", {})
    if not profile.get("runtime_expert_placement_available") or not shards:
        raise RuntimeError("vLLM did not return actual expert placement")
    return {
        expert_key: sorted(
            int(row["ep_rank"])
            for row in rows
            if isinstance(row, dict) and row.get("ep_rank") is not None
        )
        for expert_key, rows in shards.items()
    }


def infer(endpoint, model_name, prompt, timeout):
    response = post_json(
        endpoint,
        "v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout,
    )
    if response.get("error") or not response.get("choices"):
        raise RuntimeError(f"Inference failed: {response}")
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/spotserve/"
            "config-vllm-expert-remap-dp2-coordinated-performance.json"
        ),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8343")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--event-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/spotserve_expert_remap_dp2_coordinated_performance/"
            "sequential-verification.json"
        ),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = f"vllm-dp2-remap-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["router_config"]["metrics_path"] = str(
        args.output.with_suffix(".jsonl").resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto", namespace="sllm")
    if ray.available_resources().get("GPU", 0) < 2:
        raise RuntimeError("Two free Ray GPUs are required for this test")

    registered = False
    try:
        registered = True
        post_json(args.endpoint, "register", config, args.ready_timeout)
        instance_id, state = wait_for_instance(model_name, args.ready_timeout)

        infer(
            args.endpoint,
            model_name,
            "Describe mixture-of-experts briefly.",
            args.request_timeout,
        )
        instance_id, state = wait_for_instance(
            model_name, args.ready_timeout, require_idle=True
        )
        before = runtime_profile(instance_id, state)
        before_owners = expert_owners(before)

        event = post_json(
            args.endpoint,
            "spot/event",
            {
                "event": "add",
                "node_id": "0",
                "model_name": model_name,
                "node_info": {
                    "free_gpu": 2,
                    "total_gpu": 4,
                    "state": "ready",
                },
            },
            args.event_timeout,
        )
        event_succeeded = bool(
            event.get("success") is True or event.get("status") == "ok"
        )
        if not event_succeeded:
            raise RuntimeError(f"Placement event failed: {event}")

        _, state = wait_for_instance(
            model_name, args.ready_timeout, require_idle=True
        )
        after = runtime_profile(instance_id, state)
        after_owners = expert_owners(after)
        infer(
            args.endpoint,
            model_name,
            "Describe expert parallelism briefly.",
            args.request_timeout,
        )

        replan = event.get("result", {}).get(model_name, {}).get(
            "reparallelization", {}
        )
        execution = replan.get("execution", {})
        runtime = execution.get("expert_placement_runtime", {})
        required_counts = {
            "apply_success_count": runtime.get("apply_success_count", 0),
            "verify_success_count": runtime.get("verify_success_count", 0),
            "physical_weight_migration_count": runtime.get(
                "physical_weight_migration_count", 0
            ),
        }
        if any(int(value or 0) < 1 for value in required_counts.values()):
            raise RuntimeError(
                "DP2 remap was not runtime-applied and verified: "
                f"runtime={runtime}, execution={execution}"
            )
        if not after.get("expert_placement_runtime_verified_placement"):
            raise RuntimeError("Post-remap runtime placement is not verified")
        if not after.get("expert_placement_physical_weight_migration"):
            raise RuntimeError("Post-remap runtime did not report physical migration")

        changed = {
            key: {
                "before_ep_ranks": before_owners[key],
                "after_ep_ranks": after_owners.get(key, []),
            }
            for key in before_owners
            if before_owners[key] != after_owners.get(key, [])
        }
        if not changed:
            raise RuntimeError("No expert ownership changed during DP2 remap")

        report = {
            "model": model_name,
            "instance_id": instance_id,
            "pre_request_completed": True,
            "router_idle_before_event": True,
            "post_request_completed": True,
            "changed_expert_count": len(changed),
            "changed_experts": changed,
            "runtime": runtime,
            "runtime_verified_placement": True,
            "physical_weight_migration": True,
            "spot_event": event,
        }
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "DP2 coordinated remap verified: "
            f"{len(changed)} expert shards changed; report={args.output}"
        )
    finally:
        if registered:
            try:
                post_json(args.endpoint, "delete", {"model": model_name}, 120)
            except Exception as exc:
                print(f"Cleanup warning for {model_name}: {exc}")


if __name__ == "__main__":
    main()
