"""Verify quiescent in-place vLLM Ray-DP EP2 to EP4 resizing."""

import argparse
import concurrent.futures
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import ray


def query_gpu_memory():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, free_mib, total_mib = (part.strip() for part in line.split(","))
        rows.append(
            {
                "index": int(index),
                "free_mib": int(free_mib),
                "total_mib": int(total_mib),
            }
        )
    return rows


def require_free_gpu_memory(expected_gpus, minimum_free_mib):
    probe = ray.remote(num_cpus=0, resources={"worker_node": 0.01})(
        query_gpu_memory
    )
    rows = ray.get(probe.remote(), timeout=30)
    if len(rows) < expected_gpus:
        raise RuntimeError(
            f"Expected {expected_gpus} worker GPUs, found {len(rows)}: {rows}"
        )
    blocked = [
        row
        for row in rows[:expected_gpus]
        if row["free_mib"] < minimum_free_mib
    ]
    if blocked:
        details = ", ".join(
            f"GPU {row['index']}={row['free_mib']} MiB free"
            for row in rows[:expected_gpus]
        )
        raise RuntimeError(
            "Elastic EP2 -> EP4 requires four physically free GPUs before "
            f"model registration (minimum {minimum_free_mib} MiB each); "
            f"observed {details}"
        )
    return rows[:expected_gpus]


def post_json(endpoint, path, payload, timeout):
    request = Request(
        f"{endpoint.rstrip('/')}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_ready_instance(model_name, timeout):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            for instance_id, state in latest.items():
                if (
                    state.get("pool") == "ready"
                    and state.get("state") == "ready"
                ):
                    return instance_id, state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"No ready instance for {model_name}; states={latest}")


def wait_for_active_request(model_name, instance_id, timeout):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            state = latest.get(instance_id, {})
            if int(state.get("concurrency", 0) or 0) > 0:
                return state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(0.05)
    raise TimeoutError(
        f"Request never became active on {instance_id}; states={latest}"
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


def wait_for_runtime_profile(
    instance_id,
    state,
    expected_worker_count,
    timeout,
    require_verified=False,
):
    deadline = time.monotonic() + timeout
    latest = {}
    latest_error = None
    while time.monotonic() < deadline:
        try:
            latest = runtime_profile(instance_id, state)
            worker_count = int(
                latest.get("runtime_expert_placement_worker_count", 0) or 0
            )
            verified = bool(
                latest.get(
                    "expert_placement_runtime_verified_placement", False
                )
            )
            if worker_count == expected_worker_count and (
                verified or not require_verified
            ):
                return latest
            latest_error = None
        except (ray.exceptions.GetTimeoutError, KeyError) as exc:
            latest_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(
        "Runtime placement did not converge: "
        f"expected_workers={expected_worker_count}, "
        f"require_verified={require_verified}, latest_error={latest_error}, "
        f"latest={latest}"
    )


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


def inference(
    endpoint,
    model_name,
    timeout,
    *,
    max_tokens=8,
    ignore_eos=False,
):
    response = post_json(
        endpoint,
        "v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Count to three."}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
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
            "examples/spotserve/config-vllm-elastic-ep-resize-performance.json"
        ),
    )
    parser.add_argument("--model-path", default="/models/Qwen2-MoE-Tiny")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8343")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--event-timeout", type=float, default=600)
    parser.add_argument(
        "--minimum-free-gpu-memory-mib",
        type=int,
        default=8192,
        help="Fail before model registration if any of the four GPUs is busy.",
    )
    parser.add_argument(
        "--active-request-drain",
        action="store_true",
        help=(
            "Trigger resize while a request is active and verify that the "
            "request drains successfully before the quiescent resize."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/spotserve_elastic_ep_resize_performance/report.json"
        ),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = f"vllm-elastic-ep-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["backend_config"]["pretrained_model_name_or_path"] = args.model_path
    config["router_config"]["metrics_path"] = str(
        args.output.with_suffix(".jsonl").resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto", namespace="sllm")
    if ray.available_resources().get("GPU", 0) < 4:
        raise RuntimeError("Four free Ray GPUs are required for this test")
    gpu_memory_before = require_free_gpu_memory(
        expected_gpus=4,
        minimum_free_mib=args.minimum_free_gpu_memory_mib,
    )

    registered = False
    try:
        registered = True
        post_json(args.endpoint, "register", config, args.ready_timeout)
        source_id, source_state = wait_for_ready_instance(
            model_name, args.ready_timeout
        )
        before = wait_for_runtime_profile(
            source_id,
            source_state,
            expected_worker_count=2,
            timeout=args.ready_timeout,
        )
        before_owners = expert_owners(before)
        if before.get("runtime_expert_placement_worker_count") != 2:
            raise RuntimeError(f"Initial runtime is not EP2: {before}")
        inference(args.endpoint, model_name, args.event_timeout)

        active_request_observed = False
        active_request_response = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            active_request = None
            if args.active_request_drain:
                active_request = executor.submit(
                    inference,
                    args.endpoint,
                    model_name,
                    args.event_timeout,
                    max_tokens=128,
                    ignore_eos=True,
                )
                wait_for_active_request(
                    model_name,
                    source_id,
                    min(args.ready_timeout, 120),
                )
                if active_request.done():
                    raise RuntimeError(
                        "Request completed before resize event was emitted"
                    )
                active_request_observed = True

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
            if active_request is not None:
                active_request_response = active_request.result(
                    timeout=args.event_timeout
                )
        target_id, target_state = wait_for_ready_instance(
            model_name, args.ready_timeout
        )
        if target_id != source_id:
            raise RuntimeError(
                "Elastic EP resize recreated the SLLM actor: "
                f"{source_id} -> {target_id}"
            )

        # The router remains ready while an in-place resize runs, so router
        # readiness is not a resize-completion signal. Wait until runtime
        # metadata includes every new EngineCore and the backend has committed
        # its placement verification.
        after = wait_for_runtime_profile(
            target_id,
            target_state,
            expected_worker_count=4,
            timeout=args.event_timeout,
            require_verified=True,
        )
        after_owners = expert_owners(after)
        if (
            after.get("runtime_expert_placement_worker_count") != 4
            or not after.get("expert_placement_runtime_verified_placement")
        ):
            raise RuntimeError(f"Target runtime is not verified EP4: {after}")
        if before_owners.keys() != after_owners.keys():
            raise RuntimeError("Expert coverage changed across elastic resize")

        replan = event.get("result", {}).get(model_name, {}).get(
            "reparallelization", {}
        )
        execution = replan.get("execution", {})
        if execution.get("reparallelization_execution_model") != (
            "in_place_elastic_ep_resize"
        ):
            raise RuntimeError(f"Unexpected execution model: {execution}")
        if not execution.get("in_place_ep_resize"):
            raise RuntimeError("Resize did not report in_place_ep_resize")
        if not execution.get("elastic_ep_admission_drained"):
            raise RuntimeError("Resize did not report an admission drain")
        drained_request_count = int(
            execution.get("elastic_ep_drained_request_count", 0) or 0
        )
        if args.active_request_drain and drained_request_count < 1:
            raise RuntimeError(
                "Active-request resize did not report a drained request: "
                f"{execution}"
            )
        if execution.get("instance_ids") != [source_id]:
            raise RuntimeError(f"Unexpected target instances: {execution}")

        inference(args.endpoint, model_name, args.event_timeout)
        changed = {
            key: {
                "before_ep_ranks": before_owners[key],
                "after_ep_ranks": after_owners[key],
            }
            for key in before_owners
            if before_owners[key] != after_owners[key]
        }
        report = {
            "model": model_name,
            "instance_id": source_id,
            "spot_event": event,
            "before_worker_count": 2,
            "after_worker_count": 4,
            "before_expert_owners": before_owners,
            "after_expert_owners": after_owners,
            "changed_experts": changed,
            "runtime_verified_placement": True,
            "execution_model": execution[
                "reparallelization_execution_model"
            ],
            "actor_recreated": False,
            "gpu_memory_before_registration": gpu_memory_before,
            "active_request_drain_requested": args.active_request_drain,
            "active_request_observed": active_request_observed,
            "active_request_completed": active_request_response is not None,
            "elastic_ep_admission_drained": True,
            "elastic_ep_drained_request_count": drained_request_count,
            "source_effective_expert_parallel_size": execution[
                "source_effective_expert_parallel_size"
            ],
            "target_effective_expert_parallel_size": execution[
                "target_effective_expert_parallel_size"
            ],
        }
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "In-place elastic EP2 -> EP4 verified on actor "
            f"{source_id}; changed_experts={len(changed)}; "
            f"report={args.output}"
        )
    finally:
        if registered:
            try:
                post_json(
                    args.endpoint, "delete", {"model": model_name}, 180
                )
            except Exception as exc:
                print(f"Cleanup warning for {model_name}: {exc}")


if __name__ == "__main__":
    main()
