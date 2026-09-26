"""Compare real vLLM A2A payload before and after a physical expert remap."""

import argparse
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import ray


COUNTER_FIELDS = (
    "collective_calls",
    "observed_input_bytes",
    "observed_output_bytes",
    "internode_calls",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def base_endpoint(endpoint):
    suffix = "/v1/chat/completions"
    return endpoint[: -len(suffix)] if endpoint.endswith(suffix) else endpoint


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
    raise TimeoutError(f"No ready instance for {model_name}; states={latest}")


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


def counters(profile):
    return {
        "available": bool(profile.get("all_to_all_counters_available")),
        "collective_calls": int(profile.get("all_to_all_collective_calls", 0)),
        "observed_input_bytes": int(
            profile.get("all_to_all_observed_input_bytes", 0)
        ),
        "observed_output_bytes": int(
            profile.get("all_to_all_observed_output_bytes", 0)
        ),
        "internode_calls": int(profile.get("all_to_all_internode_calls", 0)),
    }


def counter_delta(after, before):
    delta = {field: int(after[field]) - int(before[field]) for field in COUNTER_FIELDS}
    delta["observed_payload_bytes"] = (
        delta["observed_input_bytes"] + delta["observed_output_bytes"]
    )
    return delta


def response_signature(response):
    return [
        {
            "content": choice.get("message", {}).get("content"),
            "finish_reason": choice.get("finish_reason"),
        }
        for choice in response.get("choices", [])
    ]


def run_workload(endpoint, model_name, workload, timeout):
    responses = []
    latencies = []
    started = time.monotonic()
    for row in workload:
        payload = {
            key: value
            for key, value in row.items()
            if key not in {"request_id", "time", "phase", "benchmark_phase"}
        }
        payload["model"] = model_name
        request_started = time.monotonic()
        response = post_json(
            endpoint, "v1/chat/completions", payload, timeout
        )
        latencies.append((time.monotonic() - request_started) * 1000.0)
        if response.get("error") or not response.get("choices"):
            raise RuntimeError(f"Inference failed: {response}")
        responses.append(response_signature(response))
    elapsed = time.monotonic() - started
    ordered = sorted(latencies)
    p95_index = max(
        0,
        min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1),
    )
    return {
        "responses": responses,
        "latency_ms": latencies,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p95_ms": ordered[p95_index],
        "throughput_req_s": len(workload) / elapsed,
    }


def expert_owners(profile):
    shards = profile.get("runtime_expert_placement_shards", {})
    return {
        key: sorted(
            int(row["ep_rank"])
            for row in rows
            if isinstance(row, dict) and row.get("ep_rank") is not None
        )
        for key, rows in shards.items()
    }


def measure_windows(
    endpoint,
    model_name,
    instance_id,
    workload,
    repetitions,
    ready_timeout,
    request_timeout,
):
    trials = []
    for _ in range(repetitions):
        _, state = wait_for_instance(
            model_name, ready_timeout, require_idle=True
        )
        before = counters(runtime_profile(instance_id, state))
        trial = run_workload(
            endpoint, model_name, workload, request_timeout
        )
        _, state = wait_for_instance(
            model_name, ready_timeout, require_idle=True
        )
        after = counters(runtime_profile(instance_id, state))
        trial["counters"] = counter_delta(after, before)
        if trial["counters"]["collective_calls"] <= 0:
            raise RuntimeError("Measurement window observed no real A2A collectives")
        trials.append(trial)

    expected_responses = trials[0]["responses"]
    if any(trial["responses"] != expected_responses for trial in trials[1:]):
        raise RuntimeError("Repeated measurement windows produced different outputs")
    median_counters = {
        field: int(statistics.median(trial["counters"][field] for trial in trials))
        for field in (*COUNTER_FIELDS, "observed_payload_bytes")
    }
    calls = median_counters["collective_calls"]
    return {
        "trial_count": len(trials),
        "responses": expected_responses,
        "latency_p95_ms": statistics.median(
            trial["latency_p95_ms"] for trial in trials
        ),
        "throughput_req_s": statistics.median(
            trial["throughput_req_s"] for trial in trials
        ),
        "counters": median_counters,
        "observed_payload_bytes_per_collective": (
            median_counters["observed_payload_bytes"] / calls
        ),
        "trials": trials,
    }


def cleanup_model(endpoint, model_name, instance_ids):
    try:
        post_json(endpoint, "delete", {"model": model_name}, 120)
    finally:
        deadline = time.monotonic() + 30
        pending = set(instance_ids)
        while time.monotonic() < deadline:
            pending.update(
                row["name"]
                for row in ray.util.list_named_actors(all_namespaces=True)
                if row["namespace"] == "sllm"
                and row["name"].startswith(f"{model_name}_")
            )
            live = set()
            for instance_id in pending:
                try:
                    actor = ray.get_actor(instance_id, namespace="sllm")
                    ray.kill(actor, no_restart=True)
                    live.add(instance_id)
                except ValueError:
                    pass
            if not live:
                return
            pending = live
            time.sleep(0.5)
        raise RuntimeError(f"Timed out cleaning backend actors: {sorted(pending)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--event-timeout", type=float)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    args = parser.parse_args()

    matrix = load_json(args.config)
    endpoint = base_endpoint(args.endpoint or matrix["endpoint"])
    request_timeout = args.request_timeout or float(
        matrix.get("request_timeout_s", 240)
    )
    event_timeout = args.event_timeout or float(matrix.get("event_timeout_s", 600))
    ready_timeout = float(matrix.get("ready_timeout_s", 600))
    repetitions = max(1, int(matrix.get("measurement_repetitions", 3)))
    output = Path(matrix["output"])
    workload = load_jsonl(Path(matrix["workload"]))
    if not workload or len(workload) % 2:
        raise RuntimeError("A2A reduction workload must contain an even request count")

    config = load_json(Path(matrix["deploy_config"]))
    model_name = f"vllm-a2a-reduction-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["backend_config"]["enable_prefix_caching"] = False
    config["router_config"]["metrics_path"] = str(
        output.with_suffix(".jsonl").resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address=args.ray_address, namespace=args.ray_namespace)
    if ray.available_resources().get("GPU", 0) < 2:
        raise RuntimeError("Two free Ray GPUs are required for this experiment")

    registered = False
    instance_ids = set()
    try:
        registered = True
        post_json(endpoint, "register", config, ready_timeout)
        instance_id, state = wait_for_instance(model_name, ready_timeout)
        instance_ids.add(instance_id)
        run_workload(
            endpoint,
            model_name,
            [{
                "messages": [{
                    "role": "user",
                    "content": "Warm up mixture-of-experts inference.",
                }],
                "max_tokens": 8,
                "temperature": 0,
            }],
            request_timeout,
        )
        instance_id, state = wait_for_instance(
            model_name, ready_timeout, require_idle=True
        )
        baseline_owners = expert_owners(runtime_profile(instance_id, state))
        baseline = measure_windows(
            endpoint,
            model_name,
            instance_id,
            workload,
            repetitions,
            ready_timeout,
            request_timeout,
        )

        event = post_json(
            endpoint,
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
            event_timeout,
        )
        if not (event.get("success") is True or event.get("status") == "ok"):
            raise RuntimeError(f"Placement event failed: {event}")
        replan = event.get("result", {}).get(model_name, {}).get(
            "reparallelization", {}
        )
        execution = replan.get("execution", {})
        runtime = execution.get("expert_placement_runtime", {})
        if not (
            int(runtime.get("apply_success_count", 0) or 0) > 0
            and int(runtime.get("verify_success_count", 0) or 0) > 0
            and int(runtime.get("physical_weight_migration_count", 0) or 0) > 0
        ):
            raise RuntimeError(f"Physical expert remap was not verified: {runtime}")

        _, state = wait_for_instance(
            model_name, ready_timeout, require_idle=True
        )
        candidate_profile = runtime_profile(instance_id, state)
        candidate_owners = expert_owners(candidate_profile)
        candidate = measure_windows(
            endpoint,
            model_name,
            instance_id,
            workload,
            repetitions,
            ready_timeout,
            request_timeout,
        )

        if baseline["responses"] != candidate["responses"]:
            raise RuntimeError("Baseline and candidate inference outputs differ")
        baseline_bytes = baseline["counters"]["observed_payload_bytes"]
        candidate_bytes = candidate["counters"]["observed_payload_bytes"]
        if baseline_bytes <= 0:
            raise RuntimeError("Baseline observed no A2A tensor payload")
        reduction_ratio = (baseline_bytes - candidate_bytes) / baseline_bytes
        traffic_reduced = candidate_bytes < baseline_bytes
        backend = str(config["backend_config"].get("all2all_backend", ""))
        if traffic_reduced:
            interpretation = "measured_collective_payload_reduction"
        elif candidate_bytes == baseline_bytes and backend == (
            "allgather_reducescatter"
        ):
            interpretation = "payload_invariant_for_allgather_reducescatter"
        else:
            interpretation = "measured_collective_payload_not_reduced"

        changed = {
            key: {
                "baseline_ep_ranks": baseline_owners.get(key, []),
                "candidate_ep_ranks": candidate_owners.get(key, []),
            }
            for key in sorted(set(baseline_owners) | set(candidate_owners))
            if baseline_owners.get(key, []) != candidate_owners.get(key, [])
        }
        report = {
            "experiment": matrix.get("name"),
            "model": model_name,
            "all_to_all_backend": backend,
            "request_count_per_window": len(workload),
            "measurement_repetitions": repetitions,
            "same_workload": True,
            "prefix_caching_enabled": False,
            "outputs_match": True,
            "physical_weight_migration": True,
            "runtime_verified_placement": True,
            "changed_expert_count": len(changed),
            "changed_experts": changed,
            "baseline": baseline,
            "candidate": candidate,
            "baseline_observed_payload_bytes": baseline_bytes,
            "candidate_observed_payload_bytes": candidate_bytes,
            "payload_reduction_bytes": baseline_bytes - candidate_bytes,
            "payload_reduction_ratio": reduction_ratio,
            "traffic_reduced": traffic_reduced,
            "interpretation": interpretation,
            "claim_supported": traffic_reduced,
            "runtime": runtime,
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("A2A reduction experiment completed")
        print(f"  baseline_payload_bytes={baseline_bytes}")
        print(f"  candidate_payload_bytes={candidate_bytes}")
        print(f"  payload_reduction_ratio={reduction_ratio:.6f}")
        print(f"  traffic_reduced={traffic_reduced}")
        print(f"  interpretation={interpretation}")
        print(f"  report={output}")
        if matrix.get("require_traffic_reduction") and not traffic_reduced:
            raise RuntimeError("Candidate did not reduce observed A2A payload")
    finally:
        if registered:
            try:
                cleanup_model(endpoint, model_name, instance_ids)
            except Exception as exc:
                print(f"Cleanup warning for {model_name}: {exc}")


if __name__ == "__main__":
    main()
