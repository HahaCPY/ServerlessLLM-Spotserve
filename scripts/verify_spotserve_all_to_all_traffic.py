"""Measure vLLM EP collective payload and optionally verify a reduction."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import ray


def post_json(endpoint: str, payload: dict, timeout: float) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ready_instance(model_name: str, timeout: float) -> tuple[str, dict]:
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        router = ray.get_actor(model_name, namespace="models")
        latest = ray.get(router.get_instance_states.remote(), timeout=15)
        for instance_id, state in latest.items():
            if state.get("pool") == "ready" and state.get("state") == "ready":
                return instance_id, state
        time.sleep(1)
    raise TimeoutError(f"No ready instance for {model_name}; states={latest}")


def counters(instance_id: str, state: dict) -> dict[str, int | bool]:
    actor = ray.get_actor(instance_id, namespace="sllm")
    metadata = ray.get(
        actor.get_runtime_metadata.remote(
            instance_id=instance_id,
            node_id=str(state.get("node_id") or ""),
        ),
        timeout=90,
    )
    profile = metadata["model_resource_profile"]
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


def subtract(after: dict, before: dict) -> dict[str, int]:
    return {
        key: int(after[key]) - int(before[key])
        for key in (
            "collective_calls",
            "observed_input_bytes",
            "observed_output_bytes",
            "internode_calls",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8343/v1/chat/completions"
    )
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ray.init(address="auto", namespace="sllm")
    instance_id, state = ready_instance(args.model, args.ready_timeout)
    before = counters(instance_id, state)
    for index in range(args.requests):
        response = post_json(
            args.endpoint,
            {
                "model": args.model,
                "messages": [{
                    "role": "user",
                    "content": f"Explain expert parallelism briefly. Run {index}.",
                }],
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
            args.timeout,
        )
        if "error" in response:
            raise RuntimeError(f"Inference failed: {response['error']}")
    after = counters(instance_id, state)
    delta = subtract(after, before)
    if delta["collective_calls"] <= 0:
        raise RuntimeError(
            "No runtime all-to-all calls observed; rebuild the image with the "
            "collective hook and set VLLM_SPOTSERVE_A2A_TRACE=1"
        )

    report = {
        "model": args.model,
        "instance_id": instance_id,
        "requests": args.requests,
        "counter_kind": "runtime_collective_tensor_payload",
        "before": before,
        "after": after,
        "delta": delta,
        "observed_payload_bytes": (
            delta["observed_input_bytes"] + delta["observed_output_bytes"]
        ),
    }
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_bytes = int(baseline.get("observed_payload_bytes", 0))
        candidate_bytes = int(report["observed_payload_bytes"])
        if baseline_bytes <= 0:
            raise RuntimeError("Baseline report has no observed payload")
        report["baseline_observed_payload_bytes"] = baseline_bytes
        report["payload_reduction_ratio"] = (
            baseline_bytes - candidate_bytes
        ) / baseline_bytes
        report["traffic_reduced"] = candidate_bytes < baseline_bytes
        if not report["traffic_reduced"]:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError("Candidate did not reduce observed all-to-all payload")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
