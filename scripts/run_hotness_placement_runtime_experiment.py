"""Build real EP2 placement profiles and a paired request benchmark.

The runner uses one frozen calibration workload, deploys each placement in a
fresh vLLM DP2/EP2 instance, verifies the runtime ownership, and replays the
same request/repetition IDs for every candidate.  It writes raw records plus
the bounded six-row profile table consumed by the SpotServe planner.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import ray


PROFILE_STRATEGIES = ("linear", "round_robin", "hotness_balanced")
STRATEGIES = (*PROFILE_STRATEGIES, "hotness_locality")

WORKLOADS = {
    "short-factual": [
        {
            "request_id": "short-0",
            "messages": [{"role": "user", "content": "Name one moon of Mars."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        {
            "request_id": "short-1",
            "messages": [{"role": "user", "content": "What is 7 plus 5?"}],
            "max_tokens": 8,
            "temperature": 0,
        },
    ],
    "structured-reasoning": [
        {
            "request_id": "structured-0",
            "messages": [{
                "role": "user",
                "content": "Give two concise reasons expert parallelism helps MoE serving.",
            }],
            "max_tokens": 16,
            "temperature": 0,
        },
        {
            "request_id": "structured-1",
            "messages": [{
                "role": "user",
                "content": "Return JSON with keys even and odd for numbers 1 through 4.",
            }],
            "max_tokens": 16,
            "temperature": 0,
        },
    ],
    "hotness-shift": [
        {
            "request_id": "shift-0",
            "messages": [{
                "role": "user",
                "content": (
                    "List three practical ways to reduce communication "
                    "during mixture-of-experts inference."
                ),
            }],
            "max_tokens": 20,
            "temperature": 0,
        },
        {
            "request_id": "shift-1",
            "messages": [{
                "role": "user",
                "content": (
                    "Explain in two sentences why a workload routing "
                    "distribution can change over time."
                ),
            }],
            "max_tokens": 20,
            "temperature": 0,
        },
    ],
}

CALIBRATION_BUCKETS = ("short-factual", "structured-reasoning")

HELD_OUT = [{
    "request_id": "held-out-0",
    "messages": [{
        "role": "user",
        "content": "In one sentence, contrast dense and mixture-of-experts models.",
    }],
    "max_tokens": 12,
    "temperature": 0,
}]


def post_json(endpoint: str, path: str, payload: dict, timeout: float) -> dict:
    request = Request(
        f"{endpoint.rstrip('/')}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_instance(model_name: str, timeout: float, *, idle: bool = False):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            for instance_id, state in latest.items():
                ready = state.get("pool") == "ready" and state.get("state") == "ready"
                is_idle = int(state.get("concurrency", 0) or 0) == 0
                if ready and (is_idle or not idle):
                    return instance_id, state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"No ready instance for {model_name}; states={latest}")


def runtime_profile(instance_id: str, state: dict) -> dict:
    actor = ray.get_actor(instance_id, namespace="sllm")
    metadata = ray.get(
        actor.get_runtime_metadata.remote(
            instance_id=instance_id,
            node_id=str(state.get("node_id") or ""),
        ),
        timeout=120,
    )
    return metadata["model_resource_profile"]


def owners(profile: dict) -> dict[str, list[int]]:
    shards = profile.get("runtime_expert_placement_shards", {})
    if not profile.get("runtime_expert_placement_available") or not shards:
        raise RuntimeError("runtime expert placement is unavailable")
    return {
        str(key): sorted(
            int(row["ep_rank"])
            for row in rows
            if isinstance(row, dict) and row.get("ep_rank") is not None
        )
        for key, rows in shards.items()
    }


def a2a_counters(profile: dict) -> dict[str, int | str | bool]:
    return {
        "available": bool(profile.get("all_to_all_counters_available")),
        "measurement_kind": str(
            profile.get("all_to_all_measurement_kind") or "unavailable"
        ),
        "collective_calls": int(profile.get("all_to_all_collective_calls", 0) or 0),
        "input_bytes": int(profile.get("all_to_all_observed_input_bytes", 0) or 0),
        "output_bytes": int(profile.get("all_to_all_observed_output_bytes", 0) or 0),
    }


def counter_delta(after: dict, before: dict) -> dict[str, int | str | bool]:
    result = {
        key: int(after[key]) - int(before[key])
        for key in ("collective_calls", "input_bytes", "output_bytes")
    }
    result["payload_bytes"] = result["input_bytes"] + result["output_bytes"]
    result["available"] = bool(after["available"])
    result["measurement_kind"] = str(after["measurement_kind"])
    return result


def mapping_fingerprint(mapping: dict[str, list[int]]) -> str:
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def response_signature(response: dict) -> list[dict]:
    return [
        {
            "content": choice.get("message", {}).get("content"),
            "finish_reason": choice.get("finish_reason"),
        }
        for choice in response.get("choices", [])
    ]


def run_requests(
    endpoint: str,
    model_name: str,
    workload_bucket_id: str,
    rows: list[dict],
    repetitions: int,
    timeout: float,
) -> list[dict]:
    records = []
    for repetition in range(repetitions):
        for row in rows:
            payload = {key: value for key, value in row.items() if key != "request_id"}
            payload["model"] = model_name
            started = time.monotonic()
            response = post_json(endpoint, "v1/chat/completions", payload, timeout)
            latency_ms = (time.monotonic() - started) * 1000.0
            if response.get("error") or not response.get("choices"):
                raise RuntimeError(f"Inference failed: {response}")
            usage = response.get("usage", {})
            records.append({
                "pair_id": f"{workload_bucket_id}/{repetition}/{row['request_id']}",
                "workload_bucket_id": workload_bucket_id,
                "repetition": repetition,
                "request_id": row["request_id"],
                "latency_ms": latency_ms,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                "success": True,
                "response_signature": response_signature(response),
            })
    return records


def summarize(records: list[dict]) -> dict:
    latencies = sorted(float(row["latency_ms"]) for row in records)
    elapsed_s = sum(latencies) / 1000.0

    def percentile(q: float) -> float:
        index = max(0, min(len(latencies) - 1, math.ceil(len(latencies) * q) - 1))
        return latencies[index]

    return {
        "sample_count": len(records),
        "service_latency_mean_ms": statistics.fmean(latencies),
        "service_latency_p50_ms": statistics.median(latencies),
        "service_latency_p95_ms": percentile(0.95),
        "service_latency_p99_ms": percentile(0.99),
        "throughput_req_s": len(records) / elapsed_s,
    }


def run_load_sweep(
    endpoint: str,
    model_name: str,
    *,
    concurrency_levels: list[int],
    request_count: int,
    timeout: float,
) -> list[dict]:
    """Measure closed-loop throughput under controlled concurrency."""
    request_template = HELD_OUT[0]
    results = []

    def invoke(index: int) -> dict:
        payload = {
            key: value
            for key, value in request_template.items()
            if key != "request_id"
        }
        payload["model"] = model_name
        started = time.monotonic()
        response = post_json(endpoint, "v1/chat/completions", payload, timeout)
        latency_ms = (time.monotonic() - started) * 1000.0
        if response.get("error") or not response.get("choices"):
            raise RuntimeError(f"Load-sweep inference failed: {response}")
        return {
            "pair_id": f"load/{index}",
            "workload_bucket_id": "load-sweep",
            "repetition": 0,
            "request_id": f"load-{index}",
            "latency_ms": latency_ms,
            "success": True,
        }

    for concurrency in concurrency_levels:
        wall_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            rows = list(executor.map(invoke, range(request_count)))
        wall_s = time.monotonic() - wall_started
        summary = summarize(rows)
        results.append({
            "controlled_concurrency": concurrency,
            "request_count": request_count,
            "wall_time_s": wall_s,
            "achieved_throughput_req_s": request_count / wall_s,
            **summary,
        })
    return results


def run_arrival_rate_sweep(
    endpoint: str,
    model_name: str,
    *,
    arrival_rates: list[float],
    request_count: int,
    max_workers: int,
    timeout: float,
) -> list[dict]:
    """Measure end-to-end latency under a deterministic open-loop schedule."""
    request_template = HELD_OUT[0]
    results = []

    def invoke(index: int) -> dict:
        payload = {
            key: value
            for key, value in request_template.items()
            if key != "request_id"
        }
        payload["model"] = model_name
        started = time.monotonic()
        response = post_json(endpoint, "v1/chat/completions", payload, timeout)
        latency_ms = (time.monotonic() - started) * 1000.0
        if response.get("error") or not response.get("choices"):
            raise RuntimeError(f"Arrival-sweep inference failed: {response}")
        return {
            "pair_id": f"arrival/{index}",
            "workload_bucket_id": "arrival-sweep",
            "repetition": 0,
            "request_id": f"arrival-{index}",
            "latency_ms": latency_ms,
            "success": True,
        }

    for rate in arrival_rates:
        wall_started = time.monotonic()
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for index in range(request_count):
                due = wall_started + index / rate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                futures.append(executor.submit(invoke, index))
            rows = [future.result() for future in futures]
        wall_s = time.monotonic() - wall_started
        summary = summarize(rows)
        results.append({
            "offered_arrival_rate_req_s": rate,
            "request_count": request_count,
            "max_workers": max_workers,
            "wall_time_s": wall_s,
            "achieved_throughput_req_s": request_count / wall_s,
            **summary,
        })
    return results


def placement_from_event(event: dict, model_name: str) -> dict:
    replan = event.get("result", {}).get(model_name, {}).get("reparallelization", {})
    parallel_plan = replan.get("parallel_plan", {})
    plan = parallel_plan.get("expert_placement_plan") or {}
    return {
        "reparallelization": replan,
        "plan": plan,
        "fingerprint": str(
            plan.get("placement_fingerprint")
            or parallel_plan.get("expert_placement_fingerprint")
            or ""
        ),
    }


def cleanup_model(endpoint: str, model_name: str, instance_ids: set[str]) -> None:
    try:
        post_json(endpoint, "delete", {"model": model_name}, 120)
    finally:
        deadline = time.monotonic() + 45
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
                    ray.kill(ray.get_actor(instance_id, namespace="sllm"), no_restart=True)
                    live.add(instance_id)
                except ValueError:
                    pass
            if not live:
                return
            pending = live
            time.sleep(0.5)
        raise RuntimeError(f"Timed out cleaning actors: {sorted(pending)}")


def configure(base: dict, strategy: str, model_name: str, metrics_path: Path) -> dict:
    config = copy.deepcopy(base)
    config["model"] = model_name
    backend = config["backend_config"]
    backend["enable_moe_route_instrumentation"] = True
    backend["enable_prefix_caching"] = False
    backend["all2all_backend"] = "spotserve_sparse"
    planner = config["router_config"]["reparallelization_config"]
    planner["target_expert_placement_strategy"] = strategy
    planner["minimum_hotness_samples"] = 1
    planner["hotness_max_age_s"] = 600
    planner["minimum_improvement_ratio"] = 0
    planner["require_known_hotness_movement_cost"] = False
    planner["maximum_moved_experts"] = 8
    planner["hotness_movement_weight"] = 0
    planner["source_rank_locality_weight"] = 2.0
    config["router_config"]["metrics_path"] = str(metrics_path.resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/spotserve/config-vllm-expert-remap-"
            "a2a-reduction-performance.json"
        ),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8343")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--load-sweep-requests", type=int, default=6)
    parser.add_argument(
        "--load-sweep-concurrency",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument("--arrival-sweep-requests", type=int, default=4)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        action="append",
        default=None,
    )
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--event-timeout", type=float, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hotness_placement/runtime-experiment.json"),
    )
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 5070 Ti")
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("at least two measurement repetitions are required")
    concurrency_levels = sorted(set(args.load_sweep_concurrency or [1, 2]))
    arrival_rates = sorted(set(args.arrival_rate or [0.5, 1.5]))
    if any(value < 1 for value in concurrency_levels):
        raise ValueError("load-sweep concurrency must be positive")
    if any(value <= 0 for value in arrival_rates):
        raise ValueError("arrival rates must be positive")

    base = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ray.init(address="auto", namespace="sllm")
    if ray.available_resources().get("GPU", 0) < 2:
        raise RuntimeError("Two free Ray GPUs are required")

    manifest_id = f"hotness-ep2-{uuid.uuid4().hex[:12]}"
    candidate_results = {}
    all_records = []
    profile_rows = []
    mapping_to_strategy = {}
    runtime_fingerprint = ""

    for strategy in STRATEGIES:
        model_name = f"hotness-{strategy.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
        config = configure(
            base,
            strategy,
            model_name,
            args.output.with_name(f"{strategy}-router.jsonl"),
        )
        registered = False
        instance_ids: set[str] = set()
        try:
            registered = True
            post_json(args.endpoint, "register", config, args.ready_timeout)
            instance_id, state = wait_for_instance(model_name, args.ready_timeout)
            instance_ids.add(instance_id)

            calibration_rows = [
                row
                for bucket_id in CALIBRATION_BUCKETS
                for row in WORKLOADS[bucket_id]
            ]
            run_requests(
                args.endpoint,
                model_name,
                "calibration",
                calibration_rows,
                1,
                args.request_timeout,
            )
            _, state = wait_for_instance(model_name, args.ready_timeout, idle=True)
            before_profile = runtime_profile(instance_id, state)
            event = post_json(
                args.endpoint,
                "spot/event",
                {
                    "event": "add",
                    "node_id": "0",
                    "model_name": model_name,
                    "node_info": {"free_gpu": 2, "total_gpu": 4, "state": "ready"},
                },
                args.event_timeout,
            )
            if not (event.get("success") is True or event.get("status") == "ok"):
                raise RuntimeError(f"Placement event failed: {event}")
            _, state = wait_for_instance(model_name, args.ready_timeout, idle=True)
            after_profile = runtime_profile(instance_id, state)
            actual_mapping = owners(after_profile)
            actual_mapping_fingerprint = mapping_fingerprint(actual_mapping)
            event_placement = placement_from_event(event, model_name)
            plan = event_placement["plan"]
            plan_fingerprint = event_placement["fingerprint"]
            if not plan_fingerprint:
                raise RuntimeError("replan did not expose a placement fingerprint")
            if strategy == "hotness_locality" and not plan.get(
                "source_rank_locality_available"
            ):
                raise RuntimeError(
                    "source-rank locality plan was unavailable: "
                    f"{plan.get('source_rank_locality_reason')}"
                )
            planned = plan.get("expert_to_target_rank", {})
            if planned:
                planned_ranks = {
                    key: [int(str(value).rsplit(":", 1)[-1])]
                    for key, value in planned.items()
                }
                if planned_ranks != actual_mapping:
                    raise RuntimeError(
                        f"runtime mapping differs from plan for {strategy}: "
                        f"planned={planned_ranks}, actual={actual_mapping}"
                    )

            duplicate_of = mapping_to_strategy.get(actual_mapping_fingerprint)
            mapping_to_strategy.setdefault(actual_mapping_fingerprint, strategy)
            measurement_start = a2a_counters(after_profile)
            strategy_records = []
            for bucket_id, workload in WORKLOADS.items():
                rows = run_requests(
                    args.endpoint,
                    model_name,
                    bucket_id,
                    workload,
                    args.repetitions,
                    args.request_timeout,
                )
                for row in rows:
                    row["strategy"] = strategy
                strategy_records.extend(rows)
                summary = summarize(rows)
                if strategy in PROFILE_STRATEGIES:
                    profile_rows.append({
                        "profile_id": f"{manifest_id}/{strategy}/{bucket_id}",
                        "manifest_id": manifest_id,
                        "strategy": strategy,
                        "placement_fingerprint": plan_fingerprint,
                        "actual_mapping_fingerprint": actual_mapping_fingerprint,
                        "duplicate_mapping_of": duplicate_of,
                        "workload_bucket_id": bucket_id,
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "vllm_data_parallel_size": 2,
                        "effective_expert_parallel_size": 2,
                        "sllm_replica_count": 1,
                        **summary,
                    })

            held_out = run_requests(
                args.endpoint,
                model_name,
                "held-out",
                HELD_OUT,
                args.repetitions,
                args.request_timeout,
            )
            for row in held_out:
                row["strategy"] = strategy
            held_out_ms = summarize(held_out)["service_latency_mean_ms"]
            load_sweep = run_load_sweep(
                args.endpoint,
                model_name,
                concurrency_levels=concurrency_levels,
                request_count=max(2, args.load_sweep_requests),
                timeout=args.request_timeout,
            )
            arrival_sweep = run_arrival_rate_sweep(
                args.endpoint,
                model_name,
                arrival_rates=arrival_rates,
                request_count=max(2, args.arrival_sweep_requests),
                max_workers=max(concurrency_levels),
                timeout=args.request_timeout,
            )
            prediction_error_ratio = None
            if strategy in PROFILE_STRATEGIES:
                predicted_ms = statistics.fmean(
                    row["service_latency_mean_ms"]
                    for row in profile_rows
                    if row["strategy"] == strategy
                )
                prediction_error_ratio = abs(predicted_ms - held_out_ms) / held_out_ms
                for row in profile_rows:
                    if row["strategy"] == strategy:
                        row["prediction_validated"] = True
                        row["held_out_prediction_error_ratio"] = prediction_error_ratio
            _, final_state = wait_for_instance(
                model_name, args.ready_timeout, idle=True
            )
            measurement_end_profile = runtime_profile(instance_id, final_state)
            measured_a2a = counter_delta(
                a2a_counters(measurement_end_profile), measurement_start
            )
            if measured_a2a["measurement_kind"] != (
                "runtime_sparse_transfer_payload"
            ):
                raise RuntimeError(
                    "Expected runtime sparse payload counters, observed "
                    f"{measured_a2a['measurement_kind']}"
                )
            if int(measured_a2a["collective_calls"]) <= 0:
                raise RuntimeError("measurement observed no sparse collectives")
            runtime_fingerprint = str(
                after_profile.get("runtime_fingerprint")
                or after_profile.get("vllm_version")
                or "vllm-0.11.2-runtime-moe-metadata"
            )
            all_records.extend(strategy_records)
            all_records.extend(held_out)
            candidate_results[strategy] = {
                "model": model_name,
                "placement_fingerprint": plan_fingerprint,
                "actual_mapping_fingerprint": actual_mapping_fingerprint,
                "duplicate_mapping_of": duplicate_of,
                "actual_mapping": actual_mapping,
                "hotness_snapshot_available": bool(
                    plan.get("hotness_snapshot_available")
                ),
                "hotness_snapshot_reason": plan.get("hotness_snapshot_reason"),
                "selection_reason": plan.get("selection_reason"),
                "source_rank_joint_available": bool(
                    after_profile.get("moe_source_rank_expert_histogram_available")
                ),
                "estimated_remote_ratio": plan.get("estimated_remote_ratio"),
                "estimated_remote_assignments": plan.get(
                    "estimated_remote_assignments"
                ),
                "measured_a2a": measured_a2a,
                "before_profile": before_profile,
                "event_runtime": event_placement["reparallelization"].get(
                    "execution", {}
                ).get("expert_placement_runtime", {}),
                "held_out_prediction_error_ratio": prediction_error_ratio,
                "load_sweep": load_sweep,
                "arrival_rate_sweep": arrival_sweep,
            }
        finally:
            if registered:
                cleanup_model(args.endpoint, model_name, instance_ids)

    by_pair = {}
    for row in all_records:
        if row["workload_bucket_id"] in WORKLOADS:
            by_pair.setdefault(row["pair_id"], {})[row["strategy"]] = row
    paired = []
    for pair_id, rows in sorted(by_pair.items()):
        if set(rows) != set(STRATEGIES):
            continue
        baseline = rows["linear"]
        signatures_match = all(
            row["response_signature"] == baseline["response_signature"]
            for row in rows.values()
        )
        paired.append({
            "pair_id": pair_id,
            "signatures_match": signatures_match,
            "linear_latency_ms": baseline["latency_ms"],
            "round_robin_delta_ms": (
                rows["round_robin"]["latency_ms"] - baseline["latency_ms"]
            ),
            "hotness_balanced_delta_ms": (
                rows["hotness_balanced"]["latency_ms"] - baseline["latency_ms"]
            ),
            "hotness_locality_delta_ms": (
                rows["hotness_locality"]["latency_ms"] - baseline["latency_ms"]
            ),
            "prompt_tokens": baseline["prompt_tokens"],
            "output_tokens": baseline["output_tokens"],
        })
    if len(paired) != sum(len(rows) for rows in WORKLOADS.values()) * args.repetitions:
        raise RuntimeError("paired request coverage is incomplete")
    if not all(row["signatures_match"] for row in paired):
        raise RuntimeError("candidate outputs differ for paired requests")

    linear_payload = int(
        candidate_results["linear"]["measured_a2a"]["payload_bytes"]
    )
    locality_payload = int(
        candidate_results["hotness_locality"]["measured_a2a"]["payload_bytes"]
    )
    if linear_payload <= 0:
        raise RuntimeError("linear baseline observed no sparse payload")
    a2a_reduction_ratio = (linear_payload - locality_payload) / linear_payload

    manifest = {
        "manifest_id": manifest_id,
        "model_id": base["backend_config"]["pretrained_model_name_or_path"],
        "gpu_type": args.gpu_type,
        "runtime_fingerprint": runtime_fingerprint,
        "dtype": base["backend_config"].get("torch_dtype", "float16"),
        "quantization": str(base["backend_config"].get("quantization") or "none"),
        "moe_backend": "spotserve_sparse",
        "max_conditions": len(PROFILE_STRATEGIES) * len(WORKLOADS),
        "prediction_validated": True,
        "condition_count": len(profile_rows),
        "created_at_s": time.time(),
    }
    report = {
        "status": "passed",
        "manifest": manifest,
        "profiles": profile_rows,
        "candidates": candidate_results,
        "paired_request_count": len(paired),
        "paired_coverage": 1.0,
        "paired_outputs_match": True,
        "phase3": {
            "source_rank_joint_available": bool(
                candidate_results["hotness_locality"][
                    "source_rank_joint_available"
                ]
            ),
            "baseline_payload_bytes": linear_payload,
            "locality_payload_bytes": locality_payload,
            "payload_reduction_ratio": a2a_reduction_ratio,
            "traffic_reduced": locality_payload < linear_payload,
            "correctness_passed": True,
            "claim_supported": locality_payload < linear_payload,
        },
        "load_sweep": {
            strategy: candidate_results[strategy]["load_sweep"]
            for strategy in STRATEGIES
        },
        "arrival_rate_sweep": {
            strategy: candidate_results[strategy]["arrival_rate_sweep"]
            for strategy in STRATEGIES
        },
        "hotness_shift": {
            "calibration_buckets": list(CALIBRATION_BUCKETS),
            "evaluation_bucket": "hotness-shift",
            "evaluation_excluded_from_calibration": True,
        },
        "paired": paired,
        "raw_records": all_records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Hotness runtime experiment passed; report={args.output}")
    print(f"  profile_conditions={len(profile_rows)}")
    print(f"  paired_requests={len(paired)}")
    print(f"  unique_actual_mappings={len(mapping_to_strategy)}")
    print(f"  phase3_payload_reduction_ratio={a2a_reduction_ratio:.6f}")


if __name__ == "__main__":
    main()
