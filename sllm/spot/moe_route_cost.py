"""Independent empirical route-conditioned remaining-service predictions.

Uses observed prefix routes, never future target completion times. This is a
small fixed-k nearest-neighbour pilot estimator, not expert locality, per-expert
timing, or arbitrary weight placement. Generic costs and transition costs are
shared by both versions; only neighbour selection uses expert distributions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics

from .moe_tp_profiles import profile_runtime_signature
from .reparallelization import ParallelPlan, plan_dynamic_reparallelization


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def route_vector(routes, layers=32, experts=40, top_k=8):
    histogram = routes.get("histogram", {})
    if (not histogram or routes.get("origin") != "live_frontend_routed_experts_chunks"
            or routes.get("histogram_sha256") != digest(histogram)):
        raise ValueError("independent live routing histogram/hash is unavailable")
    values = [0.0] * (layers * experts)
    totals = [0] * layers
    for key, count in histogram.items():
        try:
            a, b = key.split("/")
            if not a.startswith("layer:") or not b.startswith("expert:"):
                raise ValueError("invalid key")
            layer, expert = int(a[6:]), int(b[7:])
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed expert routing") from exc
        if (not 0 <= layer < layers or not 0 <= expert < experts
                or type(count) is not int or count <= 0):
            raise ValueError("expert route outside checkpoint bounds")
        values[layer * experts + expert] = count
        totals[layer] += count
    if len(set(totals)) != 1 or not totals[0] or totals[0] % top_k:
        raise ValueError("incomplete per-layer top-k observation")
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def validate_calibration(data):
    if data.get("status") != "passed" or data.get("scope") != "independent_prior_calibration_not_formal_runs":
        raise ValueError("a completed independent prior GPU calibration is required")
    protocol = data["protocol"]
    if data.get("protocol_sha256") != digest(protocol):
        raise ValueError("calibration protocol hash mismatch")
    if (protocol["initial_prompt_tokens"], protocol["source_output_tokens"],
            protocol["native_freeze_generated_tokens"], protocol["concurrency"]) != (4096, 512, 256, 1):
        raise ValueError("this pilot estimator only supports registered 4K C=1 256 boundary")
    training = sorted(key for key in data["source_cases"] if key.startswith("train-"))
    if len(training) < 6:
        raise ValueError("need at least six independent training workloads")
    if len({data["source_cases"][key]["frozen"]["full_prefix_sha256"] for key in training}) != len(training):
        raise ValueError("training workloads must have distinct actual prefixes")
    shared_runtime = None
    for candidate in data["candidates"].values():
        runtime = candidate["runtime"]
        if candidate["runtime_signature"] != profile_runtime_signature(runtime):
            raise ValueError("candidate runtime signature mismatch")
        if runtime["execution"]["model_revision"] != protocol["revision"]:
            raise ValueError("checkpoint revision differs from protocol")
        if runtime["execution"]["physical_gpu_indices"] != candidate["physical_gpu_indices"]:
            raise ValueError("physical group differs from actual calibration")
        signature = profile_runtime_signature(dict(runtime, tp=1))
        if shared_runtime is not None and shared_runtime != signature:
            raise ValueError("candidate runtime modes differ")
        shared_runtime = signature
        ready = candidate["ready_samples"]
        if len(ready) < 3 or len({r["artifact"] for r in ready}) != len(ready):
            raise ValueError("three independent engine transition artifacts are required")
        for row in ready:
            if row["runtime_signature"] != candidate["runtime_signature"]:
                raise ValueError("transition runtime differs")
            for field in ("ready_ms", "host_handoff_ms"):
                if type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError("unknown transition component cannot be zero-filled")
        for key in training:
            source = data["source_cases"][key]
            frozen = source["frozen"]
            if (len(frozen["tokens"]) != 4352 or frozen["remaining_tokens"] != 256
                    or frozen["completed_tokens"] != 256 or not frozen["strict_requested_boundary_verified"]
                    or frozen["full_prefix_sha256"] != digest(frozen["tokens"])):
                raise ValueError("training prefix/boundary mismatch")
            route_vector(source["routes"])
            samples = candidate["cases"][key]
            if len(samples) < 3 or len({r["request_id"] for r in samples}) != len(samples):
                raise ValueError("three distinct complete training measurements are required")
            for row in samples:
                if (not row["matches_uninterrupted_reference"] or row["timed_jit_warnings"]
                        or row["generated_tokens"] != 256 or row["finish_reason"] != "length"
                        or row["handoff_ack"]["frozen_prefix_sha256"] != frozen["full_prefix_sha256"]
                        or row["handoff_ack"]["install_mode"] != "host_token_handoff_excluding_gpu_prefill"
                        or not math.isfinite(row["latency_s"]) or row["latency_s"] <= 0):
                    raise ValueError("training output/timing/handoff evidence invalid")
    return training


def predict_candidates(data, source_case, signal="routes", seed=20260914, variable_shape=False):
    """Predict a held-out prefix; return estimates explicitly, never fake samples."""
    training = validate_calibration(data)
    if source_case.get("workload_id") in training:
        raise ValueError("evaluation workload leaked into training")
    if source_case["frozen"]["full_prefix_sha256"] in {
            data["source_cases"][key]["frozen"]["full_prefix_sha256"] for key in training}:
        raise ValueError("evaluation prefix leaked into training")
    frozen = source_case["frozen"]
    prefix_tokens, remaining = len(frozen["tokens"]), frozen["remaining_tokens"]
    if frozen["full_prefix_sha256"] != digest(frozen["tokens"]):
        raise ValueError("evaluation prefix hash differs from actual tokens")
    if variable_shape:
        if (not 4096 < prefix_tokens < 4608 or type(remaining) is not int or not 0 < remaining < 512
                or frozen["completed_tokens"] + remaining != 512
                or prefix_tokens != 4096 + frozen["completed_tokens"]):
            raise ValueError("variable shape is outside registered 4K incomplete-request bounds")
    elif (prefix_tokens, remaining, frozen["completed_tokens"]) != (4352, 256, 256):
        raise ValueError("evaluation workload differs from calibrated shape")
    query = route_vector(source_case["routes"])
    vectors = [route_vector(data["source_cases"][key]["routes"]) for key in training]
    if signal == "shuffled":
        random.Random(seed).shuffle(vectors)
    elif signal not in {"routes", "removed"}:
        raise ValueError("unknown route-signal diagnostic")
    neighbours = sorted((math.sqrt(sum((a - b) ** 2 for a, b in zip(query, vector))), key)
                        for vector, key in zip(vectors, training))
    exact = [(distance, key) for distance, key in neighbours if distance < 1e-12]
    chosen = exact or neighbours[:3]
    weights = [(key, 1.0 if exact else 1.0 / max(distance, 1e-6)) for distance, key in chosen]
    total = sum(weight for _, weight in weights)
    predictions = {}
    for cid, candidate in data["candidates"].items():
        means = {key: 1000 * statistics.mean(row["latency_s"] for row in candidate["cases"][key])
                 for key in training}
        generic = statistics.mean(means.values())
        conditioned = generic if signal == "removed" else sum(means[key] * weight for key, weight in weights) / total
        if variable_shape:
            ttfts = {key: 1000 * statistics.mean(row["ttft_s"] for row in candidate["cases"][key])
                     for key in training}
            steps = {key: (means[key] - ttfts[key]) / 255 for key in training}
            if any(step <= 0 or not math.isfinite(step) for step in steps.values()):
                raise ValueError("invalid calibrated decode-step slope")
            def scaled(ids):
                denom = sum(weight for _, weight in ids)
                first = sum(ttfts[key] * weight for key, weight in ids) / denom
                step = sum(steps[key] * weight for key, weight in ids) / denom
                return first * prefix_tokens / 4352 + step * (remaining - 1)
            generic = scaled([(key, 1) for key in training])
            conditioned = generic if signal == "removed" else scaled(weights)
        predictions[cid] = {"candidate_id": cid, "tensor_parallel_size": candidate["tp"],
            "physical_gpu_indices": candidate["physical_gpu_indices"],
            "profile_runtime_signature": candidate["runtime_signature"],
            "generic_service_ms": generic, "route_service_ms": conditioned,
            "engine_ready_ms": statistics.mean(row["ready_ms"] for row in candidate["ready_samples"]),
            "host_handoff_ms": statistics.mean(row["host_handoff_ms"] for row in candidate["ready_samples"]),
            "nearest_training_workloads": [{"id": key, "distance": distance} for distance, key in chosen],
            "estimate_definition": ("prefix_scaled_TTFT_plus_linear_decode_step_from_prior_routes"
                                    if variable_shape else "prior_prefix_route_kNN_full_prefill_plus_remaining_decode"),
            "variable_shape_prediction_requires_independent_shape_validation": variable_shape,
            "physical_dispatch_traffic_verified": False,
            "measured_current_or_future_run_latency_used": False}
    return predictions


def plan_calibrated_recovery(data, source_case, kind, moe_aware, available=(1, 2, 3),
                             resident_groups=(), signal="routes"):
    if kind not in {"migration", "reparallelization", "overall"}:
        raise ValueError("unknown comparison kind")
    predictions = predict_candidates(data, source_case, signal=signal if moe_aware else "removed",
                                     variable_shape=kind == "overall")
    group_set = {tuple(group) for group in resident_groups}
    allowed = ({(1,), (2,), (3,)} if kind == "migration" else
               {(1,), (2,), (3,), (1, 3)} if kind == "overall" else {(1,), (1, 3)})
    rows = []
    for row in predictions.values():
        group = tuple(row["physical_gpu_indices"])
        if group not in allowed or not set(group) <= set(available):
            continue
        cost = row["route_service_ms"] if moe_aware else row["generic_service_ms"]
        ready = 0.0 if group in group_set else row["engine_ready_ms"]
        rows.append({**row, "resident_engine_verified": group in group_set,
                     "engine_ready_cost_ms": ready, "selected_service_ms": cost,
                     "predicted_completion_ms": ready + row["host_handoff_ms"] + cost})
    if not rows:
        raise ValueError("no calibrated physical GPU plan is available")
    selected = min(rows, key=lambda row: (row["predicted_completion_ms"], row["candidate_id"]))
    planner = None
    if kind != "migration":
        # The TP scorer accepts configurations, not distinct physical groups
        # with identical TP. Compare each TP's best legal physical group first;
        # otherwise deduplication could overwrite TP1 with a more costly board.
        best_by_tp = {}
        for row in rows:
            tp = row["tensor_parallel_size"]
            if tp not in best_by_tp or row["predicted_completion_ms"] < best_by_tp[tp]["predicted_completion_ms"]:
                best_by_tp[tp] = row
        candidates = [{"tensor_parallel_size": row["tensor_parallel_size"],
            "pipeline_parallel_size": 1, "data_parallel_size": 1, "replica_count": 1,
            "enable_expert_parallel": False, "num_gpus": row["tensor_parallel_size"],
            "latency_estimate_ms": row["selected_service_ms"], "throughput_estimate_req_s": 0,
            "load_time_estimate_ms": row["engine_ready_cost_ms"],
            "migration_cost_estimate_ms": row["host_handoff_ms"]} for row in best_by_tp.values()]
        planner = plan_dynamic_reparallelization("granite-moe",
            {f"gpu-{g}": {"free_gpu": 1, "total_gpu": 1, "state": "ready"} for g in available},
            model_config={"backend": "vllm", "num_gpus": 1,
                          "backend_capability": {"supported_configs": candidates}},
            planner_config={"enable_workload_cost_model": True, "disable_logical_expert_placement": True,
                "min_tensor_parallel_size": 1, "max_tensor_parallel_size": 2,
                "batch_size": 1, "arrival_rate_req_s": 0, "base_score_weight": 0,
                "throughput_score_weight": 0, "latency_penalty_weight": 1,
                "load_time_penalty_weight": 1, "migration_cost_penalty_weight": 1,
                "queue_penalty_weight": 0, "expert_weight_movement_penalty_weight": 0,
                "replan_window_penalty_weight": 0}, event="calibrated_forced_recovery", backend="vllm")
        selected = min((row for row in rows if row["tensor_parallel_size"] == planner["selected_tensor_parallel_size"]),
                       key=lambda row: (row["predicted_completion_ms"], row["candidate_id"]))
    tp, group = selected["tensor_parallel_size"], selected["physical_gpu_indices"]
    plan = ParallelPlan("granite-moe", "vllm", tp, 1, num_gpus=tp,
                        target_nodes=[f"gpu-{g}" for g in group], reason="prior_route_calibrated_recovery")
    return {"kind": kind, "moe_aware": moe_aware, "selected_candidate": selected["candidate_id"],
            "selected_cost": selected, "candidates": rows, "parallel_plan": plan.to_dict(),
            "planner_decision": planner, "frozen_prefix_sha256": source_case["frozen"]["full_prefix_sha256"],
            "baseline_definition": "shared_empirical_generic_mean_without_expert_route_neighbours",
            "calibration_protocol_sha256": data["protocol_sha256"],
            "configuration_applied": False, "formal_experiment_eligible": False}


def validate_held_out_predictions(data):
    rows = []
    for key in data["protocol"]["validation_ids"]:
        source = data["source_cases"][key]
        predicted = {signal: predict_candidates(data, source, signal)
                     for signal in ("removed", "routes", "shuffled")}
        for cid, candidate in data["candidates"].items():
            samples = candidate["cases"][key]
            actual = 1000 * statistics.mean(row["latency_s"] for row in samples)
            rows.append({"workload_id": key, "candidate": cid, "actual_service_ms": actual,
                         **{signal + "_prediction_ms": predicted[signal][cid]["route_service_ms"]
                            for signal in predicted}})
    mae = {signal: statistics.mean(abs(row[signal + "_prediction_ms"] - row["actual_service_ms"])
                                   for row in rows) for signal in ("removed", "routes", "shuffled")}
    return {"scope": "held_out_forecast_diagnostic_not_optimization_speedup", "rows": rows,
            "mean_absolute_error_ms": mae,
            "routes_better_than_generic_on_this_validation": mae["routes"] < mae["removed"],
            "causal_moe_specific_benefit_established": False,
            "validation_does_not_tune_k_or_force_plan_difference": True}
