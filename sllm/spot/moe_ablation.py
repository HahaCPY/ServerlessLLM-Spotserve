"""Measured, fail-closed planning bridge for the next single-event ablations.

Both versions share generic calibration and the completion-time objective.
Only the MoE-aware version uses independent source-route conditioning. This
does not implement expert weight placement, acquire GPUs, or certify a formal
experiment. The legacy recovery harness is deliberately left unchanged.
"""

from __future__ import annotations

import math
from typing import Mapping

from .moe_tp_profiles import (
    measured_supported_configs, profile_runtime_signature,
    route_conditioned_service_cost, service_cost_for_workload,
)
from .reparallelization import ParallelPlan, plan_dynamic_reparallelization


def _gpu_group(values, tp):
    if (not isinstance(values, list) or len(values) != tp
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != tp):
        raise ValueError("incomplete or duplicate physical GPU group")
    return list(values)


def _transition_cost(evidence, profile, query):
    if (evidence.get("verified") is not True
            or evidence.get("runtime_tp_verified") is not True
            or not evidence.get("artifact")
            or evidence.get("calibration_scope") != "independent_prior_calibration"
            or evidence.get("cost_partition") != "ready_then_install_then_warmed_service"):
        raise ValueError("independent, non-overlapping transition evidence is required")
    shape = [query[key] for key in ("prompt_tokens", "output_tokens", "concurrency")]
    if evidence.get("workload") != shape:
        raise ValueError("transition evidence is for a different workload")
    if evidence.get("profile_runtime_signature") != profile_runtime_signature(profile):
        raise ValueError("transition evidence uses a different runtime")
    if evidence.get("physical_gpu_indices") != profile["execution"]["physical_gpu_indices"]:
        raise ValueError("transition evidence uses a different physical GPU group")
    for key in ("frozen_prefix_sha256", "background_trace_sha256", "cache_policy_id"):
        if not query.get(key) or evidence.get(key) != query[key]:
            raise ValueError(f"transition evidence does not match {key}")
    samples = evidence.get("samples", [])
    if len(samples) < 3:
        raise ValueError("need at least three complete transition measurements")
    if len({row.get("artifact") for row in samples}) != len(samples):
        raise ValueError("transition repetitions must have distinct artifacts")
    result = dict(evidence)
    for field in ("engine_ready_ms", "state_install_ms"):
        values = []
        for sample in samples:
            value = sample.get(field)
            if (sample.get("verified") is not True or not sample.get("artifact")
                    or sample.get("frozen_prefix_sha256") != query["frozen_prefix_sha256"]
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("unknown or invalid repeated transition component")
            values.append(value)
        result[field] = sum(values) / len(values)
    return result


def _require_three_service_batches(profile, query):
    cell = next(cell for cell in profile["cells"] if (
        cell["prompt_tokens"], cell["max_new_tokens"], cell["concurrency"]
    ) == (query["prompt_tokens"], query["output_tokens"], query["concurrency"]))
    repeats = [sample.get("repeat") for sample in cell["samples"]]
    if (len(repeats) < 3 or any(type(value) is not int for value in repeats)
            or len(set(repeats)) != len(repeats)):
        raise ValueError("need at least three complete, distinct service measurements")


def plan_measured_ablation(kind: str, calibrations: Mapping, query: Mapping,
                           available_gpu_indices: list[int], moe_aware: bool,
                           model_name: str = "granite-moe") -> dict:
    """Plan one frozen batch after a source becomes unavailable.

    ``migration`` requires three independent TP=1 targets; ``reparallelization``
    requires one TP=1 and one TP=2 shape, one replica, no EP. Candidate GPU IDs
    are local physical indices, NOT Ray placement node IDs. A deployment adapter
    must bind these exact groups and provide independent runtime evidence.
    """
    if kind not in {"migration", "reparallelization"}:
        raise ValueError("unknown ablation kind")
    if query.get("source_unavailable") is not True:
        raise ValueError("this bridge is for forced recovery, not optional keep/replan")
    for key in ("prompt_tokens", "output_tokens", "concurrency"):
        if type(query.get(key)) is not int or query[key] <= 0:
            raise ValueError("positive exact workload dimensions are required")
    available = _gpu_group(available_gpu_indices, len(available_gpu_indices))
    source = _gpu_group(query.get("source_physical_gpu_indices"), 1)
    if set(source) & set(available) or len(available) > 3:
        raise ValueError("source must be outside the at-most-three-GPU target pool")
    if len(calibrations) != (3 if kind == "migration" else 2):
        raise ValueError("incomplete preregistered candidate set")
    rows, generic_profiles, transitions = {}, {}, {}
    shared_signature = None
    for candidate_id, bundle in calibrations.items():
        profile = bundle["profile"]
        tp = profile.get("tp")
        if type(tp) is not int or tp not in (1, 2) or (kind == "migration" and tp != 1):
            raise ValueError("only verified TP=1/2 shapes are allowed")
        group = _gpu_group(profile.get("execution", {}).get("physical_gpu_indices"), tp)
        if set(group) & set(source):
            raise ValueError("a target calibration includes the source GPU")
        normalized = dict(profile, tp=1)
        signature = profile_runtime_signature(normalized)
        if shared_signature is not None and signature != shared_signature:
            raise ValueError("candidates do not share checkpoint/runtime settings")
        shared_signature = signature
        if profile.get("calibration_scope") != "independent_prior_calibration":
            raise ValueError("service calibration must precede the evaluated run")
        context = profile.get("calibration_context", {})
        for key in ("background_trace_sha256", "cache_policy_id"):
            if not query.get(key) or context.get(key) != query[key]:
                raise ValueError(f"generic calibration does not match {key}")
        cost = service_cost_for_workload(profile, query["prompt_tokens"],
                                         query["output_tokens"], query["concurrency"])
        _require_three_service_batches(profile, query)
        evidence = _transition_cost(bundle.get("transition", {}), profile, query)
        specific = None
        if moe_aware:
            specific = route_conditioned_service_cost(
                profile, bundle.get("conditioned_profile", {}), query)
            _require_three_service_batches(bundle["conditioned_profile"], query)
            cost = specific
        rows[candidate_id] = {
            "candidate_id": candidate_id, "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1, "data_parallel_size": 1,
            "replica_count": 1, "enable_expert_parallel": False, "num_gpus": tp,
            "physical_gpu_indices": group,
            "profile_runtime_signature": profile_runtime_signature(profile),
            "latency_estimate_ms": cost["latency_estimate_ms"],
            "throughput_estimate_req_s": cost["throughput_estimate_req_s"],
            "load_time_estimate_ms": evidence["engine_ready_ms"],
            "migration_cost_estimate_ms": evidence["state_install_ms"],
            "predicted_completion_ms": evidence["engine_ready_ms"]
                + evidence["state_install_ms"] + cost["latency_estimate_ms"],
            "route_conditioned_cost": specific,
            "eligible": set(group) <= set(available),
            "transition_artifact": evidence["artifact"],
        }
        generic_profiles[candidate_id] = profile
        transitions[candidate_id] = evidence
    if kind == "migration":
        groups = [row["physical_gpu_indices"][0] for row in rows.values()]
        if len(set(groups)) != 3:
            raise ValueError("migration needs three independent physical targets")
    else:
        by_tp = {row["tensor_parallel_size"]: key for key, row in rows.items()}
        if set(by_tp) != {1, 2}:
            raise ValueError("both TP=1 and TP=2 measured shapes are required")
        # Reuse the exact-profile guard, including same-prefix checks.
        measured_supported_configs(
            {tp: generic_profiles[key] for tp, key in by_tp.items()},
            query["prompt_tokens"], query["output_tokens"], query["concurrency"],
            {tp: transitions[key] for tp, key in by_tp.items()})
    eligible = [row for row in rows.values() if row["eligible"]]
    if not eligible:
        raise ValueError("no calibrated physical GPU group is available")
    planner_decision = None
    if kind == "reparallelization":
        nodes = {f"gpu-{gpu}": {"total_gpu": 1, "free_gpu": 1, "state": "ready"}
                 for gpu in available}
        planner_decision = plan_dynamic_reparallelization(
            model_name, nodes,
            model_config={"backend": "vllm", "num_gpus": 1,
                          "backend_capability": {"supported_configs": eligible}},
            planner_config={
                "enable_workload_cost_model": True,
                "disable_logical_expert_placement": True,
                "min_tensor_parallel_size": 1, "max_tensor_parallel_size": 2,
                "batch_size": query["concurrency"], "arrival_rate_req_s": 0,
                "base_score_weight": 0, "throughput_score_weight": 0,
                "latency_penalty_weight": 1, "load_time_penalty_weight": 1,
                "migration_cost_penalty_weight": 1, "queue_penalty_weight": 0,
                "expert_weight_movement_penalty_weight": 0,
                "replan_window_penalty_weight": 0,
            }, event="measured_forced_recovery", backend="vllm")
        tp = planner_decision["selected_tensor_parallel_size"]
        selected = next(row for row in eligible if row["tensor_parallel_size"] == tp)
        # Abstract worker nodes cannot express a calibrated multi-GPU group.
        # Bind it explicitly; never silently run the first two available GPUs.
        planner_decision["parallel_plan"]["target_nodes"] = [
            f"gpu-{gpu}" for gpu in selected["physical_gpu_indices"]]
        plan = planner_decision["parallel_plan"]
    else:
        selected = min(eligible, key=lambda row: (row["predicted_completion_ms"],
                                                 str(row["candidate_id"])))
        plan = ParallelPlan(model_name, "vllm", 1, 1, num_gpus=1,
                            target_nodes=[f"gpu-{selected['physical_gpu_indices'][0]}"],
                            reason="measured_migration_target_selection").to_dict()
    return {
        "kind": kind, "moe_aware": moe_aware,
        "baseline_definition": "shared_measured_cost_baseline_without_route_conditioning",
        "objective": "ready_plus_install_plus_mean_remaining_completion_ms",
        "selected_candidate": selected["candidate_id"], "selected_cost": selected,
        "candidates": list(rows.values()), "parallel_plan": plan,
        "planner_decision": planner_decision,
        "frozen_prefix_sha256": query["frozen_prefix_sha256"],
        "configuration_applied": False, "formal_experiment_eligible": False,
        "scope": "single_frozen_batch_planning_not_runtime_or_spot_churn_evidence",
        "logical_expert_movement": "not_used_engine_loading_already_measured",
    }


def verify_measured_runtime(decision: Mapping, runtime: Mapping) -> dict:
    """Check independent worker observations before traffic switches.

    The caller must obtain these fields from workers, not echo the requested
    plan. Passing this check alone does not certify delivery or experiment gates.
    """
    selected = decision["selected_cost"]
    plan = decision["parallel_plan"]
    tp = selected["tensor_parallel_size"]
    if (plan["tensor_parallel_size"] != tp or plan["num_gpus"] != tp
            or plan["target_nodes"] != [f"gpu-{gpu}" for gpu in selected["physical_gpu_indices"]]
            or plan["replica_count"] != 1 or plan["pipeline_parallel_size"] != 1
            or plan["data_parallel_size"] != 1 or plan["enable_expert_parallel"]):
        raise ValueError("deployment plan no longer matches measured selection")
    if (not runtime.get("artifact")
            or runtime.get("observation_source") != "actual_worker_runtime"
            or runtime.get("profile_runtime_signature") != selected["profile_runtime_signature"]
            or runtime.get("physical_gpu_indices") != selected["physical_gpu_indices"]
            or runtime.get("tensor_parallel_size") != tp
            or runtime.get("pipeline_parallel_size") != 1
            or runtime.get("data_parallel_size") != 1
            or runtime.get("replica_count") != 1
            or runtime.get("enable_expert_parallel") is not False
            or runtime.get("state_install_verified") is not True
            or runtime.get("frozen_prefix_sha256") != decision["frozen_prefix_sha256"]):
        raise ValueError("actual deployment does not match runtime/group/state evidence")
    ranks = runtime.get("ranks", [])
    if (len(ranks) != tp or {row.get("rank") for row in ranks} != set(range(tp))
            or sorted(row.get("physical_gpu_index", -1) for row in ranks)
            != sorted(selected["physical_gpu_indices"])
            or not all(row.get("granite_moe_model") is True
                       and row.get("expert_modules") is True for row in ranks)):
        raise ValueError("actual MoE/TP rank evidence is incomplete")
    return {"configuration_applied": True, "runtime_artifact": runtime["artifact"],
            "formal_experiment_eligible": False,
            "remaining_gates": ["prefix_delivery_and_output_correctness",
                                "four_gpu_environment", "paired_three_run_protocol"]}
