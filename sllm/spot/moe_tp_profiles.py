"""Exact-workload pilot profile lookup with fail-closed transition evidence."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping


PROFILE_RUNTIME_FIELDS = (
    "config_sha256", "vllm_version", "torch_version", "dtype",
    "cpu_offload_gb", "routing_capture", "model_runner", "enforce_eager",
    "moe_backend", "enable_prefix_caching", "max_model_len", "max_num_seqs",
    "max_num_batched_tokens", "gpu_memory_utilization",
    "async_scheduling", "scheduler_cls",
)


def profile_runtime_signature(report: Mapping) -> str:
    if any(field not in report or report[field] is None
           for field in PROFILE_RUNTIME_FIELDS):
        raise ValueError("checkpoint/runtime settings are incomplete")
    revision = report.get("execution", {}).get("model_revision")
    if not revision:
        raise ValueError("checkpoint revision is unavailable; config alone is insufficient")
    payload = {field: report[field] for field in PROFILE_RUNTIME_FIELDS}
    payload.update({"tp": report.get("tp"), "model_revision": revision})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def route_conditioned_service_cost(generic: Mapping, conditioned: Mapping,
                                   query: Mapping) -> dict:
    """Use an independently calibrated, source-route-conditioned service cell.

    This measures total service for a route-conditioned workload, not per-expert
    queues, physical dispatch traffic, or arbitrary expert weight movement.
    No current-run/future completion time is substituted into a prediction.
    """
    if profile_runtime_signature(generic) != profile_runtime_signature(conditioned):
        raise ValueError("conditioned and generic profiles use different runtime/checkpoint")
    if generic.get("execution", {}).get("physical_gpu_indices") != conditioned.get(
        "execution", {}
    ).get("physical_gpu_indices"):
        raise ValueError("conditioned profile uses a different physical GPU group")
    condition = conditioned.get("route_conditioning", {})
    if (condition.get("kind") != "source_runtime_expert_routes"
            or condition.get("calibration_scope") != "independent_prior_calibration"
            or not condition.get("routing_artifact")):
        raise ValueError("independent runtime routing calibration is unavailable")
    for field in ("background_trace_sha256", "cache_policy_id"):
        if (not query.get(field)
                or generic.get("calibration_context", {}).get(field) != query[field]
                or conditioned.get("calibration_context", {}).get(field) != query[field]):
            raise ValueError(f"generic and conditioned calibration do not share {field}")
    for field in ("source_route_signature", "frozen_prefix_sha256",
                  "background_trace_sha256", "cache_policy_id"):
        if not query.get(field) or query[field] != condition.get(field):
            raise ValueError(f"route-conditioned observation does not match {field}")
    shape = (query["prompt_tokens"], query["output_tokens"], query["concurrency"])
    original_cost = service_cost_for_workload(generic, *shape)
    specific_cost = service_cost_for_workload(conditioned, *shape)
    hashes = query.get("prompt_sha256s", [])
    if len(hashes) != shape[2] or not all(isinstance(value, str) and value for value in hashes):
        raise ValueError("complete frozen batch prompt hashes are required")
    cell = next(cell for cell in conditioned["cells"] if (
        cell["prompt_tokens"], cell["max_new_tokens"], cell["concurrency"]
    ) == shape)
    if any(sorted(row.get("prompt_sha256", "") for row in sample["requests"])
           != sorted(hashes) for sample in cell["samples"]):
        raise ValueError("conditioned measurement prompts do not match the frozen batch")
    return {**specific_cost,
            "generic_latency_ms": original_cost["latency_estimate_ms"],
            "service_delta_ms": specific_cost["latency_estimate_ms"]
            - original_cost["latency_estimate_ms"],
            "routing_artifact": condition["routing_artifact"],
            "cost_definition": "measured_route_conditioned_service_delta_ms",
            "physical_dispatch_traffic_verified": False}


def audit_profile_log(log: str) -> dict:
    active = None
    completed = 0
    timed_warnings, untimed_warnings, errors = [], [], []
    key_fields = ("tp", "prompt_tokens", "concurrency", "repeat")
    for line in log.splitlines():
        if line.startswith("MOE_TP_PROFILE_EVENT="):
            try:
                value = json.loads(line.split("=", 1)[1])
            except (ValueError, TypeError):
                errors.append("malformed_profile_event")
                continue
            key = {field: value.get(field) for field in key_fields}
            if value.get("phase") == "measurement_start":
                if active is not None:
                    errors.append("nested_measurement_start")
                active = key
            elif value.get("phase") == "measurement_end":
                if active is None or key != active:
                    errors.append("unmatched_measurement_end")
                else:
                    completed += 1
                active = None
        if ("Triton kernel JIT compilation during inference:" in line
                or "Autotuning process starts" in line):
            item = {"message": line, "measurement": active}
            (timed_warnings if active is not None else untimed_warnings).append(item)
    if active is not None:
        errors.append("incomplete_measurement_window")
    if log.count("MOE_TP_PROFILE_JSON=") != 1:
        errors.append("missing_or_duplicate_final_profile")
    return {
        "complete": not errors and completed > 0,
        "completed_measurement_windows": completed,
        "timed_compile_or_autotune_warning_count": len(timed_warnings),
        "untimed_compile_or_autotune_warning_count": len(untimed_warnings),
        "timed_warnings": timed_warnings, "untimed_warnings": untimed_warnings,
        "errors": errors,
        "scope": "stdout_event_order_audit_not_a_kernel_profiler",
    }


def service_cost_for_workload(report: Mapping, prompt_tokens: int,
                              output_tokens: int, concurrency: int) -> dict:
    if report.get("status") != "passed" or report.get("scope") != (
        "warmed_service_cost_pilot_not_formal_ablation"
    ):
        raise ValueError("need a completed measured service profile")
    if report.get("runtime_moe_module_verified") is not True:
        raise ValueError("actual runtime MoE modules were not verified")
    if len(report.get("runtime_rank_checks", [])) != report.get("tp") or not all(
        row.get("granite_moe_model") is True and row.get("expert_modules") is True
        for row in report.get("runtime_rank_checks", [])
    ):
        raise ValueError("actual MoE/TP rank evidence is incomplete")
    audit = report.get("log_audit", {})
    if audit.get("complete") is not True or audit.get(
        "timed_compile_or_autotune_warning_count"
    ) != 0:
        raise ValueError("profile has incomplete or JIT-contaminated timing windows")
    expected_windows = sum(len(cell.get("samples", []))
                           for cell in report.get("cells", []))
    if audit.get("completed_measurement_windows") != expected_windows:
        raise ValueError("timing-window log does not cover the complete profile")
    matches = [cell for cell in report.get("cells", []) if (
        cell.get("prompt_tokens"), cell.get("max_new_tokens"), cell.get("concurrency")
    ) == (prompt_tokens, output_tokens, concurrency)]
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise ValueError("exact prompt/output/concurrency cell is unavailable")
    cell = matches[0]
    samples = cell.get("samples", [])
    if len(samples) < 2 or len(samples) != report.get("repeats"):
        raise ValueError("need repeated complete measurements")
    requests, wall_s = [], 0.0
    for sample in samples:
        wall = sample.get("batch_wall_s")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("invalid batch time")
        batch = sample.get("requests", [])
        if len(batch) != concurrency:
            raise ValueError("partial batch")
        for row in batch:
            latency = row.get("latency_s")
            if (row.get("generated_tokens") != output_tokens
                    or row.get("finish_reason") != "length"
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency) or latency <= 0):
                raise ValueError("failed output or invalid latency")
        requests.extend(batch)
        wall_s += wall
    return {
        "latency_estimate_ms": 1000 * sum(row["latency_s"] for row in requests)
        / len(requests),
        "throughput_estimate_req_s": len(requests) / wall_s,
        "measurement_batches": len(samples), "request_count": len(requests),
        "prompt_tokens": prompt_tokens, "max_new_tokens": output_tokens,
        "concurrency": concurrency,
        "capacity_scope": "closed_batch_not_open_loop_queue_or_slo_capacity",
    }


def measured_supported_configs(reports: Mapping[int, Mapping], prompt_tokens: int,
                               output_tokens: int, concurrency: int,
                               transition_evidence: Mapping[int, Mapping]) -> list:
    """Return scorer inputs only with explicit measured transition components.

    This does not infer TP actuation or claim open-loop SLO capacity. The caller
    must provide matching frozen-prefix and deployment evidence for transitions.
    """
    if set(reports) != {1, 2}:
        raise ValueError("both actual TP profiles are required")
    same_fields = PROFILE_RUNTIME_FIELDS
    if any(field not in reports[tp] or reports[tp][field] is None
           for field in same_fields for tp in (1, 2)):
        raise ValueError("checkpoint/runtime settings are incomplete")
    if any(reports[1].get(field) != reports[2].get(field) for field in same_fields):
        raise ValueError("TP profiles do not share checkpoint/runtime settings")
    signatures = {tp: profile_runtime_signature(reports[tp]) for tp in (1, 2)}
    if reports[1]["execution"]["model_revision"] != reports[2]["execution"]["model_revision"]:
        raise ValueError("TP profiles use different checkpoint revisions")
    candidates = []
    for tp in (1, 2):
        if reports[tp].get("tp") != tp:
            raise ValueError("profile TP does not match its candidate")
        cost = service_cost_for_workload(reports[tp], prompt_tokens, output_tokens,
                                         concurrency)
        evidence = transition_evidence.get(tp, {})
        if (evidence.get("verified") is not True or not evidence.get("artifact")
                or evidence.get("runtime_tp_verified") is not True
                or not evidence.get("frozen_prefix_sha256")):
            raise ValueError("missing measured transition evidence; never assume zero")
        if evidence.get("profile_runtime_signature") != signatures[tp]:
            raise ValueError("transition evidence does not match the profile runtime")
        if (not reports[tp].get("execution", {}).get("physical_gpu_indices")
                or evidence.get("physical_gpu_indices") != reports[tp][
                    "execution"]["physical_gpu_indices"]):
            raise ValueError("transition and profile use different physical GPU groups")
        if tuple(evidence.get("workload", [])) != (prompt_tokens, output_tokens,
                                                   concurrency):
            raise ValueError("transition evidence is for a different workload")
        components = {}
        for field in ("engine_ready_ms", "state_install_ms"):
            value = evidence.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError("unknown or invalid transition component")
            components[field] = value
        candidates.append({
            "tensor_parallel_size": tp, "pipeline_parallel_size": 1,
            "data_parallel_size": 1, "replica_count": 1, "num_gpus": tp,
            "enable_expert_parallel": False,
            "latency_estimate_ms": cost["latency_estimate_ms"],
            "throughput_estimate_req_s": cost["throughput_estimate_req_s"],
            "load_time_estimate_ms": components["engine_ready_ms"],
            "migration_cost_estimate_ms": components["state_install_ms"],
            "transition_artifact": evidence["artifact"],
            "capacity_scope": cost["capacity_scope"],
            "reason": "measured_exact_workload_tp_profile_with_transition_evidence",
        })
    if transition_evidence[1]["frozen_prefix_sha256"] != transition_evidence[2][
        "frozen_prefix_sha256"
    ]:
        raise ValueError("candidate transitions were not measured from the same prefix")
    return candidates
