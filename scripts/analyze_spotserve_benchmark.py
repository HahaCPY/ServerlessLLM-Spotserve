import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_trace_replay(
    run_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    trace_status = read_json(run_dir / "trace_status.json")
    if not trace_status:
        trace_status = {
            key: metadata[key]
            for key in (
                "trace_replay_started",
                "trace_replay_success",
                "trace_replay_failed",
                "trace_replay_error",
                "trace_replay_exit_code",
                "trace_replay_log_path",
            )
            if key in metadata
        }
    return {
        "trace_replay_status_available": 1 if trace_status else 0,
        "trace_replay_started": int(
            bool(trace_status.get("trace_replay_started", False))
        ),
        "trace_replay_success": int(
            bool(trace_status.get("trace_replay_success", False))
        ),
        "trace_replay_failed": int(
            bool(trace_status.get("trace_replay_failed", False))
        ),
        "trace_replay_exit_code": safe_int(
            trace_status.get("trace_replay_exit_code"), 0
        ),
        "trace_replay_error": str(
            trace_status.get("trace_replay_error", "") or ""
        ),
        "trace_replay_log_path": str(
            trace_status.get("trace_replay_log_path", "") or ""
        ),
    }


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        max(0, round((pct / 100.0) * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


def summarize_requests(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    successes = [row for row in rows if row.get("success")]
    failed = len(rows) - len(successes)
    first_sent = min((row.get("sent_at", 0.0) for row in rows), default=0.0)
    last_done = max((row.get("completed_at", 0.0) for row in rows), default=0.0)
    elapsed = max(last_done - first_sent, 0.0)
    return {
        "requests": len(rows),
        "successes": len(successes),
        "failures": failed,
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "latency_avg_ms": mean(latencies) if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "throughput_req_s": len(rows) / elapsed if elapsed > 0 else 0.0,
    }


def safe_metric_name(value: Any) -> str:
    text = str(value).strip().lower()
    chars = []
    previous_underscore = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        elif not previous_underscore:
            chars.append("_")
            previous_underscore = True
    name = "".join(chars).strip("_")
    return name or "unknown"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_values(values: List[str]) -> str:
    unique = sorted({value for value in values if value})
    return ",".join(unique)


def merge_int_maps(values: List[Any]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            merged[str(key)] = (
                merged.get(str(key), 0) + safe_int(count, 0)
            )
    return merged


def summarize_phase_requests(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    rows_by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        phase = row.get("benchmark_phase")
        if not phase:
            continue
        rows_by_phase.setdefault(safe_metric_name(phase), []).append(row)

    summary: Dict[str, Any] = {}
    for phase, phase_rows in sorted(rows_by_phase.items()):
        phase_summary = summarize_requests(phase_rows)
        for key, value in phase_summary.items():
            summary[f"phase_{phase}_{key}"] = value
    return summary


def resolve_optional_path(
    path_value: Any, run_dir: Path
) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        run_dir / path,
        run_dir.parents[1] / path if len(run_dir.parents) > 1 else path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def matching_router_request_metrics(
    metric_rows: List[Dict[str, Any]], request_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    request_ids = {
        row.get("request_id") for row in request_rows if row.get("request_id")
    }
    if not request_ids:
        return []

    first_sent = min(
        (float(row.get("sent_at", 0.0)) for row in request_rows),
        default=0.0,
    )
    last_done = max(
        (float(row.get("completed_at", 0.0)) for row in request_rows),
        default=0.0,
    )
    window_start = first_sent - 5.0
    window_end = last_done + 5.0

    latest_by_request: Dict[str, Dict[str, Any]] = {}
    for row in metric_rows:
        if row.get("type") != "request":
            continue
        request_id = row.get("request_id")
        if request_id not in request_ids:
            continue
        timestamp = float(row.get("timestamp", 0.0))
        if first_sent and not (window_start <= timestamp <= window_end):
            continue
        previous = latest_by_request.get(request_id)
        if previous is None or timestamp >= float(
            previous.get("timestamp", 0.0)
        ):
            latest_by_request[request_id] = row

    return sorted(
        latest_by_request.values(),
        key=lambda row: str(row.get("request_id", "")),
    )


def metrics_in_request_window(
    metric_rows: List[Dict[str, Any]], request_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not request_rows:
        return []

    first_sent = min(
        (float(row.get("sent_at", 0.0)) for row in request_rows),
        default=0.0,
    )
    last_done = max(
        (float(row.get("completed_at", 0.0)) for row in request_rows),
        default=0.0,
    )
    if not first_sent or not last_done:
        return []

    window_start = first_sent - 5.0
    window_end = last_done + 5.0
    return [
        row
        for row in metric_rows
        if window_start <= float(row.get("timestamp", 0.0)) <= window_end
    ]


def summarize_router_metrics(
    rows: List[Dict[str, Any]],
    request_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    request_denominator = (
        len(request_rows) if request_rows is not None else len(rows)
    )
    router_successes = sum(1 for row in rows if bool(row.get("success", False)))
    failed_attempts_total = sum(
        safe_int(row.get("failed_attempts", 0)) for row in rows
    )
    retry_count_total = sum(
        safe_int(row.get("retry_count", 0)) for row in rows
    )
    recovered_tokens_total = sum(
        safe_int(row.get("recovered_tokens", 0)) for row in rows
    )
    recovery_fallback_count = sum(
        1 for row in rows if bool(row.get("recovery_fallback", False))
    )
    recovery_triggered_requests = sum(
        1
        for row in rows
        if safe_int(row.get("failed_attempts", 0)) > 0
        or safe_int(row.get("retry_count", 0)) > 0
    )
    replay_succeeded_requests = sum(
        1
        for row in rows
        if safe_int(row.get("recovered_tokens", 0)) > 0
        and not bool(row.get("recovery_fallback", False))
    )
    replay_not_needed_requests = sum(
        1
        for row in rows
        if safe_int(row.get("failed_attempts", 0)) == 0
        and safe_int(row.get("recovered_tokens", 0)) == 0
        and not bool(row.get("recovery_fallback", False))
    )
    state_restore_attempts_total = sum(
        safe_int(row.get("state_restore_attempts", 0)) for row in rows
    )
    state_restore_successes_total = sum(
        safe_int(row.get("state_restore_successes", 0)) for row in rows
    )
    state_restore_fallback_count = sum(
        1 for row in rows if bool(row.get("state_restore_fallback", False))
    )
    fallback_request_count = sum(
        1
        for row in rows
        if bool(row.get("recovery_fallback", False))
        or bool(row.get("state_restore_fallback", False))
    )
    clean_success_count = sum(
        1
        for row in rows
        if bool(row.get("success", False))
        and not bool(row.get("recovery_fallback", False))
        and not bool(row.get("state_restore_fallback", False))
    )
    state_restored_tokens_total = sum(
        safe_int(row.get("state_restored_tokens", 0)) for row in rows
    )
    state_restored_blocks_total = sum(
        safe_int(row.get("state_restored_blocks", 0)) for row in rows
    )
    supports_state_restore_requests = sum(
        1 for row in rows if bool(row.get("supports_state_restore", False))
    )
    state_restore_staged_count = sum(
        1 for row in rows if bool(row.get("state_restore_staged", False))
    )
    state_restore_durations = [
        safe_float(row.get("state_restore_duration_ms"), 0.0)
        for row in rows
        if safe_float(row.get("state_restore_duration_ms"), 0.0) > 0.0
    ]
    state_restore_finished_times = [
        safe_float(row.get("state_restore_finished_at_s"), 0.0)
        for row in rows
        if safe_float(row.get("state_restore_finished_at_s"), 0.0) > 0.0
    ]
    true_kv_restore_successes_total = sum(
        1
        for row in rows
        if safe_int(row.get("state_restore_successes", 0)) > 0
        and safe_int(row.get("state_restored_blocks", 0)) > 0
        and str(row.get("state_kind", "")) == "vllm_kv_snapshot"
    )
    restore_attempt_request_count = sum(
        1 for row in rows if safe_int(row.get("state_restore_attempts", 0)) > 0
    )
    return {
        "router_metrics_rows": len(rows),
        "router_successes": router_successes,
        "clean_success_count": clean_success_count,
        "clean_success_denominator": request_denominator,
        "clean_success_rate": (
            clean_success_count / request_denominator
            if request_denominator
            else 0.0
        ),
        "router_clean_success_rate": (
            clean_success_count / len(rows) if rows else 0.0
        ),
        "fallback_request_count": fallback_request_count,
        "fallback_denominator": request_denominator,
        "fallback_rate": (
            fallback_request_count / request_denominator
            if request_denominator
            else 0.0
        ),
        "router_fallback_rate": (
            fallback_request_count / len(rows) if rows else 0.0
        ),
        "failed_attempts_total": failed_attempts_total,
        "retry_count_total": retry_count_total,
        "recovered_tokens_total": recovered_tokens_total,
        "recovery_fallback_count": recovery_fallback_count,
        "recovery_fallback_rate": (
            recovery_fallback_count / len(rows) if rows else 0.0
        ),
        "recovery_triggered_requests": recovery_triggered_requests,
        "recovery_triggered_rate": (
            recovery_triggered_requests / len(rows) if rows else 0.0
        ),
        "replay_succeeded_requests": replay_succeeded_requests,
        "replay_not_needed_requests": replay_not_needed_requests,
        "state_restore_attempts_total": state_restore_attempts_total,
        "state_restore_successes_total": state_restore_successes_total,
        "state_restore_success_rate": (
            state_restore_successes_total / state_restore_attempts_total
            if state_restore_attempts_total
            else 0.0
        ),
        "state_restore_fallback_count": state_restore_fallback_count,
        "state_restore_fallback_rate": (
            state_restore_fallback_count / restore_attempt_request_count
            if restore_attempt_request_count
            else 0.0
        ),
        "state_restored_tokens_total": state_restored_tokens_total,
        "state_restored_blocks_total": state_restored_blocks_total,
        "supports_state_restore_requests": supports_state_restore_requests,
        "state_restore_staged_count": state_restore_staged_count,
        "state_restore_avg_duration_ms": (
            mean(state_restore_durations) if state_restore_durations else 0.0
        ),
        "state_restore_max_duration_ms": (
            max(state_restore_durations) if state_restore_durations else 0.0
        ),
        "state_restore_last_finished_at_s": (
            max(state_restore_finished_times)
            if state_restore_finished_times
            else 0.0
        ),
        "true_kv_restore_successes_total": true_kv_restore_successes_total,
        "true_kv_restore_rate": (
            true_kv_restore_successes_total / state_restore_attempts_total
            if state_restore_attempts_total
            else 0.0
        ),
        "state_kinds": compact_values(
            [str(row.get("state_kind", "")) for row in rows]
        ),
        "state_restore_reasons": compact_values(
            [str(row.get("state_restore_reason", "")) for row in rows]
        ),
    }


def summarize_response_kv_restore(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    restore_rows = []
    for row in rows:
        response = row.get("response")
        if not isinstance(response, dict):
            continue
        restore = response.get("_spotserve_kv_restore")
        if isinstance(restore, dict):
            restore_rows.append(restore)

    response_restored_blocks = sum(
        safe_int(row.get("restored_blocks", 0)) for row in restore_rows
    )
    response_cached_tokens = sum(
        safe_int(row.get("cached_tokens", 0)) for row in restore_rows
    )
    response_restore_successes = sum(
        1
        for row in restore_rows
        if bool(row.get("restored", False))
        and safe_int(row.get("restored_blocks", 0)) > 0
    )
    return {
        "response_kv_restore_events": len(restore_rows),
        "response_kv_restore_successes": response_restore_successes,
        "response_kv_restore_restored_blocks": response_restored_blocks,
        "response_kv_restore_cached_tokens": response_cached_tokens,
        "response_kv_restore_reasons": compact_values(
            [str(row.get("reason", "")) for row in restore_rows]
        ),
    }


def summarize_true_kv_restore_evidence(
    request_rows: List[Dict[str, Any]],
    router_request_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    restored_blocks_by_request: Dict[str, int] = {}
    sources = set()

    for index, row in enumerate(router_request_rows):
        restored_blocks = safe_int(row.get("state_restored_blocks", 0))
        if (
            safe_int(row.get("state_restore_successes", 0)) <= 0
            or restored_blocks <= 0
            or str(row.get("state_kind", "")) != "vllm_kv_snapshot"
        ):
            continue
        request_id = str(row.get("request_id") or f"router-{index}")
        restored_blocks_by_request[request_id] = max(
            restored_blocks_by_request.get(request_id, 0),
            restored_blocks,
        )
        sources.add("router_metrics")

    for index, row in enumerate(request_rows):
        response = row.get("response")
        if not isinstance(response, dict):
            continue
        restore = response.get("_spotserve_kv_restore")
        if not isinstance(restore, dict):
            continue
        restored_blocks = safe_int(restore.get("restored_blocks", 0))
        if not bool(restore.get("restored", False)) or restored_blocks <= 0:
            continue
        request_id = str(
            row.get("request_id")
            or response.get("id")
            or f"response-{index}"
        )
        restored_blocks_by_request[request_id] = max(
            restored_blocks_by_request.get(request_id, 0),
            restored_blocks,
        )
        sources.add("raw_response")

    return {
        "true_kv_restore_successes_total": len(restored_blocks_by_request),
        "true_kv_restored_blocks_total": sum(
            restored_blocks_by_request.values()
        ),
        "true_kv_restore_evidence_sources": compact_values(
            [str(source) for source in sources]
        ),
    }


def summarize_instance_state_metrics(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    instance_rows = [
        row for row in rows if row.get("type") == "instance_state"
    ]

    def count_to(state: str) -> int:
        return sum(1 for row in instance_rows if row.get("to") == state)

    preemption_notice_rows = [
        row
        for row in instance_rows
        if row.get("to") == "preempting"
        and safe_float(row.get("notice_time_s"), 0.0) > 0.0
    ]
    deadline_rows = [
        row
        for row in instance_rows
        if safe_float(row.get("deadline_time_s"), 0.0) > 0.0
    ]
    auto_dead_rows = [
        row
        for row in instance_rows
        if row.get("to") == "dead" and bool(row.get("auto_deadline", False))
    ]
    grace_periods = [
        safe_float(row.get("grace_period_s"), 0.0)
        for row in deadline_rows
        if safe_float(row.get("grace_period_s"), 0.0) > 0.0
    ]
    notice_to_deadline_ms = [
        (
            safe_float(row.get("deadline_time_s"), 0.0)
            - safe_float(row.get("notice_time_s"), 0.0)
        )
        * 1000
        for row in deadline_rows
        if safe_float(row.get("deadline_time_s"), 0.0) > 0.0
        and safe_float(row.get("notice_time_s"), 0.0) > 0.0
    ]
    auto_dead_lag_ms = [
        (
            safe_float(row.get("timestamp"), 0.0)
            - safe_float(row.get("deadline_time_s"), 0.0)
        )
        * 1000
        for row in auto_dead_rows
        if safe_float(row.get("deadline_time_s"), 0.0) > 0.0
        and safe_float(row.get("timestamp"), 0.0) > 0.0
    ]

    return {
        "instance_state_rows": len(instance_rows),
        "instances_marked_preempting": count_to("preempting"),
        "instances_marked_ready": count_to("ready"),
        "instances_marked_dead": count_to("dead"),
        "instances_marked_draining": count_to("draining"),
        "preemption_notice_events": len(preemption_notice_rows),
        "preemption_deadline_events": len(deadline_rows),
        "preemption_auto_deadline_events": len(auto_dead_rows),
        "preemption_avg_grace_period_s": (
            mean(grace_periods) if grace_periods else 0.0
        ),
        "preemption_avg_notice_to_deadline_ms": (
            mean(notice_to_deadline_ms)
            if notice_to_deadline_ms
            else 0.0
        ),
        "preemption_max_auto_deadline_lag_ms": (
            max(auto_dead_lag_ms) if auto_dead_lag_ms else 0.0
        ),
    }


def summarize_replanning_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    replanning_rows = [
        row for row in rows if row.get("type") == "reparallelization"
    ]

    def numeric_values(key: str) -> List[float]:
        return [
            safe_float(row.get(key), 0.0)
            for row in replanning_rows
            if row.get(key) is not None and row.get(key) != ""
        ]

    def average(values: List[float]) -> float:
        return mean(values) if values else 0.0

    no_capacity_count = sum(
        1 for row in replanning_rows if row.get("action") == "no_capacity"
    )
    selected_total_gpus = [
        int(row.get("selected_total_gpus", 0) or 0)
        for row in replanning_rows
    ]
    latest_plan = (
        replanning_rows[-1].get("parallel_plan") if replanning_rows else None
    )
    execution_statuses = [
        str(row.get("execution_status", "") or "")
        for row in replanning_rows
        if row.get("execution_status")
    ]
    execution_models = [
        str(row.get("reparallelization_execution_model", "") or "")
        for row in replanning_rows
        if row.get("reparallelization_execution_model")
    ]
    expert_execution_models = [
        str(row.get("expert_placement_execution_model", "") or "")
        for row in replanning_rows
        if row.get("expert_placement_execution_model")
    ]
    expert_contract_modes = [
        str(row.get("expert_placement_runtime_contract_mode", "") or "")
        for row in replanning_rows
        if row.get("expert_placement_runtime_contract_mode")
    ]
    latest_execution = (
        replanning_rows[-1].get("execution") if replanning_rows else None
    )
    request_migrations = [
        row.get("execution", {}).get("request_migration", {})
        for row in replanning_rows
        if isinstance(row.get("execution"), dict)
        and isinstance(
            row.get("execution", {}).get("request_migration"), dict
        )
    ]
    execution_durations = numeric_values("execution_duration_ms")
    replan_window_costs = numeric_values("selected_replan_window_cost_ms")
    load_time_costs = numeric_values("selected_load_time_estimate_ms")
    migration_costs = numeric_values(
        "selected_migration_cost_estimate_ms"
    )
    selected_expert_weight_movement_costs = numeric_values(
        "selected_expert_weight_movement_cost_estimate_ms"
    )
    queue_penalties = numeric_values("selected_queue_penalty_ms")
    throughput_estimates = numeric_values(
        "selected_throughput_estimate_req_s"
    )
    state_export_durations = [
        safe_float(row.get("state_export_duration_ms"), 0.0)
        for row in request_migrations
        if safe_float(row.get("state_export_duration_ms"), 0.0) > 0.0
    ]
    state_export_finished_times = [
        safe_float(row.get("state_export_finished_at_s"), 0.0)
        for row in request_migrations
        if safe_float(row.get("state_export_finished_at_s"), 0.0) > 0.0
    ]
    target_worker_node_counts = [
        safe_int(row.get("target_worker_node_count"), 0)
        for row in replanning_rows
    ]
    ready_worker_node_counts = [
        safe_int(row.get("ready_worker_node_count"), 0)
        for row in replanning_rows
    ]
    selected_sllm_replica_counts = [
        safe_int(
            row.get(
                "selected_sllm_replica_count",
                row.get("selected_replica_count", 0),
            ),
            0,
        )
        for row in replanning_rows
    ]
    selected_vllm_dp_sizes = [
        safe_int(
            row.get(
                "selected_vllm_data_parallel_size",
                row.get("selected_data_parallel_size", 0),
            ),
            0,
        )
        for row in replanning_rows
    ]
    selected_effective_ep_sizes = [
        safe_int(
            row.get(
                "selected_runtime_effective_expert_parallel_size",
                row.get("selected_effective_expert_parallel_size", 0),
            ),
            0,
        )
        for row in replanning_rows
    ]
    expert_plan_shards = [
        safe_int(row.get("expert_placement_plan_shards"), 0)
        for row in replanning_rows
    ]
    expert_plan_required = [
        safe_int(row.get("expert_placement_plan_required_experts"), 0)
        for row in replanning_rows
    ]
    expert_plan_covered = [
        safe_int(row.get("expert_placement_plan_covered_experts"), 0)
        for row in replanning_rows
    ]
    expert_plan_moved = [
        safe_int(row.get("expert_placement_plan_moved_experts"), 0)
        for row in replanning_rows
    ]
    expert_plan_stationary = [
        safe_int(row.get("expert_placement_plan_stationary_experts"), 0)
        for row in replanning_rows
    ]
    expert_plan_unknown_movement = [
        safe_int(
            row.get("expert_placement_plan_unknown_movement_experts"),
            0,
        )
        for row in replanning_rows
    ]
    expert_plan_moved_weight_bytes = [
        safe_int(row.get("expert_placement_plan_moved_weight_bytes"), 0)
        for row in replanning_rows
    ]
    expert_plan_weight_movement_costs = numeric_values(
        "expert_placement_plan_estimated_weight_movement_cost_ms"
    )
    expert_plan_coverage_ratios = [
        covered / required
        for covered, required in zip(
            expert_plan_covered, expert_plan_required
        )
        if required > 0
    ]
    return {
        "replanning_events": len(replanning_rows),
        "replanning_no_capacity_events": no_capacity_count,
        "replanning_max_selected_gpus": (
            max(selected_total_gpus) if selected_total_gpus else 0
        ),
        "replanning_latest_plan": (
            json.dumps(latest_plan, sort_keys=True) if latest_plan else ""
        ),
        "replanning_execution_applied": execution_statuses.count("applied"),
        "replanning_execution_failed": execution_statuses.count("failed"),
        "replanning_execution_models": compact_values(execution_models),
        "replanning_expert_placement_execution_models": compact_values(
            expert_execution_models
        ),
        "replanning_expert_placement_contract_modes": compact_values(
            expert_contract_modes
        ),
        "replanning_expert_placement_actor_recreate_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_actor_recreate")
            or row.get("expert_placement_execution_model")
            == "expert_aware_actor_recreate"
        ),
        "replanning_expert_placement_live_migration_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_live_migration")
            or row.get("expert_placement_execution_model")
            == "live_expert_weight_migration"
        ),
        "replanning_expert_placement_physical_migration_required_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_physical_migration_required")
        ),
        "replanning_latest_execution": (
            json.dumps(latest_execution, sort_keys=True)
            if latest_execution
            else ""
        ),
        "replanning_workload_cost_model_events": sum(
            1
            for row in replanning_rows
            if row.get("workload_cost_model_enabled")
        ),
        "replanning_avg_execution_duration_ms": average(
            execution_durations
        ),
        "replanning_max_execution_duration_ms": (
            max(execution_durations) if execution_durations else 0.0
        ),
        "replanning_avg_selected_replan_window_cost_ms": average(
            replan_window_costs
        ),
        "replanning_max_selected_replan_window_cost_ms": (
            max(replan_window_costs) if replan_window_costs else 0.0
        ),
        "replanning_avg_selected_load_time_estimate_ms": average(
            load_time_costs
        ),
        "replanning_avg_selected_migration_cost_estimate_ms": average(
            migration_costs
        ),
        "replanning_avg_selected_expert_weight_movement_cost_estimate_ms": (
            average(selected_expert_weight_movement_costs)
        ),
        "replanning_avg_selected_queue_penalty_ms": average(
            queue_penalties
        ),
        "replanning_avg_selected_throughput_estimate_req_s": average(
            throughput_estimates
        ),
        "replanning_state_export_events": sum(
            safe_int(row.get("state_exported"), 0)
            for row in request_migrations
        ),
        "replanning_avg_state_export_duration_ms": average(
            state_export_durations
        ),
        "replanning_max_state_export_duration_ms": (
            max(state_export_durations) if state_export_durations else 0.0
        ),
        "replanning_last_state_export_finished_at_s": (
            max(state_export_finished_times)
            if state_export_finished_times
            else 0.0
        ),
        "replanning_cross_node_targets": sum(
            1 for row in replanning_rows if row.get("cross_node_target")
        ),
        "replanning_multi_worker_targets": sum(
            1 for row in replanning_rows if row.get("multi_worker_target")
        ),
        "replanning_max_target_worker_node_count": (
            max(target_worker_node_counts)
            if target_worker_node_counts
            else 0
        ),
        "replanning_max_ready_worker_node_count": (
            max(ready_worker_node_counts) if ready_worker_node_counts else 0
        ),
        "replanning_max_selected_sllm_replica_count": (
            max(selected_sllm_replica_counts)
            if selected_sllm_replica_counts
            else 0
        ),
        "replanning_max_selected_vllm_data_parallel_size": (
            max(selected_vllm_dp_sizes) if selected_vllm_dp_sizes else 0
        ),
        "replanning_max_selected_effective_expert_parallel_size": (
            max(selected_effective_ep_sizes)
            if selected_effective_ep_sizes
            else 0
        ),
        "replanning_moe_route_histogram_available_events": sum(
            1
            for row in replanning_rows
            if row.get("moe_route_histogram_available")
        ),
        "replanning_moe_route_histogram_sources": compact_values(
            [
                str(row.get("moe_route_histogram_source", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_moe_route_histogram_kinds": compact_values(
            [
                str(row.get("moe_route_histogram_kind", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_moe_expert_parallel_size_sources": compact_values(
            [
                str(row.get("moe_expert_parallel_size_source", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_plan_available_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_plan_available")
        ),
        "replanning_expert_placement_plan_epochs": compact_values(
            [
                str(row.get("expert_placement_plan_epoch", ""))
                for row in replanning_rows
                if row.get("expert_placement_plan_epoch") not in (None, "")
            ]
        ),
        "replanning_expert_placement_plan_fingerprints": compact_values(
            [
                str(row.get("expert_placement_plan_fingerprint", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_plan_sources": compact_values(
            [
                str(row.get("expert_placement_plan_source", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_plan_reasons": compact_values(
            [
                str(row.get("expert_placement_plan_reason", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_plan_physical_migration_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_plan_physical_weight_migration")
        ),
        "replanning_expert_placement_plan_movement_observation_events": sum(
            1
            for row in replanning_rows
            if row.get("expert_placement_plan_movement_observation_available")
        ),
        "replanning_expert_placement_plan_movement_sources": compact_values(
            [
                str(row.get("expert_placement_plan_movement_source", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_total_expert_placement_plan_moved_experts": sum(
            expert_plan_moved
        ),
        "replanning_max_expert_placement_plan_moved_experts": (
            max(expert_plan_moved) if expert_plan_moved else 0
        ),
        "replanning_total_expert_placement_plan_stationary_experts": sum(
            expert_plan_stationary
        ),
        "replanning_total_expert_placement_plan_unknown_movement_experts": (
            sum(expert_plan_unknown_movement)
        ),
        "replanning_total_expert_placement_plan_moved_weight_bytes": sum(
            expert_plan_moved_weight_bytes
        ),
        "replanning_avg_expert_placement_plan_weight_movement_cost_ms": (
            average(expert_plan_weight_movement_costs)
        ),
        "replanning_expert_placement_runtime_metadata_count": sum(
            safe_int(
                row.get("expert_placement_runtime_metadata_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_apply_hook_available": sum(
            safe_int(
                row.get(
                    "expert_placement_runtime_apply_hook_available_count"
                ),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_apply_attempted": sum(
            safe_int(
                row.get("expert_placement_runtime_apply_attempted_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_apply_success": sum(
            safe_int(
                row.get("expert_placement_runtime_apply_success_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_apply_reasons": compact_values(
            [
                str(row.get("expert_placement_runtime_apply_reasons", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_runtime_verify_hook_available": sum(
            safe_int(
                row.get(
                    "expert_placement_runtime_verify_hook_available_count"
                ),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_verify_attempted": sum(
            safe_int(
                row.get("expert_placement_runtime_verify_attempted_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_verify_success": sum(
            safe_int(
                row.get("expert_placement_runtime_verify_success_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_verify_reasons": compact_values(
            [
                str(row.get("expert_placement_runtime_verify_reasons", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_runtime_plan_applied": sum(
            safe_int(
                row.get("expert_placement_runtime_plan_applied_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_plan_verified": sum(
            safe_int(
                row.get("expert_placement_runtime_plan_verified_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_contract_reasons": compact_values(
            [
                str(row.get("expert_placement_runtime_contract_reasons", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_runtime_execution_models": (
            compact_values(
                [
                    str(
                        row.get(
                            "expert_placement_runtime_execution_models", ""
                        )
                    )
                    for row in replanning_rows
                ]
            )
        ),
        "replanning_expert_placement_runtime_contract_modes": compact_values(
            [
                str(row.get("expert_placement_runtime_contract_modes", ""))
                for row in replanning_rows
            ]
        ),
        "replanning_expert_placement_runtime_live_migration": sum(
            safe_int(
                row.get("expert_placement_runtime_live_migration_count"),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_expert_placement_runtime_physical_migration_required": sum(
            safe_int(
                row.get(
                    "expert_placement_runtime_physical_migration_required_count"
                ),
                0,
            )
            for row in replanning_rows
        ),
        "replanning_max_expert_placement_plan_required_experts": (
            max(expert_plan_required) if expert_plan_required else 0
        ),
        "replanning_max_expert_placement_plan_covered_experts": (
            max(expert_plan_covered) if expert_plan_covered else 0
        ),
        "replanning_max_expert_placement_plan_shards": (
            max(expert_plan_shards) if expert_plan_shards else 0
        ),
        "replanning_avg_expert_placement_plan_coverage_ratio": average(
            expert_plan_coverage_ratios
        ),
        "replanning_max_synthetic_worker_node_count": max(
            (
                safe_int(row.get("synthetic_worker_node_count"), 0)
                for row in replanning_rows
            ),
            default=0,
        ),
        "replanning_max_runtime_worker_node_count": max(
            (
                safe_int(
                    row.get(
                        "runtime_worker_node_count",
                        row.get("physical_worker_node_count", 0),
                    ),
                    0,
                )
                for row in replanning_rows
            ),
            default=0,
        ),
        "replanning_max_physical_worker_node_count": max(
            (
                safe_int(row.get("physical_worker_node_count"), 0)
                for row in replanning_rows
            ),
            default=0,
        ),
    }


def summarize_context_migration_metrics(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    migration_rows = [
        row for row in rows if row.get("type") == "context_migration"
    ]
    total_plan_count = sum(
        int(row.get("migration_plan_count", 0) or 0)
        for row in migration_rows
    )
    total_unassigned = sum(
        int(row.get("unassigned_context_count", 0) or 0)
        for row in migration_rows
    )
    total_estimated_cost = sum(
        float(row.get("total_estimated_cost", 0.0) or 0.0)
        for row in migration_rows
    )
    total_reusable_tokens = sum(
        int(row.get("total_reusable_tokens", 0) or 0)
        for row in migration_rows
    )
    total_context_tokens = sum(
        int(row.get("total_context_tokens", 0) or 0)
        for row in migration_rows
    )
    total_reusable_blocks = sum(
        int(row.get("total_reusable_context_blocks", 0) or 0)
        for row in migration_rows
    )
    total_context_blocks = sum(
        int(row.get("total_context_blocks", 0) or 0)
        for row in migration_rows
    )
    reuse_denominator = total_context_blocks or total_context_tokens
    reuse_numerator = (
        total_reusable_blocks if total_context_blocks else total_reusable_tokens
    )
    latest_event = migration_rows[-1] if migration_rows else {}
    latest_plans = latest_event.get("plans", []) if latest_event else []
    latest_selected_target_ids = latest_event.get("selected_target_ids", [])
    if not isinstance(latest_selected_target_ids, list):
        latest_selected_target_ids = []
    if not latest_selected_target_ids and isinstance(latest_plans, list):
        latest_selected_target_ids = [
            str(plan.get("new_instance_id", ""))
            for plan in latest_plans
            if isinstance(plan, dict) and plan.get("new_instance_id")
        ]
    latest_selected_request_ids = latest_event.get("selected_request_ids", [])
    if not isinstance(latest_selected_request_ids, list):
        latest_selected_request_ids = []
    if not latest_selected_request_ids and isinstance(latest_plans, list):
        latest_selected_request_ids = [
            str(plan.get("request_id", ""))
            for plan in latest_plans
            if isinstance(plan, dict) and plan.get("request_id")
        ]
    latest_target_placement_epochs = latest_event.get(
        "selected_plan_target_placement_epochs", []
    )
    if not isinstance(latest_target_placement_epochs, list):
        latest_target_placement_epochs = []
    latest_target_placement_fingerprints = latest_event.get(
        "selected_plan_target_expert_placement_fingerprints", []
    )
    if not isinstance(latest_target_placement_fingerprints, list):
        latest_target_placement_fingerprints = []
    latest_target_plan_fingerprints = latest_event.get(
        "selected_plan_target_expert_placement_plan_fingerprints", []
    )
    if not isinstance(latest_target_plan_fingerprints, list):
        latest_target_plan_fingerprints = []
    latest_target_snapshot_fingerprints = latest_event.get(
        "selected_plan_target_expert_placement_snapshot_fingerprints", []
    )
    if not isinstance(latest_target_snapshot_fingerprints, list):
        latest_target_snapshot_fingerprints = []
    latest_target_placement_sources = latest_event.get(
        "selected_plan_target_placement_sources", []
    )
    if not isinstance(latest_target_placement_sources, list):
        latest_target_placement_sources = []
    kv_rows = [
        row.get("kv_cache_migration") or {}
        for row in migration_rows
        if row.get("kv_cache_migration")
    ]
    kv_attempted = sum(int(row.get("attempted", 0) or 0) for row in kv_rows)
    kv_succeeded = sum(int(row.get("succeeded", 0) or 0) for row in kv_rows)
    kv_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in kv_rows)
    latest_kv = kv_rows[-1] if kv_rows else None
    moe_dispatch_costs = [
        safe_float(row.get("moe_estimated_dispatch_cost"), 0.0)
        for row in migration_rows
        if row.get("moe_estimated_dispatch_cost") is not None
    ]
    moe_locality_ratios = [
        safe_float(row.get("moe_hot_expert_locality_ratio"), 0.0)
        for row in migration_rows
        if row.get("moe_hot_expert_locality_ratio") is not None
    ]
    moe_remote_routing_ratios = [
        safe_float(row.get("moe_estimated_remote_routing_ratio"), 0.0)
        for row in migration_rows
        if row.get("moe_estimated_remote_routing_ratio") is not None
    ]
    moe_dispatch_rows = [
        row
        for row in migration_rows
        if safe_int(row.get("moe_dispatch_observation_available_count"), 0)
        > 0
    ]
    moe_remote_dispatch_ratios = [
        safe_float(row.get("moe_remote_routing_ratio"), 0.0)
        for row in moe_dispatch_rows
    ]
    moe_local_by_layer = merge_int_maps([
        row.get("moe_local_routed_tokens_by_layer", {})
        for row in moe_dispatch_rows
    ])
    moe_remote_by_layer = merge_int_maps([
        row.get("moe_remote_routed_tokens_by_layer", {})
        for row in moe_dispatch_rows
    ])
    moe_local_by_expert = merge_int_maps([
        row.get("moe_local_routed_tokens_by_expert", {})
        for row in moe_dispatch_rows
    ])
    moe_remote_by_expert = merge_int_maps([
        row.get("moe_remote_routed_tokens_by_expert", {})
        for row in moe_dispatch_rows
    ])
    queue_pressures = [
        safe_float(row.get("avg_queue_pressure"), 0.0)
        for row in migration_rows
        if row.get("avg_queue_pressure") is not None
    ]
    return {
        "context_migration_events": len(migration_rows),
        "context_migration_plan_count": total_plan_count,
        "context_migration_unassigned_count": total_unassigned,
        "context_migration_total_estimated_cost": total_estimated_cost,
        "context_migration_avg_estimated_cost": (
            total_estimated_cost / total_plan_count if total_plan_count else 0.0
        ),
        "context_migration_selected_target_ids": compact_values(
            [str(value) for value in latest_selected_target_ids]
        ),
        "context_migration_selected_request_ids": compact_values(
            [str(value) for value in latest_selected_request_ids]
        ),
        "context_migration_selected_plan_total_estimated_cost": sum(
            safe_float(row.get("selected_plan_total_estimated_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_kv_migration_cost": sum(
            safe_float(row.get("selected_plan_kv_migration_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_expert_dispatch_cost": sum(
            safe_float(row.get("selected_plan_expert_dispatch_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_queue_penalty_cost": sum(
            safe_float(row.get("selected_plan_queue_penalty_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_avg_queue_pressure": (
            mean([
                safe_float(
                    row.get("selected_plan_avg_queue_pressure"), 0.0
                )
                for row in migration_rows
            ])
            if migration_rows
            else 0.0
        ),
        "context_migration_selected_plan_max_queue_depth": max(
            (
                safe_int(row.get("selected_plan_max_queue_depth"), 0)
                for row in migration_rows
            ),
            default=0,
        ),
        "context_migration_selected_plan_avg_hot_expert_locality_ratio": (
            mean([
                safe_float(
                    row.get(
                        "selected_plan_avg_hot_expert_locality_ratio"
                    ),
                    0.0,
                )
                for row in migration_rows
            ])
            if migration_rows
            else 0.0
        ),
        "context_migration_selected_plan_avg_estimated_remote_routing_ratio": (
            mean([
                safe_float(
                    row.get(
                        "selected_plan_avg_estimated_remote_routing_ratio"
                    ),
                    0.0,
                )
                for row in migration_rows
            ])
            if migration_rows
            else 0.0
        ),
        "context_migration_selected_plan_estimated_remote_routed_tokens": sum(
            safe_int(
                row.get("selected_plan_estimated_remote_routed_tokens"), 0
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_placement_epochs": compact_values(
            [str(value) for value in latest_target_placement_epochs]
        ),
        "context_migration_selected_target_expert_placement_fingerprints": (
            compact_values([
                str(value)
                for value in latest_target_placement_fingerprints
            ])
        ),
        "context_migration_selected_target_expert_placement_plan_fingerprints": (
            compact_values([
                str(value) for value in latest_target_plan_fingerprints
            ])
        ),
        "context_migration_selected_target_expert_placement_snapshot_fingerprints": (
            compact_values([
                str(value) for value in latest_target_snapshot_fingerprints
            ])
        ),
        "context_migration_selected_target_expert_placement_contracts": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_contract_available_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_plan_applied": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_plan_applied_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_plan_verified": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_plan_verified_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_contract_reasons": (
            compact_values([
                str(reason)
                for row in migration_rows
                for reason in (
                    row.get(
                        "selected_plan_target_expert_placement_contract_reasons",
                        [],
                    )
                    or []
                )
            ])
        ),
        "context_migration_selected_target_expert_placement_apply_hook_available": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_apply_hook_available_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_apply_attempted": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_apply_attempted_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_apply_success": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_apply_success_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_apply_reasons": (
            compact_values([
                str(reason)
                for row in migration_rows
                for reason in (
                    row.get(
                        "selected_plan_target_expert_placement_apply_reasons",
                        [],
                    )
                    or []
                )
            ])
        ),
        "context_migration_selected_target_expert_placement_verify_hook_available": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_verify_hook_available_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_verify_attempted": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_verify_attempted_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_verify_success": sum(
            safe_int(
                row.get(
                    "selected_plan_target_expert_placement_verify_success_count"
                ),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_selected_target_expert_placement_verify_reasons": (
            compact_values([
                str(reason)
                for row in migration_rows
                for reason in (
                    row.get(
                        "selected_plan_target_expert_placement_verify_reasons",
                        [],
                    )
                    or []
                )
            ])
        ),
        "context_migration_selected_target_placement_sources": compact_values(
            [str(value) for value in latest_target_placement_sources]
        ),
        "context_migration_placement_handshake_attempts": sum(
            safe_int(row.get("placement_handshake_attempts"), 0)
            for row in migration_rows
        ),
        "context_migration_placement_handshake_successes": sum(
            safe_int(row.get("placement_handshake_successes"), 0)
            for row in migration_rows
        ),
        "context_migration_placement_handshake_failures": sum(
            safe_int(row.get("placement_handshake_failures"), 0)
            for row in migration_rows
        ),
        "context_migration_placement_handshake_stale": sum(
            safe_int(row.get("placement_handshake_stale"), 0)
            for row in migration_rows
        ),
        "context_migration_placement_handshake_reasons": compact_values([
            str(reason)
            for row in migration_rows
            for reason in (
                row.get("placement_handshake_reasons", [])
                if isinstance(row.get("placement_handshake_reasons"), list)
                else []
            )
        ]),
        "context_migration_selected_plan_moe_routed_tokens": sum(
            safe_int(row.get("selected_plan_moe_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_moe_local_routed_tokens": sum(
            safe_int(row.get("selected_plan_moe_local_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_moe_remote_routed_tokens": sum(
            safe_int(row.get("selected_plan_moe_remote_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_selected_plan_moe_avg_remote_routing_ratio": (
            mean([
                safe_float(
                    row.get(
                        "selected_plan_moe_avg_remote_routing_ratio"
                    ),
                    0.0,
                )
                for row in moe_dispatch_rows
            ])
            if moe_dispatch_rows
            else 0.0
        ),
        "context_migration_selected_plan_moe_locality_definitions": (
            compact_values([
                str(value)
                for row in migration_rows
                for value in (
                    row.get("selected_plan_moe_locality_definitions", [])
                    if isinstance(
                        row.get("selected_plan_moe_locality_definitions"),
                        list,
                    )
                    else []
                )
            ])
        ),
        "context_migration_selected_plan_moe_locality_granularities": (
            compact_values([
                str(value)
                for row in migration_rows
                for value in (
                    row.get("selected_plan_moe_locality_granularities", [])
                    if isinstance(
                        row.get("selected_plan_moe_locality_granularities"),
                        list,
                    )
                    else []
                )
            ])
        ),
        "context_migration_selected_plan_moe_remote_routing_definitions": (
            compact_values([
                str(value)
                for row in migration_rows
                for value in (
                    row.get("selected_plan_moe_remote_routing_definitions", [])
                    if isinstance(
                        row.get(
                            "selected_plan_moe_remote_routing_definitions"
                        ),
                        list,
                    )
                    else []
                )
            ])
        ),
        "context_migration_context_source_count": max(
            (
                safe_int(row.get("context_source_count"), 0)
                for row in migration_rows
            ),
            default=0,
        ),
        "context_migration_context_target_count": max(
            (
                safe_int(row.get("context_target_count"), 0)
                for row in migration_rows
            ),
            default=0,
        ),
        "context_migration_candidate_component_cost_events": sum(
            1
            for row in migration_rows
            if row.get("candidate_component_costs")
        ),
        "context_migration_reusable_tokens": total_reusable_tokens,
        "context_migration_reusable_context_blocks": total_reusable_blocks,
        "context_migration_reuse_ratio": (
            reuse_numerator / reuse_denominator if reuse_denominator else 0.0
        ),
        "context_migration_kv_migration_cost": sum(
            safe_float(row.get("kv_migration_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_queue_penalty_cost": sum(
            safe_float(row.get("queue_penalty_cost"), 0.0)
            for row in migration_rows
        ),
        "context_migration_avg_queue_pressure": (
            mean(queue_pressures) if queue_pressures else 0.0
        ),
        "context_migration_max_queue_depth": max(
            (
                safe_int(row.get("max_queue_depth"), 0)
                for row in migration_rows
            ),
            default=0,
        ),
        "context_migration_moe_route_histogram_available_count": sum(
            safe_int(row.get("moe_route_histogram_available_count"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_target_placement_available_count": sum(
            safe_int(row.get("moe_target_placement_available_count"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_route_histogram_sources": compact_values(
            [
                str(row.get("moe_route_histogram_source", ""))
                for row in migration_rows
            ]
        ),
        "context_migration_moe_route_histogram_kinds": compact_values(
            [
                str(row.get("moe_route_histogram_kind", ""))
                for row in migration_rows
            ]
        ),
        "context_migration_moe_avg_hot_expert_locality_ratio": (
            mean(moe_locality_ratios) if moe_locality_ratios else 0.0
        ),
        "context_migration_moe_avg_estimated_remote_routing_ratio": (
            mean(moe_remote_routing_ratios)
            if moe_remote_routing_ratios
            else 0.0
        ),
        "context_migration_moe_estimated_remote_routed_tokens": sum(
            safe_int(row.get("moe_estimated_remote_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_estimated_dispatch_cost": sum(
            moe_dispatch_costs
        ),
        "context_migration_moe_dispatch_observation_available_count": sum(
            safe_int(row.get("moe_dispatch_observation_available_count"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_routed_tokens": sum(
            safe_int(row.get("moe_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_local_routed_tokens": sum(
            safe_int(row.get("moe_local_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_remote_routed_tokens": sum(
            safe_int(row.get("moe_remote_routed_tokens"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_avg_remote_routing_ratio": (
            mean(moe_remote_dispatch_ratios)
            if moe_remote_dispatch_ratios
            else 0.0
        ),
        "context_migration_moe_locality_definitions": compact_values(
            [
                str(row.get("moe_locality_definition", ""))
                for row in migration_rows
            ]
        ),
        "context_migration_moe_locality_granularities": compact_values(
            [
                str(row.get("moe_locality_granularity", ""))
                for row in migration_rows
            ]
        ),
        "context_migration_moe_remote_routing_definitions": compact_values(
            [
                str(row.get("moe_remote_routing_definition", ""))
                for row in migration_rows
            ]
        ),
        "context_migration_moe_rank_locality_available_count": sum(
            safe_int(row.get("moe_rank_locality_available_count"), 0)
            for row in migration_rows
        ),
        "context_migration_moe_physical_dispatch_traffic_available_count": sum(
            safe_int(
                row.get("moe_physical_dispatch_traffic_available_count"),
                0,
            )
            for row in migration_rows
        ),
        "context_migration_moe_local_routed_tokens_by_layer": (
            json.dumps(moe_local_by_layer, sort_keys=True)
            if moe_local_by_layer
            else ""
        ),
        "context_migration_moe_remote_routed_tokens_by_layer": (
            json.dumps(moe_remote_by_layer, sort_keys=True)
            if moe_remote_by_layer
            else ""
        ),
        "context_migration_moe_local_routed_tokens_by_expert": (
            json.dumps(moe_local_by_expert, sort_keys=True)
            if moe_local_by_expert
            else ""
        ),
        "context_migration_moe_remote_routed_tokens_by_expert": (
            json.dumps(moe_remote_by_expert, sort_keys=True)
            if moe_remote_by_expert
            else ""
        ),
        "context_migration_latest_plans": (
            json.dumps(latest_plans, sort_keys=True) if latest_plans else ""
        ),
        "kv_cache_migration_attempts": kv_attempted,
        "kv_cache_migration_successes": kv_succeeded,
        "kv_cache_migration_tokens": kv_tokens,
        "kv_cache_migration_latest": (
            json.dumps(latest_kv, sort_keys=True) if latest_kv else ""
        ),
    }


def summarize_state_recovery_metrics(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    state_rows = [row for row in rows if row.get("type") == "state_recovery"]
    restore_rows = [
        row for row in state_rows if row.get("action") == "restore_state"
    ]
    fallback_rows = [row for row in state_rows if row.get("fallback_used")]
    recovered_tokens = sum(
        int(row.get("recovered_tokens", 0) or 0) for row in state_rows
    )
    locality_rows = [
        row for row in state_rows if row.get("expert_locality_available")
    ]
    moe_dispatch_rows = [
        row
        for row in locality_rows
        if row.get("moe_dispatch_observation_available")
    ]
    locality_ratios = [
        safe_float(row.get("hot_expert_locality_ratio"), 0.0)
        for row in locality_rows
    ]
    remote_routing_ratios = [
        safe_float(row.get("estimated_remote_routing_ratio"), 0.0)
        for row in locality_rows
    ]
    moe_remote_dispatch_ratios = [
        safe_float(row.get("moe_remote_routing_ratio"), 0.0)
        for row in moe_dispatch_rows
    ]
    moe_local_by_layer = merge_int_maps([
        row.get("moe_local_routed_tokens_by_layer", {})
        for row in moe_dispatch_rows
    ])
    moe_remote_by_layer = merge_int_maps([
        row.get("moe_remote_routed_tokens_by_layer", {})
        for row in moe_dispatch_rows
    ])
    moe_local_by_expert = merge_int_maps([
        row.get("moe_local_routed_tokens_by_expert", {})
        for row in moe_dispatch_rows
    ])
    moe_remote_by_expert = merge_int_maps([
        row.get("moe_remote_routed_tokens_by_expert", {})
        for row in moe_dispatch_rows
    ])
    expert_dispatch_costs = [
        safe_float(row.get("expert_dispatch_cost"), 0.0)
        for row in locality_rows
    ]
    placement_handshake_rows = [
        row for row in state_rows if row.get("placement_handshake_attempted")
    ]
    latest_plan = state_rows[-1].get("plan") if state_rows else None
    return {
        "state_recovery_events": len(state_rows),
        "state_recovery_restore_events": len(restore_rows),
        "state_recovery_fallback_events": len(fallback_rows),
        "state_recovery_recovered_tokens": recovered_tokens,
        "state_recovery_model_semantic_compatible_count": sum(
            1 for row in state_rows if row.get("model_semantic_compatible")
        ),
        "state_recovery_state_serialization_compatible_count": sum(
            1
            for row in state_rows
            if row.get("state_serialization_compatible")
        ),
        "state_recovery_kv_layout_compatible_count": sum(
            1 for row in state_rows if row.get("kv_layout_compatible")
        ),
        "state_recovery_kv_restore_compatible_count": sum(
            1 for row in state_rows if row.get("kv_restore_compatible")
        ),
        "state_recovery_ep_layout_required_count": sum(
            1 for row in state_rows if row.get("ep_layout_required")
        ),
        "state_recovery_ep_layout_compatible_count": sum(
            1 for row in state_rows if row.get("ep_layout_compatible")
        ),
        "state_recovery_expert_placement_mismatch_count": sum(
            1 for row in state_rows if row.get("expert_placement_mismatch")
        ),
        "state_recovery_expert_locality_available_count": len(locality_rows),
        "state_recovery_moe_dispatch_observation_available_count": sum(
            1 for row in moe_dispatch_rows
        ),
        "state_recovery_avg_hot_expert_locality_ratio": (
            mean(locality_ratios) if locality_ratios else 0.0
        ),
        "state_recovery_avg_estimated_remote_routing_ratio": (
            mean(remote_routing_ratios) if remote_routing_ratios else 0.0
        ),
        "state_recovery_estimated_remote_routed_tokens": sum(
            safe_int(row.get("estimated_remote_routed_tokens"), 0)
            for row in locality_rows
        ),
        "state_recovery_estimated_dispatch_cost": sum(
            expert_dispatch_costs
        ),
        "state_recovery_target_placement_epochs": compact_values([
            str(row.get("target_placement_epoch"))
            for row in state_rows
            if row.get("target_placement_epoch") is not None
        ]),
        "state_recovery_current_placement_epochs": compact_values([
            str(row.get("current_placement_epoch"))
            for row in state_rows
            if row.get("current_placement_epoch") is not None
        ]),
        "state_recovery_target_expert_placement_fingerprints": compact_values([
            str(row.get("target_expert_placement_fingerprint"))
            for row in state_rows
            if row.get("target_expert_placement_fingerprint")
        ]),
        "state_recovery_target_expert_placement_plan_fingerprints": (
            compact_values([
                str(row.get("target_expert_placement_plan_fingerprint"))
                for row in state_rows
                if row.get("target_expert_placement_plan_fingerprint")
            ])
        ),
        "state_recovery_target_expert_placement_snapshot_fingerprints": (
            compact_values([
                str(row.get("target_expert_placement_snapshot_fingerprint"))
                for row in state_rows
                if row.get("target_expert_placement_snapshot_fingerprint")
            ])
        ),
        "state_recovery_target_expert_placement_contracts": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_contract_available")
        ),
        "state_recovery_target_expert_placement_plan_applied": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_plan_applied")
        ),
        "state_recovery_target_expert_placement_plan_verified": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_plan_verified")
        ),
        "state_recovery_target_expert_placement_contract_reasons": (
            compact_values([
                str(row.get("target_expert_placement_contract_reason"))
                for row in state_rows
                if row.get("target_expert_placement_contract_reason")
            ])
        ),
        "state_recovery_target_expert_placement_apply_hook_available": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_apply_hook_available")
        ),
        "state_recovery_target_expert_placement_apply_attempted": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_apply_attempted")
        ),
        "state_recovery_target_expert_placement_apply_success": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_apply_success")
        ),
        "state_recovery_target_expert_placement_apply_reasons": (
            compact_values([
                str(row.get("target_expert_placement_apply_reason"))
                for row in state_rows
                if row.get("target_expert_placement_apply_reason")
            ])
        ),
        "state_recovery_target_expert_placement_verify_hook_available": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_verify_hook_available")
        ),
        "state_recovery_target_expert_placement_verify_attempted": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_verify_attempted")
        ),
        "state_recovery_target_expert_placement_verify_success": sum(
            1
            for row in state_rows
            if row.get("target_expert_placement_verify_success")
        ),
        "state_recovery_target_expert_placement_verify_reasons": (
            compact_values([
                str(row.get("target_expert_placement_verify_reason"))
                for row in state_rows
                if row.get("target_expert_placement_verify_reason")
            ])
        ),
        "state_recovery_current_expert_placement_fingerprints": (
            compact_values([
                str(row.get("current_expert_placement_fingerprint"))
                for row in state_rows
                if row.get("current_expert_placement_fingerprint")
            ])
        ),
        "state_recovery_placement_handshake_attempts": len(
            placement_handshake_rows
        ),
        "state_recovery_placement_handshake_successes": sum(
            1
            for row in placement_handshake_rows
            if row.get("placement_handshake_compatible")
        ),
        "state_recovery_placement_handshake_failures": sum(
            1
            for row in placement_handshake_rows
            if not row.get("placement_handshake_compatible")
        ),
        "state_recovery_placement_handshake_stale": sum(
            1
            for row in placement_handshake_rows
            if row.get("placement_handshake_stale")
        ),
        "state_recovery_placement_handshake_reasons": compact_values([
            str(row.get("placement_handshake_reason", ""))
            for row in placement_handshake_rows
        ]),
        "state_recovery_moe_routed_tokens": sum(
            safe_int(row.get("moe_routed_tokens"), 0)
            for row in moe_dispatch_rows
        ),
        "state_recovery_moe_local_routed_tokens": sum(
            safe_int(row.get("moe_local_routed_tokens"), 0)
            for row in moe_dispatch_rows
        ),
        "state_recovery_moe_remote_routed_tokens": sum(
            safe_int(row.get("moe_remote_routed_tokens"), 0)
            for row in moe_dispatch_rows
        ),
        "state_recovery_moe_avg_remote_routing_ratio": (
            mean(moe_remote_dispatch_ratios)
            if moe_remote_dispatch_ratios
            else 0.0
        ),
        "state_recovery_moe_locality_definitions": compact_values(
            [
                str(row.get("moe_locality_definition", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_moe_locality_granularities": compact_values(
            [
                str(row.get("moe_locality_granularity", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_moe_remote_routing_definitions": compact_values(
            [
                str(row.get("moe_remote_routing_definition", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_moe_rank_locality_available_count": sum(
            1 for row in state_rows if row.get("moe_rank_locality_available")
        ),
        "state_recovery_moe_physical_dispatch_traffic_available_count": sum(
            1
            for row in state_rows
            if row.get("moe_physical_dispatch_traffic_available")
        ),
        "state_recovery_moe_local_routed_tokens_by_layer": (
            json.dumps(moe_local_by_layer, sort_keys=True)
            if moe_local_by_layer
            else ""
        ),
        "state_recovery_moe_remote_routed_tokens_by_layer": (
            json.dumps(moe_remote_by_layer, sort_keys=True)
            if moe_remote_by_layer
            else ""
        ),
        "state_recovery_moe_local_routed_tokens_by_expert": (
            json.dumps(moe_local_by_expert, sort_keys=True)
            if moe_local_by_expert
            else ""
        ),
        "state_recovery_moe_remote_routed_tokens_by_expert": (
            json.dumps(moe_remote_by_expert, sort_keys=True)
            if moe_remote_by_expert
            else ""
        ),
        "state_recovery_moe_route_histogram_sources": compact_values(
            [
                str(row.get("moe_route_histogram_source", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_moe_route_histogram_kinds": compact_values(
            [
                str(row.get("moe_route_histogram_kind", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_model_semantic_reasons": compact_values(
            [
                str(row.get("model_semantic_reason", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_state_serialization_reasons": compact_values(
            [
                str(row.get("state_serialization_reason", ""))
                for row in state_rows
            ]
        ),
        "state_recovery_kv_layout_reasons": compact_values(
            [str(row.get("kv_layout_reason", "")) for row in state_rows]
        ),
        "state_recovery_ep_layout_reasons": compact_values(
            [str(row.get("ep_layout_reason", "")) for row in state_rows]
        ),
        "state_recovery_latest_plan": (
            json.dumps(latest_plan, sort_keys=True) if latest_plan else ""
        ),
    }


def summarize_risk_aware_scheduling_metrics(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    scheduling_rows = [
        row for row in rows if row.get("type") == "risk_aware_scheduling"
    ]
    selected_risks = [
        float(row.get("selected_spot_risk", 0.0) or 0.0)
        for row in scheduling_rows
    ]
    selected_scores = [
        float(row.get("selected_score", 0.0) or 0.0)
        for row in scheduling_rows
    ]
    selected_confidences = [
        float(row.get("selected_confidence", 0.0) or 0.0)
        for row in scheduling_rows
    ]
    latest_decision = (
        scheduling_rows[-1].get("decision") if scheduling_rows else None
    )
    return {
        "risk_scheduling_events": len(scheduling_rows),
        "risk_scheduling_allocations": sum(
            1 for row in scheduling_rows if row.get("action") == "allocate"
        ),
        "risk_scheduling_avg_selected_risk": (
            mean(selected_risks) if selected_risks else 0.0
        ),
        "risk_scheduling_avg_selected_score": (
            mean(selected_scores) if selected_scores else 0.0
        ),
        "risk_scheduling_latest_node": (
            scheduling_rows[-1].get("selected_node_id")
            if scheduling_rows
            else ""
        ),
        "risk_scheduling_latest_metadata_source": (
            scheduling_rows[-1].get("selected_metadata_source")
            if scheduling_rows
            else ""
        ),
        "risk_scheduling_latest_provider": (
            scheduling_rows[-1].get("selected_provider")
            if scheduling_rows
            else ""
        ),
        "risk_scheduling_avg_selected_confidence": (
            mean(selected_confidences) if selected_confidences else 0.0
        ),
        "risk_scheduling_latest_decision": (
            json.dumps(latest_decision, sort_keys=True)
            if latest_decision
            else ""
        ),
    }


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    request_rows = read_jsonl(run_dir / "raw_requests.jsonl")
    router_metrics_path = resolve_optional_path(
        metadata.get("router_metrics_path")
        or metadata.get("metrics_path")
        or metadata.get("router_config", {}).get("metrics_path"),
        run_dir,
    )
    router_request_rows = []
    router_metric_rows = []
    router_metric_rows_in_window = []
    if router_metrics_path is not None:
        router_metric_rows = read_jsonl(router_metrics_path)
        router_metric_rows_in_window = metrics_in_request_window(
            router_metric_rows, request_rows
        )
        router_request_rows = matching_router_request_metrics(
            router_metric_rows_in_window, request_rows
        )
        if router_request_rows:
            with (run_dir / "router_request_metrics.jsonl").open(
                "w", encoding="utf-8"
            ) as metrics_file:
                for row in router_request_rows:
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")

    request_summary = summarize_requests(request_rows)
    phase_summary = summarize_phase_requests(request_rows)
    response_kv_summary = summarize_response_kv_restore(request_rows)
    router_summary = summarize_router_metrics(
        router_request_rows,
        request_rows,
    )
    true_kv_summary = summarize_true_kv_restore_evidence(
        request_rows,
        router_request_rows,
    )

    summary = {
        "run_dir": str(run_dir),
        "run_name": metadata.get("name", run_dir.name),
        "policy": metadata.get("policy", "unknown"),
        "backend": metadata.get("backend", "unknown"),
        "model": metadata.get("model", "unknown"),
        "router_metrics_path": (
            str(router_metrics_path) if router_metrics_path is not None else ""
        ),
        **summarize_trace_replay(run_dir, metadata),
        **request_summary,
        **phase_summary,
        **response_kv_summary,
        **router_summary,
        **true_kv_summary,
        **summarize_instance_state_metrics(router_metric_rows_in_window),
        **summarize_replanning_metrics(router_metric_rows_in_window),
        **summarize_context_migration_metrics(router_metric_rows_in_window),
        **summarize_state_recovery_metrics(router_metric_rows_in_window),
        **summarize_risk_aware_scheduling_metrics(
            router_metric_rows
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SpotServe benchmark output"
    )
    parser.add_argument("run_dir", nargs="+")
    args = parser.parse_args()

    summaries = [analyze_run(Path(run_dir)) for run_dir in args.run_dir]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
