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


def summarize_router_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed_attempts_total = sum(
        int(row.get("failed_attempts", 0) or 0) for row in rows
    )
    retry_count_total = sum(
        int(row.get("retry_count", 0) or 0) for row in rows
    )
    recovered_tokens_total = sum(
        int(row.get("recovered_tokens", 0) or 0) for row in rows
    )
    recovery_fallback_count = sum(
        1 for row in rows if bool(row.get("recovery_fallback", False))
    )
    recovery_triggered_requests = sum(
        1
        for row in rows
        if int(row.get("failed_attempts", 0) or 0) > 0
        or int(row.get("retry_count", 0) or 0) > 0
    )
    replay_succeeded_requests = sum(
        1
        for row in rows
        if int(row.get("recovered_tokens", 0) or 0) > 0
        and not bool(row.get("recovery_fallback", False))
    )
    replay_not_needed_requests = sum(
        1
        for row in rows
        if int(row.get("failed_attempts", 0) or 0) == 0
        and int(row.get("recovered_tokens", 0) or 0) == 0
        and not bool(row.get("recovery_fallback", False))
    )
    return {
        "router_metrics_rows": len(rows),
        "failed_attempts_total": failed_attempts_total,
        "retry_count_total": retry_count_total,
        "recovered_tokens_total": recovered_tokens_total,
        "recovery_fallback_count": recovery_fallback_count,
        "recovery_triggered_requests": recovery_triggered_requests,
        "replay_succeeded_requests": replay_succeeded_requests,
        "replay_not_needed_requests": replay_not_needed_requests,
    }


def summarize_instance_state_metrics(
    rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    instance_rows = [
        row for row in rows if row.get("type") == "instance_state"
    ]

    def count_to(state: str) -> int:
        return sum(1 for row in instance_rows if row.get("to") == state)

    return {
        "instance_state_rows": len(instance_rows),
        "instances_marked_preempting": count_to("preempting"),
        "instances_marked_ready": count_to("ready"),
        "instances_marked_dead": count_to("dead"),
        "instances_marked_draining": count_to("draining"),
    }


def summarize_replanning_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    replanning_rows = [
        row for row in rows if row.get("type") == "reparallelization"
    ]
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
    return {
        "replanning_events": len(replanning_rows),
        "replanning_no_capacity_events": no_capacity_count,
        "replanning_max_selected_gpus": (
            max(selected_total_gpus) if selected_total_gpus else 0
        ),
        "replanning_latest_plan": (
            json.dumps(latest_plan, sort_keys=True) if latest_plan else ""
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
    latest_plans = migration_rows[-1].get("plans", []) if migration_rows else []
    return {
        "context_migration_events": len(migration_rows),
        "context_migration_plan_count": total_plan_count,
        "context_migration_unassigned_count": total_unassigned,
        "context_migration_total_estimated_cost": total_estimated_cost,
        "context_migration_avg_estimated_cost": (
            total_estimated_cost / total_plan_count if total_plan_count else 0.0
        ),
        "context_migration_reusable_tokens": total_reusable_tokens,
        "context_migration_reusable_context_blocks": total_reusable_blocks,
        "context_migration_reuse_ratio": (
            reuse_numerator / reuse_denominator if reuse_denominator else 0.0
        ),
        "context_migration_latest_plans": (
            json.dumps(latest_plans, sort_keys=True) if latest_plans else ""
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

    summary = {
        "run_dir": str(run_dir),
        "run_name": metadata.get("name", run_dir.name),
        "policy": metadata.get("policy", "unknown"),
        "backend": metadata.get("backend", "unknown"),
        "model": metadata.get("model", "unknown"),
        "router_metrics_path": (
            str(router_metrics_path) if router_metrics_path is not None else ""
        ),
        **summarize_requests(request_rows),
        **summarize_router_metrics(router_request_rows),
        **summarize_instance_state_metrics(router_metric_rows_in_window),
        **summarize_replanning_metrics(router_metric_rows_in_window),
        **summarize_context_migration_metrics(router_metric_rows_in_window),
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
