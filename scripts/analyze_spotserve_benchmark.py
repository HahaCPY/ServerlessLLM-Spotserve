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
    if router_metrics_path is not None:
        router_request_rows = matching_router_request_metrics(
            read_jsonl(router_metrics_path), request_rows
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
