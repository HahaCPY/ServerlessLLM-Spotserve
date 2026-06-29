import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


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


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    request_rows = read_jsonl(run_dir / "raw_requests.jsonl")
    summary = {
        "run_dir": str(run_dir),
        "run_name": metadata.get("name", run_dir.name),
        "policy": metadata.get("policy", "unknown"),
        "backend": metadata.get("backend", "unknown"),
        "model": metadata.get("model", "unknown"),
        **summarize_requests(request_rows),
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
