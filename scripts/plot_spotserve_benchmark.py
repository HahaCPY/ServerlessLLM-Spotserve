import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List


POLICY_COLORS = {
    "none": "#7a7a7a",
    "naive_retry": "#2563eb",
    "generated_token_replay": "#16a34a",
    "stateful_recovery": "#0891b2",
    "fallback": "#f97316",
    "failure": "#dc2626",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def svg_bar(label: str, value: float, max_value: float, color: str) -> str:
    width = 520
    bar_width = 0 if max_value <= 0 else int((value / max_value) * width)
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(label)}</div>
      <svg width="{width}" height="22" role="img" aria-label="{html.escape(label)}">
        <rect x="0" y="3" width="{width}" height="16" rx="3" fill="#e5e7eb"/>
        <rect x="0" y="3" width="{bar_width}" height="16" rx="3" fill="{color}"/>
      </svg>
      <div class="bar-value">{value:.2f}</div>
    </div>
    """


def latency_cdf_svg(rows: List[Dict[str, Any]]) -> str:
    latencies = sorted(float(row.get("latency_ms", 0.0)) for row in rows)
    if not latencies:
        return "<p>No latency data.</p>"
    width = 720
    height = 260
    max_latency = max(latencies) or 1.0
    points = []
    for index, latency in enumerate(latencies):
        x = (latency / max_latency) * (width - 60) + 40
        y = height - 30 - (index / max(1, len(latencies) - 1)) * (height - 60)
        points.append(f"{x:.1f},{y:.1f}")
    return f"""
    <svg class="chart" width="{width}" height="{height}" role="img" aria-label="Latency CDF">
      <line x1="40" y1="{height - 30}" x2="{width - 20}" y2="{height - 30}" stroke="#6b7280"/>
      <line x1="40" y1="20" x2="40" y2="{height - 30}" stroke="#6b7280"/>
      <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{' '.join(points)}"/>
      <text x="40" y="{height - 8}" fill="#374151">0 ms</text>
      <text x="{width - 120}" y="{height - 8}" fill="#374151">{max_latency:.1f} ms</text>
      <text x="8" y="24" fill="#374151">CDF</text>
    </svg>
    """


def render_report(run_dir: Path) -> None:
    metadata = read_json(run_dir / "run_metadata.json")
    summary = read_json(run_dir / "summary.json")
    rows = read_jsonl(run_dir / "raw_requests.jsonl")
    policy = summary.get("policy", metadata.get("policy", "none"))
    color = POLICY_COLORS.get(policy, "#2563eb")

    success_rate = float(summary.get("success_rate", 0.0)) * 100
    throughput = float(summary.get("throughput_req_s", 0.0))
    p95 = float(summary.get("latency_p95_ms", 0.0))
    p99 = float(summary.get("latency_p99_ms", 0.0))
    max_bar = max(success_rate, throughput, p95, p99, 1.0)
    failed_attempts = int(summary.get("failed_attempts_total", 0) or 0)
    retry_count = int(summary.get("retry_count_total", 0) or 0)
    recovered_tokens = int(summary.get("recovered_tokens_total", 0) or 0)
    recovery_fallbacks = int(summary.get("recovery_fallback_count", 0) or 0)
    recovery_triggered = int(
        summary.get("recovery_triggered_requests", 0) or 0
    )
    replay_succeeded = int(
        summary.get("replay_succeeded_requests", 0) or 0
    )
    state_restore_attempts = int(
        summary.get("state_restore_attempts_total", 0) or 0
    )
    state_restore_successes = int(
        summary.get("state_restore_successes_total", 0) or 0
    )
    state_restore_fallbacks = int(
        summary.get("state_restore_fallback_count", 0) or 0
    )
    state_restored_tokens = int(
        summary.get("state_restored_tokens_total", 0) or 0
    )
    state_recovery_events = int(
        summary.get("state_recovery_events", 0) or 0
    )
    state_recovery_restore_events = int(
        summary.get("state_recovery_restore_events", 0) or 0
    )
    state_recovery_fallback_events = int(
        summary.get("state_recovery_fallback_events", 0) or 0
    )
    instance_state_rows = int(summary.get("instance_state_rows", 0) or 0)
    instances_marked_preempting = int(
        summary.get("instances_marked_preempting", 0) or 0
    )
    instances_marked_ready = int(
        summary.get("instances_marked_ready", 0) or 0
    )
    instances_marked_dead = int(
        summary.get("instances_marked_dead", 0) or 0
    )
    instances_marked_draining = int(
        summary.get("instances_marked_draining", 0) or 0
    )
    migration_events = int(
        summary.get("context_migration_events", 0) or 0
    )
    migration_plan_count = int(
        summary.get("context_migration_plan_count", 0) or 0
    )
    migration_unassigned_count = int(
        summary.get("context_migration_unassigned_count", 0) or 0
    )
    migration_total_cost = float(
        summary.get("context_migration_total_estimated_cost", 0.0) or 0.0
    )
    migration_avg_cost = float(
        summary.get("context_migration_avg_estimated_cost", 0.0) or 0.0
    )
    migration_reuse_ratio = float(
        summary.get("context_migration_reuse_ratio", 0.0) or 0.0
    )

    html_report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SpotServe Benchmark Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta, .summary {{ border-collapse: collapse; margin: 16px 0 28px; min-width: 760px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f9fafb; }}
    .card {{ margin: 24px 0; max-width: 880px; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 540px 90px; align-items: center; gap: 10px; margin: 8px 0; }}
    .bar-label {{ color: #374151; }}
    .bar-value {{ font-variant-numeric: tabular-nums; }}
    .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>SpotServe Benchmark Report</h1>
  <p>Run: <code>{html.escape(run_dir.name)}</code></p>

  <h2>Run Metadata</h2>
  <table class="meta">
    <tr><th>Policy</th><td>{html.escape(str(policy))}</td></tr>
    <tr><th>Backend</th><td>{html.escape(str(summary.get("backend", "unknown")))}</td></tr>
    <tr><th>Model</th><td>{html.escape(str(summary.get("model", "unknown")))}</td></tr>
    <tr><th>Trace</th><td>{html.escape(str(metadata.get("trace", "none")))}</td></tr>
    <tr><th>Workload</th><td>{html.escape(str(metadata.get("workload", "unknown")))}</td></tr>
    <tr><th>Git Commit</th><td>{html.escape(str(metadata.get("git_commit", "unknown")))}</td></tr>
  </table>

  <h2>Summary</h2>
  <table class="summary">
    <tr>
      <th>Requests</th><th>Success Rate</th><th>P50</th><th>P95</th><th>P99</th><th>Throughput</th>
    </tr>
    <tr>
      <td>{summary.get("requests", 0)}</td>
      <td>{success_rate:.2f}%</td>
      <td>{float(summary.get("latency_p50_ms", 0.0)):.2f} ms</td>
      <td>{p95:.2f} ms</td>
      <td>{p99:.2f} ms</td>
      <td>{throughput:.2f} req/s</td>
    </tr>
  </table>

  <h2>Recovery Correctness</h2>
  <table class="summary">
    <tr>
      <th>Router Metrics Rows</th><th>Triggered Requests</th><th>Failed Attempts</th><th>Retry Count</th><th>Recovered Tokens</th><th>Fallbacks</th><th>Replay Succeeded</th>
    </tr>
    <tr>
      <td>{summary.get("router_metrics_rows", 0)}</td>
      <td>{recovery_triggered}</td>
      <td>{failed_attempts}</td>
      <td>{retry_count}</td>
      <td>{recovered_tokens}</td>
      <td>{recovery_fallbacks}</td>
      <td>{replay_succeeded}</td>
    </tr>
  </table>

  <h2>Stateful Recovery Metrics</h2>
  <table class="summary">
    <tr>
      <th>State Events</th><th>Restore Events</th><th>Fallback Events</th><th>Restore Attempts</th><th>Restore Successes</th><th>Restore Fallbacks</th><th>State Restored Tokens</th>
    </tr>
    <tr>
      <td>{state_recovery_events}</td>
      <td>{state_recovery_restore_events}</td>
      <td>{state_recovery_fallback_events}</td>
      <td>{state_restore_attempts}</td>
      <td>{state_restore_successes}</td>
      <td>{state_restore_fallbacks}</td>
      <td>{state_restored_tokens}</td>
    </tr>
  </table>

  <h2>Instance State Metrics</h2>
  <table class="summary">
    <tr>
      <th>Instance Events</th><th>Preempting</th><th>Ready</th><th>Dead</th><th>Draining</th>
    </tr>
    <tr>
      <td>{instance_state_rows}</td>
      <td>{instances_marked_preempting}</td>
      <td>{instances_marked_ready}</td>
      <td>{instances_marked_dead}</td>
      <td>{instances_marked_draining}</td>
    </tr>
  </table>

  <h2>Context Migration Metrics</h2>
  <table class="summary">
    <tr>
      <th>Migration Events</th><th>Plans</th><th>Unassigned</th><th>Total Estimated Cost</th><th>Avg Cost</th><th>Reuse Ratio</th>
    </tr>
    <tr>
      <td>{migration_events}</td>
      <td>{migration_plan_count}</td>
      <td>{migration_unassigned_count}</td>
      <td>{migration_total_cost:.2f}</td>
      <td>{migration_avg_cost:.2f}</td>
      <td>{migration_reuse_ratio:.2%}</td>
    </tr>
  </table>

  <div class="card">
    <h2>Key Metrics</h2>
    {svg_bar("Success rate (%)", success_rate, max_bar, color)}
    {svg_bar("Throughput (req/s)", throughput, max_bar, color)}
    {svg_bar("P95 latency (ms)", p95, max_bar, "#9333ea")}
    {svg_bar("P99 latency (ms)", p99, max_bar, "#be123c")}
  </div>

  <div class="card">
    <h2>Latency CDF</h2>
    {latency_cdf_svg(rows)}
  </div>
</body>
</html>
"""
    (run_dir / "report.html").write_text(html_report, encoding="utf-8")

    markdown_report = f"""# SpotServe Benchmark Report

## Run Metadata

- Run: `{run_dir.name}`
- Policy: `{policy}`
- Backend: `{summary.get("backend", "unknown")}`
- Model: `{summary.get("model", "unknown")}`
- Trace: `{metadata.get("trace", "none")}`
- Workload: `{metadata.get("workload", "unknown")}`

## Summary

| Requests | Success Rate | P50 | P95 | P99 | Throughput |
|---:|---:|---:|---:|---:|---:|
| {summary.get("requests", 0)} | {success_rate:.2f}% | {float(summary.get("latency_p50_ms", 0.0)):.2f} ms | {p95:.2f} ms | {p99:.2f} ms | {throughput:.2f} req/s |

## Recovery Correctness

| Router Metrics Rows | Triggered Requests | Failed Attempts | Retry Count | Recovered Tokens | Fallbacks | Replay Succeeded |
|---:|---:|---:|---:|---:|---:|---:|
| {summary.get("router_metrics_rows", 0)} | {recovery_triggered} | {failed_attempts} | {retry_count} | {recovered_tokens} | {recovery_fallbacks} | {replay_succeeded} |

## Stateful Recovery Metrics

| State Events | Restore Events | Fallback Events | Restore Attempts | Restore Successes | Restore Fallbacks | State Restored Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| {state_recovery_events} | {state_recovery_restore_events} | {state_recovery_fallback_events} | {state_restore_attempts} | {state_restore_successes} | {state_restore_fallbacks} | {state_restored_tokens} |

## Instance State Metrics

| Instance Events | Preempting | Ready | Dead | Draining |
|---:|---:|---:|---:|---:|
| {instance_state_rows} | {instances_marked_preempting} | {instances_marked_ready} | {instances_marked_dead} | {instances_marked_draining} |

## Context Migration Metrics

| Migration Events | Plans | Unassigned | Total Estimated Cost | Avg Cost | Reuse Ratio |
|---:|---:|---:|---:|---:|---:|
| {migration_events} | {migration_plan_count} | {migration_unassigned_count} | {migration_total_cost:.2f} | {migration_avg_cost:.2f} | {migration_reuse_ratio:.2%} |

Open `report.html` for the visual report.
"""
    (run_dir / "report.md").write_text(markdown_report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Render SpotServe benchmark report"
    )
    parser.add_argument("run_dir", nargs="+")
    args = parser.parse_args()

    for run_dir in args.run_dir:
        render_report(Path(run_dir))
        print(f"Wrote report for {run_dir}")


if __name__ == "__main__":
    main()
