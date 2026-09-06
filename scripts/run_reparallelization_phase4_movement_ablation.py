import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.metrics import JsonlMetricsWriter, make_replanning_event
from sllm.spot.reparallelization import plan_dynamic_reparallelization


DEFAULT_COMPARISON_FIELDS = [
    "selected_total_gpus",
    "selected_score",
    "selected_expert_weight_movement_cost_estimate_ms",
    "selected_expert_placement_moved_expert_count",
    "selected_expert_placement_moved_weight_bytes",
    "replanning_expert_placement_plan_movement_observation_events",
    "replanning_max_expert_placement_plan_moved_experts",
    "replanning_total_expert_placement_plan_moved_weight_bytes",
    "replanning_avg_expert_placement_plan_weight_movement_cost_ms",
    "selection_expected",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })


def load_analyzer():
    path = REPO_ROOT / "scripts" / "analyze_spotserve_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "spotserve_benchmark_analyzer", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark analyzer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merged_config(
    base_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> Dict[str, Any]:
    config = dict(base_config or {})
    config.update(dict(run_config or {}))
    return config


def build_comparisons(
    comparisons: List[Mapping[str, Any]],
    summaries: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_name = {summary.get("run_name"): summary for summary in summaries}
    rows: List[Dict[str, Any]] = []
    for comparison in comparisons:
        baseline_name = comparison.get("baseline")
        candidate_name = comparison.get("candidate")
        baseline = by_name.get(baseline_name)
        candidate = by_name.get(candidate_name)
        if baseline is None or candidate is None:
            continue
        fields = comparison.get("fields", DEFAULT_COMPARISON_FIELDS)
        row: Dict[str, Any] = {
            "name": comparison.get(
                "name", f"{candidate_name}-vs-{baseline_name}"
            ),
            "baseline": baseline_name,
            "candidate": candidate_name,
            "baseline_selected_reason": baseline.get(
                "selected_reason", ""
            ),
            "candidate_selected_reason": candidate.get(
                "selected_reason", ""
            ),
            "selection_changed": (
                baseline.get("selected_reason")
                != candidate.get("selected_reason")
            ),
        }
        for field in fields:
            if field not in baseline or field not in candidate:
                continue
            baseline_value = baseline.get(field)
            candidate_value = candidate.get(field)
            if isinstance(baseline_value, bool):
                baseline_value = 1.0 if baseline_value else 0.0
            if isinstance(candidate_value, bool):
                candidate_value = 1.0 if candidate_value else 0.0
            if not isinstance(baseline_value, (int, float)):
                continue
            if not isinstance(candidate_value, (int, float)):
                continue
            row[f"{field}_baseline"] = baseline_value
            row[f"{field}_candidate"] = candidate_value
            row[f"{field}_delta"] = candidate_value - baseline_value
        rows.append(row)
    return rows


def summarize_run(
    run: Mapping[str, Any],
    decision: Mapping[str, Any],
    event: Mapping[str, Any],
    analyzer: Any,
) -> Dict[str, Any]:
    selected_config = dict(decision.get("selected_config") or {})
    expected_reason = str(run.get("expected_selected_reason", "") or "")
    expected_total_gpus = run.get("expected_selected_total_gpus")
    expected_moved = run.get("expected_moved_experts")
    expected_moved_bytes = run.get("expected_moved_weight_bytes")
    expected_movement_cost = run.get("expected_movement_cost_ms")
    expected_observation = run.get("expected_movement_observation")

    selected_reason = str(selected_config.get("reason", "") or "")
    selected_total_gpus = int(decision.get("selected_total_gpus", 0) or 0)
    selected_moved = int(
        decision.get("selected_expert_placement_moved_expert_count", 0) or 0
    )
    selected_moved_bytes = int(
        decision.get("selected_expert_placement_moved_weight_bytes", 0) or 0
    )
    selected_movement_cost = float(
        decision.get(
            "selected_expert_weight_movement_cost_estimate_ms", 0.0
        )
        or 0.0
    )
    selected_observation = bool(
        decision.get(
            "selected_expert_placement_movement_observation_available",
            False,
        )
    )

    checks = {
        "selected_reason": (
            selected_reason == expected_reason if expected_reason else True
        ),
        "selected_total_gpus": (
            selected_total_gpus == int(expected_total_gpus)
            if expected_total_gpus is not None
            else True
        ),
        "moved_experts": (
            selected_moved == int(expected_moved)
            if expected_moved is not None
            else True
        ),
        "moved_weight_bytes": (
            selected_moved_bytes == int(expected_moved_bytes)
            if expected_moved_bytes is not None
            else True
        ),
        "movement_cost_ms": (
            selected_movement_cost == float(expected_movement_cost)
            if expected_movement_cost is not None
            else True
        ),
        "movement_observation": (
            selected_observation is bool(expected_observation)
            if expected_observation is not None
            else True
        ),
    }
    metric_summary = analyzer.summarize_replanning_metrics([dict(event)])
    return {
        "run_name": str(run.get("name", "unnamed")),
        "description": str(run.get("description", "")),
        "selected_reason": selected_reason,
        "expected_selected_reason": expected_reason,
        "selected_total_gpus": selected_total_gpus,
        "expected_selected_total_gpus": expected_total_gpus,
        "selected_score": float(decision.get("selected_score", 0.0) or 0.0),
        "selected_replan_window_cost_ms": float(
            decision.get("selected_replan_window_cost_ms", 0.0) or 0.0
        ),
        "selected_expert_weight_movement_cost_estimate_ms": (
            selected_movement_cost
        ),
        "selected_expert_placement_movement_observation_available": (
            selected_observation
        ),
        "selected_expert_placement_movement_source": str(
            decision.get(
                "selected_expert_placement_movement_source",
                "unavailable",
            )
        ),
        "selected_expert_placement_moved_expert_count": selected_moved,
        "selected_expert_placement_moved_weight_bytes": selected_moved_bytes,
        "expected_moved_experts": expected_moved,
        "expected_moved_weight_bytes": expected_moved_bytes,
        "expected_movement_cost_ms": expected_movement_cost,
        "selection_expected": all(checks.values()),
        "expectation_checks": checks,
        **metric_summary,
    }


def run_ablation(
    input_path: Path,
    output_dir: Path,
    validate: bool = True,
) -> Dict[str, Any]:
    payload = load_json(input_path)
    model = str(
        payload.get("model", "reparallelization-phase4-movement-ablation")
    )
    base_planner_config = dict(payload.get("planner_config", {}) or {})
    base_model_config = dict(payload.get("model_config", {}) or {})
    base_worker_nodes = dict(payload.get("worker_nodes", {}) or {})
    analyzer = load_analyzer()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    failed_expectations: List[Dict[str, Any]] = []
    for run in payload.get("runs", []):
        run_name = str(run.get("name", "unnamed"))
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        model_config = merged_config(
            base_model_config,
            run.get("model_config", {}),
        )
        planner_config = merged_config(
            base_planner_config,
            run.get("planner_config", {}),
        )
        worker_nodes = merged_config(
            base_worker_nodes,
            run.get("worker_nodes", {}),
        )

        event_name = str(run.get("event", "preempt"))
        node_id = run.get("node_id")
        decision = plan_dynamic_reparallelization(
            model_name=model,
            worker_nodes=worker_nodes,
            model_config=model_config,
            planner_config=planner_config,
            event=event_name,
            node_id=str(node_id) if node_id is not None else None,
            backend=str(model_config.get("backend", "vllm")),
        )
        event = make_replanning_event(
            model=model,
            event=event_name,
            decision=decision,
            node_id=str(node_id) if node_id is not None else None,
        )
        summary = summarize_run(run, decision, event, analyzer)
        summary.update({
            "model": model,
            "input": str(input_path),
            "planner_config": planner_config,
        })

        write_json(run_dir / "input.json", {
            "worker_nodes": worker_nodes,
            "model_config": model_config,
            "planner_config": planner_config,
        })
        write_json(run_dir / "replanning_decision.json", decision)
        write_json(run_dir / "summary.json", summary)
        metrics_path = run_dir / "replanning_metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        metrics_writer = JsonlMetricsWriter(metrics_path)
        metrics_writer.emit(dict(event))

        summaries.append(summary)
        if not summary["selection_expected"]:
            failed_expectations.append({
                "run_name": run_name,
                "selected_reason": summary["selected_reason"],
                "expected_selected_reason": (
                    summary["expected_selected_reason"]
                ),
                "selected_moved_experts": (
                    summary[
                        "selected_expert_placement_moved_expert_count"
                    ]
                ),
                "expected_moved_experts": (
                    summary["expected_moved_experts"]
                ),
                "expectation_checks": summary["expectation_checks"],
            })

    comparisons = build_comparisons(
        payload.get("comparisons", []),
        summaries,
    )
    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "runs": summaries,
        "comparisons": comparisons,
        "failed_expectations": failed_expectations,
        "passed": not failed_expectations,
    }
    write_json(output_dir / "latest_summary.json", summaries)
    write_csv(output_dir / "latest_summary.csv", summaries)
    write_json(output_dir / "latest_comparisons.json", comparisons)
    write_csv(output_dir / "latest_comparisons.csv", comparisons)
    write_json(output_dir / "report.json", report)

    if validate and failed_expectations:
        raise AssertionError(
            "Phase 4 movement ablation mismatch: "
            f"{failed_expectations}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic Phase 4 expert movement-diff ablation"
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "benchmarks/spotserve/"
            "reparallelization_phase4_movement_ablation.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/spotserve_reparallelization_phase4_movement_ablation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Write reports even if expected planner selections do not match.",
    )
    args = parser.parse_args()

    report = run_ablation(
        Path(args.input),
        Path(args.output_dir),
        validate=not args.no_validate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
