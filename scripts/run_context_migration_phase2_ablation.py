import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    estimate_expert_dispatch_cost,
    estimate_kv_migration_cost,
    estimate_queue_penalty_cost,
    plan_low_cost_migration_from_dict,
)
from sllm.spot.metrics import JsonlMetricsWriter, make_context_migration_event


DEFAULT_COMPARISON_FIELDS = [
    "context_migration_total_estimated_cost",
    "context_migration_kv_migration_cost",
    "context_migration_moe_estimated_dispatch_cost",
    "context_migration_queue_penalty_cost",
    "context_migration_reuse_ratio",
    "context_migration_moe_avg_hot_expert_locality_ratio",
    "context_migration_moe_avg_estimated_remote_routing_ratio",
    "context_migration_moe_estimated_remote_routed_tokens",
    "context_migration_avg_queue_pressure",
    "context_migration_max_queue_depth",
    "target_selection_expected",
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


def merged_planner_config(
    base_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> Dict[str, Any]:
    config = dict(base_config or {})
    config.update(dict(run_config or {}))
    return config


def selected_targets(decision: Mapping[str, Any]) -> List[str]:
    return [
        str(plan.get("new_instance_id", ""))
        for plan in decision.get("plans", [])
        if plan.get("new_instance_id")
    ]


def first_plan(decision: Mapping[str, Any]) -> Dict[str, Any]:
    plans = decision.get("plans", [])
    if not plans:
        return {}
    return dict(plans[0] or {})


def candidate_component_costs(
    sources: List[Mapping[str, Any]],
    targets: List[Mapping[str, Any]],
    planner_config: Mapping[str, Any],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    target_objects = [MigrationTarget.from_dict(row) for row in targets]
    for source_row in sources:
        source = ContextMetadata.from_dict(source_row)
        request_id = str(source.request_id or source.instance_id)
        rows[request_id] = {}
        for target in target_objects:
            kv = estimate_kv_migration_cost(source, target, planner_config)
            expert = estimate_expert_dispatch_cost(
                source, target, planner_config
            )
            queue = estimate_queue_penalty_cost(
                target,
                planner_config,
                planned_requests_ahead=0,
            )
            rows[request_id][target.instance_id] = {
                "total_cost": (
                    float(kv.get("cost", 0.0) or 0.0)
                    + float(expert.get("cost", 0.0) or 0.0)
                    + float(queue.get("cost", 0.0) or 0.0)
                ),
                "kv_migration_cost": float(kv.get("cost", 0.0) or 0.0),
                "expert_dispatch_cost": float(
                    expert.get("cost", 0.0) or 0.0
                ),
                "queue_penalty_cost": float(queue.get("cost", 0.0) or 0.0),
                "hot_expert_locality_ratio": float(
                    expert.get("locality_ratio", 0.0) or 0.0
                ),
                "estimated_remote_routing_ratio": float(
                    expert.get("estimated_remote_routing_ratio", 0.0) or 0.0
                ),
                "queue_pressure": float(
                    queue.get("queue_pressure", 0.0) or 0.0
                ),
            }
    return rows


def summarize_run(
    run: Mapping[str, Any],
    decision: Mapping[str, Any],
    event: Mapping[str, Any],
    candidate_costs: Mapping[str, Any],
    analyzer: Any,
) -> Dict[str, Any]:
    expected_targets = [
        str(target) for target in run.get("expected_targets", []) or []
    ]
    selected = selected_targets(decision)
    plan = first_plan(decision)
    target_selection_expected = (
        selected == expected_targets if expected_targets else True
    )
    metric_summary = analyzer.summarize_context_migration_metrics([dict(event)])
    return {
        "run_name": str(run.get("name", "unnamed")),
        "description": str(run.get("description", "")),
        "selected_targets": selected,
        "selected_target_ids": ",".join(selected),
        "expected_targets": expected_targets,
        "target_selection_expected": bool(target_selection_expected),
        "selected_kv_migration_cost": float(
            plan.get("kv_migration_cost", 0.0) or 0.0
        ),
        "selected_expert_dispatch_cost": float(
            plan.get("expert_dispatch_cost", 0.0) or 0.0
        ),
        "selected_queue_penalty_cost": float(
            plan.get("queue_penalty_cost", 0.0) or 0.0
        ),
        "selected_hot_expert_locality_ratio": float(
            plan.get("hot_expert_locality_ratio", 0.0) or 0.0
        ),
        "selected_estimated_remote_routing_ratio": float(
            plan.get("estimated_remote_routing_ratio", 0.0) or 0.0
        ),
        "selected_queue_pressure": float(
            plan.get("queue_pressure", 0.0) or 0.0
        ),
        "candidate_component_costs": dict(candidate_costs),
        **metric_summary,
    }


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
            "baseline_selected_targets": baseline.get("selected_targets", []),
            "candidate_selected_targets": candidate.get("selected_targets", []),
            "target_selection_changed": (
                baseline.get("selected_targets")
                != candidate.get("selected_targets")
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


def run_ablation(
    input_path: Path,
    output_dir: Path,
    validate: bool = True,
) -> Dict[str, Any]:
    payload = load_json(input_path)
    model = str(payload.get("model", "context-migration-phase2-ablation"))
    base_config = dict(payload.get("planner_config", {}) or {})
    analyzer = load_analyzer()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    failed_expectations: List[Dict[str, Any]] = []
    for run in payload.get("runs", []):
        run_name = str(run.get("name", "unnamed"))
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        planner_config = merged_planner_config(
            base_config,
            run.get("planner_config", {}),
        )
        run_payload = {
            "sources": payload.get("sources", []),
            "targets": payload.get("targets", []),
            "planner_config": planner_config,
        }
        decision = plan_low_cost_migration_from_dict(run_payload).to_dict()
        event = make_context_migration_event(
            model=model,
            decision=decision,
            reason="phase2_synthetic_ablation",
        )
        candidate_costs = candidate_component_costs(
            payload.get("sources", []),
            payload.get("targets", []),
            planner_config,
        )
        summary = summarize_run(
            run,
            decision,
            event,
            candidate_costs,
            analyzer,
        )
        summary.update({
            "model": model,
            "input": str(input_path),
            "planner_config": planner_config,
        })

        write_json(run_dir / "input.json", run_payload)
        write_json(run_dir / "migration_plan.json", decision)
        write_json(run_dir / "summary.json", summary)
        metrics_path = run_dir / "migration_metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        metrics_writer = JsonlMetricsWriter(metrics_path)
        metrics_writer.emit(dict(event))

        summaries.append(summary)
        if not summary["target_selection_expected"]:
            failed_expectations.append({
                "run_name": run_name,
                "selected_targets": summary["selected_targets"],
                "expected_targets": summary["expected_targets"],
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
    write_json(output_dir / "report.json", report)

    if validate and failed_expectations:
        raise AssertionError(
            "Phase 2 ablation target selection mismatch: "
            f"{failed_expectations}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic Phase 2 context migration ablation"
    )
    parser.add_argument(
        "--input",
        default="benchmarks/spotserve/context_migration_phase2_ablation.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/spotserve_context_migration_phase2_ablation",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Write reports even if expected target selections do not match.",
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
