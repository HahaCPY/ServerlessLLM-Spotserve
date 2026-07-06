import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.context_migration import plan_low_cost_migration_from_dict
from sllm.spot.metrics import JsonlMetricsWriter, make_context_migration_event


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def summarize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    plans = decision.get("plans", [])
    return {
        "action": decision.get("action"),
        "migration_plan_count": len(plans),
        "unassigned_context_count": len(
            decision.get("unassigned_contexts", [])
        ),
        "total_estimated_cost": decision.get("total_estimated_cost", 0.0),
        "avg_estimated_cost": (
            decision.get("total_estimated_cost", 0.0) / len(plans)
            if plans
            else 0.0
        ),
        "total_reusable_tokens": decision.get("total_reusable_tokens", 0),
        "total_context_tokens": decision.get("total_context_tokens", 0),
        "total_reusable_context_blocks": decision.get(
            "total_reusable_context_blocks", 0
        ),
        "total_context_blocks": decision.get("total_context_blocks", 0),
        "reuse_ratio": decision.get("reuse_ratio", 0.0),
    }


def run_benchmark(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    payload = load_json(input_path)
    model = str(payload.get("model", "context-migration-synthetic"))
    decision = plan_low_cost_migration_from_dict(payload).to_dict()
    summary = {
        "model": model,
        "input": str(input_path),
        **summarize_decision(decision),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "migration_plan.json", decision)
    write_json(output_dir / "summary.json", summary)

    metrics_path = output_dir / "migration_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    metrics_writer = JsonlMetricsWriter(metrics_path)
    metrics_writer.emit(
        make_context_migration_event(
            model=model,
            decision=decision,
            reason="synthetic_context_migration_benchmark",
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a synthetic SpotServe context migration benchmark"
    )
    parser.add_argument(
        "--input",
        default="benchmarks/spotserve/context_migration_synthetic.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/spotserve_context_migration",
    )
    args = parser.parse_args()

    summary = run_benchmark(Path(args.input), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
