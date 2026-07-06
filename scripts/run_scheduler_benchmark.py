import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.metrics import (
    JsonlMetricsWriter,
    make_risk_aware_scheduling_event,
)
from sllm.spot.risk_aware_scheduling import plan_risk_aware_scheduling

READY_STATE = "ready"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def first_ready_node(
    worker_nodes: Mapping[str, Mapping[str, Any]],
    requested_gpus: int,
) -> Optional[str]:
    for node_id, node_info in worker_nodes.items():
        if node_info.get("state", READY_STATE) != READY_STATE:
            continue
        if int(node_info.get("free_gpu", 0) or 0) >= requested_gpus:
            return str(node_id)
    return None


def consume_gpu(
    worker_nodes: Dict[str, Dict[str, Any]],
    node_id: Optional[str],
    requested_gpus: int,
) -> None:
    if node_id is None:
        return
    worker_nodes[node_id]["free_gpu"] = max(
        0,
        int(worker_nodes[node_id].get("free_gpu", 0) or 0) - requested_gpus,
    )


def run_policy(
    policy: str,
    payload: Mapping[str, Any],
    metrics_writer: Optional[JsonlMetricsWriter] = None,
) -> Dict[str, Any]:
    worker_nodes = {
        str(node_id): dict(node_info)
        for node_id, node_info in payload.get("worker_nodes", {}).items()
    }
    scheduler_config = dict(payload.get("scheduler_config", {}) or {})
    allocations = []

    for request_row in payload.get("requests", []):
        model_name = str(request_row.get("model", "scheduler-synthetic"))
        requested_gpus = int(request_row.get("num_gpus", 1) or 1)
        if policy == "health_only":
            selected_node_id = first_ready_node(worker_nodes, requested_gpus)
            decision = {
                "action": "allocate" if selected_node_id else "no_capacity",
                "model_name": model_name,
                "requested_gpus": requested_gpus,
                "selected_node_id": selected_node_id,
                "candidates": [],
                "reason": "health_only",
            }
        elif policy == "risk_aware":
            decision = plan_risk_aware_scheduling(
                model_name=model_name,
                worker_nodes=worker_nodes,
                requested_gpus=requested_gpus,
                scheduler_config=scheduler_config,
            ).to_dict()
            selected_node_id = decision.get("selected_node_id")
            if metrics_writer is not None:
                metrics_writer.emit(
                    make_risk_aware_scheduling_event(
                        model=model_name,
                        policy=policy,
                        decision=decision,
                    )
                )
        else:
            raise ValueError(f"Unsupported scheduler policy: {policy}")

        consume_gpu(worker_nodes, selected_node_id, requested_gpus)
        allocations.append(
            {
                "request_id": request_row.get("request_id"),
                "model": model_name,
                "requested_gpus": requested_gpus,
                "selected_node_id": selected_node_id,
                "decision": decision,
            }
        )

    selected_nodes = [
        row["selected_node_id"] for row in allocations if row["selected_node_id"]
    ]
    selected_risks = [
        float(
            payload.get("worker_nodes", {})
            .get(node_id, {})
            .get("spot_risk", 0.0)
            or 0.0
        )
        for node_id in selected_nodes
    ]
    return {
        "policy": policy,
        "requests": len(allocations),
        "allocations": len(selected_nodes),
        "selected_nodes": selected_nodes,
        "avg_selected_spot_risk": (
            sum(selected_risks) / len(selected_risks)
            if selected_risks
            else 0.0
        ),
        "allocation_rows": allocations,
    }


def run_benchmark(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    payload = load_json(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "scheduler_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    metrics_writer = JsonlMetricsWriter(metrics_path)

    policies = payload.get("policies", ["health_only", "risk_aware"])
    policy_summaries = [
        run_policy(policy, payload, metrics_writer=metrics_writer)
        for policy in policies
    ]
    summary = {
        "input": str(input_path),
        "policies": policy_summaries,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "allocations.json",
        {"policies": policy_summaries},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a synthetic SpotServe risk-aware scheduling benchmark"
    )
    parser.add_argument(
        "--input",
        default="benchmarks/spotserve/risk_aware_scheduling_synthetic.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/spotserve_risk_aware_scheduling",
    )
    args = parser.parse_args()

    summary = run_benchmark(Path(args.input), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
