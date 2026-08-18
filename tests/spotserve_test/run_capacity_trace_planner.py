"""Replay a capacity trace through the real SpotServe parallel planner.

This is the cheap, deterministic half of the four-GPU experiment.  It keeps
four one-GPU worker slots in the planner state, applies every add/remove event
from a model-specific trace, and records the selected ParallelPlan after each
change.  A separate GPU smoke can then execute the selected transition without
silently replacing planner decisions with a hard-coded TP value.
"""

import argparse
import json
from pathlib import Path

from sllm.spot.reparallelization import (
    apply_spot_event_to_worker_nodes,
    plan_dynamic_reparallelization,
)
from run_four_container_fleet_churn_smoke import load_fleet_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--min-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--target-replica-gpus", type=int, default=2)
    parser.add_argument("--max-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Every physical GPU is a separate resource slot.  A trace changes only
    # lifecycle state; total_gpu remains one and can never grow past four.
    worker_nodes = {
        str(index): {
            "ray_node_id": str(index),
            "address": "local",
            "free_gpu": 0,
            "total_gpu": 1,
            "state": "dead",
        }
        for index in range(4)
    }
    events = load_fleet_trace(args.trace)
    planner_config = {
        "model_gpu_requirement": max(args.min_tensor_parallel_size, 1),
        "target_replica_gpus": max(args.target_replica_gpus, 1),
        "min_tensor_parallel_size": max(args.min_tensor_parallel_size, 1),
        "max_tensor_parallel_size": max(args.max_tensor_parallel_size, 1),
        "max_pipeline_parallel_size": 1,
    }
    rows: list[dict] = []
    for event_index, event in enumerate(events):
        event_name = str(event["event"])
        if event_name == "DONE":
            rows.append({"event_index": event_index, **event, "done": True})
            break
        node_ids = [str(node).removeprefix("node-") for node in event["nodes"]]
        for node_id in node_ids:
            if node_id not in worker_nodes:
                raise ValueError(
                    f"trace references {node!r}; only four GPU slots node-0..3"
                )
            worker_nodes = apply_spot_event_to_worker_nodes(
                worker_nodes,
                event_name,
                node_id,
                node_info=(
                    {"free_gpu": 1, "total_gpu": 1}
                    if event_name == "add"
                    else None
                ),
            )
        # Nodes listed in one trace record change as one capacity update. This
        # avoids pretending that a simultaneous two-GPU add briefly leaves a
        # Qwen TP2 model with only one usable GPU.
        decision = plan_dynamic_reparallelization(
            model_name=args.model,
            worker_nodes=worker_nodes,
            model_config={
                "model": args.model,
                "backend": "vllm",
                "num_gpus": 4,
                "backend_config": {
                    "pretrained_model_name_or_path": args.model,
                },
            },
            planner_config=planner_config,
            event=event_name,
            node_id=",".join(node_ids),
            backend="vllm",
        )
        rows.append(
            {
                "event_index": event_index,
                "time_ms": event["time_ms"],
                "event": event_name,
                "nodes": [f"node-{node_id}" for node_id in node_ids],
                "worker_nodes": worker_nodes,
                "decision": decision,
            }
        )
    output = {
        "status": "passed",
        "model": args.model,
        "trace": args.trace,
        "gpu_slots": 4,
        "planner_config": planner_config,
        "rows": rows,
        "selected_plan_shapes": [
            {
                "event": row.get("event"),
                "nodes": row.get("nodes", []),
                "available_gpus": row.get("decision", {})
                .get("availability", {})
                .get("available_gpus"),
                "action": row.get("decision", {}).get("action"),
                "tensor_parallel_size": row.get("decision", {}).get(
                    "selected_tensor_parallel_size"
                ),
                "data_parallel_size": row.get("decision", {}).get(
                    "selected_data_parallel_size"
                ),
                "selected_total_gpus": row.get("decision", {}).get(
                    "selected_total_gpus"
                ),
            }
            for row in rows
            if "decision" in row
        ],
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output": args.output}))


if __name__ == "__main__":
    main()
