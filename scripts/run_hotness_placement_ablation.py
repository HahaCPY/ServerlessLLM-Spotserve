#!/usr/bin/env python3
"""Run the no-GPU hotness placement diagnostic from the implementation plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.moe_placement import build_logical_expert_placement_plan


DEFAULT_HISTOGRAM = {
    "layer:0/expert:0": 100,
    "layer:0/expert:1": 80,
    "layer:0/expert:2": 20,
    "layer:0/expert:3": 10,
}
STRATEGIES = ("linear", "round_robin", "hotness_balanced")
ALPHA_SENSITIVITY_GLOBAL = [47, 75, 8, 65]
ALPHA_SENSITIVITY_RECENT = [28, 5, 12, 56]


def _load_histograms(path: Path | None) -> tuple[dict[str, int], dict[str, int]]:
    if path is None:
        return dict(DEFAULT_HISTOGRAM), dict(DEFAULT_HISTOGRAM)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("histogram input must be a JSON object")
    global_counts = payload.get("global_expert_hotness", payload)
    recent_counts = payload.get("recent_window_expert_hotness", global_counts)
    if not isinstance(global_counts, Mapping) or not isinstance(
        recent_counts, Mapping
    ):
        raise ValueError("global/recent histograms must be JSON objects")
    return (
        {str(key): int(value) for key, value in global_counts.items()},
        {str(key): int(value) for key, value in recent_counts.items()},
    )


def run_ablation(
    *,
    global_counts: Mapping[str, int],
    recent_counts: Mapping[str, int],
    num_layers: int,
    num_experts: int,
    ep_size: int,
) -> dict[str, Any]:
    model_config = {
        "runtime_metadata": {
            "global_expert_hotness": dict(global_counts),
            "recent_window_expert_hotness": dict(recent_counts),
            "moe_route_histogram_complete_sparse": True,
            "moe_route_counter_unit": "topk_routed_assignments",
            "moe_route_aggregation_scope": "deployment",
            "moe_route_histogram_key_granularity": "layer_expert",
            "moe_route_global_reset_marker": "synthetic-ablation",
        },
        "backend_config": {
            "model_config": {
                "num_hidden_layers": num_layers,
                "num_experts": num_experts,
            }
        },
    }
    target = {
        "backend": "vllm",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": ep_size,
        "replica_count": 1,
        "enable_expert_parallel": ep_size > 1,
        "effective_expert_parallel_size": ep_size,
    }
    results: dict[str, Any] = {}
    for strategy in STRATEGIES:
        plan = build_logical_expert_placement_plan(
            model_name="hotness-ablation-moe",
            target_parallel_plan=target,
            model_config=model_config,
            planner_config={
                "target_expert_placement_strategy": strategy,
            },
        )
        if plan is None or plan.covered_expert_count != num_layers * num_experts:
            raise AssertionError(f"{strategy}: required experts were not covered")
        results[strategy] = {
            "placement_fingerprint": plan.placement_fingerprint,
            "expert_to_target_rank": dict(plan.expert_to_target_rank),
            "per_layer_rank_load": plan.to_dict()["per_layer_rank_load"],
            "per_layer_load_imbalance": dict(
                plan.per_layer_load_imbalance
            ),
            "per_layer_load_cv": dict(plan.per_layer_load_cv),
            "load_balance_penalty": plan.estimated_load_balance_penalty,
            "movement": {
                "source": plan.movement_source,
                "moved_experts": plan.moved_expert_count,
                "moved_weight_bytes": plan.moved_weight_bytes,
                "estimated_cost_ms": (
                    plan.estimated_expert_weight_movement_cost_ms
                ),
            },
            "selection_reason": plan.selection_reason,
            "hotness_snapshot_id": plan.hotness_snapshot_id,
        }
    current = {
        f"layer:0/expert:{expert_id}": {
            "rank_id": f"ep-rank:{rank}",
            "weight_size_bytes": 1024,
        }
        for expert_id, rank in enumerate((0, 0, 1, 1))
    }
    alpha_model_config = {
        "runtime_metadata": {
            "global_expert_hotness": {
                f"layer:0/expert:{index}": value
                for index, value in enumerate(ALPHA_SENSITIVITY_GLOBAL)
            },
            "recent_window_expert_hotness": {
                f"layer:0/expert:{index}": value
                for index, value in enumerate(ALPHA_SENSITIVITY_RECENT)
            },
            "moe_route_histogram_complete_sparse": True,
            "moe_route_counter_unit": "topk_routed_assignments",
            "moe_route_aggregation_scope": "deployment",
            "moe_route_histogram_key_granularity": "layer_expert",
            "moe_route_global_reset_marker": "alpha-sensitivity",
        },
        "backend_config": {
            "model_config": {"num_hidden_layers": 1, "num_experts": 4}
        },
    }
    alpha_results = {}
    for alpha in (0.8, 1.0):
        plan = build_logical_expert_placement_plan(
            model_name="hotness-alpha-sensitivity-moe",
            target_parallel_plan=target,
            model_config=alpha_model_config,
            planner_config={
                "target_expert_placement_strategy": "hotness_balanced",
                "hotness_alpha": alpha,
            },
        )
        if plan is None:
            raise AssertionError(f"alpha={alpha}: no placement plan")
        alpha_results[str(alpha)] = {
            "expert_to_target_rank": dict(plan.expert_to_target_rank),
            "load_balance_penalty": plan.estimated_load_balance_penalty,
            "hotness": dict(plan.hotness_snapshot.get("hotness", {})),
        }
    if (
        alpha_results["0.8"]["expert_to_target_rank"]
        == alpha_results["1.0"]["expert_to_target_rank"]
    ):
        raise AssertionError("alpha sensitivity did not change placement")

    movement_model_config = {
        **model_config,
        "runtime_metadata": {
            **model_config["runtime_metadata"],
            "runtime_expert_placement_shards": current,
        },
    }
    movement_results = {}
    for label, movement_weight in (("zero", 0.0), ("high", 1.0)):
        plan = build_logical_expert_placement_plan(
            model_name="hotness-movement-sensitivity-moe",
            target_parallel_plan=target,
            model_config=movement_model_config,
            planner_config={
                "target_expert_placement_strategy": "hotness_balanced",
                "current_expert_placement_snapshot": current,
                "expert_weight_movement_cost_ms_per_expert": 100,
                "hotness_movement_weight": movement_weight,
                "movement_normalization_reference_ms": 1,
            },
        )
        if plan is None:
            raise AssertionError(f"movement={label}: no placement plan")
        movement_results[label] = {
            "movement_weight": movement_weight,
            "expert_to_target_rank": dict(plan.expert_to_target_rank),
            "moved_expert_count": plan.moved_expert_count,
            "estimated_movement_cost_ms": (
                plan.estimated_expert_weight_movement_cost_ms
            ),
            "selection_reason": plan.selection_reason,
            "placement_objective": plan.placement_objective,
        }
    if movement_results["zero"]["moved_expert_count"] <= 0:
        raise AssertionError("zero movement weight did not accept a useful move")
    if movement_results["high"]["moved_expert_count"] != 0:
        raise AssertionError("high movement weight did not retain current placement")

    return {
        "model": "hotness-ablation-moe",
        "parallel_shape": target,
        "strategies": results,
        "sensitivity": {
            "alpha": alpha_results,
            "movement_weight": movement_results,
            "assertions_passed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histogram", type=Path)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    global_counts, recent_counts = _load_histograms(args.histogram)
    payload = run_ablation(
        global_counts=global_counts,
        recent_counts=recent_counts,
        num_layers=max(1, args.num_layers),
        num_experts=max(1, args.num_experts),
        ep_size=max(1, args.ep_size),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
