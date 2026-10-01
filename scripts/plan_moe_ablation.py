"""Read-only CPU planning gate; does not create engines or acquire GPUs.

Usage: python -m scripts.plan_moe_ablation --bundle calibration_bundle.json
The bundle contains kind, calibrations, query and available_gpu_indices.
Both versions must pass the same candidate gate before a paired plan is returned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sllm.spot.moe_ablation import plan_measured_ablation


def evaluate_bundle(bundle):
    decisions = {}
    for version, moe_aware in (("original", False), ("moe_aware", True)):
        decisions[version] = plan_measured_ablation(
            bundle["kind"], bundle["calibrations"], bundle["query"],
            bundle["available_gpu_indices"], moe_aware,
            bundle.get("model_name", "granite-moe"))
    return {"status": "planning_gate_passed", "decisions": decisions,
            "same_selection": decisions["original"]["selected_candidate"]
                == decisions["moe_aware"]["selected_candidate"],
            "formal_experiment_eligible": False,
            "scope": "cpu_planning_only_not_gpu_experiment"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
        result = evaluate_bundle(bundle)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "blocked", "reason": str(error),
                  "formal_experiment_eligible": False,
                  "scope": "cpu_planning_only_not_gpu_experiment"}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
