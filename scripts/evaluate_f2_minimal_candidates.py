"""Evaluate Original and MoE-aware choices from one measured F2 A/B run.

This is deliberately an offline mechanism gate for a four-GPU host.  Target A
and Target B are measured sequentially, then both policies consume the same
frozen candidate records.  It is not a substitute for a live paired-target
scheduler experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def candidate(
    target_id: str, placement: str, measurement: dict[str, Any]
) -> dict[str, Any]:
    throughput = float(measurement["throughput_req_s"])
    request_count = len(measurement["responses"])
    return {
        "target_id": target_id,
        "placement": placement,
        "request_count": request_count,
        "window_elapsed_s": request_count / throughput,
        "latency_p95_ms": float(measurement["latency_p95_ms"]),
        "throughput_req_s": throughput,
        "a2a_payload_bytes": int(
            measurement["counters"]["observed_payload_bytes"]
        ),
        "a2a_collective_calls": int(
            measurement["counters"]["collective_calls"]
        ),
    }


def evaluate(report: dict[str, Any], latency_guardrail_pct: float) -> dict[str, Any]:
    if not report.get("outputs_match"):
        raise ValueError("Target A/B outputs do not match")
    if not report.get("runtime_verified_placement"):
        raise ValueError("Target B physical placement was not runtime verified")
    if int(report.get("measurement_repetitions", 0)) != 1:
        raise ValueError("Minimal F2 requires exactly one measurement per target")

    candidates = [
        candidate("target_a", "baseline", report["baseline"]),
        candidate("target_b", "physical_remap", report["candidate"]),
    ]

    original = min(
        candidates,
        key=lambda row: (row["latency_p95_ms"], row["target_id"]),
    )
    best_latency = original["latency_p95_ms"]
    latency_limit = best_latency * (1.0 + latency_guardrail_pct / 100.0)
    eligible = [
        row for row in candidates if row["latency_p95_ms"] <= latency_limit
    ]
    moe_aware = min(
        eligible,
        key=lambda row: (
            row["a2a_payload_bytes"],
            row["latency_p95_ms"],
            row["target_id"],
        ),
    )

    selected_latency_regression_pct = (
        (moe_aware["latency_p95_ms"] - best_latency) / best_latency * 100.0
    )
    selected_a2a_reduction_pct = (
        (
            original["a2a_payload_bytes"]
            - moe_aware["a2a_payload_bytes"]
        )
        / original["a2a_payload_bytes"]
        * 100.0
    )
    return {
        "experiment": "minimal_f2_sequential_two_target_selection",
        "evidence_kind": "measured_offline_same_candidate_ablation",
        "same_candidate_set": True,
        "same_workload": bool(report.get("same_workload")),
        "outputs_match": True,
        "runtime_verified_placement": True,
        "measurement_repetitions_per_target": 1,
        "latency_guardrail_pct": latency_guardrail_pct,
        "candidate_set": candidates,
        "policies": {
            "original": "minimum measured p95 latency",
            "moe_aware": (
                "minimum measured A2A payload among candidates within the "
                "latency guardrail"
            ),
        },
        "decisions": {
            "original": original["target_id"],
            "moe_aware": moe_aware["target_id"],
            "different": original["target_id"] != moe_aware["target_id"],
        },
        "selected_latency_regression_pct": selected_latency_regression_pct,
        "selected_a2a_reduction_pct": selected_a2a_reduction_pct,
        "runtime_evidence": {
            "changed_expert_count": int(report["changed_expert_count"]),
            "changed_experts": report["changed_experts"],
            "physical_weight_migration": bool(
                report.get("physical_weight_migration")
            ),
            "traffic_reduced": bool(report.get("traffic_reduced")),
        },
        "limitations": [
            "Target A and Target B were measured sequentially on four GPUs.",
            "This is not a live scheduler choice between two READY targets.",
            "This run does not include source-to-target KV recovery.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency-guardrail-pct", type=float, default=5.0)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(report, args.latency_guardrail_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
