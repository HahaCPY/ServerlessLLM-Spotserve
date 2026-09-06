from pathlib import Path

from scripts.run_reparallelization_phase4_movement_ablation import run_ablation


def test_phase4_movement_ablation_passes(tmp_path: Path):
    report = run_ablation(
        Path("benchmarks/spotserve/reparallelization_phase4_movement_ablation.json"),
        tmp_path,
    )

    assert report["passed"] is True

    summaries = {
        row["run_name"]: row
        for row in report["runs"]
    }
    unpenalized = summaries["phase4-movement-unpenalized"]
    penalized = summaries["phase4-movement-penalized"]

    assert unpenalized["selected_reason"] == "split_across_two_ep_ranks"
    assert unpenalized[
        "selected_expert_placement_moved_expert_count"
    ] == 2
    assert unpenalized[
        "selected_expert_placement_moved_weight_bytes"
    ] == 2097152
    assert unpenalized[
        "selected_expert_weight_movement_cost_estimate_ms"
    ] == 20
    assert unpenalized[
        "replanning_expert_placement_plan_movement_observation_events"
    ] == 1

    assert penalized["selected_reason"] == "stationary_single_rank"
    assert penalized[
        "selected_expert_placement_moved_expert_count"
    ] == 0
    assert penalized[
        "selected_expert_weight_movement_cost_estimate_ms"
    ] == 0
