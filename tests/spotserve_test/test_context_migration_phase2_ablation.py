import importlib.util
import tempfile
from pathlib import Path


def load_ablation_runner():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "run_context_migration_phase2_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "spotserve_context_migration_phase2_ablation", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_ablation_selects_expected_targets():
    runner = load_ablation_runner()
    repo_root = Path(__file__).resolve().parents[2]
    input_path = (
        repo_root
        / "benchmarks"
        / "spotserve"
        / "context_migration_phase2_ablation.json"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report = runner.run_ablation(
            input_path=input_path,
            output_dir=Path(tmpdir),
        )

    assert report["passed"] is True
    summaries = {row["run_name"]: row for row in report["runs"]}
    assert summaries["phase2-kv-only"]["selected_targets"] == [
        "target-kv-busy-remote-expert"
    ]
    assert summaries["phase2-kv-plus-expert-locality"]["selected_targets"] == [
        "target-expert-busy"
    ]
    assert summaries["phase2-kv-plus-queue"]["selected_targets"] == [
        "target-idle-remote-expert"
    ]
    assert summaries["phase2-kv-plus-expert-plus-queue"]["selected_targets"] == [
        "target-expert-idle"
    ]
    assert summaries[
        "phase2-kv-plus-expert-locality"
    ]["context_migration_moe_estimated_dispatch_cost"] > 0.0
    assert summaries[
        "phase2-kv-plus-expert-locality"
    ]["context_migration_moe_route_histogram_sources"] == "synthetic"
    assert summaries[
        "phase2-kv-plus-expert-locality"
    ]["context_migration_moe_route_histogram_kinds"] == "synthetic_ablation"
    assert summaries[
        "phase2-kv-plus-queue"
    ]["context_migration_queue_penalty_cost"] == 0.0
    assert summaries[
        "phase2-kv-plus-queue"
    ]["context_migration_avg_queue_pressure"] == 0.0
    queue_costs = summaries[
        "phase2-kv-plus-queue"
    ]["candidate_component_costs"]["phase2-req-hot-expert-1"]
    assert queue_costs[
        "target-kv-busy-remote-expert"
    ]["queue_penalty_cost"] == 20.0
    assert queue_costs[
        "target-kv-busy-remote-expert"
    ]["total_estimated_cost"] > queue_costs[
        "target-idle-remote-expert"
    ]["total_estimated_cost"]
    combined_costs = summaries[
        "phase2-kv-plus-expert-plus-queue"
    ]["candidate_component_costs"]["phase2-req-hot-expert-1"]
    assert combined_costs[
        "target-expert-idle"
    ]["total_estimated_cost"] < combined_costs[
        "target-idle-remote-expert"
    ]["total_estimated_cost"]
    assert combined_costs[
        "target-expert-idle"
    ]["expert_dispatch_cost"] < combined_costs[
        "target-idle-remote-expert"
    ]["expert_dispatch_cost"]
    assert combined_costs[
        "target-idle-remote-expert"
    ]["expert_dispatch_cost"] > 0.0
    assert any(
        row["target_selection_changed"]
        for row in report["comparisons"]
    )
