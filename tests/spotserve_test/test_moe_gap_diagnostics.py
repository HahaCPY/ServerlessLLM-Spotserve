from copy import deepcopy

import pytest

from scripts.run_moe_gap_diagnostics import (
    artifact_row, compare_pair, load_functions, migration_score_probe,
    patch_hook_probe, request_rows,
)


def fixture_run(request_id="run-specific-id"):
    return {
        "mode": "original_migration",
        "prompt_audit": {request_id: {"token_ids_sha256": "prompt-hash", "token_count": 4}},
        "recovery": {
            "source_completed_tokens": {request_id: 2},
            "source_prefix_tokens": {request_id: 6},
            "target_requested_tokens": {request_id: 3},
            "moe_route_profiles": {request_id: {"per_request_expert_route_histogram": {
                "layer:0/expert:0": 9, "layer:0/expert:1": 1,
            }}},
            "configuration_applied": True, "restored_blocks": 7, "recomputed_tokens": 1,
            "sequence_checks": {request_id: {"reference_sha256": "output", "actual_sha256": "output"}},
        },
    }


def test_request_normalization_ignores_run_specific_ids():
    assert request_rows(fixture_run("first")) == request_rows(fixture_run("second"))


def test_request_normalization_detects_different_preempt_boundary():
    changed = fixture_run()
    changed["recovery"]["source_completed_tokens"]["run-specific-id"] += 1
    assert request_rows(fixture_run()) != request_rows(changed)


def test_missing_physical_observations_are_unknown_not_measured_zero(tmp_path, monkeypatch):
    from scripts import run_moe_gap_diagnostics as runner
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")
    row = artifact_row(path, fixture_run())
    assert row["configuration_applied_raw"] is True
    assert row["runtime_parallel_configuration_verified"] is None
    assert row["actual_transfer_bytes"] is None
    assert row["acknowledged_kv_blocks"] is None
    assert row["gpu_recomputed_tokens_observed"] is None
    assert row["expected_blocks_not_acknowledgements"] == 7


def test_equal_metadata_counters_do_not_prove_reduced_physical_work():
    original = {"mode": "original_migration", "path": "original", "requests": [],
                "migration_plans": [], "parallel_plan": {},
                "expected_blocks_not_acknowledgements": 7,
                "metadata_recompute_counter_not_gpu_observation": 1}
    moe = {**deepcopy(original), "path": "moe"}
    comparison = compare_pair(original, moe)
    assert comparison["expected_blocks_equal"] is True
    assert comparison["measured_physical_work_reduction"] is None


def test_score_switch_works_but_same_coverage_different_rank_is_invisible():
    probe = migration_score_probe(fixture_run())
    assert probe["passed"] is True
    assert probe["scores"]["full_coverage_on"]["locality_ratio"] == 1.0
    assert probe["scores"]["same_keys_different_rank_on"]["rank_locality_available"] is False
    assert probe["scores"]["missing_one_expert_on"]["cost"] == 9.0


def test_repository_patch_hooks_receive_contract_without_applying_placement():
    probe = patch_hook_probe()
    assert probe["passed"] is True
    assert probe["apply_result"]["physical_weight_migration"] is False
    assert probe["verify_result"]["contract_seen_by_runtime"] is True


def test_function_extraction_does_not_execute_module_main_or_imports():
    functions = load_functions("raise RuntimeError('must not execute')\ndef probe():\n    return 3\n", {"probe"}, {}, "fixture")
    assert functions["probe"]() == 3


def test_function_extraction_rejects_missing_definition():
    with pytest.raises(ValueError):
        load_functions("def other():\n    pass\n", {"probe"}, {}, "fixture")
