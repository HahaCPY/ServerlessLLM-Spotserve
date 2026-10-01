"""CPU fixtures only: never interpreted as MoE performance measurements."""

from copy import deepcopy

import pytest

from sllm.spot.moe_route_cost import (
    digest, plan_calibrated_recovery, predict_candidates, route_vector,
    validate_calibration, validate_held_out_predictions,
)
from sllm.spot.moe_tp_profiles import profile_runtime_signature


def source_case(key, offset):
    histogram = {f"layer:{layer}/expert:{(expert + offset) % 40}": 4351
                 for layer in range(32) for expert in range(8)}
    tokens = [offset + 1] * 4352
    return {"workload_id": key, "frozen": {"tokens": tokens,
        "full_prefix_sha256": digest(tokens), "remaining_tokens": 256,
        "completed_tokens": 256, "strict_requested_boundary_verified": True},
        "routes": {"histogram": histogram, "histogram_sha256": digest(histogram),
                   "origin": "live_frontend_routed_experts_chunks"}}


def calibration():
    data = {"status": "passed", "scope": "independent_prior_calibration_not_formal_runs",
        "protocol": {"revision": "CPU-fixture", "initial_prompt_tokens": 4096,
            "source_output_tokens": 512, "native_freeze_generated_tokens": 256,
            "concurrency": 1, "validation_ids": ["validation-a"]},
        "source_cases": {f"train-{i}": source_case(f"train-{i}", i) for i in range(6)},
        "candidates": {}}
    data["source_cases"]["validation-a"] = source_case("validation-a", 9)
    data["protocol_sha256"] = digest(data["protocol"])
    for group in ([1], [2], [3], [1, 3]):
        cid = "gpu" + "_".join(map(str, group)) + "-tp" + str(len(group))
        runtime = {"config_sha256": "fixture", "vllm_version": "fixture", "torch_version": "fixture",
            "dtype": "bfloat16", "cpu_offload_gb": 0, "routing_capture": False,
            "model_runner": "V1", "enforce_eager": True, "moe_backend": "triton",
            "enable_prefix_caching": False, "max_model_len": 8704, "max_num_seqs": 1,
            "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.8,
            "async_scheduling": False, "scheduler_cls": "fixture-native",
            "tp": len(group), "execution": {"physical_gpu_indices": group, "model_revision": "CPU-fixture"}}
        signature = profile_runtime_signature(runtime)
        candidate = {"candidate_id": cid, "physical_gpu_indices": group, "tp": len(group),
            "runtime": runtime, "runtime_signature": signature, "ready_samples": [
                {"artifact": f"{cid}-{i}", "runtime_signature": signature,
                 "ready_ms": 100, "host_handoff_ms": 2} for i in range(3)], "cases": {}}
        for key, source in data["source_cases"].items():
            candidate["cases"][key] = [{"request_id": f"{cid}-{key}-{i}",
                "latency_s": 1 if len(group) == 1 else 2, "ttft_s": 0.1, "generated_tokens": 256,
                "finish_reason": "length", "matches_uninterrupted_reference": True,
                "timed_jit_warnings": [], "handoff_ack": {
                    "frozen_prefix_sha256": source["frozen"]["full_prefix_sha256"],
                    "install_mode": "host_token_handoff_excluding_gpu_prefill"}} for i in range(3)]
        data["candidates"][cid] = candidate
    return data


def test_same_costs_do_not_force_tp_difference():
    data = calibration()
    source = data["source_cases"]["validation-a"]
    original = plan_calibrated_recovery(data, source, "reparallelization", False)
    moe = plan_calibrated_recovery(data, source, "reparallelization", True)
    assert original["selected_candidate"] == moe["selected_candidate"] == "gpu1-tp1"
    assert moe["planner_decision"]["selected_tensor_parallel_size"] == 1
    assert moe["configuration_applied"] is False


def test_measured_tp2_win_is_permitted_without_tp2_bonus():
    data = calibration()
    for samples in data["candidates"]["gpu1_3-tp2"]["cases"].values():
        for row in samples:
            row["latency_s"] = 0.5
    decision = plan_calibrated_recovery(data, data["source_cases"]["validation-a"], "reparallelization", True)
    assert decision["parallel_plan"]["tensor_parallel_size"] == 2
    assert decision["parallel_plan"]["target_nodes"] == ["gpu-1", "gpu-3"]


def test_prior_route_neighbours_can_change_the_plan_not_the_objective():
    data = calibration()
    for key, samples in data["candidates"]["gpu1-tp1"]["cases"].items():
        for row in samples:
            row["latency_s"] = 3 if key in {"train-3", "train-4", "train-5"} else 0.1
    for samples in data["candidates"]["gpu1_3-tp2"]["cases"].values():
        for row in samples:
            row["latency_s"] = 1.8
    source = data["source_cases"]["validation-a"]
    assert plan_calibrated_recovery(data, source, "reparallelization", False)["selected_candidate"] == "gpu1-tp1"
    assert plan_calibrated_recovery(data, source, "reparallelization", True)["selected_candidate"] == "gpu1_3-tp2"


def test_migration_has_three_legal_targets_but_only_tp1():
    data = calibration()
    decision = plan_calibrated_recovery(data, data["source_cases"]["validation-a"], "migration", True,
                                       resident_groups=[[1], [2], [3]])
    assert len(decision["candidates"]) == 3
    assert all(row["tensor_parallel_size"] == 1 for row in decision["candidates"])
    assert all(row["engine_ready_cost_ms"] == 0 for row in decision["candidates"])


def test_unavailable_calibrated_group_is_not_silently_replaced():
    data = calibration()
    with pytest.raises(ValueError, match="available"):
        plan_calibrated_recovery(data, data["source_cases"]["validation-a"], "reparallelization", True,
                                 available=[2])


@pytest.mark.parametrize("damage", ["runtime", "partial", "jit", "cold", "routes", "duplicate"])
def test_incomplete_or_fabricated_evidence_is_rejected(damage):
    data = calibration()
    candidate = data["candidates"]["gpu1-tp1"]
    if damage == "runtime":
        candidate["runtime"]["async_scheduling"] = True
    elif damage == "partial":
        candidate["cases"]["train-0"][0]["generated_tokens"] = 255
    elif damage == "jit":
        candidate["cases"]["train-0"][0]["timed_jit_warnings"] = ["fixture-warning"]
    elif damage == "cold":
        candidate["ready_samples"] = candidate["ready_samples"][:1]
    elif damage == "routes":
        data["source_cases"]["train-0"]["routes"]["histogram_sha256"] = "wrong"
    else:
        data["source_cases"]["train-1"]["frozen"] = deepcopy(data["source_cases"]["train-0"]["frozen"])
    with pytest.raises(ValueError):
        validate_calibration(data)


def test_query_id_or_prefix_leakage_is_rejected():
    data = calibration()
    query = deepcopy(data["source_cases"]["train-0"])
    with pytest.raises(ValueError, match="leaked"):
        predict_candidates(data, query)
    query["workload_id"] = "new-name-same-prefix"
    with pytest.raises(ValueError, match="leaked"):
        predict_candidates(data, query)


def test_route_removal_reproduces_generic_and_validation_does_not_tune():
    data = calibration()
    rows = predict_candidates(data, data["source_cases"]["validation-a"], "removed")
    assert all(row["generic_service_ms"] == row["route_service_ms"] for row in rows.values())
    diagnostic = validate_held_out_predictions(data)
    assert diagnostic["validation_does_not_tune_k_or_force_plan_difference"]
    assert not diagnostic["causal_moe_specific_benefit_established"]


def test_out_of_bounds_expert_is_not_coverage_data():
    routes = source_case("x", 0)["routes"]
    routes["histogram"]["layer:0/expert:40"] = 1
    routes["histogram_sha256"] = digest(routes["histogram"])
    with pytest.raises(ValueError, match="bounds"):
        route_vector(routes)


def test_variable_shape_forecast_is_explicit_not_an_exact_measured_cell():
    data = calibration()
    source = deepcopy(data["source_cases"]["validation-a"])
    source["frozen"]["tokens"] = [9] * 4224
    source["frozen"]["full_prefix_sha256"] = digest(source["frozen"]["tokens"])
    source["frozen"]["completed_tokens"], source["frozen"]["remaining_tokens"] = 128, 384
    with pytest.raises(ValueError, match="shape"):
        predict_candidates(data, source)
    row = predict_candidates(data, source, "removed", variable_shape=True)["gpu1-tp1"]
    assert row["variable_shape_prediction_requires_independent_shape_validation"]
    assert row["generic_service_ms"] > 1000


def test_shape_scaling_preserves_reference_cell_prediction():
    data = calibration()
    source = data["source_cases"]["validation-a"]
    exact = predict_candidates(data, source, "routes")
    scaled = predict_candidates(data, source, "routes", variable_shape=True)
    for key in exact:
        assert scaled[key]["route_service_ms"] == pytest.approx(exact[key]["route_service_ms"])


def test_overall_compares_three_tp1_targets_and_tp2_without_duplicate_tp_overwrite():
    data = calibration()
    for samples in data["candidates"]["gpu2-tp1"]["cases"].values():
        for row in samples:
            row["latency_s"] = 0.5
    for samples in data["candidates"]["gpu3-tp1"]["cases"].values():
        for row in samples:
            row["latency_s"] = 10
    decision = plan_calibrated_recovery(data, data["source_cases"]["validation-a"], "overall", False)
    assert len(decision["candidates"]) == 4
    assert decision["selected_candidate"] == "gpu2-tp1"
    assert decision["parallel_plan"]["target_nodes"] == ["gpu-2"]
    alone = plan_calibrated_recovery(data, data["source_cases"]["validation-a"], "overall", True,
                                    available=[2])
    assert alone["selected_candidate"] == "gpu2-tp1"
