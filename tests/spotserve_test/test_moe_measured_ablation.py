"""CPU-only contract fixtures, NOT measured Granite optimization results."""

from copy import deepcopy
import json

import pytest

from sllm.spot.moe_ablation import plan_measured_ablation, verify_measured_runtime
from sllm.spot.moe_tp_profiles import profile_runtime_signature
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.reparallelization_executor import ReparallelizationExecutor
from scripts.plan_moe_ablation import evaluate_bundle, main


def query():
    return {"prompt_tokens": 4352, "output_tokens": 256, "concurrency": 1,
            "source_unavailable": True, "source_physical_gpu_indices": [0],
            "source_route_signature": "fixture-runtime-routes",
            "frozen_prefix_sha256": "fixture-prefix",
            "background_trace_sha256": "fixture-background",
            "cache_policy_id": "fixture-shared-warmup",
            "prompt_sha256s": ["fixture-frozen-prompt"]}


def bundle(tp, group, generic_s=1.0, conditioned_s=1.0):
    value = query()
    profile = {
        "status": "passed", "scope": "warmed_service_cost_pilot_not_formal_ablation",
        "tp": tp, "config_sha256": "cpu-fixture-not-checkpoint", "repeats": 3,
        "vllm_version": "fixture", "torch_version": "fixture", "dtype": "bfloat16",
        "cpu_offload_gb": 0, "routing_capture": False, "model_runner": "V1",
        "enforce_eager": True, "moe_backend": "triton", "enable_prefix_caching": False,
        "max_model_len": 8704, "max_num_seqs": 4, "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.8,
        "async_scheduling": True, "scheduler_cls": "fixture-default-scheduler",
        "execution": {"physical_gpu_indices": group, "model_revision": "fixture-revision"},
        "runtime_moe_module_verified": True,
        "runtime_rank_checks": [{"granite_moe_model": True, "expert_modules": True}
                                for _ in range(tp)],
        "log_audit": {"complete": True, "timed_compile_or_autotune_warning_count": 0,
                      "completed_measurement_windows": 3},
        "calibration_scope": "independent_prior_calibration",
        "calibration_context": {key: value[key] for key in
                                ("background_trace_sha256", "cache_policy_id")},
        "cells": [{"prompt_tokens": 4352, "max_new_tokens": 256, "concurrency": 1,
                   "status": "passed", "samples": [{"repeat": repeat,
                       "batch_wall_s": generic_s, "requests": [{"latency_s": generic_s,
                       "generated_tokens": 256, "finish_reason": "length",
                       "prompt_sha256": "fixture-frozen-prompt"}],
                   } for repeat in range(3)]}],
    }
    conditioned = deepcopy(profile)
    conditioned["route_conditioning"] = {
        "kind": "source_runtime_expert_routes",
        "calibration_scope": "independent_prior_calibration",
        "routing_artifact": "cpu-fixture-not-live-routing",
        **{key: value[key] for key in ("source_route_signature", "frozen_prefix_sha256",
                                      "background_trace_sha256", "cache_policy_id")},
    }
    for sample in conditioned["cells"][0]["samples"]:
        sample["batch_wall_s"] = conditioned_s
        sample["requests"][0]["latency_s"] = conditioned_s
    transition = {
        "verified": True, "runtime_tp_verified": True, "artifact": "fixture-transition",
        "calibration_scope": "independent_prior_calibration",
        "cost_partition": "ready_then_install_then_warmed_service",
        "physical_gpu_indices": group, "workload": [4352, 256, 1],
        "profile_runtime_signature": profile_runtime_signature(profile),
        **{key: value[key] for key in ("frozen_prefix_sha256", "background_trace_sha256",
                                      "cache_policy_id")},
        "samples": [{"verified": True, "artifact": f"fixture-transition-{repeat}",
                     "frozen_prefix_sha256": value["frozen_prefix_sha256"],
                     "engine_ready_ms": 100 + repeat, "state_install_ms": 30}
                    for repeat in range(3)],
    }
    return {"profile": profile, "conditioned_profile": conditioned,
            "transition": transition}


def tp_bundles():
    return {"tp1": bundle(1, [1], 1, 1), "tp2": bundle(2, [1, 3], 2, 2)}


def plan(calibrations=None, moe_aware=False, kind="reparallelization", observation=None,
         available=None):
    return plan_measured_ablation(kind, calibrations or tp_bundles(), observation or query(),
                                  [1, 2, 3] if available is None else available, moe_aware)


def test_identical_costs_do_not_force_an_ablation_gap():
    original, moe = plan(), plan(moe_aware=True)
    assert original["selected_candidate"] == moe["selected_candidate"] == "tp1"
    assert original["selected_cost"]["predicted_completion_ms"] == 1131
    assert original["configuration_applied"] is False
    assert moe["formal_experiment_eligible"] is False
    assert moe["planner_decision"]["expert_placement_plan"] is None


@pytest.mark.parametrize("field,value", [
    ("async_scheduling", False), ("scheduler_cls", "fixture-native-freeze"),
    ("async_scheduling", None),
])
def test_scheduler_mismatch_or_missing_observation_invalidates_calibration(field, value):
    values = tp_bundles()
    values["tp1"]["profile"][field] = value
    with pytest.raises(ValueError):
        plan(values)


def test_measured_route_conditioning_can_change_tp_without_gpu_count_bonus():
    values = {"tp1": bundle(1, [1], 1, 2), "tp2": bundle(2, [1, 3], 2, 0.5)}
    assert plan(values)["selected_candidate"] == "tp1"
    moe = plan(values, True)
    assert moe["selected_candidate"] == "tp2"
    assert moe["parallel_plan"]["target_nodes"] == ["gpu-1", "gpu-3"]
    assert moe["selected_cost"]["route_conditioned_cost"]["service_delta_ms"] == -1500
    assert moe["selected_cost"]["route_conditioned_cost"]["physical_dispatch_traffic_verified"] is False


def test_expensive_engine_recreation_can_outweigh_faster_service():
    values = {"tp1": bundle(1, [1], 1, 1), "tp2": bundle(2, [1, 3], 0.5, 0.5)}
    assert plan(values)["selected_candidate"] == "tp2"
    for sample in values["tp2"]["transition"]["samples"]:
        sample["engine_ready_ms"] = 5000
    assert plan(values)["selected_candidate"] == "tp1"


def test_unavailable_measured_tp2_group_is_not_replaced_with_other_two_gpus():
    values = {"tp1": bundle(1, [1], 1, 1), "tp2": bundle(2, [1, 3], 0.5, 0.5)}
    assert plan(values, available=[1, 2])["selected_candidate"] == "tp1"


def test_migration_can_distinguish_three_full_targets_without_fake_partial_experts():
    values = {"a": bundle(1, [1], 1, 2), "b": bundle(1, [2], 2, 0.5),
              "c": bundle(1, [3], 3, 3)}
    assert plan(values, kind="migration")["selected_candidate"] == "a"
    moe = plan(values, True, "migration")
    assert moe["selected_candidate"] == "b"
    assert all(row["tensor_parallel_size"] == 1 for row in moe["candidates"])
    assert "expert_placement_plan" not in moe["selected_cost"]


@pytest.mark.parametrize("key", ["frozen_prefix_sha256", "source_route_signature",
                                 "background_trace_sha256", "cache_policy_id"])
def test_conditioned_observation_mismatch_is_rejected(key):
    values = tp_bundles()
    values["tp1"]["conditioned_profile"]["route_conditioning"][key] = "wrong"
    with pytest.raises(ValueError, match=key):
        plan(values, True)


def test_missing_moe_costs_do_not_fall_back_to_equal_coverage():
    values = tp_bundles()
    del values["tp1"]["conditioned_profile"]
    assert plan(values)["selected_candidate"] == "tp1"
    with pytest.raises(ValueError):
        plan(values, True)


def test_same_length_different_frozen_tokens_cannot_reuse_route_cost():
    value = query()
    value["prompt_sha256s"] = ["other-prefix-tokens"]
    with pytest.raises(ValueError, match="frozen batch"):
        plan(moe_aware=True, observation=value)


def test_conditioned_background_is_not_allowed_to_differ_from_baseline():
    values = tp_bundles()
    values["tp1"]["conditioned_profile"]["calibration_context"]["background_trace_sha256"] = "other"
    with pytest.raises(ValueError, match="share background"):
        plan(values, True)


def test_one_transition_pilot_is_not_three_repetitions():
    values = tp_bundles()
    values["tp1"]["transition"]["samples"] = values["tp1"]["transition"]["samples"][:1]
    with pytest.raises(ValueError, match="three complete transition"):
        plan(values)


@pytest.mark.parametrize("profile_key", ["profile", "conditioned_profile"])
def test_two_service_batches_are_not_three_repetitions(profile_key):
    values = tp_bundles()
    profile = values["tp1"][profile_key]
    profile["repeats"] = 2
    profile["cells"][0]["samples"] = profile["cells"][0]["samples"][:2]
    profile["log_audit"]["completed_measurement_windows"] = 2
    with pytest.raises(ValueError, match="three complete"):
        plan(values, True)


def test_migration_duplicate_physical_target_is_rejected():
    values = {"a": bundle(1, [1]), "b": bundle(1, [1]), "c": bundle(1, [3])}
    with pytest.raises(ValueError, match="three independent"):
        plan(values, kind="migration")


def test_target_pool_cannot_include_source_gpu():
    with pytest.raises(ValueError, match="source must be outside"):
        plan(available=[0, 1, 2])


def test_copied_transition_artifacts_do_not_count_as_repetitions():
    values = tp_bundles()
    values["tp1"]["transition"]["samples"][1]["artifact"] = "fixture-transition-0"
    with pytest.raises(ValueError, match="distinct artifacts"):
        plan(values)


def test_copied_service_repeat_is_not_an_independent_measurement():
    values = tp_bundles()
    values["tp1"]["profile"]["cells"][0]["samples"][1]["repeat"] = 0
    with pytest.raises(ValueError, match="distinct service"):
        plan(values)


def test_unknown_transition_timing_is_not_zero():
    values = tp_bundles()
    values["tp1"]["transition"]["samples"][0]["state_install_ms"] = None
    with pytest.raises(ValueError, match="unknown"):
        plan(values)


def test_future_current_run_measurements_are_not_planner_inputs():
    values = tp_bundles()
    values["tp1"]["profile"]["calibration_scope"] = "current_formal_run"
    with pytest.raises(ValueError, match="precede"):
        plan(values)


def test_optional_replan_requires_a_keep_candidate_not_this_forced_bridge():
    value = query()
    value["source_unavailable"] = False
    with pytest.raises(ValueError, match="optional keep"):
        plan(observation=value)


def runtime(decision):
    selected = decision["selected_cost"]
    return {"artifact": "cpu-fixture-not-deployed",
            "observation_source": "actual_worker_runtime",
            "profile_runtime_signature": selected["profile_runtime_signature"],
            "physical_gpu_indices": selected["physical_gpu_indices"],
            "tensor_parallel_size": selected["tensor_parallel_size"],
            "pipeline_parallel_size": 1, "data_parallel_size": 1,
            "replica_count": 1, "enable_expert_parallel": False,
            "state_install_verified": True,
            "frozen_prefix_sha256": decision["frozen_prefix_sha256"],
            "ranks": [{"rank": rank, "physical_gpu_index": gpu,
                       "granite_moe_model": True, "expert_modules": True}
                      for rank, gpu in enumerate(selected["physical_gpu_indices"])]}


def test_verified_runtime_is_not_automatically_a_formal_experiment():
    decision = plan()
    proof = verify_measured_runtime(decision, runtime(decision))
    assert proof["configuration_applied"] is True
    assert proof["formal_experiment_eligible"] is False


@pytest.mark.parametrize("key,value", [("tensor_parallel_size", 2),
                                       ("physical_gpu_indices", [2]),
                                       ("state_install_verified", False),
                                       ("observation_source", "planner_echo")])
def test_runtime_plan_echo_wrong_tp_or_group_is_rejected(key, value):
    decision = plan()
    observed = runtime(decision)
    observed[key] = value
    with pytest.raises(ValueError, match="actual deployment"):
        verify_measured_runtime(decision, observed)


@pytest.mark.asyncio
async def test_executor_runtime_failure_stops_new_target_before_drain_or_switch():
    decision, events = plan(), []
    def create(target_plan):
        events.append(("create", target_plan.tensor_parallel_size))
        return "new"
    def verify(worker, target_plan):
        events.append(("verify", worker))
        observed = runtime(decision)
        observed["ranks"][0]["expert_modules"] = False
        return verify_measured_runtime(decision, observed)
    executor = ReparallelizationExecutor(
        create, lambda *_: True, lambda *_: events.append(("switch", "new")),
        lambda worker: events.append(("drain", worker)),
        lambda worker: events.append(("stop", worker)), "old", verify_runtime=verify)
    with pytest.raises(ValueError, match="MoE/TP rank"):
        await executor.apply(ParallelPlan.from_dict(decision["parallel_plan"]))
    assert events == [("create", 1), ("verify", "new"), ("stop", "new")]
    assert executor.current == "old"


@pytest.mark.asyncio
async def test_executor_verified_measured_selection_reaches_switch():
    decision, events = plan(), []
    executor = ReparallelizationExecutor(
        lambda _: "new", lambda *_: True,
        lambda *_: events.append("switch"), lambda _: events.append("drain"),
        lambda _: events.append("stop"), "old",
        verify_runtime=lambda *_: verify_measured_runtime(decision, runtime(decision)))
    await executor.apply(ParallelPlan.from_dict(decision["parallel_plan"]))
    assert events == ["drain", "switch", "stop"]
    assert executor.current == "new"


def test_cli_evaluates_both_versions_but_does_not_launch_gpu_workers():
    result = evaluate_bundle({"kind": "reparallelization", "calibrations": tp_bundles(),
                              "query": query(), "available_gpu_indices": [1, 2, 3]})
    assert result["status"] == "planning_gate_passed"
    assert result["same_selection"] is True
    assert result["formal_experiment_eligible"] is False


def test_cli_missing_calibration_is_blocked_not_a_partial_baseline_result(tmp_path, capsys):
    path = tmp_path / "cpu_fixture.json"
    path.write_text(json.dumps({"kind": "migration", "calibrations": {},
                                "query": query(), "available_gpu_indices": [1, 2, 3]}))
    assert main(["--bundle", str(path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert "decisions" not in result
    assert result["formal_experiment_eligible"] is False
