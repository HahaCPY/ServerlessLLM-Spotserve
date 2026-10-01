"""Synthetic contract tests; numbers are not measured Granite performance."""

from copy import deepcopy
import json

import pytest

from sllm.spot.moe_tp_profiles import (
    audit_profile_log, measured_supported_configs, profile_runtime_signature,
    service_cost_for_workload,
)


def log_event(phase):
    return "MOE_TP_PROFILE_EVENT=" + json.dumps({
        "phase": phase, "tp": 1, "prompt_tokens": 4096, "concurrency": 1,
        "repeat": 0,
    })


def test_jit_during_warmup_does_not_contaminate_measurement():
    log = "\n".join(["Triton kernel JIT compilation during inference: warm_kernel",
                     log_event("measurement_start"), log_event("measurement_end"),
                     "MOE_TP_PROFILE_JSON={}"])
    audit = audit_profile_log(log)
    assert audit["complete"] is True
    assert audit["timed_compile_or_autotune_warning_count"] == 0
    assert audit["untimed_compile_or_autotune_warning_count"] == 1


def test_jit_in_a_measurement_is_not_hidden_by_warmup():
    log = "\n".join([log_event("measurement_start"),
                     "Triton kernel JIT compilation during inference: late_kernel",
                     log_event("measurement_end"), "MOE_TP_PROFILE_JSON={}"])
    assert audit_profile_log(log)["timed_compile_or_autotune_warning_count"] == 1


def test_truncated_log_cannot_claim_clean_measurements():
    assert audit_profile_log(log_event("measurement_start"))["complete"] is False


def profile(tp):
    return {
        "status": "passed", "scope": "warmed_service_cost_pilot_not_formal_ablation",
        "tp": tp, "config_sha256": "fixture-not-a-checkpoint", "repeats": 2,
        "vllm_version": "fixture", "torch_version": "fixture", "dtype": "bfloat16",
        "cpu_offload_gb": 0, "routing_capture": False, "model_runner": "V1",
        "enforce_eager": True, "moe_backend": "triton", "enable_prefix_caching": False,
        "max_model_len": 8704, "max_num_seqs": 4, "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.8,
        "async_scheduling": True, "scheduler_cls": "fixture-default-scheduler",
        "execution": {"physical_gpu_indices": [1] if tp == 1 else [1, 3],
                      "model_revision": "cpu-fixture-not-a-weight-checksum"},
        "runtime_moe_module_verified": True,
        "runtime_rank_checks": [{"granite_moe_model": True, "expert_modules": True}
                                for _ in range(tp)],
        "log_audit": {"complete": True, "timed_compile_or_autotune_warning_count": 0,
                      "completed_measurement_windows": 2},
        "cells": [{"prompt_tokens": 4096, "max_new_tokens": 512, "concurrency": 1,
                   "status": "passed", "samples": [{
                       "batch_wall_s": 2.0, "requests": [{"latency_s": 2.0,
                       "generated_tokens": 512, "finish_reason": "length"}],
                   } for _ in range(2)]}],
    }


def transitions():
    return {tp: {"verified": True, "artifact": "cpu-fixture-not-runtime",
                 "workload": [4096, 512, 1], "engine_ready_ms": 100,
                 "runtime_tp_verified": True, "frozen_prefix_sha256": "fixture-prefix",
                 "profile_runtime_signature": profile_runtime_signature(profile(tp)),
                 "physical_gpu_indices": [1] if tp == 1 else [1, 3],
                 "state_install_ms": 0} for tp in (1, 2)}


def test_exact_workload_lookup_returns_measured_service_fields():
    cost = service_cost_for_workload(profile(1), 4096, 512, 1)
    assert cost["latency_estimate_ms"] == 2000
    assert cost["throughput_estimate_req_s"] == 0.5


@pytest.mark.parametrize("workload", [(4352, 256, 1), (4096, 512, 2)])
def test_remaining_work_or_concurrency_is_not_silently_extrapolated(workload):
    with pytest.raises(ValueError, match="exact"):
        service_cost_for_workload(profile(1), *workload)


def test_jit_contaminated_profile_cannot_feed_a_decision():
    value = profile(1)
    value["log_audit"]["timed_compile_or_autotune_warning_count"] = 1
    with pytest.raises(ValueError, match="JIT"):
        service_cost_for_workload(value, 4096, 512, 1)


def test_missing_transition_cost_is_not_a_measured_zero():
    with pytest.raises(ValueError, match="never assume zero"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1, {})


def test_unknown_transition_component_is_rejected():
    evidence = transitions()
    evidence[2]["state_install_ms"] = None
    with pytest.raises(ValueError, match="unknown"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1,
                                   evidence)


def test_mismatched_checkpoint_cannot_form_a_fair_candidate_set():
    reports = {1: profile(1), 2: profile(2)}
    reports[2]["config_sha256"] = "different-checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        measured_supported_configs(reports, 4096, 512, 1, transitions())


def test_explicit_verified_transition_zero_is_allowed_not_fabricated():
    candidates = measured_supported_configs({1: profile(1), 2: profile(2)},
                                              4096, 512, 1, transitions())
    assert [row["tensor_parallel_size"] for row in candidates] == [1, 2]
    assert all(row["migration_cost_estimate_ms"] == 0 for row in candidates)


def test_transition_for_another_workload_is_rejected():
    evidence = deepcopy(transitions())
    evidence[2]["workload"] = [4352, 256, 1]
    with pytest.raises(ValueError, match="different workload"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1,
                                   evidence)


def test_incomplete_runtime_settings_are_not_treated_as_matching():
    reports = {1: profile(1), 2: profile(2)}
    del reports[1]["model_runner"]
    del reports[2]["model_runner"]
    with pytest.raises(ValueError, match="incomplete"):
        measured_supported_configs(reports, 4096, 512, 1, transitions())


def test_missing_measurement_windows_block_profile_lookup():
    value = profile(1)
    value["log_audit"]["completed_measurement_windows"] = 1
    with pytest.raises(ValueError, match="complete profile"):
        service_cost_for_workload(value, 4096, 512, 1)


def test_different_physical_gpu_groups_block_transition_reuse():
    evidence = transitions()
    evidence[2]["physical_gpu_indices"] = [2, 3]
    with pytest.raises(ValueError, match="physical GPU"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1,
                                   evidence)


def test_different_frozen_prefixes_cannot_form_paired_transition_costs():
    evidence = transitions()
    evidence[2]["frozen_prefix_sha256"] = "other-prefix"
    with pytest.raises(ValueError, match="same prefix"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1,
                                   evidence)


def test_same_config_but_different_weights_revision_is_rejected():
    reports = {1: profile(1), 2: profile(2)}
    reports[2]["execution"]["model_revision"] = "different-weights"
    with pytest.raises(ValueError, match="revisions"):
        measured_supported_configs(reports, 4096, 512, 1, transitions())


def test_transition_with_a_different_runtime_is_rejected():
    evidence = transitions()
    evidence[2]["profile_runtime_signature"] = "wrong-runtime"
    with pytest.raises(ValueError, match="profile runtime"):
        measured_supported_configs({1: profile(1), 2: profile(2)}, 4096, 512, 1,
                                   evidence)
