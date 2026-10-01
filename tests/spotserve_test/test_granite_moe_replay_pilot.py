"""CPU state/transport contracts, not a measured preemption experiment."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from scripts.run_granite_moe_replay_pilot import (
    backfill_client_prefix, digest, inspect_ranks, reference_output_hash,
    validate_frozen_prefix, wait_event,
)


def metadata():
    return {"found": True, "prompt_tokens": [1, 2], "output_tokens": [3, 4, 5],
            "tokens": [1, 2, 3, 4, 5], "completed_tokens": 3,
            "computed_tokens": 4, "allocated_kv_block_count": 1}


def test_overshoot_is_recorded_not_reported_as_an_exact_barrier():
    frozen = validate_frozen_prefix(metadata(), [1, 2], 6, 2)
    assert frozen["remaining_tokens"] == 3
    assert frozen["freeze_overshoot_tokens"] == 1
    assert frozen["strict_requested_boundary_verified"] is False
    assert frozen["full_prefix_sha256"] == digest([1, 2, 3, 4, 5])


def test_exact_observed_boundary_is_reported_as_exact():
    assert validate_frozen_prefix(metadata(), [1, 2], 6, 3)[
        "strict_requested_boundary_verified"] is True


@pytest.mark.parametrize("field,value", [
    ("found", False), ("prompt_tokens", [9]), ("completed_tokens", 4),
    ("tokens", [1, 2, 3]), ("output_tokens", [3]),
])
def test_invalid_frozen_state_fails_closed(field, value):
    state = deepcopy(metadata())
    state[field] = value
    with pytest.raises(ValueError):
        validate_frozen_prefix(state, [1, 2], 6, 2)


def test_already_complete_request_is_not_an_inflight_recovery():
    with pytest.raises(ValueError, match="already complete"):
        validate_frozen_prefix(metadata(), [1, 2], 3, 2)


def test_dense_or_wrong_rank_count_cannot_verify_runtime_moe_tp():
    with pytest.raises(ValueError):
        inspect_ranks(["GraniteForCausalLM"], 1)
    with pytest.raises(ValueError):
        inspect_ranks(["GraniteMoeForCausalLM GraniteMoeMoE FusedMoE"], 2)
    assert len(inspect_ranks(["GraniteMoeForCausalLM GraniteMoeMoE FusedMoE"] * 2, 2)) == 2


def test_reference_requires_repeated_matching_prompts_and_stable_outputs():
    profile = {"status": "passed", "cells": [{
        "prompt_tokens": 4096, "concurrency": 1, "max_new_tokens": 512,
        "status": "passed", "samples": [{"requests": [{
            "prompt_sha256": "fixture", "generated_tokens": 512,
            "finish_reason": "length", "output_sha256": "same-output",
        }]} for _ in range(2)],
    }]}
    assert reference_output_hash(profile, "fixture", 4096, 512) == "same-output"
    profile["cells"][0]["samples"][1]["requests"][0]["output_sha256"] = "different"
    with pytest.raises(ValueError, match="deterministic"):
        reference_output_hash(profile, "fixture", 4096, 512)


def test_worker_flags_preserve_legacy_defaults_and_allow_safe_pilot(monkeypatch):
    path = Path(__file__).with_name("cross_container_nixl_worker.py")
    spec = importlib.util.spec_from_file_location("nixl_worker_args_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["worker", "--role", "source", "--model", "/fixture", "--control-socket",
            "/fixture.sock", "--side-channel-host", "127.0.0.1", "--side-channel-port", "1"]
    monkeypatch.setattr("sys.argv", argv)
    args = module.parse_args()
    assert args.disable_routed_experts_capture is False
    assert args.no_trust_remote_code is False
    monkeypatch.setattr("sys.argv", argv + ["--disable-routed-experts-capture",
                                           "--no-trust-remote-code"])
    args = module.parse_args()
    assert args.disable_routed_experts_capture is True
    assert args.no_trust_remote_code is True


def test_worker_errors_are_not_silently_ignored():
    class Connection:
        def poll(self, _):
            return True

        def recv(self):
            return {"event": "generation_error", "traceback": "fixture failure"}

    with pytest.raises(RuntimeError, match="fixture failure"):
        wait_event(Connection(), lambda event: False, 1)


def test_computed_but_undelivered_source_tokens_are_backfilled():
    frozen = validate_frozen_prefix(metadata(), [1, 2], 6, 2)
    client_prefix, pending = backfill_client_prefix(frozen, [3, 4])
    assert pending == [5]
    assert client_prefix == frozen["generated_prefix"]


@pytest.mark.parametrize("visible", [[9], [3, 4, 5, 6]])
def test_inconsistent_client_prefix_is_not_hidden_by_metadata(visible):
    frozen = validate_frozen_prefix(metadata(), [1, 2], 6, 2)
    with pytest.raises(ValueError, match="consumer-visible"):
        backfill_client_prefix(frozen, visible)
