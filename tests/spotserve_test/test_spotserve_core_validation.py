"""CPU-only evidence-gate tests. Fixtures are not GPU-transfer evidence."""

import copy

import pytest

from sllm.backends.vllm_response_state import completed_response_tokens
from sllm.spot.core_validation import audit_live_kv_core


def evidence():
    baseline = [{
        "request_id": "req-0", "benchmark_phase": "measured", "success": True,
        "response": {"_spotserve_prompt_token_ids": [10, 11],
                     "_spotserve_token_ids": [20, 21, 22, 23],
                     "usage": {"completion_tokens": 4}},
    }]
    recovery = copy.deepcopy(baseline)
    recovery[0]["response"]["_spotserve_kv_restore"] = {
        "restored": True, "restored_blocks": 1, "cached_tokens": 3,
    }
    metrics = [
        {"type": "reparallelization", "event": "preempt", "execution_status": "applied",
         "target_nodes": ["4", "5"], "event_state_marked_at_s": 100,
         "planner_started_at_s": 101},
        {"type": "native_kv_receipt", "request_id": "req-0", "cached_tokens": 3,
         "restored_blocks": 1, "source_instance_id": "old-a", "target_instance_id": "new-a",
         "timestamp": 110},
        {"type": "request", "request_id": "req-0", "success": True,
         "recovery_fallback": False, "state_restore_fallback": False},
        {"type": "native_kv_source_release_authorized", "request_ids": ["req-0"],
         "source_instance_ids": ["old-a"], "timestamp": 111, "deadline_time_s": 130},
    ]
    initial = {"old-a": {"member_node_ids": ["0", "1"]}}
    final = {"new-a": {"pool": "ready", "state": "ready", "member_node_ids": ["4", "5"]}}
    return baseline, recovery, metrics, initial, final, "1", 4


def test_audited_live_kv_is_not_claimed_as_full_spotserve():
    result = audit_live_kv_core(*evidence())
    assert result["status"] == "live_kv_core_verified"
    assert all(result["checks"].values())
    assert result["full_spotserve_verified"] is False
    assert "persistent_context_owner" in result["not_verified"]


@pytest.mark.parametrize("failure", [
    "replay", "zero_cached", "planned_only", "wrong_workers", "old_target",
    "unrelated_target", "released_early", "past_deadline", "missing_prompt",
    "short_response", "different_tokens", "duplicate_request", "missing_fallback_audit",
])
def test_core_probe_rejects_missing_or_contradictory_evidence(failure):
    baseline, recovery, metrics, initial, final, worker, tokens = evidence()
    if failure == "replay":
        metrics[2]["recovery_fallback"] = True
    elif failure == "zero_cached":
        recovery[0]["response"]["_spotserve_kv_restore"]["cached_tokens"] = 0
    elif failure == "planned_only":
        metrics[0]["execution_status"] = "failed"
    elif failure == "wrong_workers":
        final["new-a"]["member_node_ids"] = ["6", "7"]
    elif failure == "old_target":
        final["old-a"] = final.pop("new-a")
    elif failure == "unrelated_target":
        metrics[1]["target_instance_id"] = "other"
    elif failure == "released_early":
        metrics[3]["timestamp"] = 109
    elif failure == "past_deadline":
        metrics[3]["timestamp"] = 131
    elif failure == "missing_prompt":
        baseline[0]["response"].pop("_spotserve_prompt_token_ids")
        recovery[0]["response"].pop("_spotserve_prompt_token_ids")
    elif failure == "short_response":
        recovery[0]["response"]["usage"]["completion_tokens"] = 2
    elif failure == "different_tokens":
        recovery[0]["response"]["_spotserve_token_ids"][0] = 99
    elif failure == "duplicate_request":
        recovery.append(copy.deepcopy(recovery[0]))
    elif failure == "missing_fallback_audit":
        metrics[2].pop("state_restore_fallback")
    result = audit_live_kv_core(baseline, recovery, metrics, initial, final, worker, tokens)
    assert result["status"] == "blocked"
    assert not all(result["checks"].values())


def test_completed_response_retains_committed_tokens_and_checks_boundary():
    assert completed_response_tokens([20, 21], [22, 23], 2, [10, 11, 20, 21]) == [20, 21, 22, 23]


@pytest.mark.parametrize("prefix,suffix,prompt,inputs", [
    ([20], [21], 2, [10, 11, 19]),
    ([20], [21], 1, [10, 11, 20]),
    ([True], [21], 2, [10, 11, True]),
    ([20], [-1], 2, [10, 11, 20]),
    ([20], [21], None, [10, 11, 20]),
])
def test_invalid_committed_boundary_is_not_silently_joined(prefix, suffix, prompt, inputs):
    with pytest.raises(ValueError):
        completed_response_tokens(prefix, suffix, prompt, inputs)
