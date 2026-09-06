import pytest

from sllm.backends.dummy_backend import DummyBackend
from sllm.spot.metrics import make_state_recovery_event
from sllm.spot.stateful_recovery import (
    InferenceState,
    plan_compatible_state_target,
    plan_stateful_recovery,
)


@pytest.fixture(autouse=True)
def clear_forced_failures():
    DummyBackend._forced_failures_seen.clear()


def test_stateful_recovery_plan_uses_backend_restore_when_supported():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-stateful",
        instance_id="old-a",
        completed_tokens=3,
        state_kind="dummy_token_state",
        supports_restore=True,
    )

    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=True,
    )

    assert decision.action == "restore_state"
    assert decision.fallback_used is False
    assert decision.recovered_tokens == 3
    assert decision.plan.target_instance_id == "new-a"


def test_stateful_recovery_plan_falls_back_without_backend_restore():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-stateful",
        instance_id="old-a",
        completed_tokens=3,
        supports_restore=False,
    )

    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=False,
    )

    assert decision.action == "fallback_token_replay"
    assert decision.fallback_used is True
    assert decision.recovered_tokens == 3


def test_planner_selects_only_compatible_ready_target():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-planner",
        instance_id="source",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "cache_block_size": 16,
            "cache_dtype": "float16",
            "can_restore_same_node": True,
        },
    )
    decision = plan_compatible_state_target(
        state,
        [
            {
                "instance_id": "tp2-target",
                "node_id": "node-0",
                "ready": True,
                "supports_state_restore": True,
                "backend": "vllm",
                "model_name": "model",
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "cache_block_size": 16,
            },
            {
                "instance_id": "tp1-target",
                "node_id": "node-0",
                "ready": True,
                "supports_state_restore": True,
                "backend": "vllm",
                "model_name": "model",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "cache_block_size": 16,
            },
        ],
        source_instance_id="source",
    )

    assert decision["action"] == "restore_state"
    assert decision["target_instance_id"] == "tp1-target"


def test_planner_does_not_reject_ep_mismatch_without_hard_dependency():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-ep-locality",
        instance_id="source",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "effective_expert_parallel_size": 1,
            "expert_parallel_enabled": False,
            "cache_block_size": 16,
            "can_restore_same_node": True,
        },
    )

    decision = plan_compatible_state_target(
        state,
        [{
            "instance_id": "ep-target",
            "node_id": "node-0",
            "ready": True,
            "supports_state_restore": True,
            "backend": "vllm",
            "model_name": "model",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "effective_expert_parallel_size": 2,
            "expert_parallel_enabled": True,
            "cache_block_size": 16,
        }],
        source_instance_id="source",
    )

    assert decision["action"] == "restore_state"
    assert decision["target_instance_id"] == "ep-target"
    assert decision["selected_candidate"]["ep_layout_required"] is False
    assert decision["selected_candidate"]["expert_placement_mismatch"] is True
    assert decision["selected_candidate"]["kv_restore_compatible"] is True


def test_planner_rejects_ep_mismatch_when_state_requires_ep_layout():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-ep-hard",
        instance_id="source",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "effective_expert_parallel_size": 1,
            "expert_parallel_enabled": False,
            "state_restore_requires_ep_layout": True,
            "can_restore_same_node": True,
        },
    )

    decision = plan_compatible_state_target(
        state,
        [{
            "instance_id": "ep-target",
            "node_id": "node-0",
            "ready": True,
            "supports_state_restore": True,
            "backend": "vllm",
            "model_name": "model",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "effective_expert_parallel_size": 2,
            "expert_parallel_enabled": True,
        }],
        source_instance_id="source",
    )

    assert decision["action"] == "fallback_token_replay"
    assert decision["candidates"][0]["instance_id"] == "ep-target"
    assert decision["candidates"][0]["reason"] == (
        "effective_expert_parallel_size_mismatch"
    )
    assert decision["candidates"][0]["compatibility"][
        "ep_layout_required"
    ] is True


def test_planner_ranks_compatible_targets_by_expert_locality():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-moe-locality",
        instance_id="source",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "cache_block_size": 16,
            "can_restore_same_node": True,
            "moe_route_histogram_available": True,
            "moe_route_histogram_source": "vllm_runtime_topk",
            "moe_route_histogram_kind": "runtime_observed_topk",
            "per_request_expert_route_histogram": {
                "req-moe-locality": {
                    "layer:0/expert:1": 10,
                }
            },
        },
    )

    decision = plan_compatible_state_target(
        state,
        [
            {
                "instance_id": "remote-expert-target",
                "node_id": "node-0",
                "ready": True,
                "supports_state_restore": True,
                "backend": "vllm",
                "model_name": "model",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "cache_block_size": 16,
                "expert_placement_available": True,
                "expert_placement_snapshot": {
                    "layer:0/expert:2": {"rank_id": "rank-0"}
                },
            },
            {
                "instance_id": "local-expert-target",
                "node_id": "node-0",
                "ready": True,
                "supports_state_restore": True,
                "backend": "vllm",
                "model_name": "model",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "cache_block_size": 16,
                "expert_placement_available": True,
                "expert_placement_snapshot": {
                    "layer:0/expert:1": {"rank_id": "rank-1"}
                },
                "placement_epoch": 3,
                "expert_placement_fingerprint": "placement-local",
                "expert_placement_plan_fingerprint": "plan-local",
                "expert_placement_snapshot_fingerprint": "snapshot-local",
                "expert_placement_contract_available": True,
                "expert_placement_plan_applied": False,
                "expert_placement_plan_verified": False,
                "expert_placement_contract_reason": "runtime_not_applied",
                "expert_placement_apply_hook_available": False,
                "expert_placement_apply_attempted": False,
                "expert_placement_apply_success": False,
                "expert_placement_apply_reason": "runtime_apply_hook_unavailable",
                "expert_placement_verify_hook_available": False,
                "expert_placement_verify_attempted": False,
                "expert_placement_verify_success": False,
                "expert_placement_verify_reason": "runtime_verify_hook_unavailable",
                "placement_source": "runtime_fixture",
            },
        ],
        source_instance_id="source",
        planner_config={
            "enable_moe_expert_locality": True,
            "expert_dispatch_weight": 10.0,
        },
    )

    assert decision["action"] == "restore_state"
    assert decision["target_instance_id"] == "local-expert-target"
    selected = decision["selected_candidate"]
    assert selected["expert_locality_available"] is True
    assert selected["hot_expert_locality_ratio"] == 1.0
    assert selected["estimated_remote_routing_ratio"] == 0.0
    assert selected["moe_dispatch_observation_available"] is True
    assert selected["moe_routed_tokens"] == 10
    assert selected["moe_local_routed_tokens"] == 10
    assert selected["moe_remote_routed_tokens"] == 0
    assert selected["moe_remote_routing_ratio"] == 0.0
    assert selected["target_placement_epoch"] == 3
    assert selected["target_expert_placement_fingerprint"] == (
        "placement-local"
    )
    assert selected["target_expert_placement_plan_fingerprint"] == (
        "plan-local"
    )
    assert selected["target_expert_placement_snapshot_fingerprint"] == (
        "snapshot-local"
    )
    assert selected["target_expert_placement_contract_available"] is True
    assert selected["target_expert_placement_plan_applied"] is False
    assert selected["target_expert_placement_plan_verified"] is False
    assert selected["target_expert_placement_contract_reason"] == (
        "runtime_not_applied"
    )
    assert selected["target_expert_placement_apply_hook_available"] is False
    assert selected["target_expert_placement_apply_attempted"] is False
    assert selected["target_expert_placement_apply_success"] is False
    assert selected["target_expert_placement_apply_reason"] == (
        "runtime_apply_hook_unavailable"
    )
    assert selected["target_expert_placement_verify_hook_available"] is False
    assert selected["target_expert_placement_verify_attempted"] is False
    assert selected["target_expert_placement_verify_success"] is False
    assert selected["target_expert_placement_verify_reason"] == (
        "runtime_verify_hook_unavailable"
    )
    assert selected["target_placement_source"] == "runtime_fixture"
    assert selected["expert_dispatch_cost"] == 0.0
    assert selected["moe_route_histogram_source"] == "vllm_runtime_topk"
    assert selected["moe_route_histogram_kind"] == "runtime_observed_topk"


def test_planner_rejects_cross_node_without_capability():
    state = InferenceState.from_tokens(
        tokens=[1],
        request_id="req-cross-node",
        instance_id="source",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={"can_restore_cross_node": False},
    )
    decision = plan_compatible_state_target(
        state,
        [{
            "instance_id": "target",
            "node_id": "node-1",
            "ready": True,
            "supports_state_restore": True,
            "backend": "vllm",
            "model_name": "model",
        }],
        source_instance_id="source",
    )

    assert decision["action"] == "fallback_token_replay"
    assert decision["reason"] == "no_compatible_ready_target"


def test_state_recovery_metric_contains_restore_summary():
    state = InferenceState.from_tokens(
        tokens=[1, 2],
        request_id="req-stateful",
        supports_restore=True,
    )
    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=True,
    )

    event = make_state_recovery_event(
        model="dummy-stateful",
        request_id="req-stateful",
        decision=decision.to_dict(),
    )

    assert event["type"] == "state_recovery"
    assert event["action"] == "restore_state"
    assert event["recovered_tokens"] == 2
    assert event["fallback_used"] is False


def test_state_recovery_metric_contains_phase3_locality_summary():
    state = InferenceState.from_tokens(
        tokens=[1, 2],
        request_id="req-stateful",
        instance_id="old-a",
        node_id="node-0",
        backend="vllm",
        model_name="model",
        completed_tokens=2,
        state_kind="vllm_kv_snapshot",
        supports_restore=True,
        metadata={
            "moe_route_histogram_available": True,
            "moe_route_histogram_source": "vllm_runtime_topk",
            "moe_route_histogram_kind": "runtime_observed_topk",
            "per_request_expert_route_histogram": {
                "req-stateful": {"layer:0/expert:1": 2}
            },
        },
    )
    target_selection = plan_compatible_state_target(
        state,
        [{
            "instance_id": "new-a",
            "node_id": "node-0",
            "ready": True,
            "supports_state_restore": True,
            "backend": "vllm",
            "model_name": "model",
            "expert_placement_available": True,
            "expert_placement_snapshot": {
                "layer:0/expert:2": {"rank_id": "rank-0"}
            },
        }],
        source_instance_id="old-a",
        planner_config={
            "enable_moe_expert_locality": True,
            "expert_dispatch_weight": 5.0,
        },
    )
    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=True,
        target_selection=target_selection,
    )

    event = make_state_recovery_event(
        model="dummy-stateful",
        request_id="req-stateful",
        decision=decision.to_dict(),
    )

    assert event["kv_restore_compatible"] is True
    assert event["ep_layout_required"] is False
    assert event["expert_locality_available"] is True
    assert event["hot_expert_locality_ratio"] == 0.0
    assert event["estimated_remote_routed_tokens"] == 2
    assert event["moe_dispatch_observation_available"] is True
    assert event["moe_routed_tokens"] == 2
    assert event["moe_local_routed_tokens"] == 0
    assert event["moe_remote_routed_tokens"] == 2
    assert event["moe_remote_routing_ratio"] == 1.0
    assert event["moe_locality_definition"] == "target_placement_coverage"
    assert event["moe_locality_granularity"] == (
        "target_instance_or_deployment"
    )
    assert event["moe_remote_routing_definition"] == (
        "missing_from_target_placement_snapshot"
    )
    assert event["moe_rank_locality_available"] is False
    assert event["moe_physical_dispatch_traffic_available"] is False
    assert event["target_placement_epoch"] is None
    assert event["placement_handshake_attempted"] is False
    assert event["moe_remote_routed_tokens_by_layer"] == {"layer:0": 2}
    assert event["moe_remote_routed_tokens_by_expert"] == {
        "layer:0/expert:1": 2
    }
    assert event["expert_dispatch_cost"] == 5.0
    assert event["moe_route_histogram_source"] == "vllm_runtime_topk"


def test_inference_state_preserves_runtime_restore_payload():
    state = InferenceState.from_dict(
        {
            "request_id": "req-stateful",
            "tokens": [1, 2],
            "supports_restore": True,
            "runtime_state": {
                "snapshot_handle": "snapshot-1",
                "expires_at": 123,
            },
        }
    )

    assert state.to_dict()["runtime_state"] == {
        "snapshot_handle": "snapshot-1",
        "expires_at": 123,
    }


@pytest.mark.asyncio
async def test_dummy_backend_exports_and_restores_inference_state():
    source = DummyBackend("dummy-stateful", {})
    target = DummyBackend("dummy-stateful", {})
    request = {
        "model": "dummy-stateful",
        "request_id": "stateful-restore-1",
        "messages": [{"role": "user", "content": "restore state"}],
        "max_tokens": 6,
        "token_latency": 0.0,
        "force_failure": "preempted",
        "force_fail_after_tokens": 2,
        "force_fail_once": True,
    }

    first_result = await source.generate(request)
    state = await source.export_inference_state(
        request_data=request,
        current_output=first_result["current_output"],
        completed_tokens=first_result["completed_tokens"],
    )
    restored = await target.restore_inference_state(
        state=state,
        request_data=request,
    )
    second_result = await target.generate(request)

    assert first_result["preempted"] is True
    assert state["supports_restore"] is True
    assert restored["restored"] is True
    assert second_result["usage"]["completion_tokens"] == 6
