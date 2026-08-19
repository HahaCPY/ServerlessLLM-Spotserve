from sllm.backends.vllm_state_metadata import get_vllm_inference_state
from sllm.spot.stateful_recovery import InferenceState


class FakeKvTransferParams:
    def __init__(self):
        self.block_ids = [21, 22]
        self.block_table = {"req-3": [21, 22]}


def test_vllm_inference_state_exports_token_snapshot():
    state = get_vllm_inference_state(
        model_name="vllm-moe",
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=2,
        instance_id="old-vllm-0",
        node_id="node-0",
    )

    assert state["request_id"] == "req-1"
    assert state["instance_id"] == "old-vllm-0"
    assert state["node_id"] == "node-0"
    assert state["backend"] == "vllm"
    assert state["model_name"] == "vllm-moe"
    assert state["tokens"] == [1, 2, 3]
    assert state["completed_tokens"] == 2
    assert state["state_kind"] == "token_snapshot"
    assert state["supports_restore"] is False
    assert state["metadata"]["cache_engine"] == "vllm"
    assert state["metadata"]["kv_block_count"] == 0
    assert state["metadata"]["can_restore_same_node"] is False
    assert state["metadata"]["can_restore_cross_node"] is False

    parsed = InferenceState.from_dict(state)
    assert parsed.tokens == [1, 2, 3]
    assert parsed.supports_restore is False


def test_vllm_inference_state_can_use_runtime_tokens():
    state = get_vllm_inference_state(
        model_name="vllm-dense",
        runtime_metadata={
            "request_id": "req-2",
            "prompt_tokens": [10, 11],
            "output_tokens": [12],
            "kv_block_count": 3,
            "expert_parallel_enabled": True,
            "expert_parallel_size": 2,
            "planned_expert_parallel_size": 2,
            "effective_expert_parallel_size": 2,
            "planned_effective_expert_parallel_size": 2,
            "expert_parallel_size_verified": True,
            "expert_parallel_size_source": "derived_from_tp_dp",
        },
    )

    assert state["request_id"] == "req-2"
    assert state["tokens"] == [10, 11, 12]
    assert state["completed_tokens"] == 3
    assert state["metadata"]["prompt_token_count"] == 2
    assert state["metadata"]["generated_token_count"] == 1
    assert state["metadata"]["kv_block_count"] == 3
    assert state["metadata"]["expert_parallel_enabled"] is True
    assert state["metadata"]["expert_parallel_size"] == 2
    assert state["metadata"]["planned_expert_parallel_size"] == 2
    assert state["metadata"]["effective_expert_parallel_size"] == 2
    assert state["metadata"]["planned_effective_expert_parallel_size"] == 2
    assert state["metadata"]["expert_parallel_size_verified"] is True
    assert state["metadata"]["expert_parallel_size_source"] == "derived_from_tp_dp"


def test_vllm_inference_state_preserves_kv_transfer_metadata():
    state = get_vllm_inference_state(
        model_name="vllm-dense",
        runtime_metadata={
            "request_id": "req-3",
            "prompt_tokens": [1],
            "output_tokens": [2, 3],
            "kv_transfer_params": FakeKvTransferParams(),
        },
    )

    assert state["metadata"]["kv_block_count"] == 2
    assert state["metadata"]["block_ids"] == [21, 22]
    assert state["metadata"]["block_table"] == {"req-3": [21, 22]}
    assert state["supports_restore"] is False
