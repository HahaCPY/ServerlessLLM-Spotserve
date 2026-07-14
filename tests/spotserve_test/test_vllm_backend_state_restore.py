import asyncio

import pytest


pytest.importorskip("vllm")

from sllm.backends.backend_utils import BackendStatus
from sllm.backends.vllm_backend import LLMEngineStatusDict, VllmBackend


class FakeStatefulEngine:
    def supports_state_restore(self):
        return True

    def export_inference_state(self, request_id, **kwargs):
        return {
            "request_id": request_id,
            "instance_id": "old-vllm-0",
            "node_id": "node-0",
            "state_kind": "vllm_kv_snapshot",
            "supports_restore": True,
            "runtime_state": {"snapshot_handle": "snapshot-1"},
            "metadata": {
                "kv_block_count": 2,
                "block_ids": [10, 11],
                "block_table": {request_id: [10, 11]},
                "cache_block_size": 16,
                "cache_dtype": "auto",
                "can_restore_same_node": True,
                "can_restore_cross_node": False,
            },
        }

    def restore_inference_state(self, state, request_id, **kwargs):
        return {
            "restored": True,
            "restored_blocks": len(state["metadata"]["block_ids"]),
        }


class FakeMetadataEngine:
    def get_all_request_kv_metadata(self):
        return [self.get_request_kv_metadata("req-live")]

    def get_request_kv_metadata(self, request_id):
        return {
            "request_id": request_id,
            "found": True,
            "tokens": [1, 2, 3],
            "prompt_tokens": [1, 2],
            "output_tokens": [3],
            "completed_tokens": 3,
            "kv_block_count": 2,
            "allocated_kv_block_count": 3,
            "block_ids": [10, 11, 20],
            "kv_block_ids_by_group": [[10, 11], [20]],
            "raw_block_ids_by_group": [[10, 11], [0, 20]],
            "null_block_mask_by_group": [
                [False, False],
                [True, False],
            ],
            "block_table": {
                "group_0": [10, 11],
                "group_1": [0, 20],
            },
            "cache_block_size": 16,
            "cache_dtype": "torch.float16",
            "cache_layout": "NHD",
            "reusable_tokens_by_target": {},
            "reusable_blocks_by_target": {},
        }

def make_backend(engine=None, **backend_config):
    backend = VllmBackend.__new__(VllmBackend)
    backend.model_name = "model"
    backend.backend_config = backend_config
    backend.engine = engine
    backend.request_trace = LLMEngineStatusDict()
    backend.status = BackendStatus.RUNNING
    backend.status_lock = asyncio.Lock()
    return backend


@pytest.mark.asyncio
async def test_export_is_restorable_only_when_runtime_hooks_exist():
    unsupported = make_backend(object())
    state = await unsupported.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    assert state["supports_restore"] is False
    assert state["state_kind"] == "token_snapshot"

    supported = make_backend(FakeStatefulEngine())
    state = await supported.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    assert state["supports_restore"] is True
    assert state["state_kind"] == "vllm_kv_snapshot"
    assert state["metadata"]["kv_block_count"] == 2


@pytest.mark.asyncio
async def test_restore_attaches_state_and_rejects_incompatible_cache():
    backend = make_backend(
        FakeStatefulEngine(), block_size=16, kv_cache_dtype="auto"
    )
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    restored = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert restored == {
        "restored": True,
        "restored_blocks": 2,
        "state_kind": "vllm_kv_snapshot",
        "recovered_tokens": 3,
        "restore_scope": "same_node",
    }

    incompatible = make_backend(FakeStatefulEngine(), block_size=32)
    result = await incompatible.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert result["restored"] is False
    assert result["reason"] == "incompatible_cache_config"


@pytest.mark.asyncio
async def test_restore_rejects_cross_node_without_transfer_support():
    backend = make_backend(FakeStatefulEngine())
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"}, current_output=[[1]]
    )
    result = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-1"}
    )
    assert result["restored"] is False
    assert result["reason"] == "cross_node_restore_unsupported"


@pytest.mark.asyncio
async def test_context_metadata_reads_active_requests_from_runtime():
    backend = make_backend(FakeMetadataEngine())

    contexts = await backend.get_context_metadata("instance-0", "node-0")

    assert len(contexts) == 1
    assert contexts[0]["request_id"] == "req-live"
    assert contexts[0]["num_tokens"] == 3
    assert contexts[0]["context_blocks"] == 2
    assert contexts[0]["metadata"]["block_table"] == {
        "group_0": [10, 11],
        "group_1": [0, 20],
    }
    assert contexts[0]["metadata"]["cache_dtype"] == "torch.float16"
    assert contexts[0]["supports_state_restore"] is False


@pytest.mark.asyncio
async def test_export_queries_runtime_when_request_trace_has_no_output():
    backend = make_backend(FakeMetadataEngine())

    state = await backend.export_inference_state(
        request_data={"request_id": "req-live"}
    )

    assert state["tokens"] == [1, 2, 3]
    assert state["completed_tokens"] == 3
    assert state["metadata"]["kv_block_count"] == 2
    assert state["metadata"]["cache_layout"] == "NHD"
    assert state["supports_restore"] is False
