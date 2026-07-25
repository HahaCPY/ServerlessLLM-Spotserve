from sllm.backends.vllm_context_metadata import get_vllm_context_metadata
from sllm.spot.context_migration import ContextMetadata


class FakeKvTransferParams:
    def __init__(self):
        self.block_ids = [10, 11, 12]


def test_vllm_context_metadata_reports_token_count_from_tokens():
    metadata = get_vllm_context_metadata(
        model_name="vllm-moe",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-1",
            "tokens": [1, 2, 3, 4],
        },
    )

    assert metadata["request_id"] == "req-1"
    assert metadata["instance_id"] == "old-vllm-0"
    assert metadata["node_id"] == "node-0"
    assert metadata["model_name"] == "vllm-moe"
    assert metadata["backend"] == "vllm"
    assert metadata["num_tokens"] == 4
    assert metadata["context_blocks"] == 0
    assert metadata["reusable_tokens_by_target"] == {}
    assert metadata["reusable_blocks_by_target"] == {}
    assert metadata["supports_state_export"] is False
    assert metadata["supports_state_restore"] is False

    context = ContextMetadata.from_dict(metadata)
    assert context.num_tokens == 4
    assert context.context_blocks == 0


def test_vllm_context_metadata_preserves_explicit_reuse_maps():
    metadata = get_vllm_context_metadata(
        model_name="vllm-moe",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-1",
            "num_tokens": 8,
            "context_blocks": 2,
            "reusable_tokens_by_target": {
                "new-vllm-0": 8,
                "node-0": 8,
            },
            "reusable_blocks_by_target": {
                "new-vllm-0": 2,
                "node-0": 2,
            },
        },
    )

    assert metadata["num_tokens"] == 8
    assert metadata["context_blocks"] == 2
    assert metadata["reusable_tokens_by_target"] == {
        "new-vllm-0": 8,
        "node-0": 8,
    }
    assert metadata["reusable_blocks_by_target"] == {
        "new-vllm-0": 2,
        "node-0": 2,
    }


def test_vllm_context_metadata_counts_prompt_and_output_tokens():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-2",
            "prompt_tokens": [1, 2],
            "output_tokens": [3, 4, 5],
        },
    )

    assert metadata["num_tokens"] == 5


def test_vllm_context_metadata_preserves_backend_metadata():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-3",
            "tokens": [1, 2, 3],
            "metadata": {
                "prompt_token_count": 1,
                "generated_token_count": 2,
            },
        },
    )

    assert metadata["metadata"] == {
        "prompt_token_count": 1,
        "generated_token_count": 2,
    }


def test_vllm_context_metadata_counts_explicit_kv_blocks():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-4",
            "tokens": [1, 2, 3],
            "kv_block_count": 4,
        },
    )

    assert metadata["context_blocks"] == 4


def test_vllm_context_metadata_preserves_runtime_kv_fields():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-4b",
            "tokens": [1, 2, 3, 4, 5, 6, 7, 8],
            "kv_block_count_by_group": [2, 2],
            "raw_block_ids_by_group": [[10, 11], [20, 21]],
            "cache_block_size": 4,
            "cache_dtype": "torch.float16",
            "cache_layout": "NHD",
            "cache_config_fingerprint": "cache-fp",
            "model_revision": "rev-1",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
    )

    assert metadata["context_blocks"] == 2
    assert metadata["cache_block_size"] == 4
    assert metadata["cache_dtype"] == "torch.float16"
    assert metadata["cache_layout"] == "NHD"
    assert metadata["metadata"]["raw_block_ids_by_group"] == [
        [10, 11],
        [20, 21],
    ]
    assert metadata["metadata"]["cache_config_fingerprint"] == "cache-fp"
    context = ContextMetadata.from_dict(metadata)
    assert context.metadata["model_revision"] == "rev-1"


def test_vllm_context_metadata_uses_nested_runtime_metadata():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-4c",
            "tokens": [1, 2, 3, 4],
            "metadata": {
                "kv_block_count": 1,
                "cache_block_size": 4,
                "cache_dtype": "torch.float16",
                "cache_layout": "NHD",
            },
        },
    )

    assert metadata["context_blocks"] == 1
    assert metadata["cache_block_size"] == 4
    assert metadata["cache_dtype"] == "torch.float16"
    assert metadata["cache_layout"] == "NHD"


def test_vllm_context_metadata_counts_kv_transfer_params_blocks():
    metadata = get_vllm_context_metadata(
        model_name="vllm-dense",
        instance_id="old-vllm-0",
        node_id="node-0",
        runtime_metadata={
            "request_id": "req-5",
            "tokens": [1, 2, 3],
            "kv_transfer_params": FakeKvTransferParams(),
        },
    )

    assert metadata["context_blocks"] == 3
