from sllm.backends.vllm_runtime_metadata import (
    get_vllm_model_resource_profile,
    get_vllm_runtime_metadata,
)
from sllm.spot.risk_aware_scheduling import node_risk_score


def test_vllm_model_resource_profile_reports_parallel_shape():
    profile = get_vllm_model_resource_profile(
        model_name="vllm-moe",
        backend_config={
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 1024,
            "max_num_seqs": 1,
        },
        runtime_metadata={
            "estimated_load_time_s": 12.5,
            "gpu_memory_required_gb": 28.0,
        },
    )

    assert profile["model_name"] == "vllm-moe"
    assert profile["backend"] == "vllm"
    assert profile["num_gpus"] == 4
    assert profile["tensor_parallel_size"] == 2
    assert profile["pipeline_parallel_size"] == 2
    assert profile["data_parallel_size"] == 1
    assert profile["expert_parallel_enabled"] is True
    assert profile["estimated_load_time_s"] == 12.5
    assert profile["gpu_memory_required_gb"] == 28.0


def test_vllm_runtime_metadata_can_feed_risk_score():
    metadata = get_vllm_runtime_metadata(
        model_name="vllm-moe",
        backend_config={"tensor_parallel_size": 4},
        instance_id="vllm-moe-0",
        node_id="node-0",
        runtime_metadata={
            "load_time_s": 8.0,
            "free_gpu": 4,
            "total_gpu": 4,
            "spot_risk": 0.2,
            "remaining_lifetime_s": 1800,
        },
    )

    assert metadata["loading_cost"] == 8.0
    assert metadata["model_resource_profile"]["num_gpus"] == 4

    score = node_risk_score(
        node_id=metadata["node_id"],
        node_info=metadata,
        requested_gpus=4,
    )
    assert score.free_gpu == 4
    assert score.total_gpu == 4
    assert score.loading_cost == 8.0
    assert score.spot_risk == 0.2


def test_vllm_runtime_metadata_omits_unknown_spot_signals():
    metadata = get_vllm_runtime_metadata(
        model_name="vllm-dense",
        backend_config={"tensor_parallel_size": 1},
        instance_id="vllm-dense-0",
        node_id="node-0",
        runtime_metadata={"load_time_s": 3.0},
    )

    assert metadata["loading_cost"] == 3.0
    assert "spot_risk" not in metadata
    assert "remaining_lifetime_s" not in metadata
