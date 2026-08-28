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
            "planned_expert_parallel_size": 2,
            "expert_parallel_size": 2,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 1024,
            "max_num_seqs": 1,
        },
        runtime_metadata={
            "estimated_load_time_s": 12.5,
            "gpu_memory_required_gb": 28.0,
            "expert_parallel_size": 2,
            "expert_parallel_size_verified": True,
            "expert_parallel_size_source": "engine_args",
        },
    )

    assert profile["model_name"] == "vllm-moe"
    assert profile["backend"] == "vllm"
    assert profile["num_gpus"] == 4
    assert profile["tensor_parallel_size"] == 2
    assert profile["pipeline_parallel_size"] == 2
    assert profile["data_parallel_size"] == 1
    assert profile["vllm_data_parallel_size"] == 1
    assert profile["sllm_replica_count"] == 1
    assert profile["expert_parallel_enabled"] is True
    assert profile["planned_effective_expert_parallel_size"] == 2
    assert profile["planned_expert_parallel_size"] == 2
    assert profile["effective_expert_parallel_size"] == 2
    assert profile["expert_parallel_size"] == 2
    assert profile["expert_parallel_size_verified"] is True
    assert profile["expert_parallel_size_source"] == "engine_args"
    assert profile["expert_placement_available"] is False
    assert profile["placement_source"] == "unavailable"
    assert profile["moe_route_histogram_available"] is False
    assert profile["moe_route_histogram_source"] == "unavailable"
    assert profile["parallel_plan_mismatch"] is False
    assert profile["estimated_load_time_s"] == 12.5
    assert profile["gpu_memory_required_gb"] == 28.0


def test_vllm_model_resource_profile_derives_ep_from_tp_dp():
    profile = get_vllm_model_resource_profile(
        model_name="vllm-moe",
        backend_config={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
            "planned_effective_expert_parallel_size": 2,
        },
    )

    assert profile["expert_parallel_enabled"] is True
    assert profile["planned_effective_expert_parallel_size"] == 2
    assert profile["planned_expert_parallel_size"] == 2
    assert profile["effective_expert_parallel_size"] == 2
    assert profile["expert_parallel_size"] == 2
    assert profile["expert_parallel_size_verified"] is True
    assert profile["expert_parallel_size_source"] == "derived_from_tp_dp"
    assert profile["parallel_plan_mismatch"] is False


def test_vllm_model_resource_profile_reports_parallel_plan_mismatch():
    profile = get_vllm_model_resource_profile(
        model_name="vllm-moe",
        backend_config={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
            "planned_effective_expert_parallel_size": 4,
        },
    )

    assert profile["effective_expert_parallel_size"] == 2
    assert profile["planned_effective_expert_parallel_size"] == 4
    assert profile["parallel_plan_mismatch"] is True


def test_vllm_model_resource_profile_prefers_runtime_effective_ep():
    profile = get_vllm_model_resource_profile(
        model_name="vllm-moe",
        backend_config={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
            "planned_effective_expert_parallel_size": 4,
            "replica_count": 3,
        },
        runtime_metadata={
            "effective_expert_parallel_size": 4,
            "expert_placement_snapshot": {
                "layer:0/expert:1": {"rank_id": "rank-0"}
            },
            "placement_epoch": 7,
            "placement_source": "runtime_snapshot",
            "per_request_expert_route_histogram": {
                "req-1": {"layer:0/expert:1": 4}
            },
            "moe_route_histogram_source": "instrumentation",
        },
    )

    assert profile["effective_expert_parallel_size"] == 4
    assert profile["runtime_effective_expert_parallel_size"] == 4
    assert profile["derived_effective_expert_parallel_size"] == 2
    assert profile["expert_parallel_size"] == 4
    assert profile["expert_parallel_size_source"] == "runtime_metadata"
    assert profile["parallel_plan_mismatch"] is False
    assert profile["sllm_replica_count"] == 3
    assert profile["vllm_data_parallel_size"] == 1
    assert profile["expert_placement_available"] is True
    assert profile["placement_epoch"] == 7
    assert profile["placement_source"] == "runtime_snapshot"
    assert profile["moe_route_histogram_available"] is True
    assert profile["moe_route_histogram_source"] == "instrumentation"


def test_vllm_model_resource_profile_requires_canonical_route_histogram():
    profile = get_vllm_model_resource_profile(
        model_name="vllm-moe",
        backend_config={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "enable_expert_parallel": True,
        },
        runtime_metadata={
            "moe_route_histogram_available": True,
            "per_request_routed_tokens_by_expert": {
                "req-1": {"layer:0/expert:1": 4}
            },
            "expert_route_histogram": {
                "layer:0/expert:1": 4
            },
        },
    )

    assert profile["moe_route_histogram_available"] is False
    assert "per_request_expert_route_histogram" not in profile


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
    assert metadata["vllm_data_parallel_size"] == 1
    assert metadata["sllm_replica_count"] == 1
    assert metadata["expert_placement_available"] is False
    assert metadata["moe_route_histogram_available"] is False

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
