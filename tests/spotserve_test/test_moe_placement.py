from sllm.spot.moe_placement import (
    ExpertPlacementPlan,
    ExpertPlacementState,
    ExpertShard,
    build_logical_expert_placement_plan,
)


def test_expert_placement_state_serializes_snapshot_metadata():
    shard = ExpertShard(
        layer_id=0,
        expert_id=1,
        physical_expert_id=3,
        rank_id="rank-0",
        node_id="node-0",
        gpu_id="gpu-0",
        routed_tokens=8,
    )
    state = ExpertPlacementState(
        model_name="moe",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        vllm_data_parallel_size=1,
        sllm_replica_count=2,
        expert_parallel_enabled=True,
        effective_expert_parallel_size=2,
        expert_parallel_size_source="runtime_metadata",
        expert_physical_replication_factor=1,
        placement_epoch=4,
        placement_source="runtime_snapshot",
        shards=(shard,),
    )

    payload = state.to_dict()

    assert payload["expert_placement_available"] is True
    assert payload["sllm_replica_count"] == 2
    assert payload["vllm_data_parallel_size"] == 1
    assert payload["placement_epoch"] == 4
    assert payload["shards"][0]["physical_expert_id"] == 3


def test_expert_placement_plan_serializes_cost_fields():
    plan = ExpertPlacementPlan(
        model_name="moe",
        target_parallel_plan={"tensor_parallel_size": 2},
        expert_to_target_rank={"layer:0/expert:1": "rank-0"},
        placement_epoch=5,
        moved_expert_count=1,
        moved_weight_bytes=1024,
        estimated_dispatch_cost=0.25,
    )

    payload = plan.to_dict()

    assert payload["placement_epoch"] == 5
    assert payload["expert_to_target_rank"] == {
        "layer:0/expert:1": "rank-0"
    }
    assert payload["moved_weight_bytes"] == 1024
    assert payload["estimated_dispatch_cost"] == 0.25
    assert payload["movement_observation_available"] is False
    assert payload["movement_source"] == "unavailable"


def test_build_logical_expert_placement_plan_covers_required_experts():
    plan = build_logical_expert_placement_plan(
        model_name="tiny-moe",
        target_parallel_plan={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "replica_count": 2,
            "enable_expert_parallel": True,
            "effective_expert_parallel_size": 2,
            "target_nodes": ["node-a", "node-b"],
        },
        model_config={
            "backend_config": {
                "model_config": {
                    "num_hidden_layers": 2,
                    "num_experts": 3,
                }
            }
        },
        placement_epoch=6,
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["expert_placement_available"] is True
    assert payload["placement_epoch"] == 6
    assert payload["placement_source"] == "logical_reparallelization_planner"
    assert payload["required_expert_count"] == 6
    assert payload["covered_expert_count"] == 6
    assert payload["planned_shard_count"] == 12
    assert payload["target_rank_count"] == 2
    assert payload["sllm_replica_count"] == 2
    assert payload["physical_weight_migration"] is False
    assert payload["placement_fingerprint"]
    assert payload["expert_to_target_ranks"]["layer:0/expert:1"] == [
        "replica:0/ep-rank:1",
        "replica:1/ep-rank:1",
    ]
    assert payload["expert_placement_snapshot"]["layer:0/expert:1"][
        "node_id"
    ] == "node-b"


def test_logical_expert_placement_plan_reports_stationary_experts():
    snapshot = {
        "layer:0/expert:0": {
            "layer_id": 0,
            "expert_id": 0,
            "rank_id": "ep-rank-0",
            "node_id": "node-a",
            "gpu_id": "0",
            "weight_size_bytes": 2048,
        },
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "ep-rank-1",
            "node_id": "node-b",
            "gpu_id": "1",
            "weight_size_bytes": 2048,
        },
    }

    plan = build_logical_expert_placement_plan(
        model_name="tiny-moe",
        target_parallel_plan={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "replica_count": 1,
            "enable_expert_parallel": True,
            "effective_expert_parallel_size": 2,
            "target_nodes": ["node-a", "node-b"],
        },
        model_config={
            "runtime_metadata": {
                "expert_placement_snapshot": snapshot,
            }
        },
        placement_epoch=7,
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["movement_observation_available"] is True
    assert payload["movement_source"] == "runtime_metadata"
    assert payload["moved_expert_count"] == 0
    assert payload["stationary_expert_count"] == 2
    assert payload["unknown_movement_expert_count"] == 0
    assert payload["moved_weight_bytes"] == 0
    assert payload["estimated_expert_weight_movement_cost_ms"] == 0.0
    assert payload["expert_movement_diff"] == {}


def test_logical_expert_placement_plan_estimates_moved_experts():
    snapshot = {
        "layer:0/expert:0": {
            "layer_id": 0,
            "expert_id": 0,
            "rank_id": "ep-rank-0",
            "node_id": "node-a",
            "gpu_id": "0",
            "weight_size_bytes": 4096,
        },
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "ep-rank-1",
            "node_id": "node-a",
            "gpu_id": "1",
            "weight_size_bytes": 4096,
        },
    }

    plan = build_logical_expert_placement_plan(
        model_name="tiny-moe",
        target_parallel_plan={
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "replica_count": 1,
            "enable_expert_parallel": True,
            "effective_expert_parallel_size": 2,
            "target_nodes": ["node-b", "node-c"],
        },
        model_config={
            "runtime_metadata": {
                "expert_placement_snapshot": snapshot,
            }
        },
        planner_config={
            "expert_weight_movement_cost_ms_per_expert": 5,
        },
        placement_epoch=8,
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["movement_observation_available"] is True
    assert payload["movement_source"] == "runtime_metadata"
    assert payload["moved_expert_count"] == 2
    assert payload["stationary_expert_count"] == 0
    assert payload["unknown_movement_expert_count"] == 0
    assert payload["moved_weight_bytes"] == 8192
    assert payload["estimated_expert_weight_movement_cost_ms"] == 10.0
    assert payload["expert_movement_diff"]["layer:0/expert:0"][
        "reason"
    ] == "node_changed"


def test_build_logical_expert_placement_plan_uses_runtime_snapshot_topology():
    plan = build_logical_expert_placement_plan(
        model_name="runtime-moe",
        target_parallel_plan={
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "replica_count": 1,
            "enable_expert_parallel": False,
            "target_nodes": ["node-a"],
        },
        model_config={
            "runtime_metadata": {
                "expert_placement_snapshot": {
                    "layer:0/expert:0": {
                        "layer_id": 0,
                        "expert_id": 0,
                    },
                    "layer:1/expert:3": {
                        "layer_id": 1,
                        "expert_id": 3,
                    },
                }
            }
        },
        placement_epoch=2,
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["expert_placement_available"] is True
    assert payload["required_expert_count"] == 8
    assert payload["covered_expert_count"] == 8
    assert payload["planned_shard_count"] == 8
    assert payload["target_rank_count"] == 1
    assert payload["placement_source"] == "logical_reparallelization_planner"


def test_build_logical_expert_placement_plan_uses_nested_runtime_profile():
    plan = build_logical_expert_placement_plan(
        model_name="runtime-moe",
        target_parallel_plan={
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "replica_count": 1,
            "enable_expert_parallel": False,
            "target_nodes": ["node-a"],
        },
        model_config={
            "runtime_metadata": {
                "model_resource_profile": {
                    "expert_placement_snapshot": {
                        "layer:0/expert:0": {},
                        "layer:1/expert:3": {},
                    }
                }
            }
        },
        placement_epoch=2,
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["expert_placement_available"] is True
    assert payload["required_expert_count"] == 8
    assert payload["covered_expert_count"] == 8
