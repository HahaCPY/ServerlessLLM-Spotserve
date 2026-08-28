from sllm.spot.moe_placement import (
    ExpertPlacementPlan,
    ExpertPlacementState,
    ExpertShard,
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
