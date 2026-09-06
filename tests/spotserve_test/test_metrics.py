from sllm.spot.metrics import make_replanning_event


def test_replanning_event_exposes_runtime_expert_placement_hook_status():
    event = make_replanning_event(
        model="moe-model",
        event="preempt",
        node_id="0",
        instance_id="instance-0",
        decision={
            "action": "reparallelize",
            "parallel_plan": {
                "target_nodes": ["0"],
                "expert_placement_plan": {
                    "expert_placement_available": True,
                    "placement_epoch": 1,
                    "placement_fingerprint": "plan-fp",
                    "physical_weight_migration": False,
                    "movement_observation_available": True,
                    "movement_source": "runtime_metadata",
                    "moved_expert_count": 2,
                    "stationary_expert_count": 6,
                    "unknown_movement_expert_count": 0,
                    "moved_weight_bytes": 4096,
                    "estimated_expert_weight_movement_cost_ms": 12.5,
                },
            },
            "execution": {
                "status": "applied",
                "duration_ms": 1234,
                "expert_placement_runtime": {
                    "metadata_count": 1,
                    "apply_hook_available_count": 1,
                    "apply_attempted_count": 1,
                    "apply_success_count": 0,
                    "apply_reasons": (
                        "physical_expert_placement_migration_not_supported"
                    ),
                    "verify_hook_available_count": 1,
                    "verify_attempted_count": 1,
                    "verify_success_count": 0,
                    "verify_reasons": (
                        "physical_expert_placement_verification_not_supported"
                    ),
                    "plan_applied_count": 0,
                    "plan_verified_count": 0,
                    "contract_reasons": (
                        "physical_expert_placement_migration_not_supported"
                    ),
                },
            },
        },
    )

    assert event["expert_placement_plan_available"] is True
    assert event["expert_placement_plan_movement_observation_available"] is True
    assert event["expert_placement_plan_movement_source"] == "runtime_metadata"
    assert event["expert_placement_plan_moved_experts"] == 2
    assert event["expert_placement_plan_stationary_experts"] == 6
    assert event["expert_placement_plan_unknown_movement_experts"] == 0
    assert event["expert_placement_plan_moved_weight_bytes"] == 4096
    assert (
        event["expert_placement_plan_estimated_weight_movement_cost_ms"]
        == 12.5
    )
    assert event["expert_placement_runtime_metadata_count"] == 1
    assert event["expert_placement_runtime_apply_hook_available_count"] == 1
    assert event["expert_placement_runtime_apply_attempted_count"] == 1
    assert event["expert_placement_runtime_apply_success_count"] == 0
    assert event["expert_placement_runtime_verify_hook_available_count"] == 1
    assert event["expert_placement_runtime_verify_attempted_count"] == 1
    assert event["expert_placement_runtime_verify_success_count"] == 0
    assert event["expert_placement_runtime_plan_applied_count"] == 0
    assert event["expert_placement_runtime_plan_verified_count"] == 0
