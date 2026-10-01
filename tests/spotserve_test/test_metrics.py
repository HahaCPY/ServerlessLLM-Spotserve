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
                "reparallelization_execution_model": "actor_recreate",
                "reparallelization_execution_model_reason": (
                    "vllm_actor_recreate"
                ),
                "expert_placement_execution_model": (
                    "expert_aware_actor_recreate"
                ),
                "expert_placement_execution_model_reason": (
                    "logical_expert_placement_plan_carried_into_recreated_actor"
                ),
                "expert_placement_runtime_contract_mode": (
                    "observe_only_contract"
                ),
                "expert_placement_live_migration_enabled": False,
                "expert_placement_physical_migration_required": False,
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
                    "contract_seen_count": 1,
                    "contract_seen_all_workers_count": 1,
                    "contract_seen_worker_count": 2,
                    "contract_seen_worker_total": 2,
                    "physical_weight_migration_count": 0,
                    "runtime_moved_expert_shards": 4,
                    "runtime_moved_weight_bytes": 786432,
                    "runtime_remap_duration_ms": 12.25,
                    "active_request_remap_count": 1,
                    "step_boundary_barrier_count": 1,
                    "physical_host_ids_observed_count": 1,
                    "cross_node_weight_migration_count": 1,
                    "cross_node_moved_expert_shards": 2,
                    "cross_node_moved_weight_bytes": 393216,
                    "verification_levels": "contract_seen_only",
                    "verified_placement_count": 0,
                    "can_verify_physical_placement_count": 0,
                    "can_remap_live_ep_rank_count": 0,
                    "can_measure_all_to_all_count": 0,
                    "capability_reasons": (
                        "vllm_live_ep_rank_remap_not_supported"
                    ),
                    "plan_applied_count": 0,
                    "plan_verified_count": 0,
                    "contract_reasons": (
                        "physical_expert_placement_migration_not_supported"
                    ),
                    "expert_placement_execution_models": (
                        "expert_aware_actor_recreate"
                    ),
                    "expert_placement_contract_modes": (
                        "observe_only_contract"
                    ),
                    "expert_placement_live_migration_count": 0,
                    "expert_placement_physical_migration_required_count": 0,
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
    assert event["expert_placement_runtime_contract_seen_count"] == 1
    assert (
        event["expert_placement_runtime_contract_seen_all_workers_count"] == 1
    )
    assert event["expert_placement_runtime_contract_seen_worker_count"] == 2
    assert event["expert_placement_runtime_contract_seen_worker_total"] == 2
    assert event["expert_placement_runtime_physical_weight_migration_count"] == 0
    assert event["expert_placement_runtime_moved_expert_shards"] == 4
    assert event["expert_placement_runtime_moved_weight_bytes"] == 786432
    assert event["expert_placement_runtime_remap_duration_ms"] == 12.25
    assert event["expert_placement_runtime_active_request_remap_count"] == 1
    assert event["expert_placement_runtime_step_boundary_barrier_count"] == 1
    assert event["expert_placement_runtime_physical_host_ids_observed_count"] == 1
    assert event["expert_placement_runtime_cross_node_weight_migration_count"] == 1
    assert event["expert_placement_runtime_cross_node_moved_expert_shards"] == 2
    assert event["expert_placement_runtime_cross_node_moved_weight_bytes"] == 393216
    assert event["expert_placement_runtime_verification_levels"] == (
        "contract_seen_only"
    )
    assert event["expert_placement_runtime_verified_placement_count"] == 0
    assert (
        event[
            "expert_placement_runtime_can_verify_physical_placement_count"
        ]
        == 0
    )
    assert event["expert_placement_runtime_can_remap_live_ep_rank_count"] == 0
    assert event["expert_placement_runtime_can_measure_all_to_all_count"] == 0
    assert event["expert_placement_runtime_capability_reasons"] == (
        "vllm_live_ep_rank_remap_not_supported"
    )
    assert event["expert_placement_runtime_plan_applied_count"] == 0
    assert event["expert_placement_runtime_plan_verified_count"] == 0
    assert event["reparallelization_execution_model"] == "actor_recreate"
    assert event["expert_placement_execution_model"] == (
        "expert_aware_actor_recreate"
    )
    assert event["expert_placement_runtime_contract_mode"] == (
        "observe_only_contract"
    )
    assert event["expert_placement_live_migration_enabled"] is False
    assert event["expert_placement_physical_migration_required"] is False
    assert event["expert_placement_actor_recreate"] is True
    assert event["expert_placement_live_migration"] is False
    assert event["expert_placement_runtime_execution_models"] == (
        "expert_aware_actor_recreate"
    )
    assert event["expert_placement_runtime_contract_modes"] == (
        "observe_only_contract"
    )


def test_quiescent_remap_reports_actor_recreate_and_physical_remap_separately():
    event = make_replanning_event(
        model="moe-model",
        event="preempt",
        node_id="0",
        instance_id="instance-0",
        decision={
            "action": "reparallelize",
            "execution": {
                "status": "applied",
                "reparallelization_execution_model": "actor_recreate",
                "expert_placement_execution_model": "quiescent_fixed_ep_remap",
            },
        },
    )

    assert event["expert_placement_actor_recreate"] is True
    assert event["expert_placement_quiescent_remap"] is True
    assert event["expert_placement_live_migration"] is False
