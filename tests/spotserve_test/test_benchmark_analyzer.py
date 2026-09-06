import importlib.util
from pathlib import Path


def load_analyzer():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "analyze_spotserve_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "spotserve_benchmark_analyzer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_summary_exposes_state_restore_evidence():
    analyzer = load_analyzer()

    summary = analyzer.summarize_router_metrics(
        [
            {
                "type": "request",
                "request_id": "req-1",
                "state_restore_attempts": 1,
                "state_restore_successes": 1,
                "state_restore_fallback": False,
                "state_restored_tokens": 16,
                "supports_state_restore": True,
                "state_kind": "vllm_kv_snapshot",
                "state_restore_reason": "nixl_kv_attach_completed",
                "state_restored_blocks": 6,
                "state_restore_staged": True,
            }
        ]
    )

    assert summary["state_restore_attempts_total"] == 1
    assert summary["state_restore_successes_total"] == 1
    assert summary["state_restore_fallback_count"] == 0
    assert summary["state_restored_tokens_total"] == 16
    assert summary["state_restored_blocks_total"] == 6
    assert summary["supports_state_restore_requests"] == 1
    assert summary["state_restore_staged_count"] == 1
    assert summary["true_kv_restore_successes_total"] == 1
    assert summary["state_kinds"] == "vllm_kv_snapshot"
    assert summary["state_restore_reasons"] == "nixl_kv_attach_completed"


def test_state_recovery_summary_exposes_phase3_moe_metrics():
    analyzer = load_analyzer()

    summary = analyzer.summarize_state_recovery_metrics(
        [
            {
                "type": "state_recovery",
                "action": "restore_state",
                "recovered_tokens": 16,
                "model_semantic_compatible": True,
                "model_semantic_reason": "compatible_model_semantics",
                "state_serialization_compatible": True,
                "state_serialization_reason": (
                    "compatible_state_serialization"
                ),
                "kv_layout_compatible": True,
                "kv_layout_reason": "compatible_kv_layout",
                "kv_restore_compatible": True,
                "ep_layout_required": False,
                "ep_layout_compatible": True,
                "ep_layout_reason": "ep_layout_mismatch_is_locality_only",
                "expert_placement_mismatch": True,
                "expert_locality_available": True,
                "hot_expert_locality_ratio": 0.25,
                "estimated_remote_routing_ratio": 0.75,
                "estimated_remote_routed_tokens": 12,
                "expert_dispatch_cost": 7.5,
                "target_placement_epoch": 4,
                "current_placement_epoch": 4,
                "target_expert_placement_fingerprint": "fp-a",
                "target_expert_placement_plan_fingerprint": "plan-a",
                "target_expert_placement_snapshot_fingerprint": "snapshot-a",
                "target_expert_placement_contract_available": True,
                "target_expert_placement_plan_applied": False,
                "target_expert_placement_plan_verified": False,
                "target_expert_placement_contract_reason": (
                    "runtime_not_applied"
                ),
                "target_expert_placement_apply_hook_available": False,
                "target_expert_placement_apply_attempted": False,
                "target_expert_placement_apply_success": False,
                "target_expert_placement_apply_reason": (
                    "runtime_apply_hook_unavailable"
                ),
                "target_expert_placement_verify_hook_available": False,
                "target_expert_placement_verify_attempted": False,
                "target_expert_placement_verify_success": False,
                "target_expert_placement_verify_reason": (
                    "runtime_verify_hook_unavailable"
                ),
                "current_expert_placement_fingerprint": "fp-a",
                "placement_handshake_attempted": True,
                "placement_handshake_verified": True,
                "placement_handshake_compatible": True,
                "placement_handshake_stale": False,
                "placement_handshake_reason": "stable_target_placement",
                "moe_dispatch_observation_available": True,
                "moe_routed_tokens": 16,
                "moe_local_routed_tokens": 4,
                "moe_remote_routed_tokens": 12,
                "moe_remote_routing_ratio": 0.75,
                "moe_local_routed_tokens_by_layer": {"layer:0": 4},
                "moe_remote_routed_tokens_by_layer": {"layer:0": 12},
                "moe_local_routed_tokens_by_expert": {
                    "layer:0/expert:1": 4
                },
                "moe_remote_routed_tokens_by_expert": {
                    "layer:0/expert:2": 12
                },
                "moe_locality_definition": "target_placement_coverage",
                "moe_locality_granularity": "target_instance_or_deployment",
                "moe_remote_routing_definition": (
                    "missing_from_target_placement_snapshot"
                ),
                "moe_rank_locality_available": False,
                "moe_physical_dispatch_traffic_available": False,
                "moe_route_histogram_source": "vllm_runtime_topk",
                "moe_route_histogram_kind": "runtime_observed_topk",
            }
        ]
    )

    assert summary["state_recovery_events"] == 1
    assert summary["state_recovery_restore_events"] == 1
    assert summary["state_recovery_model_semantic_compatible_count"] == 1
    assert summary["state_recovery_state_serialization_compatible_count"] == 1
    assert summary["state_recovery_kv_layout_compatible_count"] == 1
    assert summary["state_recovery_kv_restore_compatible_count"] == 1
    assert summary["state_recovery_ep_layout_required_count"] == 0
    assert summary["state_recovery_ep_layout_compatible_count"] == 1
    assert summary["state_recovery_expert_placement_mismatch_count"] == 1
    assert summary["state_recovery_expert_locality_available_count"] == 1
    assert summary["state_recovery_avg_hot_expert_locality_ratio"] == 0.25
    assert summary["state_recovery_avg_estimated_remote_routing_ratio"] == 0.75
    assert summary["state_recovery_estimated_remote_routed_tokens"] == 12
    assert summary["state_recovery_estimated_dispatch_cost"] == 7.5
    assert summary["state_recovery_target_placement_epochs"] == "4"
    assert summary["state_recovery_current_placement_epochs"] == "4"
    assert (
        summary["state_recovery_target_expert_placement_fingerprints"]
        == "fp-a"
    )
    assert (
        summary["state_recovery_target_expert_placement_plan_fingerprints"]
        == "plan-a"
    )
    assert (
        summary["state_recovery_target_expert_placement_snapshot_fingerprints"]
        == "snapshot-a"
    )
    assert summary["state_recovery_target_expert_placement_contracts"] == 1
    assert summary["state_recovery_target_expert_placement_plan_applied"] == 0
    assert summary["state_recovery_target_expert_placement_plan_verified"] == 0
    assert (
        summary["state_recovery_target_expert_placement_contract_reasons"]
        == "runtime_not_applied"
    )
    assert (
        summary[
            "state_recovery_target_expert_placement_apply_hook_available"
        ]
        == 0
    )
    assert (
        summary["state_recovery_target_expert_placement_apply_attempted"]
        == 0
    )
    assert summary["state_recovery_target_expert_placement_apply_success"] == 0
    assert (
        summary["state_recovery_target_expert_placement_apply_reasons"]
        == "runtime_apply_hook_unavailable"
    )
    assert (
        summary[
            "state_recovery_target_expert_placement_verify_hook_available"
        ]
        == 0
    )
    assert (
        summary["state_recovery_target_expert_placement_verify_attempted"]
        == 0
    )
    assert summary["state_recovery_target_expert_placement_verify_success"] == 0
    assert (
        summary["state_recovery_target_expert_placement_verify_reasons"]
        == "runtime_verify_hook_unavailable"
    )
    assert (
        summary["state_recovery_current_expert_placement_fingerprints"]
        == "fp-a"
    )
    assert summary["state_recovery_placement_handshake_attempts"] == 1
    assert summary["state_recovery_placement_handshake_successes"] == 1
    assert summary["state_recovery_placement_handshake_failures"] == 0
    assert summary["state_recovery_placement_handshake_stale"] == 0
    assert summary["state_recovery_placement_handshake_reasons"] == (
        "stable_target_placement"
    )
    assert (
        summary["state_recovery_moe_dispatch_observation_available_count"]
        == 1
    )
    assert summary["state_recovery_moe_routed_tokens"] == 16
    assert summary["state_recovery_moe_local_routed_tokens"] == 4
    assert summary["state_recovery_moe_remote_routed_tokens"] == 12
    assert summary["state_recovery_moe_avg_remote_routing_ratio"] == 0.75
    assert summary["state_recovery_moe_locality_definitions"] == (
        "target_placement_coverage"
    )
    assert summary["state_recovery_moe_locality_granularities"] == (
        "target_instance_or_deployment"
    )
    assert summary["state_recovery_moe_remote_routing_definitions"] == (
        "missing_from_target_placement_snapshot"
    )
    assert summary["state_recovery_moe_rank_locality_available_count"] == 0
    assert (
        summary[
            "state_recovery_moe_physical_dispatch_traffic_available_count"
        ]
        == 0
    )
    assert summary["state_recovery_moe_local_routed_tokens_by_layer"] == (
        '{"layer:0": 4}'
    )
    assert summary["state_recovery_moe_remote_routed_tokens_by_layer"] == (
        '{"layer:0": 12}'
    )
    assert summary["state_recovery_moe_local_routed_tokens_by_expert"] == (
        '{"layer:0/expert:1": 4}'
    )
    assert summary["state_recovery_moe_remote_routed_tokens_by_expert"] == (
        '{"layer:0/expert:2": 12}'
    )
    assert summary["state_recovery_moe_route_histogram_sources"] == (
        "vllm_runtime_topk"
    )
    assert summary["state_recovery_moe_route_histogram_kinds"] == (
        "runtime_observed_topk"
    )
    assert summary["state_recovery_ep_layout_reasons"] == (
        "ep_layout_mismatch_is_locality_only"
    )


def test_router_summary_separates_success_from_fallback():
    analyzer = load_analyzer()

    summary = analyzer.summarize_router_metrics(
        [
            {
                "type": "request",
                "request_id": "req-clean",
                "success": True,
                "failed_attempts": 0,
                "retry_count": 0,
                "recovery_fallback": False,
                "state_restore_fallback": False,
            },
            {
                "type": "request",
                "request_id": "req-fallback",
                "success": True,
                "failed_attempts": 1,
                "retry_count": 1,
                "recovered_tokens": 16,
                "recovery_fallback": True,
                "state_restore_attempts": 1,
                "state_restore_successes": 0,
                "state_restore_fallback": True,
            },
        ]
    )

    assert summary["router_successes"] == 2
    assert summary["clean_success_count"] == 1
    assert summary["clean_success_rate"] == 0.5
    assert summary["fallback_request_count"] == 1
    assert summary["fallback_rate"] == 0.5
    assert summary["recovery_triggered_requests"] == 1
    assert summary["recovery_triggered_rate"] == 0.5
    assert summary["state_restore_success_rate"] == 0.0
    assert summary["state_restore_fallback_rate"] == 1.0


def test_router_summary_uses_request_denominator_for_clean_success_rate():
    analyzer = load_analyzer()

    router_rows = [
        {
            "type": "request",
            "request_id": f"req-{index}",
            "success": True,
            "recovery_fallback": False,
            "state_restore_fallback": False,
        }
        for index in range(3)
    ]
    request_rows = [{"request_id": f"req-{index}"} for index in range(8)]

    summary = analyzer.summarize_router_metrics(router_rows, request_rows)

    assert summary["clean_success_count"] == 3
    assert summary["clean_success_denominator"] == 8
    assert summary["clean_success_rate"] == 0.375
    assert summary["router_clean_success_rate"] == 1.0
    assert summary["fallback_denominator"] == 8
    assert summary["fallback_rate"] == 0.0
    assert summary["router_fallback_rate"] == 0.0


def test_raw_response_summary_exposes_kv_restore_evidence():
    analyzer = load_analyzer()

    summary = analyzer.summarize_response_kv_restore(
        [
            {
                "response": {
                    "_spotserve_kv_restore": {
                        "restored": True,
                        "restored_blocks": 6,
                        "cached_tokens": 82,
                        "reason": "nixl_kv_attach_completed",
                    }
                }
            }
        ]
    )

    assert summary["response_kv_restore_events"] == 1
    assert summary["response_kv_restore_successes"] == 1
    assert summary["response_kv_restore_restored_blocks"] == 6
    assert summary["response_kv_restore_cached_tokens"] == 82
    assert summary["response_kv_restore_reasons"] == "nixl_kv_attach_completed"


def test_true_kv_summary_uses_response_evidence_for_legacy_router_rows():
    analyzer = load_analyzer()

    summary = analyzer.summarize_true_kv_restore_evidence(
        request_rows=[
            {
                "request_id": "req-1",
                "response": {
                    "_spotserve_kv_restore": {
                        "restored": True,
                        "restored_blocks": 6,
                        "cached_tokens": 82,
                        "reason": "nixl_kv_attach_completed",
                    }
                },
            }
        ],
        router_request_rows=[
            {
                "type": "request",
                "request_id": "req-1",
                "state_restore_attempts": 1,
                "state_restore_successes": 1,
                "state_restored_tokens": 16,
                "state_restored_blocks": 0,
            }
        ],
    )

    assert summary["true_kv_restore_successes_total"] == 1
    assert summary["true_kv_restored_blocks_total"] == 6
    assert summary["true_kv_restore_evidence_sources"] == "raw_response"


def test_replanning_summary_exposes_workload_cost_metrics():
    analyzer = load_analyzer()

    summary = analyzer.summarize_replanning_metrics(
        [
            {
                "type": "reparallelization",
                "action": "reparallelize",
                "selected_total_gpus": 2,
                "parallel_plan": {"target_nodes": ["node-b"]},
                "execution_status": "applied",
                "execution": {"status": "applied"},
                "execution_duration_ms": 2500,
                "workload_cost_model_enabled": True,
                "selected_replan_window_cost_ms": 5200,
                "selected_load_time_estimate_ms": 5000,
                "selected_migration_cost_estimate_ms": 200,
                "selected_expert_weight_movement_cost_estimate_ms": 12.5,
                "selected_queue_penalty_ms": 0,
                "selected_throughput_estimate_req_s": 2,
                "selected_sllm_replica_count": 2,
                "selected_vllm_data_parallel_size": 1,
                "selected_runtime_effective_expert_parallel_size": 2,
                "moe_route_histogram_available": False,
                "moe_route_histogram_source": "unavailable",
                "moe_route_histogram_kind": "unavailable",
                "moe_expert_parallel_size_source": "derived_from_tp_dp",
                "expert_placement_plan_available": True,
                "expert_placement_plan_epoch": 5,
                "expert_placement_plan_fingerprint": "placement-fp",
                "expert_placement_plan_source": (
                    "logical_reparallelization_planner"
                ),
                "expert_placement_plan_reason": (
                    "logical_expert_placement_plan"
                ),
                "expert_placement_plan_required_experts": 6,
                "expert_placement_plan_covered_experts": 6,
                "expert_placement_plan_shards": 12,
                "expert_placement_plan_physical_weight_migration": False,
                "expert_placement_plan_movement_observation_available": True,
                "expert_placement_plan_movement_source": "runtime_metadata",
                "expert_placement_plan_moved_experts": 2,
                "expert_placement_plan_stationary_experts": 4,
                "expert_placement_plan_unknown_movement_experts": 0,
                "expert_placement_plan_moved_weight_bytes": 4096,
                "expert_placement_plan_estimated_weight_movement_cost_ms": (
                    12.5
                ),
                "expert_placement_runtime_metadata_count": 1,
                "expert_placement_runtime_apply_hook_available_count": 1,
                "expert_placement_runtime_apply_attempted_count": 1,
                "expert_placement_runtime_apply_success_count": 0,
                "expert_placement_runtime_apply_reasons": (
                    "physical_expert_placement_migration_not_supported"
                ),
                "expert_placement_runtime_verify_hook_available_count": 1,
                "expert_placement_runtime_verify_attempted_count": 1,
                "expert_placement_runtime_verify_success_count": 0,
                "expert_placement_runtime_verify_reasons": (
                    "physical_expert_placement_verification_not_supported"
                ),
                "expert_placement_runtime_plan_applied_count": 0,
                "expert_placement_runtime_plan_verified_count": 0,
                "expert_placement_runtime_contract_reasons": (
                    "physical_expert_placement_migration_not_supported"
                ),
                "cross_node_target": True,
                "multi_worker_target": False,
                "target_worker_node_count": 1,
                "ready_worker_node_count": 2,
                "synthetic_worker_node_count": 0,
                "runtime_worker_node_count": 2,
                "physical_worker_node_count": 2,
            }
        ]
    )

    assert summary["replanning_events"] == 1
    assert summary["replanning_execution_applied"] == 1
    assert summary["replanning_workload_cost_model_events"] == 1
    assert summary["replanning_avg_execution_duration_ms"] == 2500
    assert summary["replanning_avg_selected_replan_window_cost_ms"] == 5200
    assert summary["replanning_avg_selected_load_time_estimate_ms"] == 5000
    assert summary["replanning_avg_selected_migration_cost_estimate_ms"] == 200
    assert (
        summary[
            "replanning_avg_selected_expert_weight_movement_cost_estimate_ms"
        ]
        == 12.5
    )
    assert summary["replanning_cross_node_targets"] == 1
    assert summary["replanning_max_ready_worker_node_count"] == 2
    assert summary["replanning_max_selected_sllm_replica_count"] == 2
    assert summary["replanning_max_selected_vllm_data_parallel_size"] == 1
    assert summary["replanning_max_selected_effective_expert_parallel_size"] == 2
    assert summary["replanning_moe_route_histogram_available_events"] == 0
    assert summary["replanning_moe_route_histogram_sources"] == "unavailable"
    assert summary["replanning_moe_route_histogram_kinds"] == "unavailable"
    assert summary["replanning_moe_expert_parallel_size_sources"] == "derived_from_tp_dp"
    assert summary["replanning_expert_placement_plan_available_events"] == 1
    assert summary["replanning_expert_placement_plan_epochs"] == "5"
    assert summary["replanning_expert_placement_plan_fingerprints"] == (
        "placement-fp"
    )
    assert summary["replanning_expert_placement_plan_sources"] == (
        "logical_reparallelization_planner"
    )
    assert summary["replanning_expert_placement_plan_reasons"] == (
        "logical_expert_placement_plan"
    )
    assert (
        summary["replanning_expert_placement_plan_physical_migration_events"]
        == 0
    )
    assert (
        summary[
            "replanning_expert_placement_plan_movement_observation_events"
        ]
        == 1
    )
    assert (
        summary["replanning_expert_placement_plan_movement_sources"]
        == "runtime_metadata"
    )
    assert summary[
        "replanning_total_expert_placement_plan_moved_experts"
    ] == 2
    assert summary[
        "replanning_max_expert_placement_plan_moved_experts"
    ] == 2
    assert summary[
        "replanning_total_expert_placement_plan_stationary_experts"
    ] == 4
    assert summary[
        "replanning_total_expert_placement_plan_unknown_movement_experts"
    ] == 0
    assert summary[
        "replanning_total_expert_placement_plan_moved_weight_bytes"
    ] == 4096
    assert (
        summary[
            "replanning_avg_expert_placement_plan_weight_movement_cost_ms"
        ]
        == 12.5
    )
    assert summary["replanning_expert_placement_runtime_metadata_count"] == 1
    assert (
        summary[
            "replanning_expert_placement_runtime_apply_hook_available"
        ]
        == 1
    )
    assert summary["replanning_expert_placement_runtime_apply_attempted"] == 1
    assert summary["replanning_expert_placement_runtime_apply_success"] == 0
    assert summary["replanning_expert_placement_runtime_apply_reasons"] == (
        "physical_expert_placement_migration_not_supported"
    )
    assert (
        summary[
            "replanning_expert_placement_runtime_verify_hook_available"
        ]
        == 1
    )
    assert summary["replanning_expert_placement_runtime_verify_attempted"] == 1
    assert summary["replanning_expert_placement_runtime_verify_success"] == 0
    assert summary["replanning_expert_placement_runtime_verify_reasons"] == (
        "physical_expert_placement_verification_not_supported"
    )
    assert summary["replanning_expert_placement_runtime_plan_applied"] == 0
    assert summary["replanning_expert_placement_runtime_plan_verified"] == 0
    assert summary["replanning_expert_placement_runtime_contract_reasons"] == (
        "physical_expert_placement_migration_not_supported"
    )
    assert summary["replanning_max_expert_placement_plan_required_experts"] == 6
    assert summary["replanning_max_expert_placement_plan_covered_experts"] == 6
    assert summary["replanning_max_expert_placement_plan_shards"] == 12
    assert summary["replanning_avg_expert_placement_plan_coverage_ratio"] == 1.0
    assert summary["replanning_max_runtime_worker_node_count"] == 2
    assert summary["replanning_max_physical_worker_node_count"] == 2


def test_context_migration_summary_exposes_moe_locality_metrics():
    analyzer = load_analyzer()

    summary = analyzer.summarize_context_migration_metrics(
        [
            {
                "type": "context_migration",
                "migration_plan_count": 1,
                "total_estimated_cost": 3.5,
                "total_reusable_tokens": 8,
                "total_context_tokens": 10,
                "selected_target_ids": ["target-a"],
                "selected_request_ids": ["req-a"],
                "selected_plan_total_estimated_cost": 3.5,
                "selected_plan_kv_migration_cost": 1.0,
                "selected_plan_expert_dispatch_cost": 1.5,
                "selected_plan_queue_penalty_cost": 2.0,
                "selected_plan_avg_queue_pressure": 0.5,
                "selected_plan_max_queue_depth": 2,
                "selected_plan_avg_hot_expert_locality_ratio": 0.75,
                "selected_plan_avg_estimated_remote_routing_ratio": 0.25,
                "selected_plan_estimated_remote_routed_tokens": 4,
                "selected_plan_target_placement_epochs": ["9"],
                "selected_plan_target_expert_placement_fingerprints": [
                    "fp-b"
                ],
                "selected_plan_target_expert_placement_plan_fingerprints": [
                    "plan-b"
                ],
                "selected_plan_target_expert_placement_snapshot_fingerprints": [
                    "snapshot-b"
                ],
                "selected_plan_target_expert_placement_contract_available_count": 1,
                "selected_plan_target_expert_placement_plan_applied_count": 0,
                "selected_plan_target_expert_placement_plan_verified_count": 0,
                "selected_plan_target_expert_placement_contract_reasons": [
                    "runtime_not_applied"
                ],
                "selected_plan_target_expert_placement_apply_hook_available_count": 0,
                "selected_plan_target_expert_placement_apply_attempted_count": 0,
                "selected_plan_target_expert_placement_apply_success_count": 0,
                "selected_plan_target_expert_placement_apply_reasons": [
                    "runtime_apply_hook_unavailable"
                ],
                "selected_plan_target_expert_placement_verify_hook_available_count": 0,
                "selected_plan_target_expert_placement_verify_attempted_count": 0,
                "selected_plan_target_expert_placement_verify_success_count": 0,
                "selected_plan_target_expert_placement_verify_reasons": [
                    "runtime_verify_hook_unavailable"
                ],
                "selected_plan_target_placement_sources": [
                    "runtime_fixture"
                ],
                "placement_handshake_attempts": 1,
                "placement_handshake_successes": 1,
                "placement_handshake_failures": 0,
                "placement_handshake_stale": 0,
                "placement_handshake_reasons": ["stable_target_placement"],
                "selected_plan_moe_routed_tokens": 16,
                "selected_plan_moe_local_routed_tokens": 12,
                "selected_plan_moe_remote_routed_tokens": 4,
                "selected_plan_moe_avg_remote_routing_ratio": 0.25,
                "selected_plan_moe_locality_definitions": [
                    "target_placement_coverage"
                ],
                "selected_plan_moe_locality_granularities": [
                    "target_instance_or_deployment"
                ],
                "selected_plan_moe_remote_routing_definitions": [
                    "missing_from_target_placement_snapshot"
                ],
                "context_source_count": 1,
                "context_target_count": 2,
                "candidate_component_costs": {
                    "req-a": {
                        "target-a": {"total_cost": 3.5},
                        "target-b": {"total_cost": 8.0},
                    }
                },
                "kv_migration_cost": 1.0,
                "queue_penalty_cost": 2.0,
                "avg_queue_pressure": 0.5,
                "max_queue_depth": 2,
                "moe_route_histogram_available_count": 1,
                "moe_target_placement_available_count": 2,
                "moe_route_histogram_source": "instrumentation",
                "moe_route_histogram_kind": "request_fixture",
                "moe_hot_expert_locality_ratio": 0.75,
                "moe_estimated_remote_routing_ratio": 0.25,
                "moe_estimated_remote_routed_tokens": 4,
                "moe_estimated_dispatch_cost": 1.5,
                "moe_dispatch_observation_available_count": 1,
                "moe_routed_tokens": 16,
                "moe_local_routed_tokens": 12,
                "moe_remote_routed_tokens": 4,
                "moe_remote_routing_ratio": 0.25,
                "moe_local_routed_tokens_by_layer": {"layer:0": 12},
                "moe_remote_routed_tokens_by_layer": {"layer:0": 4},
                "moe_local_routed_tokens_by_expert": {
                    "layer:0/expert:1": 12
                },
                "moe_remote_routed_tokens_by_expert": {
                    "layer:0/expert:2": 4
                },
                "moe_locality_definition": "target_placement_coverage",
                "moe_locality_granularity": "target_instance_or_deployment",
                "moe_remote_routing_definition": (
                    "missing_from_target_placement_snapshot"
                ),
                "moe_rank_locality_available_count": 0,
                "moe_physical_dispatch_traffic_available_count": 0,
            }
        ]
    )

    assert summary["context_migration_events"] == 1
    assert summary["context_migration_selected_target_ids"] == "target-a"
    assert summary["context_migration_selected_request_ids"] == "req-a"
    assert summary["context_migration_selected_plan_total_estimated_cost"] == 3.5
    assert summary["context_migration_selected_plan_kv_migration_cost"] == 1.0
    assert (
        summary["context_migration_selected_plan_expert_dispatch_cost"]
        == 1.5
    )
    assert summary["context_migration_selected_plan_queue_penalty_cost"] == 2.0
    assert summary["context_migration_selected_plan_avg_queue_pressure"] == 0.5
    assert summary["context_migration_selected_plan_max_queue_depth"] == 2
    assert (
        summary[
            "context_migration_selected_plan_avg_hot_expert_locality_ratio"
        ]
        == 0.75
    )
    assert (
        summary[
            "context_migration_selected_plan_avg_estimated_remote_routing_ratio"
        ]
        == 0.25
    )
    assert (
        summary["context_migration_selected_plan_estimated_remote_routed_tokens"]
        == 4
    )
    assert summary["context_migration_selected_target_placement_epochs"] == "9"
    assert (
        summary[
            "context_migration_selected_target_expert_placement_fingerprints"
        ]
        == "fp-b"
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_plan_fingerprints"
        ]
        == "plan-b"
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_snapshot_fingerprints"
        ]
        == "snapshot-b"
    )
    assert (
        summary["context_migration_selected_target_expert_placement_contracts"]
        == 1
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_plan_applied"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_plan_verified"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_contract_reasons"
        ]
        == "runtime_not_applied"
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_apply_hook_available"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_apply_attempted"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_apply_success"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_apply_reasons"
        ]
        == "runtime_apply_hook_unavailable"
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_verify_hook_available"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_verify_attempted"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_verify_success"
        ]
        == 0
    )
    assert (
        summary[
            "context_migration_selected_target_expert_placement_verify_reasons"
        ]
        == "runtime_verify_hook_unavailable"
    )
    assert (
        summary["context_migration_selected_target_placement_sources"]
        == "runtime_fixture"
    )
    assert summary["context_migration_placement_handshake_attempts"] == 1
    assert summary["context_migration_placement_handshake_successes"] == 1
    assert summary["context_migration_placement_handshake_failures"] == 0
    assert summary["context_migration_placement_handshake_stale"] == 0
    assert summary["context_migration_placement_handshake_reasons"] == (
        "stable_target_placement"
    )
    assert summary["context_migration_selected_plan_moe_routed_tokens"] == 16
    assert (
        summary["context_migration_selected_plan_moe_local_routed_tokens"]
        == 12
    )
    assert (
        summary["context_migration_selected_plan_moe_remote_routed_tokens"]
        == 4
    )
    assert (
        summary["context_migration_selected_plan_moe_avg_remote_routing_ratio"]
        == 0.25
    )
    assert (
        summary["context_migration_selected_plan_moe_locality_definitions"]
        == "target_placement_coverage"
    )
    assert (
        summary["context_migration_selected_plan_moe_locality_granularities"]
        == "target_instance_or_deployment"
    )
    assert (
        summary[
            "context_migration_selected_plan_moe_remote_routing_definitions"
        ]
        == "missing_from_target_placement_snapshot"
    )
    assert summary["context_migration_context_source_count"] == 1
    assert summary["context_migration_context_target_count"] == 2
    assert summary["context_migration_candidate_component_cost_events"] == 1
    assert summary["context_migration_moe_route_histogram_available_count"] == 1
    assert summary["context_migration_moe_target_placement_available_count"] == 2
    assert summary["context_migration_moe_route_histogram_sources"] == "instrumentation"
    assert (
        summary["context_migration_moe_route_histogram_kinds"]
        == "request_fixture"
    )
    assert summary["context_migration_kv_migration_cost"] == 1.0
    assert summary["context_migration_queue_penalty_cost"] == 2.0
    assert summary["context_migration_avg_queue_pressure"] == 0.5
    assert summary["context_migration_max_queue_depth"] == 2
    assert summary["context_migration_moe_avg_hot_expert_locality_ratio"] == 0.75
    assert summary["context_migration_moe_avg_estimated_remote_routing_ratio"] == 0.25
    assert summary["context_migration_moe_estimated_remote_routed_tokens"] == 4
    assert summary["context_migration_moe_estimated_dispatch_cost"] == 1.5
    assert (
        summary[
            "context_migration_moe_dispatch_observation_available_count"
        ]
        == 1
    )
    assert summary["context_migration_moe_routed_tokens"] == 16
    assert summary["context_migration_moe_local_routed_tokens"] == 12
    assert summary["context_migration_moe_remote_routed_tokens"] == 4
    assert summary["context_migration_moe_avg_remote_routing_ratio"] == 0.25
    assert summary["context_migration_moe_locality_definitions"] == (
        "target_placement_coverage"
    )
    assert summary["context_migration_moe_locality_granularities"] == (
        "target_instance_or_deployment"
    )
    assert summary["context_migration_moe_remote_routing_definitions"] == (
        "missing_from_target_placement_snapshot"
    )
    assert summary["context_migration_moe_rank_locality_available_count"] == 0
    assert (
        summary[
            "context_migration_moe_physical_dispatch_traffic_available_count"
        ]
        == 0
    )
    assert (
        summary["context_migration_moe_local_routed_tokens_by_layer"]
        == '{"layer:0": 12}'
    )
    assert (
        summary["context_migration_moe_remote_routed_tokens_by_layer"]
        == '{"layer:0": 4}'
    )
    assert (
        summary["context_migration_moe_local_routed_tokens_by_expert"]
        == '{"layer:0/expert:1": 12}'
    )
    assert (
        summary["context_migration_moe_remote_routed_tokens_by_expert"]
        == '{"layer:0/expert:2": 4}'
    )
