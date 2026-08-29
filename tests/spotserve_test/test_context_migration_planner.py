from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    estimate_expert_dispatch_cost,
    estimate_kv_migration_cost,
    estimate_migration_cost,
    estimate_queue_penalty_cost,
    plan_low_cost_migration,
    plan_low_cost_migration_from_dict,
)
from sllm.spot.metrics import make_context_migration_event


def test_migration_cost_reuses_same_node_context():
    source = ContextMetadata(
        request_id="req-a",
        instance_id="old-a",
        node_id="node-0",
        num_tokens=100,
        context_blocks=10,
    )
    same_node_target = MigrationTarget(
        instance_id="new-a",
        node_id="node-0",
    )
    cross_node_target = MigrationTarget(
        instance_id="new-b",
        node_id="node-1",
    )

    same_cost, same_tokens, same_blocks = estimate_migration_cost(
        source, same_node_target
    )
    cross_cost, cross_tokens, cross_blocks = estimate_migration_cost(
        source, cross_node_target
    )

    assert same_cost < cross_cost
    assert same_tokens == 100
    assert same_blocks == 10
    assert cross_tokens == 0
    assert cross_blocks == 0


def test_low_cost_mapping_prefers_minimum_total_cost():
    sources = [
        ContextMetadata(
            request_id="req-a",
            instance_id="old-a",
            node_id="node-0",
            num_tokens=100,
            context_blocks=10,
        ),
        ContextMetadata(
            request_id="req-b",
            instance_id="old-b",
            node_id="node-1",
            num_tokens=100,
            context_blocks=10,
        ),
    ]
    targets = [
        MigrationTarget(instance_id="new-b", node_id="node-1"),
        MigrationTarget(instance_id="new-a", node_id="node-0"),
    ]

    decision = plan_low_cost_migration(sources, targets)
    plans_by_request = {plan.request_id: plan for plan in decision.plans}

    assert decision.action == "migrate"
    assert len(decision.plans) == 2
    assert plans_by_request["req-a"].new_instance_id == "new-a"
    assert plans_by_request["req-b"].new_instance_id == "new-b"
    assert decision.reuse_ratio == 1.0


def test_target_capacity_expands_into_multiple_slots():
    sources = [
        ContextMetadata(
            request_id="req-a",
            instance_id="old-a",
            node_id="node-0",
        ),
        ContextMetadata(
            request_id="req-b",
            instance_id="old-b",
            node_id="node-0",
        ),
    ]
    targets = [
        MigrationTarget(
            instance_id="new-a",
            node_id="node-0",
            capacity=2,
        )
    ]

    decision = plan_low_cost_migration(sources, targets)

    assert len(decision.plans) == 2
    assert not decision.unassigned_contexts
    assert {plan.new_instance_id for plan in decision.plans} == {"new-a"}


def test_warmup_cost_is_charged_once_per_target():
    sources = [
        ContextMetadata(
            request_id="req-a",
            instance_id="old-a",
            node_id="node-0",
        ),
        ContextMetadata(
            request_id="req-b",
            instance_id="old-b",
            node_id="node-0",
        ),
    ]
    targets = [
        MigrationTarget(
            instance_id="new-a",
            node_id="node-0",
            capacity=2,
            warmup_cost=100.0,
        )
    ]

    decision = plan_low_cost_migration(
        sources,
        targets,
        planner_config={
            "base_migration_cost": 0.0,
            "token_transfer_cost": 0.0,
            "context_block_transfer_cost": 0.0,
            "cross_node_penalty": 0.0,
        },
    )

    assert len(decision.plans) == 2
    assert decision.total_estimated_cost == 100.0
    assert sum(plan.estimated_cost for plan in decision.plans) == 100.0


def test_kv_migration_cost_reports_component_breakdown():
    source = ContextMetadata(
        request_id="req-kv",
        instance_id="old-a",
        node_id="node-0",
        num_tokens=10,
        context_blocks=3,
        reusable_tokens_by_target={"new-a": 4},
        reusable_blocks_by_target={"new-a": 1},
    )
    target = MigrationTarget(
        instance_id="new-a",
        node_id="node-1",
        warmup_cost=5.0,
    )

    cost = estimate_kv_migration_cost(
        source,
        target,
        planner_config={
            "base_migration_cost": 1.0,
            "token_transfer_cost": 2.0,
            "context_block_transfer_cost": 3.0,
            "cross_node_penalty": 7.0,
        },
    )

    assert cost["reusable_tokens"] == 4
    assert cost["reusable_context_blocks"] == 1
    assert cost["non_reusable_tokens"] == 6
    assert cost["non_reusable_context_blocks"] == 2
    assert cost["token_migration_cost"] == 12.0
    assert cost["context_block_migration_cost"] == 6.0
    assert cost["warmup_cost"] == 5.0
    assert cost["cross_node_penalty_cost"] == 7.0
    assert cost["cost"] == 31.0


def test_expert_locality_cost_prefers_target_with_hot_experts():
    source = ContextMetadata(
        request_id="req-hot",
        instance_id="old-a",
        node_id="node-0",
        num_tokens=10,
        metadata={
            "moe_route_histogram_available": True,
            "moe_route_histogram_source": "instrumentation",
            "per_request_expert_route_histogram": {
                "req-hot": {"layer:0/expert:1": 10}
            },
        },
    )
    remote_expert_target = MigrationTarget(
        instance_id="remote-expert",
        node_id="node-1",
        metadata={
            "expert_placement_available": True,
            "expert_placement_snapshot": {
                "layer:0/expert:2": {"rank_id": "rank-0"}
            },
        },
    )
    local_expert_target = MigrationTarget(
        instance_id="local-expert",
        node_id="node-2",
        metadata={
            "expert_placement_available": True,
            "expert_placement_snapshot": {
                "layer:0/expert:1": {"rank_id": "rank-1"}
            },
        },
    )

    decision = plan_low_cost_migration(
        sources=[source],
        targets=[remote_expert_target, local_expert_target],
        planner_config={
            "enable_moe_expert_locality": True,
            "expert_dispatch_weight": 1.0,
            "token_transfer_cost": 0.0,
            "context_block_transfer_cost": 0.0,
            "cross_node_penalty": 0.0,
        },
    )

    assert decision.plans[0].new_instance_id == "local-expert"
    assert decision.plans[0].expert_locality_available is True
    assert decision.plans[0].kv_migration_cost == 0.0
    assert decision.plans[0].hot_expert_locality_ratio == 1.0
    assert decision.plans[0].estimated_remote_routing_ratio == 0.0
    assert decision.plans[0].estimated_remote_routed_tokens == 0
    assert decision.total_expert_dispatch_cost == 0.0
    assert decision.avg_hot_expert_locality_ratio == 1.0
    assert decision.avg_estimated_remote_routing_ratio == 0.0
    assert decision.total_estimated_remote_routed_tokens == 0
    assert decision.moe_route_histogram_available_count == 1
    assert decision.moe_target_placement_available_count == 2


def test_expert_dispatch_cost_reports_unavailable_without_metadata():
    result = estimate_expert_dispatch_cost(
        ContextMetadata(
            request_id="req-no-moe",
            instance_id="old-a",
            node_id="node-0",
        ),
        MigrationTarget(instance_id="new-a", node_id="node-1"),
        planner_config={"enable_moe_expert_locality": True},
    )

    assert result["available"] is False
    assert result["cost"] == 0.0


def test_expert_dispatch_cost_ignores_legacy_histogram_aliases():
    result = estimate_expert_dispatch_cost(
        ContextMetadata(
            request_id="req-legacy",
            instance_id="old-a",
            node_id="node-0",
            metadata={
                "moe_route_histogram_available": True,
                "per_request_routed_tokens_by_expert": {
                    "req-legacy": {"layer:0/expert:1": 10}
                },
                "expert_route_histogram": {
                    "layer:0/expert:1": 10
                },
                "routed_tokens_by_expert": {
                    "layer:0/expert:1": 10
                },
            },
        ),
        MigrationTarget(
            instance_id="new-a",
            node_id="node-1",
            metadata={
                "expert_placement_available": True,
                "expert_placement_snapshot": {
                    "layer:0/expert:1": {"rank_id": "rank-0"}
                },
            },
        ),
        planner_config={"enable_moe_expert_locality": True},
    )

    assert result["available"] is False
    assert result["histogram_available"] is False
    assert result["cost"] == 0.0


def test_expert_dispatch_cost_uses_routing_weighted_locality():
    result = estimate_expert_dispatch_cost(
        ContextMetadata(
            request_id="req-weighted",
            instance_id="old-a",
            node_id="node-0",
            metadata={
                "moe_route_histogram_available": True,
                "per_request_expert_route_histogram": {
                    "req-weighted": {
                        "layer:0/expert:1": 6,
                        "layer:0/expert:2": 4,
                    }
                },
            },
        ),
        MigrationTarget(
            instance_id="new-a",
            node_id="node-1",
            metadata={
                "expert_placement_available": True,
                "expert_placement_snapshot": {
                    "layer:0/expert:1": {"rank_id": "rank-0"}
                },
            },
        ),
        planner_config={
            "enable_moe_expert_locality": True,
            "expert_dispatch_weight": 2.0,
        },
    )

    assert result["available"] is True
    assert result["locality_ratio"] == 0.6
    assert result["estimated_remote_routing_ratio"] == 0.4
    assert result["estimated_remote_routed_tokens"] == 4
    assert result["cost"] == 0.8


def test_queue_penalty_prefers_less_loaded_target():
    source = ContextMetadata(
        request_id="req-queue",
        instance_id="old-a",
        node_id="node-source",
    )
    busy_target = MigrationTarget(
        instance_id="busy-target",
        node_id="node-0",
        capacity=1,
        concurrency=3,
        max_queue_length=4,
    )
    idle_target = MigrationTarget(
        instance_id="idle-target",
        node_id="node-1",
        capacity=1,
        concurrency=0,
        max_queue_length=4,
    )

    decision = plan_low_cost_migration(
        sources=[source],
        targets=[busy_target, idle_target],
        planner_config={
            "base_migration_cost": 0.0,
            "token_transfer_cost": 0.0,
            "context_block_transfer_cost": 0.0,
            "cross_node_penalty": 0.0,
            "queue_penalty_weight": 10.0,
        },
    )

    assert decision.plans[0].new_instance_id == "idle-target"
    assert decision.plans[0].queue_depth == 0
    assert decision.plans[0].queue_penalty_cost == 0.0
    assert decision.total_queue_penalty_cost == 0.0
    assert decision.max_queue_depth == 0


def test_queue_penalty_accounts_for_planned_requests_ahead():
    target = MigrationTarget(
        instance_id="target",
        node_id="node-0",
        capacity=2,
        concurrency=1,
        max_queue_length=4,
    )

    first = estimate_queue_penalty_cost(
        target,
        planner_config={"queue_penalty_weight": 10.0},
        planned_requests_ahead=0,
    )
    second = estimate_queue_penalty_cost(
        target,
        planner_config={"queue_penalty_weight": 10.0},
        planned_requests_ahead=1,
    )

    assert first["queue_depth"] == 1
    assert first["queue_pressure"] == 0.25
    assert first["cost"] == 10.0
    assert second["queue_depth"] == 2
    assert second["queue_pressure"] == 0.5
    assert second["cost"] == 20.0


def test_unassigned_contexts_when_target_capacity_is_insufficient():
    payload = {
        "sources": [
            {
                "request_id": "req-a",
                "instance_id": "old-a",
                "node_id": "node-0",
            },
            {
                "request_id": "req-b",
                "instance_id": "old-b",
                "node_id": "node-1",
            },
        ],
        "targets": [
            {
                "instance_id": "new-a",
                "node_id": "node-0",
                "capacity": 1,
            }
        ],
    }

    decision = plan_low_cost_migration_from_dict(payload)

    assert decision.action == "migrate"
    assert len(decision.plans) == 1
    assert len(decision.unassigned_contexts) == 1


def test_context_migration_metric_contains_summary_fields():
    decision = plan_low_cost_migration(
        sources=[
            ContextMetadata(
                request_id="req-a",
                instance_id="old-a",
                node_id="node-0",
                num_tokens=10,
                context_blocks=2,
            )
        ],
        targets=[
            MigrationTarget(instance_id="new-a", node_id="node-0"),
        ],
    )

    event = make_context_migration_event(
        model="test-model",
        decision=decision.to_dict(),
    )

    assert event["type"] == "context_migration"
    assert event["context_migration_plan_count"] == 1
    assert event["migration_plan_count"] == 1
    assert event["selected_plan_count"] == 1
    assert event["selected_target_ids"] == ["new-a"]
    assert event["selected_source_instance_ids"] == ["old-a"]
    assert event["selected_request_ids"] == ["req-a"]
    assert event["selected_plan_total_estimated_cost"] == 0.0
    assert event["selected_plan_kv_migration_cost"] == 0.0
    assert event["selected_plan_expert_dispatch_cost"] == 0.0
    assert event["selected_plan_queue_penalty_cost"] == 0.0
    assert event["selected_plan_avg_queue_pressure"] == 0.0
    assert event["selected_plan_max_queue_depth"] == 0
    assert event["reuse_ratio"] == 1.0
    assert event["kv_migration_cost"] == 0.0
    assert event["queue_penalty_cost"] == 0.0
    assert event["avg_queue_pressure"] == 0.0
    assert event["max_queue_depth"] == 0
    assert event["moe_route_histogram_available"] is False
    assert event["moe_target_placement_available"] is False
    assert event["moe_hot_expert_locality_ratio"] == 0.0
    assert event["moe_estimated_remote_routing_ratio"] == 0.0
    assert event["moe_estimated_remote_routed_tokens"] == 0
    assert event["moe_estimated_dispatch_cost"] == 0.0
    assert event["context_source_count"] == 1
    assert event["context_target_count"] == 0
    assert event["candidate_component_costs_enabled"] is False
    assert event["candidate_component_costs"] is None
    assert event["prefix_warmup_attempts"] == 0
    assert event["kv_restore_attempts"] == 0
    assert event["true_kv_block_transfer"] is False
