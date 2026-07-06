from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    estimate_migration_cost,
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
    assert event["migration_plan_count"] == 1
    assert event["reuse_ratio"] == 1.0
