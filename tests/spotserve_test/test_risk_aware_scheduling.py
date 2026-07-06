from sllm.spot.risk_aware_scheduling import (
    plan_risk_aware_scheduling,
    rank_nodes_by_spot_risk,
)


def synthetic_nodes():
    return {
        "node-0": {
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
            "spot_risk": 0.9,
            "remaining_lifetime_s": 300,
            "loading_cost": 5,
        },
        "node-1": {
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
            "spot_risk": 0.1,
            "remaining_lifetime_s": 3200,
            "loading_cost": 15,
        },
        "node-2": {
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "preempting",
            "spot_risk": 0.0,
            "remaining_lifetime_s": 3600,
            "loading_cost": 1,
        },
    }


def test_risk_aware_ranking_prefers_low_risk_long_lived_node():
    ranked = rank_nodes_by_spot_risk(
        synthetic_nodes(),
        requested_gpus=1,
        scheduler_config={
            "risk_weight": 1.0,
            "lifetime_weight": 0.8,
            "loading_cost_weight": 0.6,
            "max_remaining_lifetime_s": 3600,
            "max_loading_cost": 60,
        },
    )

    assert [node.node_id for node in ranked] == ["node-1", "node-0"]
    assert ranked[0].spot_risk == 0.1


def test_risk_aware_plan_returns_selected_node_and_candidates():
    decision = plan_risk_aware_scheduling(
        model_name="risk-model",
        worker_nodes=synthetic_nodes(),
        requested_gpus=1,
    )

    assert decision.action == "allocate"
    assert decision.selected_node_id == "node-1"
    assert len(decision.candidates) == 2


def test_risk_aware_plan_reports_no_capacity():
    decision = plan_risk_aware_scheduling(
        model_name="risk-model",
        worker_nodes=synthetic_nodes(),
        requested_gpus=3,
    )

    assert decision.action == "no_capacity"
    assert decision.selected_node_id is None
    assert decision.candidates == []
