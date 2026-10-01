"""CPU contract fixtures, not measured GPU profiles or performance results."""

from copy import deepcopy

from sllm.spot.reparallelization import plan_dynamic_reparallelization


def nodes():
    return {
        str(index): {"ray_node_id": f"node-{index}", "address": f"10.0.0.{index+1}",
                     "total_gpu": 1, "free_gpu": 1, "state": "ready"}
        for index in range(3)
    }


def config():
    return {
        "backend": "vllm", "num_gpus": 1,
        "backend_capability": {"supported_configs": [
            {"tensor_parallel_size": 1, "num_gpus": 1, "replica_count": 1,
             "latency_estimate_ms": 500, "throughput_estimate_req_s": 2,
             "load_time_estimate_ms": 50, "reason": "fixture_tp1"},
            {"tensor_parallel_size": 2, "num_gpus": 2, "replica_count": 1,
             "latency_estimate_ms": 333, "throughput_estimate_req_s": 3,
             "load_time_estimate_ms": 1000, "reason": "fixture_tp2"},
            {"tensor_parallel_size": 3, "num_gpus": 3, "replica_count": 1,
             "latency_estimate_ms": 1, "throughput_estimate_req_s": 1000,
             "reason": "forbidden_fixture_tp3"},
        ]},
    }


def decide(arrival_rate, model_config=None):
    return plan_dynamic_reparallelization(
        model_name="tp12-contract-fixture", worker_nodes=nodes(),
        model_config=model_config or config(), backend="vllm",
        planner_config={
            "min_tensor_parallel_size": 1, "max_tensor_parallel_size": 2,
            "max_pipeline_parallel_size": 1, "enable_workload_cost_model": True,
            "arrival_rate_req_s": arrival_rate, "batch_size": 1,
            "base_score_weight": 0, "throughput_score_weight": 0,
            "latency_penalty_weight": 1, "load_time_penalty_weight": 1,
            "migration_cost_penalty_weight": 1, "queue_penalty_weight": 1,
            "queue_penalty_ms_per_req_s": 1000,
        },
    )


def test_backend_capability_respects_tp2_maximum():
    decision = decide(100)
    assert decision["candidate_count"] == 2
    assert {row["tensor_parallel_size"] for row in decision["top_candidates"]} == {1, 2}


def test_same_candidate_profiles_choose_different_tp_for_current_load():
    low = decide(1)
    high = decide(4)
    assert low["candidate_count"] == high["candidate_count"] == 2
    assert low["selected_tensor_parallel_size"] == 1
    assert high["selected_tensor_parallel_size"] == 2
    assert low["workload_cost_model"]["enabled"] is True
    assert high["selected_queue_penalty_ms"] < high["top_candidates"][1]["queue_penalty_ms"]


def test_expensive_transition_can_outweigh_tp2_throughput():
    costly = deepcopy(config())
    costly["backend_capability"]["supported_configs"][1]["migration_cost_estimate_ms"] = 20000
    assert decide(4, costly)["selected_tensor_parallel_size"] == 1


def test_tp2_is_not_fabricated_when_backend_supports_only_tp1():
    restricted = config()
    restricted["backend_capability"]["supported_configs"] = restricted[
        "backend_capability"
    ]["supported_configs"][:1]
    decision = decide(4, restricted)
    assert decision["candidate_count"] == 1
    assert decision["selected_tensor_parallel_size"] == 1
