from sllm.spot.reparallelization import (
    ParallelPlan,
    apply_spot_event_to_worker_nodes,
    generate_parallel_candidates,
    plan_dynamic_reparallelization,
    summarize_gpu_availability,
)


def synthetic_nodes():
    return {
        "0": {
            "ray_node_id": "node-0",
            "address": "10.0.0.1",
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
        },
        "1": {
            "ray_node_id": "node-1",
            "address": "10.0.0.2",
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
        },
    }


def test_gpu_availability_changes_after_preempt_and_recover():
    nodes = synthetic_nodes()
    preempted = apply_spot_event_to_worker_nodes(nodes, "preempt", "0")
    availability = summarize_gpu_availability(preempted)

    assert availability.total_gpus == 4
    assert availability.available_gpus == 2
    assert availability.unavailable_gpus == 2
    assert availability.ready_nodes == ["1"]
    assert availability.preempting_nodes == ["0"]

    recovered = apply_spot_event_to_worker_nodes(preempted, "recover", "0")
    availability = summarize_gpu_availability(recovered)

    assert availability.available_gpus == 4
    assert availability.ready_nodes == ["0", "1"]


def test_parallel_candidate_generation_prefers_current_replica_shape():
    candidates = generate_parallel_candidates(
        4,
        {
            "target_replica_gpus": 2,
            "max_tensor_parallel_size": 4,
            "max_pipeline_parallel_size": 2,
        },
    )

    assert candidates
    selected = candidates[0]
    assert selected.total_gpus == 4
    assert selected.tensor_parallel_size == 2
    assert selected.pipeline_parallel_size == 1
    assert selected.data_parallel_size == 1
    assert selected.replica_count == 2


def test_dynamic_reparallelization_plan_after_gpu_loss():
    nodes = apply_spot_event_to_worker_nodes(
        synthetic_nodes(), "preempt", "0"
    )

    decision = plan_dynamic_reparallelization(
        model_name="dummy-reparallelization",
        worker_nodes=nodes,
        model_config={"num_gpus": 2},
        planner_config={
            "max_tensor_parallel_size": 4,
            "max_pipeline_parallel_size": 2,
        },
        event="preempt",
        node_id="0",
    )

    assert decision["action"] == "reparallelize"
    assert decision["availability"]["available_gpus"] == 2
    assert decision["selected_total_gpus"] == 2
    assert decision["selected_data_parallel_size"] == 1
    assert decision["selected_replica_count"] == 1
    assert decision["selected_enable_expert_parallel"] is False
    assert decision["selected_effective_expert_parallel_size"] == 1
    assert decision["selected_expert_parallel_size"] == 1
    assert decision["parallel_plan"] == {
        "model_name": "dummy-reparallelization",
        "backend": "unknown",
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "replica_count": 1,
        "enable_expert_parallel": False,
        "effective_expert_parallel_size": 1,
        "expert_parallel_size": 1,
        "num_replicas": 1,
        "num_gpus": 2,
        "target_nodes": ["1"],
        "reason": "preempt_replan",
    }


def test_reparallelization_respects_backend_capability_supported_shape():
    decision = plan_dynamic_reparallelization(
        model_name="qwen3-dense",
        worker_nodes={
            "0": {
                "ray_node_id": "node-0",
                "address": "10.0.0.1",
                "free_gpu": 4,
                "total_gpu": 4,
                "state": "ready",
            }
        },
        model_config={
            "model": "qwen3-dense",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {
                "tensor_parallel_size": 2,
            },
        },
        event="manual",
        backend="vllm",
    )

    assert decision["action"] == "reparallelize"
    assert decision["parallel_plan"]["tensor_parallel_size"] == 2
    assert decision["parallel_plan"]["data_parallel_size"] == 1
    assert decision["parallel_plan"]["num_gpus"] == 4
    assert decision["selected_config"]["reason"] == "current_vllm_config"


def test_reparallelization_does_not_fallback_when_capability_has_no_capacity():
    decision = plan_dynamic_reparallelization(
        model_name="qwen3-dense",
        worker_nodes={
            "0": {
                "ray_node_id": "node-0",
                "address": "10.0.0.1",
                "free_gpu": 2,
                "total_gpu": 2,
                "state": "ready",
            }
        },
        model_config={
            "model": "qwen3-dense",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {
                "tensor_parallel_size": 2,
            },
        },
        event="manual",
        backend="vllm",
    )

    assert decision["action"] == "no_capacity"
    assert decision["parallel_plan"] is None
    assert decision["candidate_count"] == 0


def test_workload_cost_model_is_opt_in_for_reparallelization_score():
    decision = plan_dynamic_reparallelization(
        model_name="cost-aware-vllm",
        worker_nodes={
            "0": {
                "ray_node_id": "node-0",
                "address": "10.0.0.1",
                "free_gpu": 4,
                "total_gpu": 4,
                "state": "ready",
            }
        },
        model_config={
            "model": "cost-aware-vllm",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_capability": {
                "supported_configs": [
                    {
                        "tensor_parallel_size": 4,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 1,
                        "num_gpus": 4,
                        "reason": "expensive_full_gpu_plan",
                    },
                    {
                        "tensor_parallel_size": 2,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 1,
                        "num_gpus": 2,
                        "reason": "cheaper_partial_gpu_plan",
                    },
                ]
            },
        },
        planner_config={
            "load_time_per_gpu_ms": 1000,
            "base_score_weight": 0,
        },
        event="manual",
        backend="vllm",
    )

    assert decision["workload_cost_model"]["enabled"] is False
    assert decision["selected_total_gpus"] == 4
    assert decision["selected_config"]["reason"] == "expensive_full_gpu_plan"


def test_workload_cost_model_can_prefer_lower_replan_cost():
    decision = plan_dynamic_reparallelization(
        model_name="cost-aware-vllm",
        worker_nodes={
            "0": {
                "ray_node_id": "node-0",
                "address": "10.0.0.1",
                "free_gpu": 4,
                "total_gpu": 4,
                "state": "ready",
            }
        },
        model_config={
            "model": "cost-aware-vllm",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {"max_num_seqs": 1},
            "backend_capability": {
                "supported_configs": [
                    {
                        "tensor_parallel_size": 4,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 1,
                        "num_gpus": 4,
                        "reason": "expensive_full_gpu_plan",
                    },
                    {
                        "tensor_parallel_size": 2,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 1,
                        "num_gpus": 2,
                        "reason": "cheaper_partial_gpu_plan",
                    },
                ]
            },
        },
        planner_config={
            "enable_workload_cost_model": True,
            "base_score_weight": 0,
            "throughput_score_weight": 0,
            "load_time_per_gpu_ms": 1000,
            "load_time_penalty_weight": 1,
            "migration_cost_penalty_weight": 0,
            "queue_penalty_weight": 0,
            "latency_estimate_ms": 1000,
            "batch_size": 1,
        },
        event="manual",
        backend="vllm",
    )

    assert decision["workload_cost_model"]["enabled"] is True
    assert decision["selected_total_gpus"] == 2
    assert decision["selected_config"]["reason"] == "cheaper_partial_gpu_plan"
    assert decision["selected_load_time_estimate_ms"] == 2000
    assert decision["selected_replan_window_cost_ms"] == 2000
    assert decision["selected_score"] > decision["top_candidates"][1]["score"]


def test_workload_cost_model_can_prefer_higher_throughput_under_queue_pressure():
    decision = plan_dynamic_reparallelization(
        model_name="cost-aware-vllm",
        worker_nodes={
            "0": {
                "ray_node_id": "node-0",
                "address": "10.0.0.1",
                "free_gpu": 4,
                "total_gpu": 4,
                "state": "ready",
            }
        },
        model_config={
            "model": "cost-aware-vllm",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {"max_num_seqs": 1},
            "backend_capability": {
                "supported_configs": [
                    {
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 1,
                        "num_gpus": 1,
                        "latency_estimate_ms": 2000,
                        "load_time_estimate_ms": 500,
                        "migration_cost_estimate_ms": 100,
                        "reason": "low_cost_single_replica",
                    },
                    {
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                        "replica_count": 2,
                        "num_gpus": 2,
                        "latency_estimate_ms": 1000,
                        "load_time_estimate_ms": 1500,
                        "migration_cost_estimate_ms": 300,
                        "reason": "higher_throughput_replicas",
                    },
                ]
            },
        },
        planner_config={
            "enable_workload_cost_model": True,
            "arrival_rate_req_s": 2.5,
            "batch_size": 1,
            "base_score_weight": 0,
            "throughput_score_weight": 1000,
            "latency_penalty_weight": 0,
            "load_time_penalty_weight": 0.1,
            "migration_cost_penalty_weight": 0.1,
            "queue_penalty_weight": 1,
        },
        event="manual",
        backend="vllm",
    )

    assert decision["selected_config"]["reason"] == "higher_throughput_replicas"
    assert decision["selected_replica_count"] == 2
    assert decision["selected_latency_estimate_ms"] == 1000
    assert decision["selected_throughput_estimate_req_s"] == 2.0
    assert decision["selected_queue_penalty_ms"] == 500
    assert decision["selected_load_time_estimate_ms"] == 1500
    assert decision["selected_migration_cost_estimate_ms"] == 300
    assert decision["selected_replan_window_cost_ms"] == 2300
    assert decision["selected_score"] > decision["top_candidates"][1]["score"]


def test_parallel_plan_shared_interface_serializes_to_dict():
    plan = ParallelPlan(
        model_name="moe-model",
        backend="vllm",
        tensor_parallel_size=2,
        data_parallel_size=1,
        replica_count=4,
        enable_expert_parallel=True,
        num_gpus=8,
        target_nodes=["node-a", "node-b"],
        reason="spot_preempt",
    )

    assert plan.to_dict() == {
        "model_name": "moe-model",
        "backend": "vllm",
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "replica_count": 4,
        "enable_expert_parallel": True,
        "effective_expert_parallel_size": 2,
        "expert_parallel_size": 2,
        "num_replicas": 4,
        "num_gpus": 8,
        "target_nodes": ["node-a", "node-b"],
        "reason": "spot_preempt",
    }
