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
    assert selected.data_parallel_size == 2


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
    assert decision["parallel_plan"] == {
        "model_name": "dummy-reparallelization",
        "backend": "unknown",
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "num_replicas": 1,
        "num_gpus": 2,
        "target_nodes": ["1"],
        "reason": "preempt_replan",
    }


def test_parallel_plan_shared_interface_serializes_to_dict():
    plan = ParallelPlan(
        model_name="moe-model",
        backend="vllm",
        tensor_parallel_size=2,
        data_parallel_size=4,
        expert_parallel_size=2,
        num_replicas=4,
        num_gpus=8,
        target_nodes=["node-a", "node-b"],
        reason="spot_preempt",
    )

    assert plan.to_dict() == {
        "model_name": "moe-model",
        "backend": "vllm",
        "tensor_parallel_size": 2,
        "data_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 2,
        "num_replicas": 4,
        "num_gpus": 8,
        "target_nodes": ["node-a", "node-b"],
        "reason": "spot_preempt",
    }
