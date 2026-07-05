import json

import pytest

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.utils import InstanceHandle, InstanceState


@pytest.mark.asyncio
async def test_preempting_instance_cannot_accept_new_requests():
    instance = InstanceHandle(
        instance_id="instance-0",
        max_queue_length=2,
        num_gpu=1,
    )
    await instance.mark_ready(node_id="node-0")

    assert await instance.can_accept_request()

    await instance.mark_preempting()

    assert instance.state == InstanceState.PREEMPTING
    assert not await instance.can_accept_request()


@pytest.mark.asyncio
async def test_mark_ready_does_not_revive_preempting_instance():
    instance = InstanceHandle(
        instance_id="instance-0",
        max_queue_length=1,
        num_gpu=1,
    )

    await instance.mark_preempting()
    marked_ready = await instance.mark_ready(node_id="node-0")

    assert marked_ready is False
    assert instance.state == InstanceState.PREEMPTING
    assert not await instance.can_accept_request()


@pytest.mark.asyncio
async def test_recover_only_revives_preempting_instance():
    instance = InstanceHandle(
        instance_id="instance-0",
        max_queue_length=1,
        num_gpu=1,
    )
    await instance.mark_ready(node_id="node-0")
    await instance.mark_preempting()

    recovered = await instance.mark_recovered()

    assert recovered is True
    assert instance.state == InstanceState.READY
    assert await instance.can_accept_request()


@pytest.mark.asyncio
async def test_recover_does_not_revive_dead_instance():
    instance = InstanceHandle(
        instance_id="instance-0",
        max_queue_length=1,
        num_gpu=1,
    )
    await instance.mark_dead()

    recovered = await instance.mark_recovered()

    assert recovered is False
    assert instance.state == InstanceState.DEAD
    assert not await instance.can_accept_request()


@pytest.mark.asyncio
async def test_busy_instance_can_accept_when_concurrency_has_capacity():
    instance = InstanceHandle(
        instance_id="instance-0",
        max_queue_length=2,
        num_gpu=1,
    )
    await instance.mark_ready(node_id="node-0")
    await instance.add_requests(1)
    instance.state = InstanceState.BUSY

    assert await instance.can_accept_request()


@pytest.mark.asyncio
async def test_router_marks_matching_node_instances_preempting():
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 0},
        backend="dummy",
        backend_config={},
        router_config={},
    )
    target = InstanceHandle(
        instance_id="instance-target",
        max_queue_length=1,
        num_gpu=0,
    )
    other = InstanceHandle(
        instance_id="instance-other",
        max_queue_length=1,
        num_gpu=0,
    )
    await target.mark_ready(node_id="node-0")
    await other.mark_ready(node_id="node-1")

    router.ready_inference_instances[target.instance_id] = target
    router.ready_inference_instances[other.instance_id] = other

    result = await router.handle_preemption(node_id="node-0")

    assert result["instances"] == ["instance-target"]
    assert target.state == InstanceState.PREEMPTING
    assert other.state == InstanceState.READY
    assert not await target.can_accept_request()
    assert await other.can_accept_request()


@pytest.mark.asyncio
async def test_router_marks_matching_node_instances_dead():
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 0},
        backend="dummy",
        backend_config={},
        router_config={},
    )
    target = InstanceHandle(
        instance_id="instance-target",
        max_queue_length=1,
        num_gpu=0,
    )
    other = InstanceHandle(
        instance_id="instance-other",
        max_queue_length=1,
        num_gpu=0,
    )
    await target.mark_ready(node_id="node-0")
    await other.mark_ready(node_id="node-1")

    router.ready_inference_instances[target.instance_id] = target
    router.ready_inference_instances[other.instance_id] = other

    result = await router.handle_dead(node_id="node-0")

    assert result["instances"] == ["instance-target"]
    assert target.state == InstanceState.DEAD
    assert other.state == InstanceState.READY
    assert not await target.can_accept_request()
    assert await other.can_accept_request()


@pytest.mark.asyncio
async def test_router_recovers_matching_preempting_instances():
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 0},
        backend="dummy",
        backend_config={},
        router_config={},
    )
    target = InstanceHandle(
        instance_id="instance-target",
        max_queue_length=1,
        num_gpu=0,
    )
    other = InstanceHandle(
        instance_id="instance-other",
        max_queue_length=1,
        num_gpu=0,
    )
    await target.mark_ready(node_id="node-0")
    await other.mark_ready(node_id="node-1")
    await target.mark_preempting()

    router.ready_inference_instances[target.instance_id] = target
    router.ready_inference_instances[other.instance_id] = other

    result = await router.handle_recover(node_id="node-0")

    assert result["instances"] == ["instance-target"]
    assert target.state == InstanceState.READY
    assert other.state == InstanceState.READY
    assert await target.can_accept_request()
    assert await other.can_accept_request()


@pytest.mark.asyncio
async def test_router_records_reparallelization_decision(tmp_path):
    metrics_path = tmp_path / "router-metrics.jsonl"
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
        backend="dummy",
        backend_config={},
        router_config={
            "metrics_path": str(metrics_path),
            "enable_reparallelization": True,
            "reparallelization_config": {
                "model_gpu_requirement": 2,
                "max_tensor_parallel_size": 4,
                "max_pipeline_parallel_size": 2,
                "synthetic_worker_nodes": {
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
                },
            },
        },
    )

    result = await router.handle_preemption(node_id="0")

    replanning = result["reparallelization"]
    assert replanning["action"] == "reparallelize"
    assert replanning["parallel_plan"]["backend"] == "dummy"
    assert replanning["parallel_plan"]["target_nodes"] == ["1"]

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["type"] == "reparallelization"
    assert rows[-1]["parallel_plan"] == replanning["parallel_plan"]
