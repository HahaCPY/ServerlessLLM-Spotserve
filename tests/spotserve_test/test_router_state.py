import json

import pytest

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.utils import InstanceHandle, InstanceState


class FakeContextBackend:
    async def get_current_tokens(self):
        return []

    async def get_context_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ):
        return [
            {
                "request_id": "request-live-0",
                "instance_id": instance_id,
                "node_id": node_id,
                "num_tokens": 8,
                "context_blocks": 2,
                "reusable_tokens_by_target": {"instance-target": 6},
                "reusable_blocks_by_target": {"instance-target": 2},
            }
        ]


class FakeKvSourceBackend(FakeContextBackend):
    async def get_current_tokens(self):
        return [[1, 2, 3, 4]]


class FakeKvTargetBackend:
    def __init__(self):
        self.resumed_batches = []

    async def resume_kv_cache(self, request_datas):
        self.resumed_batches.append(request_datas)
        return True


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


@pytest.mark.asyncio
async def test_router_plans_live_context_migration_on_preemption(tmp_path):
    metrics_path = tmp_path / "router-metrics.jsonl"
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 0},
        backend="dummy",
        backend_config={},
        router_config={
            "metrics_path": str(metrics_path),
            "enable_context_migration": True,
            "context_migration_config": {
                "target_warmup_cost": 1.0,
                "planner_config": {
                    "cross_node_penalty": 0.0,
                },
            },
        },
    )
    source = InstanceHandle(
        instance_id="instance-source",
        max_queue_length=1,
        num_gpu=0,
        backend_instance=FakeContextBackend(),
    )
    target = InstanceHandle(
        instance_id="instance-target",
        max_queue_length=1,
        num_gpu=0,
    )
    await source.mark_ready(node_id="node-source")
    await target.mark_ready(node_id="node-target")

    router.ready_inference_instances[source.instance_id] = source
    router.ready_inference_instances[target.instance_id] = target

    result = await router.handle_preemption(node_id="node-source")

    decision = result["context_migration"]
    assert decision["action"] == "migrate"
    assert len(decision["plans"]) == 1
    assert decision["plans"][0]["old_instance_id"] == "instance-source"
    assert decision["plans"][0]["new_instance_id"] == "instance-target"
    assert decision["plans"][0]["reusable_tokens"] == 6
    assert decision["plans"][0]["reusable_context_blocks"] == 2

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    context_rows = [
        row for row in rows if row["type"] == "context_migration"
    ]
    assert len(context_rows) == 1
    assert context_rows[0]["action"] == "migrate"
    assert context_rows[0]["migration_plan_count"] == 1
    assert context_rows[0]["plans"][0]["new_instance_id"] == "instance-target"


@pytest.mark.asyncio
async def test_router_executes_kv_cache_migration_on_preemption(tmp_path):
    metrics_path = tmp_path / "router-metrics.jsonl"
    router = RoundRobinRouter(
        model_name="test-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 0},
        backend="dummy",
        backend_config={},
        router_config={
            "metrics_path": str(metrics_path),
            "enable_context_migration": True,
            "enable_kv_cache_migration": True,
            "context_migration_config": {
                "target_warmup_cost": 1.0,
                "planner_config": {
                    "cross_node_penalty": 0.0,
                },
            },
        },
    )
    target_backend = FakeKvTargetBackend()
    source = InstanceHandle(
        instance_id="instance-source",
        max_queue_length=1,
        num_gpu=0,
        backend_instance=FakeKvSourceBackend(),
    )
    target = InstanceHandle(
        instance_id="instance-target",
        max_queue_length=1,
        num_gpu=0,
        backend_instance=target_backend,
    )
    await source.mark_ready(node_id="node-source")
    await target.mark_ready(node_id="node-target")

    router.ready_inference_instances[source.instance_id] = source
    router.ready_inference_instances[target.instance_id] = target

    result = await router.handle_preemption(node_id="node-source")

    execution = result["context_migration"]["kv_cache_migration"]
    assert execution["action"] == "resume_kv_cache"
    assert execution["attempted"] == 1
    assert execution["succeeded"] == 1
    assert execution["total_tokens"] == 4
    assert target_backend.resumed_batches == [[[1, 2, 3, 4]]]

    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    context_rows = [
        row for row in rows if row["type"] == "context_migration"
    ]
    assert context_rows[-1]["kv_cache_migration"]["succeeded"] == 1
