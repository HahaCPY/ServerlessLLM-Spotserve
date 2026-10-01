import asyncio

import pytest

import sllm.spot.vllm_deployment_adapter as adapter_module
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.vllm_deployment_adapter import VllmDeploymentAdapter
from sllm.utils import InstanceState


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Actor:
    def __init__(self):
        self.applied_expert_plans = []
        self.elastic_ep_resizes = []
        self.elastic_ep_error = None

        async def init_backend():
            return None

        async def get_runtime_metadata(**kwargs):
            return {"backend": "vllm", "status": "running"}

        async def apply_expert_placement_plan(*, expert_placement_plan):
            self.applied_expert_plans.append(dict(expert_placement_plan))
            return {"success": True, "applied": True, "verified": True}

        async def stop():
            return None

        async def scale_elastic_ep(
            *,
            new_data_parallel_size,
            drain_timeout_s,
            expert_placement_plan=None,
        ):
            self.elastic_ep_resizes.append(
                (
                    new_data_parallel_size,
                    drain_timeout_s,
                    dict(expert_placement_plan or {}),
                )
            )
            if self.elastic_ep_error is not None:
                raise self.elastic_ep_error
            return {
                "resized": True,
                "data_parallel_size": new_data_parallel_size,
            }

        self.init_backend = _Remote(init_backend)
        self.get_runtime_metadata = _Remote(get_runtime_metadata)
        self.apply_expert_placement_plan = _Remote(
            apply_expert_placement_plan
        )
        self.stop = _Remote(stop)
        self.scale_elastic_ep = _Remote(scale_elastic_ep)


class _StartInstance:
    def options(self, **kwargs):
        return self

    async def remote(self, *args):
        return _Actor()


class _Scheduler:
    def __init__(self):
        self.resizes = []
        self.distributed_allocations = []

        async def allocate_resource(**kwargs):
            return kwargs.get("target_node_id", "node-0")

        async def deallocate_resource(*args):
            return None

        async def allocate_distributed_resource(**kwargs):
            self.distributed_allocations.append(dict(kwargs))
            target_nodes = list(kwargs["target_node_ids"])
            return {
                "node_id": target_nodes[0],
                "target_node_ids": target_nodes,
                "node_allocations": {node_id: 1 for node_id in target_nodes},
                "num_gpus": kwargs["resources"]["num_gpus"],
            }

        async def resize_resource(model_name, instance_id, resources):
            self.resizes.append(
                (model_name, instance_id, dict(resources))
            )
            return {"num_gpus": resources["num_gpus"]}

        self.allocate_resource = _Remote(allocate_resource)
        self.allocate_distributed_resource = _Remote(
            allocate_distributed_resource
        )
        self.deallocate_resource = _Remote(deallocate_resource)
        self.resize_resource = _Remote(resize_resource)


def _elastic_ep_deployment(actor, scheduler):
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "enable_expert_parallel": True,
            "data_parallel_backend": "ray",
            "spotserve_elastic_ep_enabled": True,
            "spotserve_vllm_ray_dp_owns_gpus": True,
        },
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
        scheduler=scheduler,
        traffic_switcher=lambda *_: None,
    )
    source_plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        num_gpus=2,
        target_nodes=["node-0"],
    )
    handle = adapter_module.InstanceHandle(
        instance_id="i-0",
        max_queue_length=1,
        num_gpu=2,
        node_id="node-0",
        backend_instance=actor,
    )
    return adapter, adapter_module.VllmDeployment(
        plan=source_plan,
        instances={"i-0": handle},
        backend_config=dict(adapter.backend_config),
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
    )


@pytest.mark.asyncio
async def test_vllm_adapter_live_elastic_ep_scale_up():
    actor = _Actor()
    scheduler = _Scheduler()
    adapter, deployment = _elastic_ep_deployment(actor, scheduler)
    target = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        data_parallel_size=4,
        enable_expert_parallel=True,
        num_gpus=4,
        target_nodes=["node-0"],
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_fingerprint": "ep4",
        },
    )

    resized = await adapter.resize_workers_elastic_ep(deployment, target)

    assert scheduler.resizes == [("m", "i-0", {"num_cpus": 1, "num_gpus": 4})]
    assert actor.elastic_ep_resizes == [
        (
            4,
            adapter.drain_timeout_s,
            {
                "expert_placement_available": True,
                "placement_fingerprint": "ep4",
            },
        )
    ]
    assert actor.applied_expert_plans == []
    assert resized.plan == target
    assert resized.instances["i-0"].num_gpu == 4
    assert resized.instances["i-0"].ready is True
    assert resized.instances["i-0"].state == InstanceState.READY
    assert resized.backend_config["reparallelization_execution_model"] == (
        "in_place_elastic_ep_resize"
    )
    assert resized.backend_config["elastic_ep_admission_drained"] is True
    assert resized.backend_config["elastic_ep_drained_request_count"] == 0


@pytest.mark.asyncio
async def test_vllm_adapter_rolls_back_scale_up_reservation_on_failure():
    actor = _Actor()
    actor.elastic_ep_error = RuntimeError("scale failed")
    scheduler = _Scheduler()
    adapter, deployment = _elastic_ep_deployment(actor, scheduler)
    target = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        data_parallel_size=4,
        enable_expert_parallel=True,
        num_gpus=4,
        target_nodes=["node-0"],
    )

    with pytest.raises(RuntimeError, match="scale failed"):
        await adapter.resize_workers_elastic_ep(deployment, target)

    assert [row[2]["num_gpus"] for row in scheduler.resizes] == [4, 2]
    assert deployment.instances["i-0"].num_gpu == 2
    assert deployment.instances["i-0"].ready is False
    assert deployment.instances["i-0"].state == InstanceState.DRAINING


@pytest.mark.asyncio
async def test_vllm_adapter_drains_admission_before_elastic_ep_resize():
    actor = _Actor()
    scheduler = _Scheduler()
    adapter, deployment = _elastic_ep_deployment(actor, scheduler)
    handle = deployment.instances["i-0"]
    handle.ready = True
    handle.state = InstanceState.READY
    handle.concurrency = 1
    target = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        data_parallel_size=4,
        enable_expert_parallel=True,
        num_gpus=4,
        target_nodes=["node-0"],
    )

    resize = asyncio.create_task(
        adapter.resize_workers_elastic_ep(deployment, target)
    )
    await asyncio.sleep(0.1)

    assert handle.state == InstanceState.DRAINING
    assert handle.ready is False
    assert actor.elastic_ep_resizes == []
    async with handle.lock:
        handle.concurrency = 0

    resized = await resize
    assert actor.elastic_ep_resizes
    assert resized.instances["i-0"].state == InstanceState.READY
    assert resized.instances["i-0"].ready is True
    assert resized.backend_config["elastic_ep_admission_drained"] is True
    assert resized.backend_config["elastic_ep_drained_request_count"] == 1


class _SnapshotScheduler(_Scheduler):
    def __init__(self, worker_nodes):
        super().__init__()

        async def get_worker_nodes():
            return worker_nodes

        self._get_worker_nodes = _Remote(get_worker_nodes)


def test_vllm_adapter_propagates_opt_in_quiescent_remap():
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={"tensor_parallel_size": 1},
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        scheduler=_Scheduler(),
        traffic_switcher=lambda *_: None,
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=2,
        data_parallel_size=1,
        enable_expert_parallel=True,
        num_gpus=2,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_fingerprint": "remap-test",
            "live_expert_remap": True,
            "expert_placement_physical_migration_required": True,
        },
    )

    config = adapter._plan_backend_config(plan)

    assert config["expert_placement_execution_model"] == (
        "quiescent_fixed_ep_remap"
    )
    assert config["expert_placement_runtime_contract_mode"] == (
        "quiescent_fixed_ep_remap"
    )
    assert config["expert_placement_live_migration_enabled"] is False
    assert config["expert_placement_quiescent_remap_enabled"] is True
    assert config["expert_placement_physical_migration_required"] is True


@pytest.mark.asyncio
async def test_vllm_adapter_remaps_existing_workers_in_place():
    actor = _Actor()
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={
            "tensor_parallel_size": 2,
            "enable_expert_parallel": True,
        },
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
        scheduler=_Scheduler(),
        traffic_switcher=lambda *_: None,
    )
    handle = adapter_module.InstanceHandle(
        instance_id="i-0",
        max_queue_length=1,
        num_gpu=2,
        node_id="node-0",
        backend_instance=actor,
    )
    current = adapter_module.VllmDeployment(
        plan=ParallelPlan(
            model_name="m",
            backend="vllm",
            tensor_parallel_size=2,
            data_parallel_size=1,
            enable_expert_parallel=True,
            num_gpus=2,
            target_nodes=["node-0"],
        ),
        instances={"i-0": handle},
        backend_config=dict(adapter.backend_config),
        resource_requirements=dict(adapter.resource_requirements),
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=2,
        data_parallel_size=1,
        enable_expert_parallel=True,
        num_gpus=2,
        target_nodes=["node-0"],
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_fingerprint": "active-remap",
            "live_expert_remap": True,
            "allow_active_requests": True,
            "expert_placement_physical_migration_required": True,
        },
    )

    updated = await adapter.remap_workers_in_place(current, plan)

    assert updated.instances == current.instances
    assert actor.applied_expert_plans[0]["allow_active_requests"] is True
    assert updated.backend_config["reparallelization_execution_model"] == (
        "in_place_expert_remap"
    )
    assert updated.backend_config["expert_placement_execution_model"] == (
        "active_fixed_ep_remap"
    )


@pytest.mark.asyncio
async def test_vllm_adapter_creates_real_shape_and_honors_target_node(monkeypatch):
    monkeypatch.setattr(adapter_module, "start_instance", _StartInstance())
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={"tensor_parallel_size": 1},
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        scheduler=_Scheduler(),
        traffic_switcher=lambda *_: None,
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        replica_count=1,
        enable_expert_parallel=True,
        num_gpus=2,
        target_nodes=["node-1"],
        placement_epoch=7,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_epoch": 7,
            "placement_source": "logical_reparallelization_planner",
            "placement_fingerprint": "abc12345",
            "expert_placement_snapshot": {
                "layer:0/expert:1": {
                    "layer_id": 0,
                    "expert_id": 1,
                    "rank_id": "replica:0/ep-rank:1",
                    "node_id": "node-1",
                    "gpu_id": "1",
                }
            },
        },
    )
    deployment = await adapter.create_workers(plan)
    assert list(deployment.instances.values())[0].node_id == "node-1"
    assert deployment.resource_requirements["num_gpus"] == 2
    assert deployment.backend_config["tensor_parallel_size"] == 2
    assert deployment.backend_config["data_parallel_size"] == 1
    assert deployment.backend_config["vllm_data_parallel_size"] == 1
    assert deployment.backend_config["replica_count"] == 1
    assert deployment.backend_config["sllm_replica_count"] == 1
    assert deployment.backend_config["planned_effective_expert_parallel_size"] == 2
    assert deployment.backend_config["planned_expert_parallel_size"] == 2
    assert deployment.backend_config["expert_parallel_size_source"] == "derived_from_tp_dp"
    assert deployment.backend_config["enable_expert_parallel"] is True
    assert deployment.backend_config["expert_parallel_size_verified"] is False
    assert deployment.backend_config["placement_epoch"] == 7
    assert deployment.backend_config["reparallelization_execution_model"] == (
        "actor_recreate"
    )
    assert deployment.backend_config[
        "reparallelization_execution_model_reason"
    ] == "vllm_actor_recreate"
    assert deployment.backend_config["expert_placement_execution_model"] == (
        "expert_aware_actor_recreate"
    )
    assert deployment.backend_config[
        "expert_placement_execution_model_reason"
    ] == "logical_expert_placement_plan_carried_into_recreated_actor"
    assert deployment.backend_config[
        "expert_placement_runtime_contract_mode"
    ] == "observe_only_contract"
    assert (
        deployment.backend_config["expert_placement_live_migration_enabled"]
        is False
    )
    assert (
        deployment.backend_config[
            "expert_placement_physical_migration_required"
        ]
        is False
    )
    assert deployment.backend_config[
        "expert_placement_runtime_verification_level"
    ] == "unavailable"
    assert (
        deployment.backend_config[
            "expert_placement_runtime_verified_placement"
        ]
        is False
    )
    assert (
        deployment.backend_config[
            "expert_placement_runtime_can_verify_physical_placement"
        ]
        is False
    )
    assert (
        deployment.backend_config[
            "expert_placement_runtime_can_remap_live_ep_rank"
        ]
        is False
    )
    assert (
        deployment.backend_config[
            "expert_placement_runtime_can_measure_all_to_all"
        ]
        is False
    )
    assert deployment.backend_config[
        "expert_placement_runtime_capability_reason"
    ] == "awaiting_runtime_expert_placement_hook"
    assert deployment.backend_config["placement_source"] == (
        "logical_reparallelization_planner"
    )
    assert deployment.backend_config["expert_placement_fingerprint"] == "abc12345"
    assert deployment.backend_config["expert_placement_plan_fingerprint"] == (
        "abc12345"
    )
    assert deployment.backend_config["expert_placement_contract_available"] is True
    assert deployment.backend_config["expert_placement_contract_source"] == (
        "logical_reparallelization_planner"
    )
    assert deployment.backend_config["expert_placement_contract_epoch"] == 7
    assert deployment.backend_config["expert_placement_plan_applied"] is False
    assert deployment.backend_config["expert_placement_plan_verified"] is False
    assert deployment.backend_config["expert_placement_snapshot"][
        "layer:0/expert:1"
    ]["rank_id"] == "replica:0/ep-rank:1"
    assert await adapter.ready_workers(deployment, plan)


@pytest.mark.asyncio
async def test_vllm_adapter_rejects_unknown_target_node(monkeypatch):
    monkeypatch.setattr(adapter_module, "start_instance", _StartInstance())
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={"tensor_parallel_size": 1},
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        scheduler=_SnapshotScheduler({"node-0": {"free_gpu": 1}}),
        traffic_switcher=lambda *_: None,
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        replica_count=1,
        enable_expert_parallel=False,
        num_gpus=1,
        target_nodes=["synthetic-node-1"],
        placement_epoch=1,
    )

    with pytest.raises(RuntimeError, match="target_worker_node_not_in_scheduler"):
        await adapter.create_workers(plan)


@pytest.mark.asyncio
async def test_vllm_adapter_reserves_cross_host_ray_dp_workers(monkeypatch):
    monkeypatch.setattr(adapter_module, "start_instance", _StartInstance())
    scheduler = _SnapshotScheduler(
        {"0": {"free_gpu": 1}, "1": {"free_gpu": 1}}
    )
    adapter = VllmDeploymentAdapter(
        model_name="m",
        backend_config={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "data_parallel_backend": "ray",
            "enable_expert_parallel": True,
            "spotserve_vllm_ray_dp_owns_gpus": True,
            "spotserve_distributed_gpu_allocation": True,
        },
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
        scheduler=scheduler,
        traffic_switcher=lambda *_: None,
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=2,
        replica_count=1,
        enable_expert_parallel=True,
        num_gpus=2,
        target_nodes=["0", "1"],
    )

    deployment = await adapter.create_workers(plan)

    assert len(deployment.instances) == 1
    assert next(iter(deployment.instances.values())).node_id == "0"
    assert scheduler.distributed_allocations[0]["target_node_ids"] == ["0", "1"]
    assert deployment.backend_config["spotserve_reserved_worker_nodes"] == [
        "0",
        "1",
    ]
    assert deployment.backend_config["spotserve_distributed_node_allocations"] == {
        "0": 1,
        "1": 1,
    }
