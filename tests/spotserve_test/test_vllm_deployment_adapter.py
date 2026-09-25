import pytest

import sllm.spot.vllm_deployment_adapter as adapter_module
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.vllm_deployment_adapter import VllmDeploymentAdapter


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Actor:
    def __init__(self):
        self.applied_expert_plans = []

        async def init_backend():
            return None

        async def get_runtime_metadata(**kwargs):
            return {"backend": "vllm", "status": "running"}

        async def apply_expert_placement_plan(*, expert_placement_plan):
            self.applied_expert_plans.append(dict(expert_placement_plan))
            return {"success": True, "applied": True, "verified": True}

        async def stop():
            return None

        self.init_backend = _Remote(init_backend)
        self.get_runtime_metadata = _Remote(get_runtime_metadata)
        self.apply_expert_placement_plan = _Remote(
            apply_expert_placement_plan
        )
        self.stop = _Remote(stop)


class _StartInstance:
    def options(self, **kwargs):
        return self

    async def remote(self, *args):
        return _Actor()


class _Scheduler:
    def __init__(self):
        async def allocate_resource(**kwargs):
            return kwargs.get("target_node_id", "node-0")

        async def deallocate_resource(*args):
            return None

        self.allocate_resource = _Remote(allocate_resource)
        self.deallocate_resource = _Remote(deallocate_resource)


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
