import pytest

import sllm.spot.vllm_deployment_adapter as adapter_module
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.vllm_deployment_adapter import VllmDeploymentAdapter


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Actor:
    def __init__(self):
        async def init_backend():
            return None

        async def get_runtime_metadata(**kwargs):
            return {"backend": "vllm", "status": "running"}

        async def stop():
            return None

        self.init_backend = _Remote(init_backend)
        self.get_runtime_metadata = _Remote(get_runtime_metadata)
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
    assert await adapter.ready_workers(deployment, plan)
