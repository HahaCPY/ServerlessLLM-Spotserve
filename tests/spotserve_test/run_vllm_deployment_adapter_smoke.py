"""Dependency-light executable smoke for the V6 vLLM deployment adapter."""

import asyncio
import json

from sllm.spot.reparallelization import ParallelPlan
import sllm.spot.vllm_deployment_adapter as adapter_module


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


async def main():
    adapter_module.start_instance = _StartInstance()
    adapter = adapter_module.VllmDeploymentAdapter(
        "m",
        {"tensor_parallel_size": 1},
        {"num_cpus": 1, "num_gpus": 1},
        _Scheduler(),
        lambda *_: None,
    )
    plan = ParallelPlan(
        model_name="m",
        backend="vllm",
        tensor_parallel_size=2,
        data_parallel_size=1,
        num_gpus=2,
        target_nodes=["node-1"],
    )
    deployment = await adapter.create_workers(plan)
    assert list(deployment.instances.values())[0].node_id == "node-1"
    assert deployment.resource_requirements["num_gpus"] == 2
    assert await adapter.ready_workers(deployment, plan)
    print(
        json.dumps(
            {
                "status": "passed",
                "target_node": list(deployment.instances.values())[0].node_id,
                "tensor_parallel_size": deployment.backend_config[
                    "tensor_parallel_size"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
