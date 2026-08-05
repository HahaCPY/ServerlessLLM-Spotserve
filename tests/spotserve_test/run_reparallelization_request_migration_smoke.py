"""Executable smoke for V6 in-flight request snapshot/restore migration."""

import asyncio
import json

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.vllm_deployment_adapter import VllmDeployment
from sllm.utils import InstanceHandle


class _MigratingBackend:
    def __init__(self, node_id, source=False):
        self.node_id = node_id
        self.source = source
        self.started = asyncio.Event()
        self.aborted = asyncio.Event()
        self.restore_calls = 0

    async def export_inference_state(
        self, request_data=None, current_output=None, completed_tokens=None
    ):
        return {
            "request_id": (request_data or {}).get("request_id"),
            "instance_id": "source-instance",
            "node_id": "node-source",
            "backend": "vllm",
            "model_name": "migration-model",
            "tokens": [1, 2, 3, 4],
            "completed_tokens": 4,
            "state_kind": "vllm_kv_snapshot",
            "supports_restore": True,
            "runtime_state": {"handle": "snapshot-1"},
            "metadata": {
                "can_restore_same_node": True,
                "can_restore_cross_node": False,
                "kv_block_count": 1,
            },
        }

    async def abort_request(self, request_id, reason="preempted"):
        self.aborted.set()
        return {"aborted": True, "request_id": request_id, "reason": reason}

    async def supports_state_restore(self):
        return not self.source

    async def restore_inference_state(self, state, request_data=None):
        self.restore_calls += 1
        return {
            "restored": True,
            "state_kind": "vllm_kv_snapshot",
            "restored_blocks": 1,
        }

    async def generate(self, request_data):
        if self.source:
            self.started.set()
            await self.aborted.wait()
            return {
                "preempted": True,
                "_spotserve_reparallelization": True,
                "request_id": request_data.get("request_id"),
            }
        return {"choices": [], "usage": {"completion_tokens": 1}}


async def main():
    router = RoundRobinRouter(
        model_name="migration-model",
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        backend="vllm",
        backend_config={},
        router_config={"max_retries": 0},
    )
    router.running = True
    source_backend = _MigratingBackend("node-source", source=True)
    target_backend = _MigratingBackend("node-target")
    source = InstanceHandle(
        "source-instance", max_queue_length=1, num_gpu=1,
        node_id="node-source", backend_instance=source_backend,
    )
    target = InstanceHandle(
        "target-instance", max_queue_length=1, num_gpu=1,
        node_id="node-target", backend_instance=target_backend,
    )
    await source.mark_ready(node_id="node-source")
    await target.mark_ready(node_id="node-target")

    allocations = iter([source, target])

    async def allocate():
        instance = next(allocations)
        await instance.add_requests(1)
        return instance.instance_id, instance

    router._allocate_instance_for_request = allocate
    request_task = asyncio.create_task(
        router.inference(
            {
                "request_id": "external-request-1",
                "model": "migration-model",
                "prompt": "long request",
                "max_tokens": 32,
            },
            "generate",
        )
    )
    await source_backend.started.wait()

    plan = ParallelPlan(
        model_name="migration-model",
        backend="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        num_replicas=1,
        num_gpus=1,
        target_nodes=["node-target"],
    )
    deployment = VllmDeployment(plan=plan, instances={source.instance_id: source})
    migration = await router._prepare_reparallelization_requests(deployment)
    result = await request_task

    assert migration["migratable"] == 1
    assert result["choices"] == []
    assert target_backend.restore_calls == 1
    print(json.dumps({"status": "passed", "migration": migration}))


if __name__ == "__main__":
    asyncio.run(main())
