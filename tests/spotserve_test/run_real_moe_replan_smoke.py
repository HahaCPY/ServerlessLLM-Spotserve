"""Real GPU smoke for planner-selected MoE re-parallelization.

Run this on a host with the model directory and at least three visible GPUs:

    CUDA_VISIBLE_DEVICES=1,2,3 \
      python tests/spotserve_test/run_real_moe_replan_smoke.py

The extra GPU is intentional: the executor makes the TP2 target ready before
draining the TP1 source, so both deployments must fit concurrently.
"""

import asyncio
import json
import os
import time

import ray

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.spot.reparallelization import ParallelPlan


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Scheduler:
    """Dependency-light scheduler facade for a real local Ray smoke."""

    def __init__(self):
        async def allocate_resource(**kwargs):
            return str(kwargs.get("target_node_id", "0"))

        async def deallocate_resource(*args):
            return None

        self.allocate_resource = _Remote(allocate_resource)
        self.deallocate_resource = _Remote(deallocate_resource)


async def main() -> None:
    model_path = os.getenv(
        "SPOTSERVE_REAL_MOE_MODEL", "/work/spotserve-models/Qwen2-MoE-Tiny"
    )
    model_name = "moe-planner-real-smoke"
    gpu_count = int(os.getenv("SPOTSERVE_REAL_MOE_GPU_COUNT", "3"))
    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=gpu_count,
        num_cpus=4,
        resources={"worker_node": 1, "worker_id_0": 1, "worker_id_1": 1},
        _temp_dir="/tmp/spotserve-ray-planner-real-smoke",
    )

    backend_config = {
        "pretrained_model_name_or_path": model_path,
        "load_format": "auto",
        "torch_dtype": "float16",
        "gpu_memory_utilization": 0.05,
        "max_model_len": 32,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": True,
    }
    router = RoundRobinRouter(
        model_name=model_name,
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        backend="vllm",
        backend_config=backend_config,
        router_config={
            "enable_reparallelization": True,
            "reparallelization_config": {
                "model_gpu_requirement": 1,
                "target_replica_gpus": 2,
                "max_tensor_parallel_size": 2,
                "max_pipeline_parallel_size": 1,
                "drain_timeout_s": 60,
                "synthetic_worker_nodes": {
                    "0": {
                        "ray_node_id": "0",
                        "address": "0",
                        "free_gpu": 1,
                        "total_gpu": 1,
                        "state": "ready",
                    },
                    "1": {
                        "ray_node_id": "1",
                        "address": "1",
                        "free_gpu": 2,
                        "total_gpu": 2,
                        "state": "ready",
                    },
                },
            },
        },
    )
    router.model_loading_scheduler = _Scheduler()
    assert router._ensure_vllm_reparallelization_adapter()
    adapter = router.vllm_deployment_adapter
    source_plan = ParallelPlan(
        model_name=model_name,
        backend="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        num_replicas=1,
        num_gpus=1,
        target_nodes=["0"],
        reason="source",
    )
    source = None
    target = None
    started = time.monotonic()
    try:
        source = await adapter.create_workers(source_plan)
        assert await adapter.ready_workers(source, source_plan)
        router.ready_inference_instances = dict(source.instances)

        result = await router._replan_after_spot_event(
            "preempt", "0", None, list(source.instances.values())
        )
        assert result["action"] == "reparallelize", result
        assert result["execution"]["status"] == "applied", result
        assert result["execution"]["parallel_plan"]["tensor_parallel_size"] == 2

        target = adapter.snapshot(router.ready_inference_instances)
        target_actor = next(iter(target.instances.values())).backend_instance
        target_result = await target_actor.generate.remote(
            {
                "model": model_name,
                "request_id": "planner-target-1",
                "prompt": "Say TARGET.",
                "max_tokens": 3,
                "temperature": 0.0,
            }
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "elapsed_s": round(time.monotonic() - started, 2),
                    "planner_action": result["action"],
                    "execution": result["execution"],
                    "target_result": target_result,
                },
                default=str,
            )
        )
    finally:
        if target is not None:
            await adapter.stop_workers(target)
        elif source is not None:
            await adapter.stop_workers(source)
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
