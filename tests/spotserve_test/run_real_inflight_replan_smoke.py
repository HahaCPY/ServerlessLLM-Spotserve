"""Real GPU smoke for an in-flight request during planner re-parallelization.

This complements ``run_real_moe_replan_smoke.py``: the latter verifies that a
new target can serve traffic, while this script keeps a request alive long
enough for the router's export/abort/retry path to run during the replan.

The test needs three visible GPUs because the TP2 target is made ready before
the TP1 source is drained.
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
    """Small scheduler facade that leaves Ray GPU placement to the adapter."""

    def __init__(self):
        async def allocate_resource(**kwargs):
            return str(kwargs.get("target_node_id", "0"))

        async def deallocate_resource(*args):
            return None

        self.allocate_resource = _Remote(allocate_resource)
        self.deallocate_resource = _Remote(deallocate_resource)


async def _wait_until_live(router, source, request_id: str, timeout_s: float):
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if request_id in router.inflight_requests:
            try:
                rows = await source.backend_instance.get_context_metadata.remote(
                    instance_id=source.instance_id,
                    node_id=source.node_id or "",
                )
                if rows:
                    return rows
            except Exception:
                # The request may be between admission and its first scheduler
                # iteration. Keep polling until it has a real KV snapshot.
                pass
        await asyncio.sleep(0.2)
    raise TimeoutError("request did not expose live KV metadata")


async def main() -> None:
    # Keep the real GPU request alive while the TP2 replacement engine loads.
    # This is an explicit smoke-test pacing knob; production defaults to zero.
    os.environ.setdefault("SPOTSERVE_TEST_TOKEN_DELAY_S", "20")
    model_path = os.getenv(
        "SPOTSERVE_REAL_MOE_MODEL", "/work/spotserve-models/Qwen2-MoE-Tiny"
    )
    model_name = "moe-inflight-replan-real-smoke"
    gpu_count = int(os.getenv("SPOTSERVE_REAL_MOE_GPU_COUNT", "3"))
    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=gpu_count,
        num_cpus=4,
        resources={"worker_node": 1, "worker_id_0": 1, "worker_id_1": 1},
        # Keep the Unix socket path below Linux's 108-byte AF_UNIX limit.
        _temp_dir="/tmp/srirm",
    )

    backend_config = {
        "pretrained_model_name_or_path": model_path,
        "load_format": "auto",
        "torch_dtype": "float16",
        "gpu_memory_utilization": 0.05,
        # Qwen2-MoE-Tiny advertises max_position_embeddings=512.
        "max_model_len": 512,
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
            "recovery_policy": "stateful_recovery",
            "max_retries": 1,
            "reparallelization_config": {
                "model_gpu_requirement": 1,
                "target_replica_gpus": 2,
                "max_tensor_parallel_size": 2,
                "max_pipeline_parallel_size": 1,
                "drain_timeout_s": 120,
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
    source = None
    target = None
    request_id = "real-inflight-replan-1"
    allocation_count = 0
    started = time.monotonic()

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
    try:
        source = await adapter.create_workers(source_plan)
        assert await adapter.ready_workers(source, source_plan)
        router.ready_inference_instances = dict(source.instances)
        source_handle = next(iter(source.instances.values()))
        router.running = True

        async def allocate_for_test():
            nonlocal allocation_count
            handle = source_handle if allocation_count == 0 else next(
                iter(router.ready_inference_instances.values())
            )
            allocation_count += 1
            await handle.add_requests(1)
            return handle.instance_id, handle

        router._allocate_instance_for_request = allocate_for_test
        request_task = asyncio.create_task(
            router.inference(
                {
                    "request_id": request_id,
                    "model": model_name,
                    "prompt": "Continue a long deterministic explanation.",
                    "max_tokens": 480,
                    "ignore_eos": True,
                    "temperature": 0.0,
                },
                "generate",
            )
        )
        live_rows = await _wait_until_live(
            router, source_handle, request_id, timeout_s=60
        )

        result = await router._replan_after_spot_event(
            "preempt", "0", None, list(source.instances.values())
        )
        assert result["action"] == "reparallelize", result
        assert result["execution"]["status"] == "applied", result
        migration = result["execution"].get("request_migration") or {}
        assert migration.get("attempted", 0) >= 1, result
        assert migration.get("abort_requested", 0) >= 1, result

        request_result = await asyncio.wait_for(request_task, timeout=180)
        assert not request_result.get("error"), request_result
        target = adapter.snapshot(router.ready_inference_instances)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "elapsed_s": round(time.monotonic() - started, 2),
                    "live_context": live_rows[0] if live_rows else {},
                    "planner_action": result["action"],
                    "execution": result["execution"],
                    "request_result": request_result,
                    "allocation_attempts": allocation_count,
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
