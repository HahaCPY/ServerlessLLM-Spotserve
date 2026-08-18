"""Real same-node smoke for planner-selected compatible NIXL recovery.

The source and target are both created before the request starts and both use
the same TP1 configuration.  The source is forced to return a preemption
marker; the router must reserve the already-ready target through the new
stateful target planner and restore the exported KV state there.  No new
engine is allowed after the preemption event.
"""

import asyncio
import json
import os
import time
from typing import Any

import ray

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.spot.reparallelization import ParallelPlan


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Scheduler:
    def __init__(self):
        async def allocate_resource(**_kwargs):
            return "local"

        async def deallocate_resource(*_args):
            return None

        self.allocate_resource = _Remote(allocate_resource)
        self.deallocate_resource = _Remote(deallocate_resource)


class _Metrics:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


async def main() -> None:
    model_path = os.getenv(
        "SPOTSERVE_STATEFUL_PLANNER_MODEL",
        "/work/spotserve-models/Qwen2-MoE-Tiny",
    )
    model_name = "stateful-target-planner-real-smoke"
    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=2,
        num_cpus=4,
        resources={"worker_node": 1, "worker_id_local": 1},
        _temp_dir="/tmp/sstp",
    )

    backend_config = {
        "pretrained_model_name_or_path": model_path,
        "load_format": "auto",
        "torch_dtype": "float16",
        "gpu_memory_utilization": 0.05,
        "max_model_len": 512,
        "max_num_seqs": 2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": True,
        "moe_backend": "triton",
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
            "kv_buffer_device": "cuda",
            "kv_connector_extra_config": {"kv_lease_duration": 120},
        },
    }
    router = RoundRobinRouter(
        model_name=model_name,
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        backend="vllm",
        backend_config=backend_config,
        router_config={
            "recovery_policy": "stateful_recovery",
            "max_retries": 1,
            "enable_stateful_target_planner": True,
        },
    )
    router.model_loading_scheduler = _Scheduler()
    metrics = _Metrics()
    router.metrics_writer = metrics
    from sllm.spot.vllm_deployment_adapter import VllmDeploymentAdapter

    adapter = VllmDeploymentAdapter(
        model_name=model_name,
        backend_config=backend_config,
        resource_requirements={"num_cpus": 1, "num_gpus": 1},
        scheduler=router.model_loading_scheduler,
        traffic_switcher=lambda *_args: None,
    )
    source = None
    target = None
    allocation_count = 0
    worker_creation_count = 0
    started = time.monotonic()
    plan = ParallelPlan(
        model_name=model_name,
        backend="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        num_replicas=1,
        num_gpus=1,
        target_nodes=["local"],
        reason="preexisting_compatible_target",
    )
    try:
        original_create_workers = adapter.create_workers

        async def tracked_create_workers(worker_plan):
            nonlocal worker_creation_count
            worker_creation_count += 1
            return await original_create_workers(worker_plan)

        adapter.create_workers = tracked_create_workers
        source = await adapter.create_workers(plan)
        target = await adapter.create_workers(plan)
        if not await adapter.ready_workers(source, plan):
            raise RuntimeError("source worker readiness failed")
        if not await adapter.ready_workers(target, plan):
            raise RuntimeError("target worker readiness failed")
        router.ready_inference_instances = {
            **source.instances,
            **target.instances,
        }
        source_handle = next(iter(source.instances.values()))
        target_handle = next(iter(target.instances.values()))

        async def allocate_source_only():
            nonlocal allocation_count
            if allocation_count > 0:
                raise AssertionError("planner did not reserve the READY target")
            allocation_count += 1
            if not await source_handle.add_requests(1):
                raise AssertionError("source was not available")
            return source_handle.instance_id, source_handle

        router._allocate_instance_for_request = allocate_source_only
        router.running = True
        result = await asyncio.wait_for(router.inference(
            {
                "request_id": "stateful-target-planner-real-1",
                "model": model_name,
                "prompt": "Continue a deterministic migration sequence.",
                "max_tokens": 3,
                "ignore_eos": True,
                "temperature": 0.0,
                "force_failure": "preempted",
                "force_fail_after_tokens": 1,
                "force_fail_once": True,
                "_spotserve_return_token_ids": True,
            },
            "generate",
        ), timeout=180.0)
        state_events = [
            event for event in metrics.events if event.get("type") == "state_recovery"
        ]
        if not state_events:
            diagnostic = {
                "status": "failed",
                "reason": "no_state_recovery_event",
                "request_result": result,
                "metrics": metrics.events,
                "allocation_attempts": allocation_count,
            }
            print(json.dumps(diagnostic, default=str))
            raise AssertionError(diagnostic)
        state_event = state_events[-1]
        restore_result = result.get("_spotserve_kv_restore", {})
        report = {
            "status": "passed",
            "elapsed_s": round(time.monotonic() - started, 3),
            "planner_target": state_event.get("target_instance_id"),
            "planner_reason": state_event.get("reason"),
            "expected_target": target_handle.instance_id,
            "state_restore_successes": state_event.get("action") == "restore_state",
            "request_result": result,
            "kv_restore": restore_result,
            "allocation_attempts": allocation_count,
            "worker_creation_count": worker_creation_count,
            "engine_created_after_preemption": worker_creation_count > 2,
            "tensor_parallel_size": 1,
        }
        if report["planner_target"] != report["expected_target"]:
            raise AssertionError(report)
        if state_event.get("reason") != "planner_selected_compatible_ready_target":
            raise AssertionError(report)
        if not restore_result.get("restored"):
            raise AssertionError(report)
        if report["engine_created_after_preemption"]:
            raise AssertionError(report)
        print(json.dumps(report, default=str))
    finally:
        if target is not None:
            await adapter.stop_workers(target)
        if source is not None:
            await adapter.stop_workers(source)
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
