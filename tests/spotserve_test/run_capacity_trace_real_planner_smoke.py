"""Run one real-GPU planner expansion/shrink trace for a MoE model.

Unlike the four-policy recovery harness, this controller does not choose a
TP shape itself.  It exposes four one-GPU worker slots to RoundRobinRouter,
applies each trace record atomically, and lets the router's real
``plan_dynamic_reparallelization`` plus ``VllmDeploymentAdapter`` create the
selected deployment.  The smoke has no live request; request-state migration
is measured separately by the NIXL tests.
"""

import argparse
import asyncio
import copy
import json
import os
import time
from pathlib import Path

import ray

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.spot.reparallelization import (
    ParallelPlan,
    apply_spot_event_to_worker_nodes,
)

from run_four_container_fleet_churn_smoke import load_fleet_trace


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Scheduler:
    def __init__(self):
        async def allocate_resource(**kwargs):
            return str(kwargs.get("target_node_id", "0"))

        async def deallocate_resource(*args):
            return None

        self.allocate_resource = _Remote(allocate_resource)
        self.deallocate_resource = _Remote(deallocate_resource)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--min-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--target-replica-gpus", type=int, default=2)
    parser.add_argument("--max-tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.08)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    if not Path(args.model, "config.json").is_file():
        raise SystemExit(f"model config not found: {args.model}")
    min_tp = max(1, int(args.min_tensor_parallel_size))
    model_is_qwen_large = min_tp >= 2
    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=4,
        num_cpus=8,
        resources={
            "worker_node": 1,
            **{f"worker_id_{index}": 1 for index in range(4)},
        },
        _temp_dir=f"/tmp/spotserve-capacity-planner-{os.getpid()}",
    )
    backend_config = {
        "pretrained_model_name_or_path": args.model,
        "load_format": "auto",
        "torch_dtype": "bfloat16" if model_is_qwen_large else "float16",
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": 2,
        "tensor_parallel_size": min_tp,
        "pipeline_parallel_size": 1,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": True,
        "moe_backend": "triton",
    }
    if model_is_qwen_large:
        backend_config["enable_expert_parallel"] = True
    model_name = (
        "capacity-planner-qwen-a27b"
        if model_is_qwen_large
        else "capacity-planner-tiny"
    )
    initial_ready = min_tp
    worker_nodes = {
        str(index): {
            "ray_node_id": str(index),
            "address": "local",
            "free_gpu": 1 if index < initial_ready else 0,
            "total_gpu": 1,
            "state": "ready" if index < initial_ready else "dead",
        }
        for index in range(4)
    }
    planner_config = {
        "model_gpu_requirement": min_tp,
        "target_replica_gpus": max(1, int(args.target_replica_gpus)),
        "min_tensor_parallel_size": min_tp,
        "max_tensor_parallel_size": max(1, int(args.max_tensor_parallel_size)),
        "max_pipeline_parallel_size": 1,
        # This smoke has no in-flight requests, so it can release the old
        # deployment before creating a different shape within the four-GPU
        # capacity envelope.
        "allow_stop_before_recreate": True,
        "migrate_before_create": False,
        "synthetic_worker_nodes": copy.deepcopy(worker_nodes),
    }
    router = RoundRobinRouter(
        model_name=model_name,
        resource_requirements={"num_cpus": 1, "num_gpus": min_tp},
        backend="vllm",
        backend_config=backend_config,
        router_config={
            "enable_reparallelization": True,
            "reparallelization_config": planner_config,
        },
    )
    router.model_loading_scheduler = _Scheduler()
    if not router._ensure_vllm_reparallelization_adapter():
        raise RuntimeError("vLLM deployment adapter unavailable")

    # Keep the trace's four physical slots authoritative.  The Ray adapter
    # still starts real actors, but a handle's aggregated TP resources must not
    # overwrite the one-GPU slot model used for the next planner decision.
    async def snapshot_worker_nodes():
        return copy.deepcopy(worker_nodes)

    router._snapshot_reparallelization_worker_nodes = snapshot_worker_nodes
    adapter = router.vllm_deployment_adapter
    source_plan = ParallelPlan(
        model_name=model_name,
        backend="vllm",
        tensor_parallel_size=min_tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=2 if model_is_qwen_large else 1,
        num_replicas=1,
        num_gpus=min_tp,
        target_nodes=[str(index) for index in range(initial_ready)],
        reason="trace_source",
    )
    started = time.monotonic()
    current = None
    rows: list[dict] = []
    try:
        current = await adapter.create_workers(source_plan)
        if not await adapter.ready_workers(current, source_plan):
            raise RuntimeError("source deployment did not become ready")
        router.ready_inference_instances = dict(current.instances)
        router.running = True
        for event_index, event in enumerate(load_fleet_trace(args.trace)):
            event_name = str(event["event"])
            if event_name == "DONE":
                break
            updates = {}
            for node in event["nodes"]:
                node_id = str(node).removeprefix("node-")
                if node_id not in worker_nodes:
                    raise ValueError(f"trace references non-existent GPU slot {node}")
                info = {"total_gpu": 1}
                if event_name == "add":
                    info.update({"free_gpu": 1, "state": "ready"})
                else:
                    info.update({"free_gpu": 0, "state": "dead"})
                updates[node_id] = info
                worker_nodes = apply_spot_event_to_worker_nodes(
                    worker_nodes, event_name, node_id, node_info=info
                )
            transition_started = time.monotonic()
            decision = await router._replan_after_spot_event(
                event_name,
                None,
                None,
                list(router.ready_inference_instances.values()),
                worker_node_updates=updates,
            )
            current = adapter.snapshot(router.ready_inference_instances)
            rows.append(
                {
                    "event_index": event_index,
                    "time_ms": event["time_ms"],
                    "event": event_name,
                    "nodes": list(event["nodes"]),
                    "available_gpus": sum(
                        int(info.get("free_gpu", 0))
                        for info in worker_nodes.values()
                        if info.get("state") == "ready"
                    ),
                    "selected_plan": (decision or {}).get("parallel_plan"),
                    "planner_action": (decision or {}).get("action"),
                    "execution": (decision or {}).get("execution"),
                    "transition_s": round(
                        time.monotonic() - transition_started, 3
                    ),
                }
            )
        output = {
            "status": "passed",
            "model": args.model,
            "trace": args.trace,
            "gpu_slots": 4,
            "planner_config": planner_config,
            "elapsed_s": round(time.monotonic() - started, 3),
            "rows": rows,
            "plan_shapes": sorted(
                {
                    json.dumps(row["selected_plan"], sort_keys=True)
                    for row in rows
                    if row.get("selected_plan")
                }
            ),
        }
        Path(args.output).write_text(
            json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({"status": "passed", "output": args.output}))
    finally:
        if current is not None:
            await adapter.stop_workers(current)
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
