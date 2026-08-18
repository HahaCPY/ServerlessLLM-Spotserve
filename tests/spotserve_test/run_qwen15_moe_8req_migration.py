"""Run repeated live Qwen1.5-MoE planner/NIXL migration experiments.

The experiment deliberately keeps two TP2/EP2 deployments alive at once:
the source uses two GPUs and the replacement target uses the other two.  Each
round first records a deterministic no-migration baseline for eight requests,
then sends the same eight prompts while the planner creates a replacement
worker and the router performs stateful KV restore.

This is a same-host experiment.  The scheduler uses synthetic placement slots
(``0``/``1``) for planner decisions but reports one physical ``local`` node to
the vLLM backend, so the runtime is allowed to exercise same-node NIXL restore
without enabling the physical cross-node capability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import ray

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.spot.reparallelization import ParallelPlan


MODEL_DEFAULT = "/work/spotserve-models/Qwen1.5-MoE-A2.7B"
MODEL_NAME = "Qwen1.5-MoE-A2.7B"
REQUEST_COUNT = 8


class _Remote:
    def __init__(self, callback):
        self.remote = callback


class _Scheduler:
    """Local scheduler facade with two synthetic planner placement slots.

    Both slots map to one physical host for the restore compatibility check;
    Ray still allocates two distinct GPU groups because source and target are
    live concurrently.
    """

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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return float(ordered[index])


def _prompt(run_index: int, request_index: int) -> str:
    # Long enough to allocate several KV blocks while remaining below the
    # engine limit.  The text is deterministic across baseline and
    # migration phases.
    body = " ".join(
        f"segment-{run_index}-{request_index}-{part:03d}"
        for part in range(48)
    )
    return (
        "You are a deterministic migration benchmark. Continue the sequence "
        "without summarising. "
        + body
    )


def _request_payload(
    request_id: str, prompt: str, max_tokens: int, token_delay_s: float
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "_spotserve_return_token_ids": True,
        "_spotserve_token_delay_s": token_delay_s,
    }


async def _wait_for_live_kv(
    router: RoundRobinRouter,
    source: Any,
    request_ids: list[str],
    timeout_s: float,
) -> dict[str, dict[str, Any]]:
    # Wait until vLLM's output processor has registered the external IDs and
    # the first KV blocks are visible.  Calling export during prompt prefill
    # races that registration and produces a misleading request_not_active
    # token fallback even though the request is still running.
    deadline = asyncio.get_running_loop().time() + max(0.1, timeout_s)
    live: dict[str, dict[str, Any]] = {}
    while asyncio.get_running_loop().time() < deadline:
        for request_id in request_ids:
            if request_id in live:
                continue
            try:
                row = await source.backend_instance.get_request_kv_metadata.remote(
                    request_id=request_id,
                    instance_id=source.instance_id,
                    node_id=source.node_id or "",
                )
                if isinstance(row, dict) and row.get("found") is True:
                    live[request_id] = dict(row)
            except Exception:
                continue
        if len(live) == len(request_ids):
            break
        await asyncio.sleep(0.5)
    return live


async def _wait_results(
    tasks: dict[str, asyncio.Task],
    timeout_s: float,
) -> dict[str, Any]:
    gathered = await asyncio.wait_for(
        asyncio.gather(*tasks.values(), return_exceptions=True), timeout=timeout_s
    )
    return {
        request_id: result
        for request_id, result in zip(tasks, gathered, strict=True)
    }


def _initial_worker_nodes(source_slot: str) -> dict[str, dict[str, Any]]:
    target_slot = "1" if source_slot == "0" else "0"
    return {
        source_slot: {
            "ray_node_id": source_slot,
            "address": "local",
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
        },
        target_slot: {
            "ray_node_id": target_slot,
            "address": "local",
            "free_gpu": 2,
            "total_gpu": 2,
            "state": "ready",
        },
    }


async def _run_round(
    router: RoundRobinRouter,
    adapter: Any,
    source_deployment: Any,
    metrics: _Metrics,
    run_index: int,
    timeout_s: float,
    max_tokens: int,
    token_delay_s: float,
) -> tuple[dict[str, Any], Any]:
    source = next(iter(source_deployment.instances.values()))
    prompts = {
        f"run-{run_index}-request-{index}": _prompt(run_index, index)
        for index in range(REQUEST_COUNT)
    }

    async def submit(payload: dict[str, Any]) -> Any:
        return await router.inference(payload, "generate")

    # Baseline: identical requests complete on the source before migration.
    # Keep the baseline fast; the migration phase below deliberately holds
    # each live engine stream so that the planner has a real preemption
    # window rather than racing already-finished requests.
    baseline_ids = [f"baseline-{request_id}" for request_id in prompts]
    baseline_tasks = {
        request_id: asyncio.create_task(
            submit(
                _request_payload(
                    request_id, prompts[original_id], max_tokens, 0.0
                )
            )
        )
        for original_id, request_id in zip(prompts, baseline_ids, strict=True)
    }
    baseline_results = await _wait_results(baseline_tasks, timeout_s)
    baseline_tokens: dict[str, list[int]] = {}
    for original_id, baseline_id in zip(prompts, baseline_ids, strict=True):
        result = baseline_results[baseline_id]
        if isinstance(result, BaseException) or result.get("error"):
            raise RuntimeError(f"baseline failed for {baseline_id}: {result}")
        token_ids = result.get("_spotserve_token_ids")
        prompt_ids = result.get("_spotserve_prompt_token_ids")
        if not isinstance(token_ids, list) or not isinstance(prompt_ids, list):
            raise RuntimeError(f"baseline did not return token IDs: {result}")
        baseline_tokens[original_id] = [*prompt_ids, *token_ids]

    # Capture the exact source state at the migration boundary.  This is used
    # to reconstruct the target's full sequence (source prefix + target tail).
    captured_states: dict[str, Any] = {}

    async def migrate_and_capture(deployment: Any) -> dict[str, Any]:
        summary = await router._prepare_reparallelization_requests(deployment)
        async with router.inflight_requests_lock:
            for request_id, entry in router.inflight_requests.items():
                state = entry.get("migration_state")
                if state is not None:
                    captured_states[request_id] = state
        return summary

    adapter.request_migrator = migrate_and_capture

    migration_ids = [f"migration-{request_id}" for request_id in prompts]
    # Target vLLM startup includes model load, KV allocation and NIXL
    # handshake (about 40--60 s on this host).  Keep the source streams alive
    # long enough that drain/export happens while the requests are genuinely
    # in flight, rather than after they naturally finish.
    migration_hold_s = max(float(token_delay_s), 60.0)
    migration_tasks = {
        request_id: asyncio.create_task(
            submit(
                _request_payload(
                    request_id,
                    prompts[original_id],
                    max_tokens,
                    migration_hold_s,
                )
            )
        )
        for original_id, request_id in zip(prompts, migration_ids, strict=True)
    }
    # The request-specific hold keeps the request registered with the
    # frontend while the final output is being consumed.  Polling is still
    # required because the first KV block can be allocated a few event-loop
    # turns after admission.
    await _wait_for_live_kv(
        router,
        source,
        migration_ids,
        timeout_s=min(
            timeout_s,
            float(
                router.reparallelization_config.get(
                    "live_kv_wait_timeout_s", 30.0
                )
            ),
        ),
    )

    source_slot = str(run_index % 2)
    router.reparallelization_worker_nodes = _initial_worker_nodes(source_slot)
    migration_started = time.monotonic()
    decision = await router._replan_after_spot_event(
        "preempt", source_slot, None, [source]
    )
    migration_elapsed = time.monotonic() - migration_started
    migration_results = await _wait_results(migration_tasks, timeout_s)

    request_events = {
        str(event.get("request_id")): event
        for event in metrics.events
        if event.get("type") == "request"
        and str(event.get("request_id", "")).startswith("migration-")
    }
    restore_events = [
        event
        for event in metrics.events
        if event.get("type") == "state_recovery"
        and str(event.get("request_id", "")).startswith("migration-")
    ]

    sequence_equal = 0
    lost_or_error = 0
    restore_successes = 0
    restore_fallbacks = 0
    restored_blocks = 0
    target_generated_tokens = 0
    latency_ms: list[float] = []
    per_request: dict[str, Any] = {}

    for original_id, request_id in zip(prompts, migration_ids, strict=True):
        result = migration_results[request_id]
        state = captured_states.get(request_id)
        event = request_events.get(request_id, {})
        if isinstance(result, BaseException) or not isinstance(result, dict):
            lost_or_error += 1
            per_request[request_id] = {"status": "exception", "error": str(result)}
            continue
        if result.get("error") or state is None:
            lost_or_error += 1
            per_request[request_id] = {
                "status": "error",
                "error": result.get("error", "missing_migration_state"),
            }
            continue

        target_tail = result.get("_spotserve_token_ids", [])
        target_sequence = [*state.tokens, *target_tail]
        expected_sequence = baseline_tokens[original_id]
        equal = target_sequence == expected_sequence
        source_prefix_equal = expected_sequence[: len(state.tokens)] == list(
            state.tokens
        )
        mismatch_index = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(target_sequence, expected_sequence)
                )
                if actual != expected
            ),
            None,
        )
        sequence_equal += int(equal)
        target_generated_tokens += len(target_tail)
        latency_ms.append(float(event.get("latency_ms", 0.0) or 0.0))
        kv_restore = result.get("_spotserve_kv_restore", {}) or {}
        restored = bool(kv_restore.get("restored"))
        restore_successes += int(restored and not event.get("state_restore_fallback"))
        restore_fallbacks += int(bool(event.get("state_restore_fallback")))
        restored_blocks += int(kv_restore.get("restored_blocks", 0) or 0)
        per_request[request_id] = {
            "status": "passed" if equal else "sequence_mismatch",
            "source_prefix_tokens": len(state.tokens),
            "target_tail_tokens": len(target_tail),
            "target_sequence_tokens": len(target_sequence),
            "expected_sequence_tokens": len(expected_sequence),
            "sequence_equal": equal,
            "source_prefix_equal": source_prefix_equal,
            "first_mismatch_index": mismatch_index,
            "kv_restore": kv_restore,
            "request_metrics": event,
        }

    execution = decision.get("execution", {}) or {}
    migration = execution.get("request_migration", {}) or {}
    target_deployment = adapter.snapshot(router.ready_inference_instances)
    result = {
        "run": run_index,
        "planner_action": decision.get("action"),
        "planner_status": execution.get("status"),
        "selected_plan": decision.get("parallel_plan"),
        "request_migration": migration,
        "state_recovery_events": len(restore_events),
        "restore_successes": restore_successes,
        "restore_fallbacks": restore_fallbacks,
        "restored_blocks": restored_blocks,
        "lost_or_error": lost_or_error,
        "sequence_equal": sequence_equal,
        "sequence_mismatches": REQUEST_COUNT - sequence_equal - lost_or_error,
        "migration_elapsed_s": round(migration_elapsed, 3),
        "throughput_target_tokens_per_s": round(
            target_generated_tokens / max(migration_elapsed, 1e-6), 3
        ),
        "latency_ms": {
            "p50": round(_percentile(latency_ms, 50), 3),
            "p95": round(_percentile(latency_ms, 95), 3),
            "max": round(max(latency_ms, default=0.0), 3),
        },
        "requests": per_request,
        "state_samples": {
            request_id: {
                "state_kind": captured_states[request_id].state_kind,
                "supports_restore": captured_states[request_id].supports_restore,
                "completed_tokens": captured_states[request_id].completed_tokens,
                "runtime_state_keys": sorted(
                    (captured_states[request_id].runtime_state or {}).keys()
                ),
                "metadata_reason": (captured_states[request_id].metadata or {}).get(
                    "reason"
                ),
                "export_error": (captured_states[request_id].metadata or {}).get(
                    "export_error"
                ),
                "export_reason": (captured_states[request_id].metadata or {}).get(
                    "export_reason"
                ),
            }
            for request_id in list(captured_states)[:2]
        },
    }
    return result, target_deployment


async def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("SPOTSERVE_TEST_TOKEN_DELAY_S", str(args.token_delay_s))
    os.environ.setdefault("VLLM_NIXL_SIDE_CHANNEL_HOST", "127.0.0.1")
    os.environ.setdefault("VLLM_NIXL_SIDE_CHANNEL_BASE_PORT", "6200")

    model_path = str(Path(args.model).resolve())
    if not (Path(model_path) / "config.json").is_file():
        raise SystemExit(f"model config not found: {model_path}/config.json")

    ray.init(
        address=None,
        include_dashboard=False,
        num_gpus=4,
        num_cpus=8,
        resources={"worker_node": 1, "worker_id_local": 1},
        _temp_dir=f"/tmp/spotserve-qwen15-migration-{os.getpid()}",
    )

    backend_config = {
        "pretrained_model_name_or_path": model_path,
        "load_format": "auto",
        "torch_dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # Restored input tokens plus at least one generated token must fit.
        # Keep headroom so a live snapshot is not rejected at the boundary.
        "max_model_len": 8192,
        "max_num_seqs": REQUEST_COUNT,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": True,
        "enable_prefix_caching": True,
        "enforce_eager": True,
        "trust_remote_code": True,
        "moe_backend": "triton",
        "trace_debug": False,
        "test_token_delay_s": args.token_delay_s,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
            "kv_buffer_device": "cuda",
            # The target takes roughly 40 s to load this MoE. Keep the source
            # lease alive through engine startup and the first attach batch.
            "kv_connector_extra_config": {"kv_lease_duration": 300},
        },
    }
    router = RoundRobinRouter(
        model_name=MODEL_NAME,
        resource_requirements={"num_cpus": 1, "num_gpus": 2},
        backend="vllm",
        backend_config=backend_config,
        router_config={
            "enable_reparallelization": True,
            "recovery_policy": "stateful_recovery",
            "max_retries": 1,
            "reparallelization_config": {
                "model_gpu_requirement": 2,
                "target_replica_gpus": 2,
                # This checkpoint is about 27 GiB and does not fit on one
                # 16-GiB GPU. Keep planner candidates at TP2 or above when
                # capacity shrinks; TP1 is not a runnable fallback.
                "min_tensor_parallel_size": 2,
                "max_tensor_parallel_size": 2,
                "max_pipeline_parallel_size": 1,
                "drain_timeout_s": args.drain_timeout_s,
                "live_kv_wait_timeout_s": 3.0,
                # Keep source alive until target NIXL attaches; the planner
                # still marks it preempting and aborts tracked requests, but
                # the source lease must remain valid during target startup.
                "allow_stop_before_recreate": False,
                "migrate_before_create": True,
                "synthetic_worker_nodes": _initial_worker_nodes("0"),
            },
        },
    )
    router.auto_scaling_config = {
        "metric": "concurrency",
        "target": REQUEST_COUNT,
        "min_instances": 1,
        "max_instances": 1,
    }
    router.model_loading_scheduler = _Scheduler()
    metrics = _Metrics()
    router.metrics_writer = metrics
    if not router._ensure_vllm_reparallelization_adapter():
        raise RuntimeError("vLLM re-parallelization adapter unavailable")
    adapter = router.vllm_deployment_adapter
    assert adapter is not None

    async def allocate_instance() -> tuple[str, Any]:
        # Wait through the brief drain/switch window.  The default router
        # queue is intentionally bypassed because this standalone experiment
        # drives exactly eight requests against one deployment.
        while True:
            for handle in list(router.ready_inference_instances.values()):
                if await handle.add_requests(1):
                    return handle.instance_id, handle
            await asyncio.sleep(0.05)

    router._allocate_instance_for_request = allocate_instance
    router.running = True

    source_plan = ParallelPlan(
        model_name=MODEL_NAME,
        backend="vllm",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=2,
        num_replicas=1,
        num_gpus=2,
        target_nodes=["0"],
        reason="source_tp2_ep2",
    )
    source = None
    capability_diagnostics: dict[str, Any] = {}
    started = time.monotonic()
    rounds: list[dict[str, Any]] = []
    try:
        source = await adapter.create_workers(source_plan)
        if not await adapter.ready_workers(source, source_plan):
            raise RuntimeError("source worker readiness failed")
        router.ready_inference_instances = dict(source.instances)
        source_handle = next(iter(source.instances.values()))
        try:
            capability_diagnostics["source"] = await source_handle.backend_instance.get_state_restore_diagnostics.remote()
        except Exception as exc:
            capability_diagnostics["source_error"] = repr(exc)

        for run_index in range(args.runs):
            round_result, source = await _run_round(
                router,
                adapter,
                source,
                metrics,
                run_index,
                args.timeout_s,
                args.max_tokens,
                args.token_delay_s,
            )
            rounds.append(round_result)
            print(json.dumps(round_result, sort_keys=True), flush=True)

        summary = {
            "status": "passed",
            "model": model_path,
            "runs": args.runs,
            "requests_per_run": REQUEST_COUNT,
            "total_requests": args.runs * REQUEST_COUNT,
            "restore_successes": sum(item["restore_successes"] for item in rounds),
            "restore_fallbacks": sum(item["restore_fallbacks"] for item in rounds),
            "lost_or_error": sum(item["lost_or_error"] for item in rounds),
            "sequence_equal": sum(item["sequence_equal"] for item in rounds),
            "sequence_mismatches": sum(
                item["sequence_mismatches"] for item in rounds
            ),
            "restore_success_rate": round(
                sum(item["restore_successes"] for item in rounds)
                / max(args.runs * REQUEST_COUNT, 1),
                4,
            ),
            "fallback_rate": round(
                sum(item["restore_fallbacks"] for item in rounds)
                / max(args.runs * REQUEST_COUNT, 1),
                4,
            ),
            "elapsed_s": round(time.monotonic() - started, 2),
            "rounds": rounds,
            "capability_diagnostics": capability_diagnostics,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.output:
            Path(args.output).write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
    finally:
        if source is not None:
            await adapter.stop_workers(source)
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--drain-timeout-s", type=float, default=240.0)
    parser.add_argument("--token-delay-s", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument(
        "--output",
        default="/tmp/spotserve-qwen15-moe-8req-migration.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
