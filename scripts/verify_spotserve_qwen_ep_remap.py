"""End-to-end quiescent expert remap smoke on Qwen2-MoE-Tiny."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def _model_topology(model: str) -> tuple[int, int]:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layers = int(config.get("num_hidden_layers", 0) or 0)
    experts = int(config.get("num_experts", 0) or 0)
    if layers < 1 or experts < 2 or experts % 2:
        raise RuntimeError(
            "smoke requires an even number of experts and at least one layer"
        )
    return layers, experts


def _round_robin_plan(model: str) -> dict[str, Any]:
    layers, experts = _model_topology(model)
    expert_to_target_rank = {
        f"layer:{layer}/expert:{expert}": (
            f"replica:0/ep-rank:{expert % 2}"
        )
        for layer in range(layers)
        for expert in range(experts)
    }
    fingerprint = hashlib.sha256(
        json.dumps(expert_to_target_rank, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "model_name": model,
        "placement_epoch": 1,
        "placement_fingerprint": fingerprint,
        "placement_source": "qwen_ep2_remap_smoke",
        "live_expert_remap": True,
        "expert_placement_physical_migration_required": True,
        "target_rank_count": 2,
        "sllm_replica_count": 1,
        "expert_physical_replication_factor": 1,
        "target_parallel_plan": {
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "enable_expert_parallel": True,
        },
        "expert_to_target_rank": expert_to_target_rank,
    }


async def _generate(
    engine: Any, prompt: str, request_id: str, max_tokens: int = 8
) -> list[int]:
    final = None
    params = SamplingParams(
        temperature=0, max_tokens=max_tokens, seed=1, ignore_eos=True
    )
    async for output in engine.generate(prompt, params, request_id):
        final = output
    if final is None or not final.outputs:
        raise RuntimeError(f"request produced no output: {request_id}")
    return list(final.outputs[0].token_ids)


async def _remap_during_request(
    engine: Any, plan: dict[str, Any], prompt: str
) -> tuple[dict[str, Any], list[int], list[int]]:
    baseline = await _generate(engine, prompt, "active-baseline", max_tokens=80)
    first_token = asyncio.Event()

    async def consume() -> list[int]:
        final = None
        params = SamplingParams(
            temperature=0, max_tokens=80, seed=1, ignore_eos=True
        )
        async for output in engine.generate(prompt, params, "active-remap"):
            final = output
            if output.outputs and output.outputs[0].token_ids:
                first_token.set()
        if final is None or not final.outputs:
            raise RuntimeError("active request produced no output")
        return list(final.outputs[0].token_ids)

    request = asyncio.create_task(consume())
    await asyncio.wait_for(first_token.wait(), timeout=120)
    if request.done():
        raise RuntimeError("request completed before remap could begin")
    plan["allow_active_requests"] = True
    apply_result = await engine.apply_expert_placement_plan(plan)
    active_tokens = await request
    if not apply_result.get("applied") or not apply_result.get(
        "active_requests_at_barrier"
    ) or not apply_result.get("step_boundary_barrier"):
        raise RuntimeError(f"active-request remap failed: {apply_result}")
    if baseline != active_tokens:
        raise RuntimeError(
            f"active request diverged: baseline={baseline}, remap={active_tokens}"
        )
    return apply_result, baseline, active_tokens


async def _run(args: argparse.Namespace) -> None:
    os.environ["VLLM_SPOTSERVE_EXPERT_REMAP"] = "1"
    if args.active_request:
        os.environ["VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP"] = "1"
    if args.physical_host_id:
        os.environ["SPOTSERVE_PHYSICAL_HOST_ID"] = args.physical_host_id
    os.environ.setdefault("VLLM_ALL2ALL_BACKEND", "allgather_reducescatter")
    plan = _round_robin_plan(args.model)
    if args.require_cross_node:
        plan["require_cross_node"] = True
    engine_args = AsyncEngineArgs(
        model=args.model,
        load_format="auto",
        dtype="float16",
        tensor_parallel_size=2,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        distributed_executor_backend="mp",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=128,
        max_num_seqs=1,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    try:
        before_metadata = await engine.get_moe_runtime_metadata()
        if args.active_request:
            apply_result, before_tokens, after_tokens = (
                await _remap_during_request(engine, plan, args.prompt)
            )
        else:
            before_tokens = await _generate(engine, args.prompt, "before-remap")
            apply_result = await engine.apply_expert_placement_plan(plan)
        if not apply_result.get("applied"):
            raise RuntimeError(f"apply failed: {apply_result}")
        verify_result = await engine.verify_expert_placement_plan(plan)
        if not verify_result.get("verified"):
            raise RuntimeError(f"verify failed: {verify_result}")
        if not apply_result.get("physical_weight_migration"):
            raise RuntimeError(f"no physical movement reported: {apply_result}")
        if args.require_cross_node:
            if not apply_result.get("physical_host_ids_observed"):
                raise RuntimeError(
                    f"runtime did not observe physical host ids: {apply_result}"
                )
            if not apply_result.get("cross_node_weight_migration"):
                raise RuntimeError(
                    f"no cross-node weight movement reported: {apply_result}"
                )
        after_metadata = await engine.get_moe_runtime_metadata()
        if not args.active_request:
            after_tokens = await _generate(engine, args.prompt, "after-remap")
        if before_tokens != after_tokens:
            raise RuntimeError(
                f"token mismatch: before={before_tokens}, after={after_tokens}"
            )
        report = {
            "passed": True,
            "active_request": args.active_request,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "apply": apply_result,
            "verify": verify_result,
            "before_runtime_placement": before_metadata.get(
                "runtime_expert_placement_shards", {}
            ),
            "after_runtime_placement": after_metadata.get(
                "runtime_expert_placement_shards", {}
            ),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"report={args.output}")
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SPOTSERVE_MOE_REMAP_MODEL", "/models/Qwen2-MoE-Tiny"
        ),
    )
    parser.add_argument(
        "--prompt", default="Explain why expert routing matters in one sentence."
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--active-request", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--physical-host-id",
        default=os.environ.get("SPOTSERVE_PHYSICAL_HOST_ID", ""),
        help=(
            "Physical host id reported by every local EP worker. Same-host mp "
            "runs should use one id; true cross-node validation needs distinct "
            "ids per physical host from the deployment runtime."
        ),
    )
    parser.add_argument(
        "--require-cross-node",
        action="store_true",
        help=(
            "Fail unless runtime observes distinct physical host ids and moves "
            "at least one expert shard across those hosts."
        ),
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
