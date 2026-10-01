"""Small EP feasibility/communication probe, not a SpotServe comparison."""

import argparse
import asyncio
import json
import time
import traceback

from scripts import moe_diagnostic_trace as trace
from scripts.profile_granite_moe_tp import digest


async def all_rank_rpc(engine, method, args=()):
    core = engine.engine_core
    if engine.vllm_config.parallel_config.data_parallel_size == 1:
        return await engine.collective_rpc(method, timeout=120, args=args)
    # This pinned DPLB client executes utility RPCs on all engines but
    # returns only the first result. Request all results explicitly.
    results = await asyncio.gather(*[
        core._call_utility_async("collective_rpc", method, 120, args, {}, engine=identity)
        for identity in core.core_engines])
    return [row for group in results for row in group]


async def run(args):
    trace.emit("ep_frontend_entry")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM
    report = {"status": "running", "tp": args.tp, "dp": args.dp, "ep_flag": args.ep,
              "cells": [], "communication_profiles": [], "formal_experiment": False}
    engine = None
    try:
        with trace.span("ep_engine_constructor"):
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                model=args.model, tensor_parallel_size=args.tp,
                data_parallel_size=args.dp, data_parallel_size_local=args.dp,
                enable_expert_parallel=args.ep, all2all_backend="allgather_reducescatter",
                worker_cls="scripts.moe_architecture_diagnostic_worker.ArchitectureDiagnosticWorker",
                enforce_eager=True, dtype="bfloat16", seed=0,
                trust_remote_code=False, cpu_offload_gb=0,
                max_model_len=4608, max_num_seqs=2, max_num_batched_tokens=2048,
                gpu_memory_utilization=0.7, enable_prefix_caching=False,
                enable_return_routed_experts=False, async_scheduling=False,
                moe_backend="triton", disable_log_stats=True))
        report["placement"] = await all_rank_rpc(engine, "get_expert_placement_evidence")
        report["startup"] = await all_rank_rpc(engine, "get_diagnostic_snapshot")
        parallel = engine.vllm_config.parallel_config
        report["observed_runtime"] = {
            "tp": parallel.tensor_parallel_size, "dp": parallel.data_parallel_size,
            "ep_flag": parallel.enable_expert_parallel,
            "all2all_backend": parallel.all2all_backend,
            "scheduler": str(engine.vllm_config.scheduler_config.scheduler_cls)}

        async def generate(tokens, length, identifier, rank):
            before = time.monotonic()
            previous = []
            first = None
            kwargs = {"data_parallel_rank": rank} if args.dp > 1 else {}
            async for item in engine.generate(TokensPrompt(prompt_token_ids=tokens),
                    SamplingParams(temperature=0, max_tokens=length, ignore_eos=True, seed=0),
                    identifier, **kwargs):
                current = list(item.outputs[0].token_ids)
                if len(current) < len(previous) or current[:len(previous)] != previous:
                    raise ValueError("stream prefix was revised")
                previous = current
                if current and first is None:
                    first = time.monotonic()
            if len(previous) != length or item.outputs[0].finish_reason != "length":
                raise ValueError("incomplete generation")
            return {"latency_s": time.monotonic() - before,
                    "ttft_s": first - before, "output_token_ids": previous,
                    "output_sha256": digest(previous), "prompt_sha256": digest(tokens)}

        for prompt in (512, 4096):
            for output in (1, 64):
                prompts = [[17 + index] * prompt for index in range(2)]
                async def batch(label):
                    before = time.monotonic()
                    requests = await asyncio.gather(*[
                        generate(tokens, output, f"{label}-{index}", index)
                        for index, tokens in enumerate(prompts)])
                    return {"batch_wall_s": time.monotonic() - before, "requests": requests}
                # All profiling is disabled for warmed service measurements.
                await all_rank_rpc(engine, "set_diagnostic_measurement", (None,))
                await batch(f"warm-{prompt}-{output}")
                samples = [await batch(f"measure-{prompt}-{output}-{i}") for i in range(3)]
                report["cells"].append({"prompt_tokens": prompt, "output_tokens": output,
                                        "concurrency": 2, "samples": samples})
                label = f"profile-{prompt}-{output}"
                await all_rank_rpc(engine, "set_diagnostic_measurement", (label,))
                sample = await batch(label)
                snapshots = await all_rank_rpc(engine, "get_diagnostic_snapshot")
                report["communication_profiles"].append({"prompt_tokens": prompt,
                    "output_tokens": output, "sample": sample, "ranks": snapshots})
                await all_rank_rpc(engine, "set_diagnostic_measurement", (None,))
                trace.emit("ep_cell_complete", tp=args.tp, dp=args.dp, ep=args.ep,
                           prompt=prompt, output=output)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        if engine is not None:
            engine.shutdown()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", action="store_true")
    args = parser.parse_args()
    if args.tp * args.dp != 2:
        raise ValueError("this feasibility probe requires exactly two GPUs")
    report = asyncio.run(run(args))
    print("MOE_EP_PROBE_JSON=" + json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
