"""Screen a Granite EP shape with native workers and one ordinary request."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import traceback


def digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


async def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    from scripts.probe_granite_ep import all_rank_rpc

    report = {"status": "running", "tp": args.tp, "dp": args.dp,
              "ep": True, "physical_gpu_count": args.tp * args.dp,
              "model": args.model, "request": {}}
    engine = None
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False)
        seed = tokenizer.encode(
            "Explain why a distributed sparse model must retain every expert "
            "after a GPU failure. ")
        tokens = (seed * (512 // len(seed) + 1))[:512]
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, tensor_parallel_size=args.tp,
            data_parallel_size=args.dp, data_parallel_size_local=args.dp,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            worker_cls=("scripts.moe_architecture_diagnostic_worker."
                        "ArchitectureDiagnosticWorker"),
            enforce_eager=True, dtype="bfloat16", seed=0,
            trust_remote_code=False, cpu_offload_gb=0,
            max_model_len=8704, max_num_seqs=4,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.7,
            enable_prefix_caching=False, enable_return_routed_experts=False,
            async_scheduling=False, moe_backend="triton",
            disable_log_stats=True))
        report["engine_ready_s"] = time.monotonic() - started
        placement = await all_rank_rpc(engine, "get_expert_placement_evidence")
        report["placement"] = placement
        parallel = engine.vllm_config.parallel_config
        report["observed_runtime"] = {
            "tp": parallel.tensor_parallel_size,
            "dp": parallel.data_parallel_size,
            "ep": parallel.enable_expert_parallel,
            "all2all_backend": parallel.all2all_backend,
        }
        before = time.monotonic()
        last = None
        output = []
        kwargs = {"data_parallel_rank": 0} if args.dp > 1 else {}
        async for item in engine.generate(
            TokensPrompt(prompt_token_ids=tokens),
            SamplingParams(temperature=0, max_tokens=32,
                           ignore_eos=True, seed=0),
            "ep-shape-screen", **kwargs
        ):
            current = list(item.outputs[0].token_ids)
            if len(current) < len(output) or current[:len(output)] != output:
                raise ValueError("stream revised delivered tokens")
            output, last = current, item
        if last is None or len(output) != 32 or last.outputs[0].finish_reason != "length":
            raise ValueError("ordinary request did not complete 32 tokens")
        report["request"] = {
            "prompt_sha256": digest(tokens), "output_sha256": digest(output),
            "output_token_ids": output, "complete_s": time.monotonic() - before,
            "finish_reason": last.outputs[0].finish_reason,
        }
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        if engine is not None:
            engine.shutdown()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dp", type=int, required=True)
    args = parser.parse_args()
    if args.tp * args.dp not in (2, 3, 4):
        raise ValueError("shape screen only covers physical EP2/EP3/EP4")
    report = asyncio.run(run(args))
    print("MOE_EP_SHAPE_JSON=" + json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
