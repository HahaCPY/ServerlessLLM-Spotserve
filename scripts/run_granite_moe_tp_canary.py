"""Live Granite MoE TP=1/2 correctness and context canary, not an ablation.

Run in the existing GPU runtime. JSON is printed after a sentinel so the host
can retain evidence separately from vLLM startup logs. No checkpoint is edited.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_routes(payload, layers: int, experts: int, top_k: int) -> dict:
    if payload is None:
        return {"verified": False, "reason": "missing_runtime_routed_experts"}
    rows = payload.tolist() if hasattr(payload, "tolist") else payload
    histogram = Counter()
    invalid = 0
    for token in rows:
        if len(token) != layers:
            invalid += 1
            continue
        for layer, expert_ids in enumerate(token):
            if len(expert_ids) != top_k or len(set(expert_ids)) != top_k:
                invalid += 1
            for expert in expert_ids:
                if not isinstance(expert, int) or not 0 <= expert < experts:
                    invalid += 1
                    continue
                histogram[f"layer:{layer}/expert:{expert}"] += 1
    layer_ids = {int(key.split("/")[0].split(":")[1]) for key in histogram}
    return {
        "verified": bool(histogram) and invalid == 0 and len(layer_ids) == layers,
        "source": "live_vllm_completion_routed_experts",
        "routed_token_rows": len(rows), "observed_layer_count": len(layer_ids),
        "assignments": sum(histogram.values()), "invalid_count": invalid,
        "histogram_sha256": digest(dict(histogram)),
        "top_expert_assignments": dict(histogram.most_common(12)),
    }


async def run(args) -> dict:
    config = json.loads(Path(args.model, "config.json").read_text())
    if config.get("architectures") != ["GraniteMoeForCausalLM"]:
        raise ValueError("refusing a checkpoint without the expected MoE architecture")
    layers, experts, top_k = (
        config["num_hidden_layers"], config["num_local_experts"],
        config["num_experts_per_tok"],
    )
    if not 0 < top_k < experts:
        raise ValueError("checkpoint must use sparse expert routing")
    if max(args.prompt_tokens) + args.max_new_tokens > args.max_model_len:
        raise ValueError("prompt plus output must fit the engine context")

    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                             trust_remote_code=False)
    started = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, load_format="safetensors", dtype="bfloat16",
        tensor_parallel_size=args.tp, enforce_eager=True, seed=0,
        trust_remote_code=False, cpu_offload_gb=0,
        max_model_len=args.max_model_len, max_num_seqs=args.concurrency,
        max_num_batched_tokens=2048, gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False, enable_return_routed_experts=True,
        moe_backend="triton", disable_log_stats=True,
    ))
    startup_s = time.monotonic() - started
    try:
        inspections = await engine.collective_rpc("get_model_inspection", timeout=60)
        rank_checks = [
            {"rank": rank, "inspection_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "granite_moe_model": text.lstrip().startswith("GraniteMoeForCausalLM"),
             "expert_modules": "GraniteMoeMoE" in text and "FusedMoE" in text}
            for rank, text in enumerate(inspections)
        ]
        if len(rank_checks) != args.tp or not all(
            row["granite_moe_model"] and row["expert_modules"] for row in rank_checks
        ):
            raise RuntimeError("actual worker model/TP inspection did not verify MoE")

        async def generate(request_id, token_ids, output_tokens):
            began = time.monotonic()
            first_token = None
            generated = []
            routes = None
            finish_reason = None
            async for output in engine.generate(
                TokensPrompt(prompt_token_ids=token_ids),
                SamplingParams(temperature=0, max_tokens=output_tokens,
                               min_tokens=output_tokens, ignore_eos=True),
                request_id,
            ):
                completion = output.outputs[0]
                current = list(completion.token_ids)
                if current and first_token is None:
                    first_token = time.monotonic()
                if len(current) >= len(generated):
                    generated = current
                if completion.routed_experts is not None:
                    routes = completion.routed_experts
                if completion.finish_reason:
                    finish_reason = completion.finish_reason
            ended = time.monotonic()
            return {
                "request_id": request_id, "prompt_tokens": len(token_ids),
                "prompt_sha256": digest(token_ids), "generated_tokens": len(generated),
                "output_sha256": digest(generated), "finish_reason": finish_reason,
                "latency_s": ended - began,
                "ttft_s": None if first_token is None else first_token - began,
                "routing_payload": routes,
            }

        warmup = tokenizer.encode("Explain sparse mixture of experts inference.")
        await generate("warmup", warmup, 16)
        cases = []
        for prompt_tokens in args.prompt_tokens:
            prompts = []
            for index in range(args.concurrency):
                seed = tokenizer.encode(
                    f"Request {index}: Explain expert routing, KV cache and GPU scheduling. "
                )
                prompts.append((seed * (prompt_tokens // len(seed) + 1))[:prompt_tokens])
            batch_started = time.monotonic()
            requests = await asyncio.gather(*[
                generate(f"context-{prompt_tokens}-{index}", tokens, args.max_new_tokens)
                for index, tokens in enumerate(prompts)
            ])
            wall_s = time.monotonic() - batch_started
            for row in requests:
                row["routing"] = audit_routes(
                    row.pop("routing_payload"), layers, experts, top_k
                )
            cases.append({
                "prompt_tokens": prompt_tokens, "concurrency": args.concurrency,
                "batch_wall_s": wall_s,
                "throughput_req_s": len(requests) / wall_s,
                "throughput_output_tokens_s": sum(
                    row["generated_tokens"] for row in requests
                ) / wall_s,
                "requests": requests,
            })
        passed = all(
            row["generated_tokens"] == args.max_new_tokens
            and row["finish_reason"] == "length" and row["routing"]["verified"]
            for case in cases for row in case["requests"]
        )
        return {
            "status": "passed" if passed else "failed",
            "scope": "live_moe_tp_context_canary_not_formal_ablation",
            "model": args.model, "config_sha256": digest(config),
            "vllm_version": vllm.__version__, "torch_version": torch.__version__,
            "tp": args.tp, "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens, "cpu_offload_gb": 0,
            "dtype": "bfloat16", "startup_s": startup_s,
            "runtime_rank_checks": rank_checks, "cases": cases,
            "runtime_moe_verified": passed, "formal_experiment_eligible": False,
            "kv_recovery_verified": None,
        }
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception:
        report = {"status": "failed", "tp": args.tp, "traceback": traceback.format_exc(),
                  "runtime_moe_verified": False, "formal_experiment_eligible": False}
    print("MOE_TP_CANARY_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
