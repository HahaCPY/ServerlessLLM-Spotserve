"""Measure warmed Granite MoE service costs; never claim a formal ablation.

Run each TP in a fresh GPU container with VLLM_USE_V2_MODEL_RUNNER=0. Structured
events delimit warmup/measurement so child-process JIT warnings remain auditable.
The host retains stdout using apply_patch; this script does not edit checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import uuid


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical_gpu_uuid(value):
    # NVML prefixes UUIDs with GPU-; torch's CUDA properties do not.
    return "GPU-" + str(uuid.UUID(str(value).lower().removeprefix("gpu-")))


def validate_frozen_batch(batch, prompt_tokens, concurrency, output_tokens):
    """Require the actual calibration tokens, not only a matching length."""
    prompts = batch.get("prompt_token_ids", [])
    if (len(prompt_tokens) != 1 or concurrency != [len(prompts)] or not prompts
            or any(not isinstance(prompt, list) or len(prompt) != prompt_tokens[0]
                   or any(type(token) is not int or token < 0 for token in prompt)
                   for prompt in prompts)
            or batch.get("remaining_tokens") != output_tokens):
        raise ValueError("frozen batch does not match the exact workload")
    if (not batch.get("source_artifact") or not batch.get("source_artifact_sha256")
            or batch.get("prompt_sha256s") != [digest(prompt) for prompt in prompts]):
        raise ValueError("frozen batch token hashes/source provenance are incomplete")
    if len(prompts) == 1 and batch.get("frozen_prefix_sha256") != digest(prompts[0]):
        raise ValueError("frozen prefix hash does not match its tokens")
    return prompts


def event(phase: str, **fields) -> None:
    print("MOE_TP_PROFILE_EVENT=" + json.dumps({
        "phase": phase, "unix_time_s": time.time(), **fields,
    }, sort_keys=True), flush=True)


def validate_workload(prompt_tokens, concurrency, output_tokens, max_model_len,
                      repeats) -> None:
    values = [*prompt_tokens, *concurrency, output_tokens, max_model_len, repeats]
    if not prompt_tokens or not concurrency or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise ValueError("workload counts must be positive integers")
    if len(set(prompt_tokens)) != len(prompt_tokens) or len(set(concurrency)) != len(
        concurrency
    ):
        raise ValueError("workload cells must not be duplicated")
    if max(prompt_tokens) + output_tokens > max_model_len:
        raise ValueError("prompt plus output exceeds max_model_len")
    if repeats < 2:
        raise ValueError("a repeated profile requires at least two measurements")


def output_counts(args):
    counts = getattr(args, "output_tokens", None) or [args.max_new_tokens]
    if (len(set(counts)) != len(counts)
            or any(type(value) is not int or value <= 0 for value in counts)):
        raise ValueError("output lengths must be distinct positive integers")
    return counts


def summarize_samples(samples, concurrency: int, output_tokens: int) -> dict:
    if not samples:
        raise ValueError("cannot summarize an empty profile")
    wall_times, latencies, ttfts = [], [], []
    for sample in samples:
        wall = sample["batch_wall_s"]
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError("invalid batch wall time")
        wall_times.append(wall)
        if len(sample["requests"]) != concurrency:
            raise ValueError("partial batches cannot become capacity profiles")
        for row in sample["requests"]:
            if row["generated_tokens"] != output_tokens or row["finish_reason"] != "length":
                raise ValueError("incomplete output cannot become a capacity profile")
            latency, ttft = row["latency_s"], row["ttft_s"]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or value <= 0 for value in (latency, ttft)):
                raise ValueError("invalid request latency or TTFT")
            if ttft > latency:
                raise ValueError("first token must precede request completion")
            latencies.append(latency)
            ttfts.append(ttft)
    count = len(latencies)
    ordered = sorted(latencies)
    return {
        "measurement_batches": len(samples), "request_count": count,
        "total_batch_wall_s": sum(wall_times),
        "batch_wall_mean_s": statistics.mean(wall_times),
        "batch_wall_population_std_s": statistics.pstdev(wall_times),
        "batch_wall_sample_sd_s": statistics.stdev(wall_times) if len(wall_times) > 1 else None,
        "latency_mean_s": statistics.mean(latencies),
        "latency_median_s": statistics.median(latencies),
        "latency_p95_nearest_rank_s": ordered[math.ceil(0.95 * count) - 1],
        "ttft_mean_s": statistics.mean(ttfts),
        "decode_after_first_mean_s": statistics.mean(
            latency - first for latency, first in zip(latencies, ttfts)),
        "per_remaining_token_mean_s": (statistics.mean(
            latency - first for latency, first in zip(latencies, ttfts)) / (output_tokens - 1)
            if output_tokens > 1 else None),
        "ttft_definition": "frontend_first_token_receipt_not_pure_GPU_prefill_time",
        "throughput_req_s": count / sum(wall_times),
        "throughput_output_tokens_s": count * output_tokens / sum(wall_times),
        "capacity_scope": "closed_batch_not_open_loop_queue_or_slo_capacity",
    }


async def run(args) -> dict:
    lengths = output_counts(args)
    for length in lengths:
        validate_workload(args.prompt_tokens, args.concurrency, length,
                          args.max_model_len, args.repeats)
    config = json.loads(Path(args.model, "config.json").read_text())
    if config.get("architectures") != ["GraniteMoeForCausalLM"] or not (
        0 < config.get("num_experts_per_tok", 0) < config.get("num_local_experts", 0)
    ):
        raise ValueError("refusing a non-sparse or non-Granite-MoE checkpoint")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise ValueError("set VLLM_USE_V2_MODEL_RUNNER=0 for the recovery-compatible runner")
    frozen_batch = None
    frozen_prompts = None
    if args.frozen_batch is not None:
        if len(lengths) != 1:
            raise ValueError("a frozen batch specifies exactly one remaining output length")
        frozen_batch = json.loads(args.frozen_batch.read_text(encoding="utf-8"))
        frozen_prompts = validate_frozen_batch(frozen_batch, args.prompt_tokens,
                                               args.concurrency, lengths[0])
    if any((args.physical_gpu_indices, args.physical_gpu_uuids, args.model_revision)):
        if (not args.model_revision or len(args.physical_gpu_indices or []) != args.tp
                or len(args.physical_gpu_uuids or []) != args.tp
                or len(set(args.physical_gpu_indices)) != args.tp
                or len(set(args.physical_gpu_uuids)) != args.tp):
            raise ValueError("complete physical group/UUID/revision provenance is required")
    if args.measurement_barrier is not None and (
        len(args.prompt_tokens) != 1 or len(args.concurrency) != 1
    ):
        raise ValueError("a shared timing barrier currently supports exactly one cell")

    from transformers import AutoTokenizer
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                             trust_remote_code=False)
    report = {
        "status": "running", "scope": "warmed_service_cost_pilot_not_formal_ablation",
        "model": args.model, "config_sha256": digest(config), "tp": args.tp,
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "dtype": "bfloat16", "cpu_offload_gb": 0,
        "routing_capture": getattr(args, "capture_routes", False),
        "model_runner": "V1", "enforce_eager": True,
        "moe_backend": "triton", "enable_prefix_caching": False,
        "max_model_len": args.max_model_len, "max_num_seqs": max(args.concurrency),
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_new_tokens": lengths[0] if len(lengths) == 1 else None,
        "output_token_counts": lengths, "repeats": args.repeats,
        "startup_excluded_from_warmed_service": True,
        "cells": [], "formal_experiment_eligible": False,
        "transition_cost_verified": False, "kv_recovery_verified": None,
    }
    if frozen_batch is not None:
        report["calibration_input"] = {
            key: value for key, value in frozen_batch.items() if key != "prompt_token_ids"}
        report["calibration_input_sha256"] = digest(frozen_batch)
    if args.model_revision:
        # CDI exposes only the requested group. Verify its actual CUDA UUIDs,
        # rather than merely copying the host's requested device indices.
        observed = [canonical_gpu_uuid(torch.cuda.get_device_properties(index).uuid)
                    for index in range(torch.cuda.device_count())]
        expected = [canonical_gpu_uuid(value) for value in args.physical_gpu_uuids]
        if sorted(observed) != sorted(expected):
            raise ValueError(f"actual visible CUDA UUIDs {observed} differ from {expected}")
        report["execution"] = {"physical_gpu_indices": args.physical_gpu_indices,
                               "model_revision": args.model_revision,
                               "actual_visible_gpu_uuids": observed,
                               "physical_group_verified_via_cuda_uuid": True}
    engine = None
    try:
        event("engine_start", tp=args.tp)
        began = time.monotonic()
        scheduling = ({"async_scheduling": False}
                      if getattr(args, "synchronous_scheduling", False) else {})
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, load_format="safetensors", dtype="bfloat16",
            tensor_parallel_size=args.tp, enforce_eager=True, seed=0,
            trust_remote_code=False, cpu_offload_gb=0,
            max_model_len=args.max_model_len, max_num_seqs=max(args.concurrency),
            max_num_batched_tokens=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            enable_return_routed_experts=report["routing_capture"],
            moe_backend="triton", disable_log_stats=True,
            **scheduling,
        ))
        report["engine_construction_s"] = time.monotonic() - began
        report["async_scheduling"] = engine.vllm_config.scheduler_config.async_scheduling
        report["scheduler_cls"] = str(engine.vllm_config.scheduler_config.scheduler_cls)
        report["observed_runtime"] = {
            "tensor_parallel_size": engine.vllm_config.parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": engine.vllm_config.parallel_config.pipeline_parallel_size,
            "data_parallel_size": engine.vllm_config.parallel_config.data_parallel_size,
            "enable_expert_parallel": engine.vllm_config.parallel_config.enable_expert_parallel,
            "dtype": str(engine.vllm_config.model_config.dtype),
            "routing_capture": engine.vllm_config.model_config.enable_return_routed_experts,
            "max_model_len": engine.vllm_config.model_config.max_model_len,
            "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
            "enable_prefix_caching": engine.vllm_config.cache_config.enable_prefix_caching,
            "cpu_offload_gb": engine.vllm_config.offload_config.uva.cpu_offload_gb,
            "enforce_eager": engine.vllm_config.model_config.enforce_eager,
            "async_scheduling": engine.vllm_config.scheduler_config.async_scheduling,
        }
        if (getattr(args, "synchronous_scheduling", False)
                and report["async_scheduling"] is not False):
            raise RuntimeError("actual scheduling differs from requested synchronous runtime")
        inspections = await engine.collective_rpc("get_model_inspection", timeout=60)
        report["runtime_rank_checks"] = [
            {"rank": rank, "inspection_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "granite_moe_model": text.lstrip().startswith("GraniteMoeForCausalLM"),
             "expert_modules": "GraniteMoeMoE" in text and "FusedMoE" in text}
            for rank, text in enumerate(inspections)
        ]
        if len(inspections) != args.tp or not all(
            row["granite_moe_model"] and row["expert_modules"]
            for row in report["runtime_rank_checks"]
        ):
            raise RuntimeError("actual runtime MoE/TP inspection failed")
        report["runtime_moe_module_verified"] = True

        async def generate(request_id, token_ids, output_tokens):
            started = time.monotonic()
            first_token = None
            generated = []
            finish_reason = None
            async for output in engine.generate(
                TokensPrompt(prompt_token_ids=token_ids),
                SamplingParams(temperature=0, max_tokens=output_tokens,
                               min_tokens=output_tokens, ignore_eos=True, seed=0),
                request_id,
            ):
                completion = output.outputs[0]
                current = list(completion.token_ids)
                if len(current) < len(generated) or current[:len(generated)] != generated:
                    raise RuntimeError("already returned token prefix was truncated or revised")
                if current and first_token is None:
                    first_token = time.monotonic()
                if len(current) >= len(generated):
                    generated = current
                if completion.finish_reason:
                    finish_reason = completion.finish_reason
            ended = time.monotonic()
            return {
                "request_id": request_id, "generated_tokens": len(generated),
                "finish_reason": finish_reason, "latency_s": ended - started,
                "ttft_s": None if first_token is None else first_token - started,
                "output_token_ids": generated,
            }

        event("short_warmup_start", tp=args.tp)
        short_started = time.monotonic()
        short = await generate("short-warmup", tokenizer.encode(
            "Explain sparse mixture of experts inference."
        ), 16)
        if short["generated_tokens"] != 16:
            raise RuntimeError("short warmup failed")
        report["short_warmup_s"] = time.monotonic() - short_started
        report["cold_engine_and_short_warmup_s"] = time.monotonic() - began
        event("short_warmup_end", tp=args.tp)

        for prompt_tokens, concurrency, length in (
                (p, c, n) for p in args.prompt_tokens for c in args.concurrency for n in lengths):
                cell = {"prompt_tokens": prompt_tokens, "concurrency": concurrency,
                        "max_new_tokens": length,
                        "status": "warming", "samples": []}
                report["cells"].append(cell)
                prompts = frozen_prompts
                if prompts is None:
                    prompts = []
                    for index in range(concurrency):
                        seed = tokenizer.encode(
                            f"Request {index}: Explain expert routing, KV cache and GPU scheduling. "
                        )
                        prompts.append((seed * (prompt_tokens // len(seed) + 1))[:prompt_tokens])
                prompt_hashes = [digest(tokens) for tokens in prompts]
                fields = {"tp": args.tp, "prompt_tokens": prompt_tokens,
                          "concurrency": concurrency, "max_new_tokens": length}
                event("cell_warmup_start", **fields)
                warm_started = time.monotonic()
                warm_length = length if getattr(args, "output_tokens", None) else 64
                cell["shape_warmup_output_tokens"] = warm_length
                warm_rows = await asyncio.gather(*[
                    generate(f"warm-{prompt_tokens}-{concurrency}-{length}-{index}", tokens, warm_length)
                    for index, tokens in enumerate(prompts)
                ])
                cell["shape_warmup_s"] = time.monotonic() - warm_started
                if any(row["generated_tokens"] != warm_length for row in warm_rows):
                    raise RuntimeError("shape warmup failed")
                event("cell_warmup_end", **fields)
                if args.measurement_barrier is not None:
                    event("measurement_barrier_wait", **fields)
                    deadline = time.monotonic() + args.barrier_timeout_s
                    while not args.measurement_barrier.is_file():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("shared measurement barrier timed out")
                        await asyncio.sleep(0.25)
                    event("measurement_barrier_released", **fields)
                cell["status"] = "measuring"
                for repeat in range(args.repeats):
                    event("measurement_start", repeat=repeat, **fields)
                    batch_started = time.monotonic()
                    requests = await asyncio.gather(*[
                        generate(f"measure-{prompt_tokens}-{concurrency}-{length}-{repeat}-{index}",
                                 tokens, length)
                        for index, tokens in enumerate(prompts)
                    ])
                    wall_s = time.monotonic() - batch_started
                    event("measurement_end", repeat=repeat, batch_wall_s=wall_s, **fields)
                    for index, row in enumerate(requests):
                        row["prompt_sha256"] = prompt_hashes[index]
                        row["output_sha256"] = digest(row["output_token_ids"])
                        if not getattr(args, "retain_output_tokens", False):
                            row.pop("output_token_ids")
                    sample = {"repeat": repeat, "batch_wall_s": wall_s,
                              "requests": requests}
                    cell["samples"].append(sample)
                    # Fail closed immediately; retain partial cells if a later batch fails.
                    summarize_samples([sample], concurrency, length)
                cell["summary"] = summarize_samples(cell["samples"], concurrency,
                                                    length)
                cell["status"] = "passed"
                event("cell_complete", summary=cell["summary"], **fields)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        if engine is not None:
            event("shutdown_start", tp=args.tp)
            started = time.monotonic()
            engine.shutdown()
            report["shutdown_s"] = time.monotonic() - started
            event("shutdown_end", tp=args.tp)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", nargs="+", type=int,
                        help="Independent output-length sweep; overrides --max-new-tokens.")
    parser.add_argument("--capture-routes", action="store_true")
    parser.add_argument("--synchronous-scheduling", action="store_true")
    parser.add_argument("--retain-output-tokens", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--frozen-batch", type=Path,
                        help="Prior calibration token batch with source provenance; not a freeze proof.")
    parser.add_argument("--physical-gpu-indices", nargs="+", type=int)
    parser.add_argument("--physical-gpu-uuids", nargs="+")
    parser.add_argument("--model-revision")
    parser.add_argument("--measurement-barrier", type=Path,
                        help="Wait after warmup for a host-created all-workers-ready file.")
    parser.add_argument("--barrier-timeout-s", type=float, default=900)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception:
        report = {"status": "failed", "tp": args.tp, "traceback": traceback.format_exc(),
                  "formal_experiment_eligible": False}
    print("MOE_TP_PROFILE_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
