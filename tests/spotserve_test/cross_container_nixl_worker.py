"""Control-plane worker for the cross-container NIXL smoke.

The process is intentionally small: the host-side test drives it over a
shared Unix socket, while NIXL itself uses the container network and the
worker's side-channel TCP port.  ``source`` and ``target`` therefore have
different hostnames/network namespaces even though they share one physical
machine.
"""

import argparse
import asyncio
import os
import sys
import traceback
from multiprocessing.connection import Client

if __name__ == "__main__" and "--startup-profiling" in sys.argv:
    from scripts.moe_diagnostic_trace import emit, span
    emit("recovery_frontend_entry")


def routed_expert_metadata(engine, request_id: str) -> dict:
    """Read the real vLLM frontend routing buffer for one live request."""
    output_processor = getattr(engine, "output_processor", None)
    if output_processor is None:
        return {
            "moe_route_histogram_available": False,
            "moe_route_histogram_source": "unavailable",
            "moe_route_histogram_kind": "unavailable",
            "per_request_expert_route_histogram": {},
        }
    request_states = getattr(output_processor, "request_states", {})
    internal_ids = list(
        getattr(output_processor, "external_req_ids", {}).get(request_id, [])
    )
    if not internal_ids and request_id in request_states:
        internal_ids = [request_id]
    histogram: dict[str, int] = {}
    for internal_id in internal_ids:
        state = request_states.get(internal_id)
        for chunk in getattr(state, "routed_experts_chunks", []) or []:
            payload = chunk.tolist() if hasattr(chunk, "tolist") else chunk
            if not isinstance(payload, (list, tuple)):
                continue
            for token_routes in payload:
                if not isinstance(token_routes, (list, tuple)):
                    continue
                for layer_id, expert_ids in enumerate(token_routes):
                    if not isinstance(expert_ids, (list, tuple)):
                        continue
                    for expert_id in expert_ids:
                        try:
                            parsed = int(expert_id)
                        except (TypeError, ValueError):
                            continue
                        if parsed < 0:
                            continue
                        key = f"layer:{layer_id}/expert:{parsed}"
                        histogram[key] = histogram.get(key, 0) + 1
    available = bool(histogram)
    return {
        "moe_route_histogram_available": available,
        "moe_route_histogram_source": (
            "vllm_runtime_topk" if available else "unavailable"
        ),
        "moe_route_histogram_kind": (
            "runtime_observed_topk" if available else "unavailable"
        ),
        "per_request_expert_route_histogram": histogram,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", choices=("source", "target", "observer"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-socket", required=True)
    parser.add_argument("--side-channel-host", required=True)
    parser.add_argument("--side-channel-port", type=int, required=True)
    parser.add_argument("--token-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum and minimum generated tokens for the controlled request.",
    )
    parser.add_argument(
        "--pause-after-new-tokens",
        type=int,
        default=1,
        help=(
            "Pause an in-flight request after this many generated tokens so "
            "the controller can snapshot/export it."
        ),
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="Optional vLLM CPU weight offload for models larger than one GPU.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.08,
        help="Fraction of each GPU memory available to the vLLM executor.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="Maximum number of concurrent sequences in the engine.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional cap for one scheduler batch, useful for small checkpoints.",
    )
    parser.add_argument(
        "--kv-transfer-mode",
        choices=("nixl", "none"),
        default="nixl",
        help="Use the NIXL connector or run without a KV transfer connector.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help="Maximum sequence length for the vLLM engine.",
    )
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--disable-routed-experts-capture", action="store_true")
    parser.add_argument("--capture-routed-experts", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--native-freeze-barrier", action="store_true",
                        help="Opt-in synchronous EngineCore C=1 source token barrier.")
    parser.add_argument("--startup-profiling", action="store_true",
                        help="Diagnostic-only timestamped worker, no execution policy changes.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.native_freeze_barrier and args.max_num_seqs not in (1, 2):
        raise ValueError("native freeze currently requires C=1 or C=2")
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = args.side_channel_host
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(args.side_channel_port)

    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    if args.startup_profiling:
        emit("recovery_frontend_imports_complete")

    role = "kv_producer" if args.role == "source" else "kv_consumer"
    engine_kwargs = dict(
        model=args.model,
        tensor_parallel_size=max(int(args.tensor_parallel_size), 1),
        data_parallel_size=max(int(args.data_parallel_size), 1),
        data_parallel_size_local=max(int(args.data_parallel_size), 1),
        enable_expert_parallel=args.enable_expert_parallel,
        all2all_backend="allgather_reducescatter" if args.enable_expert_parallel else None,
        enforce_eager=True,
        gpu_memory_utilization=min(
            max(float(args.gpu_memory_utilization), 0.01), 0.99
        ),
        cpu_offload_gb=max(float(args.cpu_offload_gb), 0.0),
        max_model_len=max(int(args.max_model_len), 256),
        max_num_seqs=max(int(args.max_num_seqs), 1),
        trust_remote_code=not args.no_trust_remote_code,
        seed=0,
        enable_prefix_caching=False,
        # Upstream vLLM currently rejects routed-expert capture together
        # with every KV connector. Non-connector runs capture live routing;
        # NIXL runs consume a separately audited calibration on the host.
        enable_return_routed_experts=(
            args.kv_transfer_mode == "none"
            and (args.role == "source" or args.capture_routed_experts)
            and not args.disable_routed_experts_capture
        ),
        # Avoid a first-run FlashInfer CUTLASS JIT inside each ephemeral
        # container.  Triton is still a real CUDA execution backend and
        # keeps the cross-container test focused on NIXL transport.
        moe_backend="triton",
    )
    if args.native_freeze_barrier:
        engine_kwargs.update({
            "async_scheduling": False,
            "scheduler_cls": "sllm.spot.vllm_benchmark_scheduler.NativeFreezeScheduler",
            "worker_extension_cls": "sllm.spot.vllm_benchmark_worker.BenchmarkWorkerExtension",
        })
    if args.startup_profiling:
        engine_kwargs["worker_cls"] = "scripts.moe_architecture_diagnostic_worker.ArchitectureDiagnosticWorker"
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = max(
            int(args.max_num_batched_tokens), 1
        )
    if args.kv_transfer_mode == "nixl":
        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=role,
            kv_buffer_device="cuda",
        )
    engine_args = AsyncEngineArgs(**engine_kwargs)
    if args.startup_profiling:
        with span("recovery_engine_constructor"):
            engine = AsyncLLM.from_engine_args(engine_args)
    else:
        engine = AsyncLLM.from_engine_args(engine_args)
    conn = None
    generation_tasks: dict[str, asyncio.Task] = {}
    pause_events: dict[str, asyncio.Event] = {}
    # Keep the EngineCore sequence id returned by the first metadata snapshot.
    # The output processor normally retains the external->internal mapping,
    # but an output/lease cleanup can race with the host control request.  A
    # remembered internal id lets export address the live scheduler request
    # directly during that short window.
    request_internal_ids: dict[str, str] = {}
    last_metadata: dict[str, dict] = {}
    replay_inputs: dict[str, list[int]] = {}
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            conn.send(payload)

    async def generate(
        request_id: str,
        token_ids: list[int],
        max_new_tokens_override: int | None = None,
        pause_after_new_tokens_override: int | None = None,
        data_parallel_rank_override: int | None = None,
    ) -> None:
        pause_event = asyncio.Event()
        pause_events[request_id] = pause_event
        try:
            max_new_tokens = max(
                int(
                    args.max_new_tokens
                    if max_new_tokens_override is None
                    else max_new_tokens_override
                ),
                1,
            )
            pause_after_new_tokens = int(
                args.pause_after_new_tokens
                if pause_after_new_tokens_override is None
                else pause_after_new_tokens_override
            )
            pause_after_new_tokens = max(pause_after_new_tokens, 0)
            extra_args = None
            if args.native_freeze_barrier and pause_after_new_tokens:
                from sllm.spot.benchmark_freeze import freeze_sampling_extra_args
                extra_args = freeze_sampling_extra_args(pause_after_new_tokens, max_new_tokens)
            params = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
                min_tokens=max_new_tokens,
                ignore_eos=True,
                seed=0,
                extra_args=extra_args,
            )
            generator = engine.generate(
                TokensPrompt(prompt_token_ids=token_ids), params, request_id,
                **({"data_parallel_rank": (
                    0 if data_parallel_rank_override is None else data_parallel_rank_override)}
                   if args.data_parallel_size > 1 else {})
            )
            paused = False
            max_observed_generated_tokens = 0
            max_observed_token_ids: list[int] = []
            async for output in generator:
                generated = list(output.outputs[0].token_ids)
                # SamplingParams defaults to cumulative output, but a final
                # connector/control marker may contain an empty token list.
                # Preserve the largest observed count so batch throughput is
                # not accidentally reported as zero on a successful restore.
                max_observed_generated_tokens = max(
                    max_observed_generated_tokens, len(generated)
                )
                if len(generated) >= len(max_observed_token_ids):
                    max_observed_token_ids = generated
                await send(
                    {
                        "event": "output",
                        "request_id": request_id,
                        "token_ids": generated,
                        "cumulative_token_ids": max_observed_token_ids,
                        "generated_tokens": max_observed_generated_tokens,
                        "finish_reason": output.outputs[0].finish_reason,
                        "stop_reason": output.outputs[0].stop_reason,
                        "finished": output.finished,
                    }
                )
                if (
                    not paused
                    and pause_after_new_tokens > 0
                    and len(generated) >= pause_after_new_tokens
                ):
                    paused = True
                    # Frontend barrier only: native mode also freezes the
                    # scheduler exactly, and the controller's public pause
                    # retains the request/KV before taking its snapshot.
                    await send(
                        {
                            "event": "paused",
                            "request_id": request_id,
                            "token_ids": generated,
                            "generated_tokens": len(generated),
                            "finished": bool(output.finished),
                            "simulated": not args.native_freeze_barrier,
                            "native_freeze_barrier": args.native_freeze_barrier,
                        }
                    )
                    # Keep the request genuinely active until the host has
                    # exported/aborted it (source) or explicitly resumes it
                    # (target).  Without this barrier a short prompt can
                    # finish before the control plane reaches export().
                    await pause_event.wait()
                if args.token_delay_s > 0:
                    await asyncio.sleep(args.token_delay_s)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await send(
                {
                    "event": "generation_error",
                    "request_id": request_id,
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            pause_events.pop(request_id, None)

    try:
        conn = await asyncio.to_thread(
            Client, args.control_socket, family="AF_UNIX", authkey=b"spotserve"
        )
        if args.startup_profiling:
            emit("recovery_control_connected")
        await send(
            {
                "event": "ready",
                "role": args.role,
                "node_id": args.node_id or args.role,
                # Do not call supports_state_restore() on the ready path.
                # On long-context Qwen jobs this EngineCore utility can wait
                # behind connector initialization for several minutes.  The
                # actual export/restore calls below are the authoritative
                # capability check; advertising the connector mode here is
                # only a non-blocking hint for the controller.
                "restore_supported": args.kv_transfer_mode == "nixl",
                "side_channel_host": args.side_channel_host,
                "side_channel_port": args.side_channel_port,
            }
        )
        while True:
            command = await asyncio.to_thread(conn.recv)
            op = command["op"]
            request_id = command.get("request_id", "container-nixl-request")
            if op == "generate":
                token_ids = (replay_inputs.pop(request_id)
                             if command.get("use_installed_replay") else command["token_ids"])
                generation_tasks[request_id] = asyncio.create_task(
                    generate(
                        request_id,
                        token_ids,
                        command.get("max_new_tokens"),
                        command.get("pause_after_new_tokens"),
                        command.get("data_parallel_rank"),
                    )
                )
                await send({"event": "generate_started", "request_id": request_id})
            elif op == "install_replay_tokens":
                import hashlib
                import json
                tokens = command["token_ids"]
                if (not tokens or len(tokens) >= engine.vllm_config.model_config.max_model_len
                        or any(type(token) is not int or token < 0 for token in tokens)
                        or request_id in replay_inputs or request_id in generation_tasks):
                    raise ValueError("invalid or duplicate host token handoff")
                replay_inputs[request_id] = list(tokens)
                await send({"event": "replay_tokens_installed", "request_id": request_id,
                            "token_count": len(tokens),
                            "frozen_prefix_sha256": hashlib.sha256(
                                json.dumps(tokens, sort_keys=True).encode()).hexdigest(),
                            "install_mode": "host_token_handoff_excluding_gpu_prefill",
                            "gpu_kv_restored": False})
            elif op == "inspect":
                import json
                import torch
                import vllm

                async def every_dp_rank(method):
                    if args.data_parallel_size == 1:
                        return await engine.collective_rpc(method, timeout=60)
                    # DPLB's public collective_rpc returns only core0's
                    # result. Inspect both physical EP ranks explicitly.
                    groups = await asyncio.gather(*[
                        engine.engine_core._call_utility_async(
                            "collective_rpc", method, 60, (), {}, engine=identity)
                        for identity in engine.engine_core.core_engines
                    ])
                    return [item for group in groups for item in group]

                inspections = await every_dp_rank("get_model_inspection")
                rank_identities = (await every_dp_rank("get_benchmark_rank_identity")
                                   if args.native_freeze_barrier else None)
                if args.data_parallel_size > 1:
                    for dp_rank, identity in enumerate(rank_identities):
                        identity["worker_local_rank"] = identity["rank"]
                        identity["dp_rank"] = dp_rank
                        identity["rank"] = dp_rank * args.tensor_parallel_size + identity["rank"]
                    print("EP_DP_INSPECT_IDENTITIES=" + json.dumps(rank_identities), flush=True)
                diagnostics = (await engine.collective_rpc("get_diagnostic_snapshot", timeout=60)
                               if args.startup_profiling else None)
                await send({
                    "event": "inspection", "models": inspections,
                    "rank_identities": rank_identities,
                    **({"startup_diagnostics": diagnostics} if args.startup_profiling else {}),
                    "runtime": {
                        "vllm_version": vllm.__version__,
                        "torch_version": torch.__version__,
                        "tensor_parallel_size": (
                            engine.vllm_config.parallel_config.tensor_parallel_size
                        ),
                        "dtype": str(engine.vllm_config.model_config.dtype),
                        "routing_capture": engine.vllm_config.model_config.enable_return_routed_experts,
                        "pipeline_parallel_size": engine.vllm_config.parallel_config.pipeline_parallel_size,
                        "data_parallel_size": engine.vllm_config.parallel_config.data_parallel_size,
                        "enable_expert_parallel": engine.vllm_config.parallel_config.enable_expert_parallel,
                        "max_model_len": engine.vllm_config.model_config.max_model_len,
                        "cpu_offload_gb": engine.vllm_config.offload_config.uva.cpu_offload_gb,
                        "enforce_eager": engine.vllm_config.model_config.enforce_eager,
                        "enable_prefix_caching": engine.vllm_config.cache_config.enable_prefix_caching,
                        "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
                        "gpu_memory_utilization": engine.vllm_config.cache_config.gpu_memory_utilization,
                        "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
                        "async_scheduling": engine.vllm_config.scheduler_config.async_scheduling,
                        "scheduler_cls": str(engine.vllm_config.scheduler_config.scheduler_cls),
                        "actual_visible_gpu_uuids": [
                            str(torch.cuda.get_device_properties(index).uuid)
                            for index in range(torch.cuda.device_count())
                        ],
                    },
                })
            elif op == "metadata":
                # The engine has been frozen before this command.  Query
                # EngineCore directly so the snapshot cannot queue behind the
                # paused frontend output loop.
                output_processor = getattr(engine, "output_processor", None)
                mapped_ids = list(
                    getattr(output_processor, "external_req_ids", {}).get(
                        request_id, []
                    )
                )
                candidate_ids = mapped_ids or [request_id]
                result = {"found": False, "reason": "request_not_active"}
                for candidate_id in candidate_ids:
                    try:
                        candidate = await asyncio.wait_for(
                            engine.engine_core.call_utility_async(
                                "get_request_kv_metadata", candidate_id
                            ),
                            timeout=30.0,
                        )
                    except asyncio.TimeoutError:
                        candidate = {
                            "found": False,
                            "reason": "metadata_timeout",
                        }
                    if candidate.get("found", False):
                        result = candidate
                        result["request_id"] = request_id
                        result["sequence_id"] = candidate_id
                        break
                if result.get("found", False) and result.get("sequence_id"):
                    request_internal_ids[request_id] = str(result["sequence_id"])
                last_metadata[request_id] = result
                result.update(routed_expert_metadata(engine, request_id))
                await send({
                    "event": "metadata",
                    "request_id": request_id,
                    "result": result,
                })
            elif op == "route_metadata":
                await send({
                    "event": "route_metadata",
                    "request_id": request_id,
                    "result": routed_expert_metadata(engine, request_id),
                })
            elif op == "pause_generation":
                # Freeze EngineCore itself, not only this worker's async
                # generator consumer.  ``keep`` preserves live requests and
                # ``clear_cache=False`` preserves the KV blocks that the
                # controller is about to export.
                await engine.pause_generation(mode="keep", clear_cache=False)
                await send({"event": "generation_paused"})
            elif op == "resume_generation":
                await engine.resume_generation()
                await send({"event": "generation_resumed"})
            elif op == "export":
                # Try the external id first, then the remembered EngineCore
                # sequence id.  Retry briefly because the first output is
                # delivered through a separate output-processor task.
                candidates = [request_id]
                internal_id = request_internal_ids.get(request_id)
                if internal_id and internal_id not in candidates:
                    candidates.append(internal_id)
                result = {"supports_restore": False, "reason": "request_not_active"}
                for _ in range(8):
                    for candidate_id in candidates:
                        # Go straight to EngineCore first.  The async
                        # frontend can wait indefinitely for its external ->
                        # internal request mapping while a request is held at
                        # the preemption barrier; the scheduler already owns
                        # the same request id and can export it directly.
                        try:
                            result = await asyncio.wait_for(
                                engine.engine_core.call_utility_async(
                                    "export_inference_state", candidate_id
                                ),
                                timeout=30.0,
                            )
                        except asyncio.TimeoutError:
                            result = {
                                "supports_restore": False,
                                "reason": "engine_core_export_timeout",
                            }
                        if result.get("supports_restore", False):
                            # Preserve the host-visible request id while the
                            # connector payload retains its internal source
                            # sequence id for NIXL lookup.
                            result["request_id"] = request_id
                            metadata = dict(result.get("metadata", {}) or {})
                            metadata.update(
                                routed_expert_metadata(engine, request_id)
                            )
                            result["metadata"] = metadata
                            break
                    if result.get("supports_restore", False):
                        break
                    try:
                        live = await asyncio.wait_for(
                            engine.get_all_request_kv_metadata(), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        live = []
                    for candidate in live:
                        if not candidate.get("found", False):
                            continue
                        sequence_id = candidate.get("sequence_id")
                        if not sequence_id:
                            continue
                        if str(sequence_id) not in candidates:
                            candidates.append(str(sequence_id))
                    await asyncio.sleep(0.05)
                if not result.get("supports_restore", False):
                    # Include a compact lifecycle snapshot in the control
                    # response.  This makes a failed experimental cell
                    # diagnosable instead of reducing every race to the same
                    # request_not_active message.
                    try:
                        core_live = await asyncio.wait_for(
                            engine.engine_core.call_utility_async(
                                "get_all_request_kv_metadata"
                            ),
                            timeout=5.0,
                        )
                    except Exception as exc:
                        core_live = [{"debug_error": repr(exc)}]
                    result["debug"] = {
                        "metadata": last_metadata.get(request_id),
                        "candidates": candidates,
                        "core_live": core_live,
                    }
                await send(
                    {
                        "event": "export",
                        "request_id": request_id,
                        "result": result,
                    }
                )
            elif op == "restore":
                result = engine.restore_inference_state(
                    command["state"], request_id
                )
                await send(
                    {
                        "event": "restore",
                        "request_id": request_id,
                        "result": result,
                    }
                )
            elif op == "abort":
                pause_events.get(request_id, asyncio.Event()).set()
                await engine.abort(request_id)
                await send({"event": "aborted", "request_id": request_id})
            elif op == "resume":
                pause_events.get(request_id, asyncio.Event()).set()
                await send({"event": "resumed", "request_id": request_id})
            elif op == "shutdown":
                break
            else:
                raise ValueError(f"unknown operation: {op}")
    except BaseException:
        if conn is not None:
            try:
                await send({"event": "fatal", "traceback": traceback.format_exc()})
            except Exception:
                pass
        raise
    finally:
        for task in generation_tasks.values():
            task.cancel()
        await asyncio.gather(*generation_tasks.values(), return_exceptions=True)
        engine.shutdown()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    asyncio.run(main())
