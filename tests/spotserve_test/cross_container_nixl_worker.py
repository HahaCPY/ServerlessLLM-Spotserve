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
import traceback
from multiprocessing.connection import Client


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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = args.side_channel_host
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(args.side_channel_port)

    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    role = "kv_producer" if args.role == "source" else "kv_consumer"
    engine_kwargs = dict(
        model=args.model,
        tensor_parallel_size=max(int(args.tensor_parallel_size), 1),
        enforce_eager=True,
        gpu_memory_utilization=min(
            max(float(args.gpu_memory_utilization), 0.01), 0.99
        ),
        cpu_offload_gb=max(float(args.cpu_offload_gb), 0.0),
        max_model_len=max(int(args.max_model_len), 256),
        max_num_seqs=2,
        trust_remote_code=True,
        # Avoid a first-run FlashInfer CUTLASS JIT inside each ephemeral
        # container.  Triton is still a real CUDA execution backend and
        # keeps the cross-container test focused on NIXL transport.
        moe_backend="triton",
    )
    if args.kv_transfer_mode == "nixl":
        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role=role,
            kv_buffer_device="cuda",
        )
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)
    conn = None
    generation_tasks: dict[str, asyncio.Task] = {}
    pause_events: dict[str, asyncio.Event] = {}
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            conn.send(payload)

    async def generate(request_id: str, token_ids: list[int]) -> None:
        pause_event = asyncio.Event()
        pause_events[request_id] = pause_event
        try:
            params = SamplingParams(
                temperature=0,
                max_tokens=256,
                min_tokens=256,
                ignore_eos=True,
            )
            generator = engine.generate(
                TokensPrompt(prompt_token_ids=token_ids), params, request_id
            )
            first = True
            async for output in generator:
                generated = list(output.outputs[0].token_ids)
                await send(
                    {
                        "event": "output",
                        "request_id": request_id,
                        "token_ids": generated,
                        "finished": output.finished,
                    }
                )
                if first:
                    first = False
                    # This vLLM build has no public pause API.  The request
                    # remains active while the controller snapshots and
                    # exports its KV state, then the controller aborts it.
                    await send(
                        {
                            "event": "paused",
                            "request_id": request_id,
                            "simulated": True,
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
        await send(
            {
                "event": "ready",
                "role": args.role,
                "node_id": args.node_id or args.role,
                "restore_supported": bool(
                    await engine.supports_state_restore()
                ),
                "side_channel_host": args.side_channel_host,
                "side_channel_port": args.side_channel_port,
            }
        )
        while True:
            command = await asyncio.to_thread(conn.recv)
            op = command["op"]
            request_id = command.get("request_id", "container-nixl-request")
            if op == "generate":
                generation_tasks[request_id] = asyncio.create_task(
                    generate(request_id, command["token_ids"])
                )
                await send({"event": "generate_started", "request_id": request_id})
            elif op == "metadata":
                result = await engine.get_request_kv_metadata(request_id)
                if not result.get("found", False):
                    # AsyncLLM keeps an external->internal request mapping in
                    # the output processor.  A request can still be live in
                    # EngineCore while that mapping is briefly unavailable;
                    # expose the live sequence id so export can use it.
                    for candidate in await engine.get_all_request_kv_metadata():
                        if candidate.get("found", False):
                            result = candidate
                            break
                await send({"event": "metadata", "result": result})
            elif op == "export":
                result = await engine.export_inference_state(request_id)
                if not result.get("supports_restore", False):
                    live = await engine.get_all_request_kv_metadata()
                    for candidate in live:
                        if not candidate.get("found", False):
                            continue
                        sequence_id = candidate.get("sequence_id")
                        if not sequence_id:
                            continue
                        result = await engine.export_inference_state(
                            str(sequence_id)
                        )
                        if result.get("supports_restore", False):
                            # Preserve the host-visible request id while the
                            # connector payload retains its internal source
                            # sequence id for NIXL lookup.
                            result["request_id"] = request_id
                            break
                await send({"event": "export", "result": result})
            elif op == "restore":
                result = engine.restore_inference_state(
                    command["state"], request_id
                )
                await send({"event": "restore", "result": result})
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
