#!/usr/bin/env python3
"""Run one recompute and one experimental NIXL recovery on two Ray GPUs.

The default random-weight tiny MoE is a mechanism canary, not a performance
model. No HTTP control plane, custom Ray resources, or Kubernetes permissions
are required. Ray carries control messages; KV is transferred by NIXL only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_tiny_model(path: Path) -> dict[str, Any]:
    marker = path / "cscc_canary_manifest.json"
    if path.exists():
        if not marker.is_file():
            raise RuntimeError(f"refusing to overwrite an existing model directory: {path}")
        manifest = json.loads(marker.read_text())
        for name, digest in manifest["sha256"].items():
            if file_hash(path / name) != digest:
                raise RuntimeError(f"existing tiny model checksum mismatch: {name}")
        return manifest

    import torch
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size=4096, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
        shared_expert_intermediate_size=512, decoder_sparse_step=1,
        max_position_embeddings=8192, use_sliding_window=False,
        bos_token_id=None, eos_token_id=None, pad_token_id=None,
        tie_word_embeddings=False,
    )
    model = Qwen2MoeForCausalLM(config).to(dtype=torch.float16)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    manifest = {
        "model_kind": "random_weight_qwen2_moe_mechanism_canary",
        "seed": 0, "parameters": sum(p.numel() for p in model.parameters()),
        "sha256": {item.name: file_hash(item) for item in sorted(path.iterdir())
                   if item.is_file()},
    }
    write_json(marker, manifest)
    return manifest


class EngineProcess:
    """Ray actor owning a GPU reservation and a local engine subprocess."""

    def start(self, config: dict[str, Any], authkey: str) -> dict[str, Any]:
        import ray

        config = {**config, "node_ip": ray.util.get_node_ip_address()}
        self._pending = []
        model = Path(config["model"])
        if not (model / "config.json").is_file():
            raise RuntimeError(f"model is not visible on worker: {model}")
        marker = model / "cscc_canary_manifest.json"
        if marker.is_file():
            manifest = json.loads(marker.read_text())
            for name, digest in manifest["sha256"].items():
                if file_hash(model / name) != digest:
                    raise RuntimeError(f"worker checkpoint checksum mismatch: {name}")
        self._root = Path(tempfile.mkdtemp(prefix="cscc-f1-canary-"))
        self._config = config
        self._authkey = authkey
        self._listener = Listener(("127.0.0.1", 0), family="AF_INET",
                                  authkey=authkey.encode())
        self._listener._listener._socket.settimeout(config["timeout_s"])
        self._audit = self._root / "scheduler.jsonl"
        self._resume = self._root / "resume"
        self._log = self._root / "engine.log"
        env = dict(os.environ)
        env.update({
            "CSCC_F1_CONTROL_HOST": self._listener.address[0],
            "CSCC_F1_CONTROL_PORT": str(self._listener.address[1]),
            "CSCC_F1_CONTROL_KEY": authkey,
            "CSCC_F1_AUDIT_PATH": str(self._audit),
            "CSCC_F1_RESUME_PATH": str(self._resume),
            "CSCC_F1_CONFIG": json.dumps(config),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
        })
        # Use the checked-out scheduler on every spawned EngineCore process.
        script = Path(config["script"])
        env["PYTHONPATH"] = os.pathsep.join(
            [str(script.parent), str(script.parents[2]), env.get("PYTHONPATH", "")]
        )
        with self._log.open("wb") as log:
            self._process = subprocess.Popen(
                [sys.executable, "-u", str(script), "--worker"], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self._conn = self._listener.accept()
        hello = self.wait("hello")
        context = ray.get_runtime_context()
        self._identity = {
            "role": config["role"], "node_id": str(context.get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                text=True,
            ).strip(),
            "model_config_sha256": file_hash(model / "config.json"),
            "hello": hello,
        }
        return self._identity

    def wait(self, event: str) -> dict[str, Any]:
        for index, message in enumerate(self._pending):
            if message["event"] == event:
                return self._pending.pop(index)
        deadline = time.monotonic() + self._config["timeout_s"]
        while time.monotonic() < deadline:
            if self._conn.poll(0.2):
                try:
                    message = self._conn.recv()
                except EOFError as exc:
                    raise RuntimeError(f"engine closed control channel:\n{self.log_tail()}") from exc
                if message["event"] == "fatal":
                    raise RuntimeError(message["traceback"] + "\n" + self.log_tail())
                if message["event"] == event:
                    return message
                self._pending.append(message)
            if self._process.poll() is not None:
                raise RuntimeError(f"engine exited {self._process.returncode}:\n{self.log_tail()}")
        raise TimeoutError(f"waiting for {event}:\n{self.log_tail()}")

    def command(self, payload: dict[str, Any], event: str) -> dict[str, Any]:
        self._conn.send(payload)
        return self.wait(event)

    def audit(self) -> list[dict[str, Any]]:
        if not self._audit.exists():
            return []
        # Ignore only an unfinished final line from the live writer.
        return [json.loads(line) for line in self._audit.read_text().splitlines(keepends=True)
                if line.endswith("\n")]

    def log_tail(self) -> str:
        if not hasattr(self, "_log") or not self._log.exists():
            return ""
        with self._log.open("rb") as stream:
            stream.seek(max(self._log.stat().st_size - 24000, 0))
            return stream.read().decode("utf-8", errors="replace")

    def stop(self) -> dict[str, Any]:
        process = getattr(self, "_process", None)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for name in ("_conn", "_listener"):
            connection = getattr(self, name, None)
            if connection is not None:
                connection.close()
        return {"stopped": process is None or process.poll() is not None,
                "exit_code": None if process is None else process.returncode}


async def engine_worker() -> None:
    config = json.loads(os.environ["CSCC_F1_CONFIG"])
    conn = Client((os.environ["CSCC_F1_CONTROL_HOST"],
                   int(os.environ["CSCC_F1_CONTROL_PORT"])),
                  family="AF_INET", authkey=os.environ["CSCC_F1_CONTROL_KEY"].encode())
    conn.send({"event": "hello", "pid": os.getpid()})
    engine = None
    tasks: list[asyncio.Task] = []
    lock = asyncio.Lock()

    async def send(message):
        async with lock:
            conn.send(message)

    try:
        import importlib.metadata
        if importlib.metadata.version("vllm") != "0.11.2":
            raise RuntimeError("this canary scheduler supports vLLM 0.11.2 only")
        from vllm import SamplingParams
        from vllm.config import KVTransferConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from f1_canary_scheduler import FREEZE_KEY

        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = config["node_ip"]
        # Each Ray worker pod has its own network namespace, TP is one.
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5600"
        kwargs = dict(
            model=config["model"], dtype=config["dtype"], skip_tokenizer_init=True,
            trust_remote_code=False,
            tensor_parallel_size=1, enforce_eager=True, seed=0,
            max_model_len=config["max_model_len"], max_num_seqs=1,
            max_num_batched_tokens=config["max_model_len"],
            gpu_memory_utilization=config["gpu_memory_utilization"], block_size=16,
            cpu_offload_gb=config["cpu_offload_gb"],
            num_gpu_blocks_override=(config["max_model_len"] + 15) // 16 + 8,
            enable_prefix_caching=False, async_scheduling=False,
            enable_chunked_prefill=False,
            scheduler_cls="f1_canary_scheduler.CanaryScheduler",
        )
        if config["mode"] == "kv_restore":
            kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector="NixlConnector",
                kv_role="kv_producer" if config["role"] == "source" else "kv_consumer",
                kv_buffer_device="cuda",
            )
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        for name in ("get_request_kv_metadata", "export_inference_state",
                     "restore_inference_state"):
            if not callable(getattr(engine, name, None)):
                raise RuntimeError(f"image is missing required KV patch API: {name}")
        await send({"event": "ready", "vllm": "0.11.2", "kwargs": {
            key: value for key, value in kwargs.items() if key != "kv_transfer_config"},
                    "connector": "NixlConnector" if config["mode"] == "kv_restore" else None})

        async def generate(command):
            try:
                freeze_at = command.get("freeze_at", 0)
                if freeze_at:
                    Path(os.environ["CSCC_F1_RESUME_PATH"]).unlink(missing_ok=True)
                params = SamplingParams(
                    temperature=0, seed=0, ignore_eos=True, detokenize=False,
                    max_tokens=command["max_tokens"], min_tokens=command["max_tokens"],
                    extra_args={FREEZE_KEY: freeze_at} if freeze_at else None,
                )
                generator = engine.generate(
                    {"prompt_token_ids": command["tokens"]}, params, command["request_id"]
                )
                longest: list[int] = []
                async for output in generator:
                    tokens = list(output.outputs[0].token_ids)
                    if len(tokens) > len(longest):
                        longest = tokens
                    if len(tokens) == 1:
                        await send({"event": "first", "tokens": tokens})
                    if freeze_at and len(tokens) == freeze_at:
                        await send({"event": "boundary", "tokens": tokens})
                    if output.finished:
                        await send({"event": "done", "tokens": longest,
                                    "finish_reason": output.outputs[0].finish_reason})
            except BaseException:
                if not isinstance(sys.exc_info()[1], asyncio.CancelledError):
                    await send({"event": "fatal", "traceback": traceback.format_exc()})

        while True:
            command = await asyncio.to_thread(conn.recv)
            op = command["op"]
            if op == "generate":
                tasks.append(asyncio.create_task(generate(command)))
                await send({"event": "started"})
            elif op == "metadata":
                result = await engine.get_request_kv_metadata(command["request_id"])
                await send({"event": "metadata", "result": result})
            elif op == "export":
                result = await engine.export_inference_state(command["request_id"])
                await send({"event": "export", "result": result})
            elif op == "restore":
                result = engine.restore_inference_state(command["state"], command["request_id"])
                await send({"event": "restore", "result": result})
            elif op == "abort":
                await engine.abort(command["request_id"])
                await send({"event": "aborted"})
            elif op == "resume":
                Path(os.environ["CSCC_F1_RESUME_PATH"]).touch()
                # Wake EngineCore's input queue after the local barrier release.
                await engine.engine_core.call_utility_async("get_all_request_kv_metadata")
                await send({"event": "resumed"})
            else:
                raise ValueError(f"unknown canary operation: {op}")
    except BaseException:
        await send({"event": "fatal", "traceback": traceback.format_exc()})
        raise
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if engine is not None:
            engine.shutdown()
        conn.close()


def restore_evidence(audit, prefix_tokens, request_id=None):
    scheduled = [row for row in audit if row["event"] == "scheduled"
                 and (request_id is None or row.get("request_id") == request_id)]
    completions = [row for row in audit if row["event"] == "connector_completion"]
    if not scheduled:
        raise RuntimeError("target has no scheduler computation evidence")
    first_compute = scheduled[0]["scheduled_tokens"]
    received_ids = sorted({req for row in completions for req in row["finished_recving"]})
    if request_id is not None:
        received_ids = [req for req in received_ids if req == request_id]
    invalid = sorted({block for row in completions for block in row["invalid_block_ids"]})
    return {
        "first_target_scheduled_tokens": first_compute,
        "prefix_recomputed_tokens": first_compute,
        "prefix_reused_tokens": prefix_tokens - first_compute,
        "finished_recving_request_ids": received_ids,
        "invalid_block_ids": invalid,
        "restore_success": bool(received_ids) and not invalid and 0 < first_compute < prefix_tokens,
        "acknowledgment_kind": "scheduler_finished_recving_then_first_target_output",
    }


def run_treatment(ray, remote_actor, args, mode, root):
    stage = "reserve_two_gpu_workers"
    workers = [remote_actor.options(num_cpus=1, num_gpus=1).remote() for _ in range(2)]
    source, target = workers
    started = time.monotonic()
    timeout = args.timeout_s
    artifacts = root / mode
    artifacts.mkdir()
    result = {"mode": mode, "status": "failed"}
    try:
        # Resolve each actor's address inside start; driver has no GPU visibility.
        shared = {"model": str(args.model.resolve()), "script": str(Path(__file__).resolve()),
                  "mode": mode, "max_model_len": args.prompt_tokens + args.output_tokens + 16,
                  "timeout_s": timeout, "dtype": args.dtype,
                  "gpu_memory_utilization": args.gpu_memory_utilization,
                  "cpu_offload_gb": args.cpu_offload_gb}
        configs = [{**shared, "role": role} for role in ("source", "target")]
        identities = ray.get([worker.start.remote(config, secrets.token_hex(24))
                              for worker, config in zip(workers, configs)], timeout=timeout + 15)
        result["workers"] = identities
        if identities[0]["node_id"] == identities[1]["node_id"]:
            raise RuntimeError("canary requires two distinct Ray worker nodes")
        if identities[0]["gpu"] == identities[1]["gpu"]:
            raise RuntimeError("workers were assigned the same GPU")
        if identities[0]["model_config_sha256"] != identities[1]["model_config_sha256"]:
            raise RuntimeError("model configuration differs across workers")

        def command(worker, payload, event):
            return ray.get(worker.command.remote(payload, event), timeout=timeout + 15)

        def wait(worker, event):
            return ray.get(worker.wait.remote(event), timeout=timeout + 15)

        stage = "engine_initialization"
        ready = ray.get([worker.wait.remote("ready") for worker in workers], timeout=timeout + 15)
        result["runtime"] = ready
        result["engine_startup_s"] = time.monotonic() - started
        prompt = [100 + index % 1000 for index in range(args.prompt_tokens)]
        result["prompt_sha256"] = token_hash(prompt)
        result["target_warmup_s"] = 0.0
        if args.warmup_target:
            stage = "target_inference_warmup"
            print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
            warmup_started = time.monotonic()
            command(target, {"op": "generate", "request_id": "target-warmup", "tokens": prompt,
                             "max_tokens": 8}, "started")
            warmup = wait(target, "done")
            if len(warmup["tokens"]) != 8 or warmup["finish_reason"] != "length":
                raise RuntimeError("target inference warm-up did not complete")
            result["target_warmup_s"] = time.monotonic() - warmup_started
        stage = "uninterrupted_reference"
        print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
        command(source, {"op": "generate", "request_id": "reference", "tokens": prompt,
                         "max_tokens": args.output_tokens}, "started")
        reference = wait(source, "done")["tokens"]
        if len(reference) != args.output_tokens:
            raise RuntimeError("reference output length mismatch")
        stage = "source_preemption_boundary"
        print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
        request_id = "f1-canary"
        command(source, {"op": "generate", "request_id": request_id, "tokens": prompt,
                         "max_tokens": args.output_tokens, "freeze_at": args.preempt_tokens},
                "started")
        boundary = wait(source, "boundary")["tokens"]
        notice = time.monotonic()
        metadata_started = time.monotonic()
        metadata = command(source, {"op": "metadata", "request_id": request_id}, "metadata")["result"]
        result["metadata_snapshot_s"] = time.monotonic() - metadata_started
        write_json(artifacts / "source_metadata.json", metadata)
        source_tokens = metadata.get("tokens", [])
        if (not metadata.get("found") or not metadata.get("block_ids")
                or boundary != reference[:args.preempt_tokens]
                or len(boundary) != args.preempt_tokens
                or source_tokens != prompt + boundary):
            raise RuntimeError("source token/KV boundary validation failed")
        result["source_prefix_tokens"] = len(source_tokens)
        result["preempt_generated_tokens"] = len(boundary)
        result["source_computed_tokens"] = metadata["completed_tokens"]
        result.update(state_export_s=0.0, source_abort_s=0.0, restore_stage_s=0.0)
        state = None
        if mode == "kv_restore":
            stage = "export_kv_lease"
            print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
            export_started = time.monotonic()
            state = command(source, {"op": "export", "request_id": request_id}, "export")["result"]
            result["state_export_s"] = time.monotonic() - export_started
            write_json(artifacts / "export_state.json", state)
            if not state.get("supports_restore"):
                raise RuntimeError(f"KV export rejected: {state}")
            exported_tokens = state.get("metadata", {}).get("tokens", [])
            if exported_tokens != source_tokens:
                raise RuntimeError("export changed the frozen source boundary")
            result["advertised_can_restore_cross_node"] = state["metadata"].get("can_restore_cross_node")
            result["experimental_direct_api"] = True
            abort_started = time.monotonic()
            command(source, {"op": "abort", "request_id": request_id}, "aborted")
            result["source_abort_s"] = time.monotonic() - abort_started
            stage = "stage_target_restore"
            restore_started = time.monotonic()
            staged = command(target, {"op": "restore", "request_id": request_id, "state": state},
                             "restore")["result"]
            result["restore_stage_s"] = time.monotonic() - restore_started
            result["restore_stage"] = staged
            if not staged.get("staged") or not staged.get("expected_blocks", 0):
                raise RuntimeError(f"target restore not staged: {staged}")
        else:
            stage = "stop_source_for_recompute"
            stop_started = time.monotonic()
            result["source_stop"] = ray.get(source.stop.remote(), timeout=25)
            result["source_stop_s"] = time.monotonic() - stop_started

        stage = "first_target_token_and_attach_evidence"
        print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
        remaining = args.output_tokens - args.preempt_tokens
        target_started = time.monotonic()
        command(target, {"op": "generate", "request_id": request_id, "tokens": source_tokens,
                         "max_tokens": remaining, "freeze_at": 1}, "started")
        first = wait(target, "boundary")["tokens"]
        result["target_first_output_s"] = time.monotonic() - target_started
        result["recovery_to_first_token_s"] = time.monotonic() - notice
        audit = ray.get(target.audit.remote(), timeout=30)
        evidence = restore_evidence(audit, len(source_tokens), request_id=request_id)
        result.update(evidence)
        if mode == "kv_restore" and not evidence["restore_success"]:
            raise RuntimeError(f"restore did not avoid prefix recomputation: {evidence}")
        if mode == "kv_restore":
            result["kv_blocks_staged"] = result["restore_stage"]["expected_blocks"]
        if mode == "recompute" and evidence["first_target_scheduled_tokens"] != len(source_tokens):
            raise RuntimeError("recompute baseline did not recompute the complete prefix")
        if first != reference[args.preempt_tokens:args.preempt_tokens + 1]:
            raise RuntimeError("first target token differs from uninterrupted reference")

        if mode == "kv_restore":
            stage = "terminate_source_after_attach"
            if time.monotonic() - notice >= args.grace_s:
                raise RuntimeError("attach missed the configured logical grace deadline")
            stop_started = time.monotonic()
            result["source_stop"] = ray.get(source.stop.remote(), timeout=25)
            result["source_stop_s"] = time.monotonic() - stop_started
            result["source_stopped_after_attach_s"] = time.monotonic() - notice
            if not result["source_stop"]["stopped"]:
                raise RuntimeError("source process did not stop")
            if result["source_stopped_after_attach_s"] > args.grace_s:
                raise RuntimeError("source termination exceeded the logical grace deadline")

        stage = "continue_after_source_exit"
        print(f"F1_CANARY_STAGE={mode}/{stage}", flush=True)
        resume_started = time.monotonic()
        command(target, {"op": "resume"}, "resumed")
        done = wait(target, "done")
        result["decode_after_resume_s"] = time.monotonic() - resume_started
        suffix = done["tokens"]
        result["recovery_s"] = time.monotonic() - notice
        result["target_generated_tokens"] = len(suffix)
        result["continued_after_source_stop"] = len(suffix) > len(first)
        result["output_sha256"] = token_hash(boundary + suffix)
        result["reference_sha256"] = token_hash(reference)
        result["full_output_matches_reference"] = boundary + suffix == reference
        write_json(artifacts / "tokens.json", {"reference": reference, "boundary": boundary,
                                                "target_suffix": suffix})
        if len(suffix) != remaining or done["finish_reason"] != "length":
            raise RuntimeError("target output incomplete")
        if not result["full_output_matches_reference"]:
            raise RuntimeError("full output differs; inspect tokens before performance conclusions")
        result["status"] = "passed"
        result["stage"] = "completed"
    except BaseException:
        result["stage"] = stage
        result["error"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.monotonic() - started
        for role, worker in zip(("source", "target"), workers):
            try:
                audit = ray.get(worker.audit.remote(), timeout=15)
                write_json(artifacts / f"{role}_scheduler.json", audit)
                log = ray.get(worker.log_tail.remote(), timeout=15)
                (artifacts / f"{role}_engine_tail.log").write_text(log, encoding="utf-8")
            except Exception:
                result[f"{role}_artifact_error"] = traceback.format_exc()
            try:
                ray.get(worker.stop.remote(), timeout=25)
            except Exception:
                result[f"{role}_cleanup_error"] = traceback.format_exc()
            ray.kill(worker, no_restart=True)
        write_json(artifacts / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepare-tiny-model", action="store_true")
    parser.add_argument("--allow-experimental-cross-worker-kv", action="store_true", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=384)
    parser.add_argument("--preempt-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--grace-s", type=float, default=120)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--warmup-target", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.preempt_tokens < args.output_tokens - 1 or args.prompt_tokens <= 0:
        parser.error("require a positive prompt and at least two post-preemption output tokens")
    if args.timeout_s <= 0 or args.grace_s <= 0:
        parser.error("timeout and grace must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.cpu_offload_gb < 0:
        parser.error("require 0 < GPU memory utilization < 1 and non-negative CPU offload")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 2, "status": "failed", "scope": "F1 recovery comparison",
        "logical_preemption": True, "grace_s": args.grace_s,
        "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
        "preempt_tokens": args.preempt_tokens, "runs_per_treatment": 1,
        "model": str(args.model.resolve()), "treatments": [],
        "dtype": args.dtype, "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb, "target_warmup": args.warmup_target,
        "not_verified": ["physical-host identity", "platform pod eviction",
                         "production cross-host support", "F2 live target selection",
                         "full four-strategy F1", "statistical significance",
                         "measured network byte counters"],
    }
    try:
        if args.prepare_tiny_model:
            result["model_manifest"] = prepare_tiny_model(args.model.resolve())
        elif not (args.model / "config.json").is_file():
            raise RuntimeError(f"model config missing: {args.model}")
        model_config = json.loads((args.model / "config.json").read_text())
        result["model_architectures"] = model_config.get("architectures")
        max_length = args.prompt_tokens + args.output_tokens + 16
        if max_length > model_config.get("max_position_embeddings", max_length):
            raise RuntimeError("workload exceeds checkpoint max_position_embeddings")
        index_path = args.model / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shards = sorted(set(index["weight_map"].values()))
            missing = [name for name in shards if not (args.model / name).is_file()]
            if missing:
                raise RuntimeError(f"checkpoint shards missing: {missing}")
            result["checkpoint_total_weight_bytes"] = index.get("metadata", {}).get("total_size")
            result["checkpoint_shards"] = shards
        elif not (args.model / "model.safetensors").is_file():
            raise RuntimeError("safetensors checkpoint or shard index is missing")
        import ray
        ray.init(address="auto", ignore_reinit_error=True)
        # Allow cleanup/log collection even if an engine wait is still pending.
        remote_actor = ray.remote(max_concurrency=2)(EngineProcess)
        for mode in ("kv_restore", "recompute"):
            print(f"F1_CANARY_STAGE={mode}", flush=True)
            row = run_treatment(ray, remote_actor, args, mode, root)
            result["treatments"].append(row)
            write_json(root / "result.json", result)
            print(json.dumps(row, indent=2, sort_keys=True), flush=True)
            if row["status"] != "passed":
                break  # One attempt, preserve evidence, no automatic retries.
        if len(result["treatments"]) == 2 and all(
            row["status"] == "passed" for row in result["treatments"]
        ):
            result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
    write_json(root / "result.json", result)
    lines = ["# CSCC Ray F1 recovery 比較結果", "",
             f"狀態：`{result['status']}`；每組一次。", "",
             f"模型：`{result['model']}`；dtype：`{args.dtype}`。",
             f"Prompt / output / preempt：{args.prompt_tokens} / {args.output_tokens} / {args.preempt_tokens}。",
             f"Target 推論 warm-up：{args.warmup_target}（僅 warm up 本地推論，不預先建立 NIXL 連線）。",
             "此比較使用 logical grace period；兩個 Ray worker 不等於兩個已驗證的實體主機。",
             "比較 KV restore 與 prefix 重算，尚不包含完整 F1 四策略或 F2 live selection。", "",
             "| 策略 | 狀態 | Elapsed (s) | Recovery (s) | 首 token (s) | Prefix 重算 | Hash 相同 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in result["treatments"]:
        lines.append(f"| {row['mode']} | {row['status']} | {row.get('elapsed_s', '—')} | "
                     f"{row.get('recovery_s', '—')} | {row.get('recovery_to_first_token_s', '—')} | "
                     f"{row.get('prefix_recomputed_tokens', '—')} | "
                     f"{row.get('full_output_matches_reference', '—')} |")
        if row["status"] != "passed":
            lines.extend(["", f"失敗階段：`{row.get('stage')}`", "", "```text",
                          row.get("error", ""), "```"])
    if "error" in result:
        lines.extend(["", "```text", result["error"], "```"])
    lines.extend(["", "## 分項計時（秒）", "",
                  "各項為 driver wall time，包含相應 RPC 與排程；不等同於純網路或 GPU 時間。", "",
                  "| 項目 | kv_restore | recompute |", "|---|---:|---:|"])
    by_mode = {row["mode"]: row for row in result["treatments"]}
    for label, key in (
        ("Engine 啟動", "engine_startup_s"), ("Target 推論 warm-up", "target_warmup_s"),
        ("Metadata snapshot", "metadata_snapshot_s"), ("KV export", "state_export_s"),
        ("Source abort", "source_abort_s"), ("Restore stage", "restore_stage_s"),
        ("Target generate 至首 token", "target_first_output_s"),
        ("Source 終止確認", "source_stop_s"), ("Resume 至剩餘 decode 完成", "decode_after_resume_s"),
    ):
        values = [by_mode.get(mode, {}).get(key, "—") for mode in ("kv_restore", "recompute")]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines.extend(["", "Recovery 從 source 凍結邊界被 driver 收到開始，包含 export、stage、",
                  "target 首 token、source 終止確認及剩餘 decode；engine startup 與 reference",
                  "和 target warm-up 只計入 elapsed。每組只有一次，不能推論統計顯著性。", ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    if "--worker" in sys.argv:
        asyncio.run(engine_worker())
    else:
        raise SystemExit(main())
