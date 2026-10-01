"""Local physical-GPU adapter for audited controlled MoE experiments."""

from __future__ import annotations

import json
from multiprocessing.connection import Listener
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import uuid

from scripts.check_granite_native_freeze import validate_route_histogram
from scripts.profile_granite_moe_tp import canonical_gpu_uuid
from scripts.run_granite_moe_replay_pilot import (
    backfill_client_prefix, digest, inspect_ranks, validate_frozen_prefix,
)
from sllm.spot.moe_tp_profiles import profile_runtime_signature


def write_artifact(path, payload):
    """Write a new run output, never overwrite a historical artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


class GenerationPreempted(RuntimeError):
    pass


def validate_stream_update(previous, event):
    """No revised already-delivered tokens, falling counts, or truncated chunks."""
    count, tokens = event["generated_tokens"], event["cumulative_token_ids"]
    if (type(count) is not int or count < len(previous) or len(tokens) != count
            or list(tokens[:len(previous)]) != previous):
        raise ValueError("streamed token count/prefix is inconsistent or revised")
    return list(tokens)


class GPUWorker:
    def __init__(self, pool, label, group, capture=False, startup_profiling=False,
                 tp=None, dp=1, ep=False):
        self.pool, self.label, self.group = pool, label, list(group)
        self.tp, self.dp, self.ep = tp or len(group), dp, ep
        if self.tp * self.dp != len(group) or (self.ep and len(group) < 2):
            raise ValueError("physical group must match TP × DP and EP needs two ranks")
        self.conn = self.process = self.thread = None
        self.closed = False
        self.records, self.log_lines, self.warnings = [], [], []
        self.active_measurement = None
        self.log_lock = threading.Lock()
        self.event_condition = threading.Condition()
        self.send_lock = threading.Lock()
        self.events, self.cancelled_requests, self.client_visible = [], set(), {}
        self.live_measurements = {}
        self.io_error, self.reader_thread = None, None
        self.name = "spotserve-route-" + uuid.uuid4().hex[:16]
        control = Path(pool.control.name)
        self.listener = Listener(str(control / (self.name + ".sock")),
                                 family="AF_UNIX", authkey=b"spotserve")
        self.listener._listener._socket.settimeout(0.5)
        self.log_path = pool.output / (label + ".log")
        self.log_stream = self.log_path.open("x")
        repo, vllm, model = pool.repo, pool.vllm, pool.model
        command = ["podman", "run", "--rm", "--name", self.name,
                   "--network", "none", "--shm-size", "1g"]
        for gpu in group:
            command += ["--device", f"nvidia.com/gpu={gpu}"]
        for src, dst, mode in ((repo, repo, "ro"), (vllm, vllm, "ro"),
                               (model, model, "ro"), (control, "/control", "rw"),
                               ("/usr/local/cuda-13.0", "/usr/local/cuda", "ro")):
            command += ["--volume", f"{src}:{dst}:{mode}"]
        command += ["--volume", f"{pool.compiler_cache}:/taskcache:rw"]
        env = {"PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
               "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_HOST_IP": "127.0.0.1",
               "NCCL_SOCKET_IFNAME": "lo", "TRITON_CACHE_DIR": "/taskcache/triton",
               "CUDA_CACHE_PATH": "/taskcache/cuda"}
        env.update(getattr(pool, "diagnostic_env", {}))
        for key, value in env.items():
            command += ["--env", f"{key}={value}"]
        command += [pool.image, str(vllm / ".venv/bin/python"), "-u",
                    str(repo / "tests/spotserve_test/cross_container_nixl_worker.py"),
                    "--model", str(model), "--control-socket", f"/control/{self.name}.sock",
                    "--role", "source" if capture else "target", "--node-id", label,
                    "--side-channel-host", "127.0.0.1", "--side-channel-port", "54301",
                    "--tensor-parallel-size", str(len(group)), "--max-model-len", "8704",
                    "--data-parallel-size", str(self.dp),
                    "--max-new-tokens", "512", "--pause-after-new-tokens", "0",
                    "--max-num-seqs", str(pool.max_num_seqs), "--max-num-batched-tokens", "2048",
                    "--gpu-memory-utilization", "0.8", "--cpu-offload-gb", "0",
                    "--kv-transfer-mode", "none", "--no-trust-remote-code",
                    "--native-freeze-barrier"]
        command[command.index("--tensor-parallel-size") + 1] = str(self.tp)
        if self.ep:
            command.append("--enable-expert-parallel")
        if not capture:
            command.append("--disable-routed-experts-capture")
        if startup_profiling:
            command.append("--startup-profiling")
        self.command = command
        self.started = time.monotonic()
        pool.workers.append(self)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._drain_logs, daemon=True)
        self.thread.start()
        try:
            deadline = self.started + pool.timeout
            while self.conn is None:
                try:
                    self.conn = self.listener.accept()
                except socket.timeout:
                    if self.process.poll() is not None:
                        raise RuntimeError("worker exited before connect: " + "".join(self.log_lines[-15:]))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("worker connect deadline exceeded")
            self.reader_thread = threading.Thread(target=self._read_events, daemon=True)
            self.reader_thread.start()
            self.wait(lambda e: e.get("event") == "ready")
            self.cold_to_engine_ready_s = time.monotonic() - self.started
            self.send({"op": "inspect"})
            inspection = self.wait(lambda e: e.get("event") == "inspection")
            self.inspection = inspection
            self.runtime = inspection["runtime"]
            self.rank_checks = inspect_ranks(inspection["models"], len(group))
            identities = inspection["rank_identities"]
            expected = {canonical_gpu_uuid(pool.gpu_uuids[g]): g for g in group}
            actual = {canonical_gpu_uuid(row["actual_gpu_uuid"]) for row in identities}
            if actual != set(expected) or {row["rank"] for row in identities} != set(range(len(group))):
                raise ValueError("actual rank GPUs differ from requested group")
            self.ranks = [{**self.rank_checks[i], "physical_gpu_index": expected[
                canonical_gpu_uuid(row["actual_gpu_uuid"])], "actual_gpu_uuid": row["actual_gpu_uuid"],
                "granite_moe_model": row["model_class"] == "GraniteMoeForCausalLM",
                "expert_modules": True} for i, row in enumerate(identities)]
            r = self.runtime
            if (r["tensor_parallel_size"] != self.tp or r["pipeline_parallel_size"] != 1
                    or r["data_parallel_size"] != self.dp or r["enable_expert_parallel"] is not self.ep
                    or r["async_scheduling"] is not False or r["routing_capture"] is not capture
                    or r["dtype"] != "torch.bfloat16" or r["cpu_offload_gb"] != 0
                    or r["max_num_seqs"] != pool.max_num_seqs or r["max_model_len"] != 8704
                    or r["enable_prefix_caching"] or not r["enforce_eager"]):
                raise ValueError("actual runtime does not match controlled protocol")
            self.profile_runtime = {**r, "dtype": "bfloat16", "tp": self.tp,
                                    "config_sha256": digest(pool.config), "model_runner": "V1",
                                    "moe_backend": "triton", "execution": {
                                        "model_revision": pool.revision, "physical_gpu_indices": group}}
            self.runtime_signature = (digest(self.profile_runtime) if self.ep
                                      else profile_runtime_signature(self.profile_runtime))
        except BaseException:
            self.close()
            raise

    def _drain_logs(self):
        for line in self.process.stdout:
            with self.log_lock:
                self.log_lines.append(line)
                self.log_stream.write(line)
                self.log_stream.flush()
                if "Triton kernel JIT compilation during inference:" in line or "Autotuning process starts" in line:
                    self.warnings.append({"measurement": self.active_measurement,
                                          "message": line.strip(), "host_monotonic_s": time.monotonic()})

    def _read_events(self):
        try:
            while True:
                event = self.conn.recv()
                event["host_received_monotonic_s"] = time.monotonic()
                with self.event_condition:
                    self.events.append(event)
                    self.event_condition.notify_all()
        except (EOFError, OSError) as exc:
            with self.event_condition:
                self.io_error = repr(exc)
                self.event_condition.notify_all()

    def send(self, command):
        with self.send_lock:
            self.conn.send(command)

    def wait(self, predicate, timeout=None, request_id=None):
        deadline = time.monotonic() + (timeout or self.pool.timeout)
        with self.event_condition:
            while time.monotonic() < deadline:
                if request_id in self.cancelled_requests:
                    raise GenerationPreempted(request_id)
                for i, event in enumerate(self.events):
                    if event.get("event") in {"fatal", "generation_error"}:
                        raise RuntimeError(event.get("traceback", repr(event)))
                    if predicate(event):
                        return self.events.pop(i)
                if self.io_error is not None:
                    raise RuntimeError("worker event stream closed: " + self.io_error)
                self.event_condition.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
        raise TimeoutError("worker event deadline exceeded")

    def cancel_measurement(self, request_id):
        with self.event_condition:
            self.cancelled_requests.add(request_id)
            self.event_condition.notify_all()

    def command_event(self, command, event):
        self.send(command)
        return self.wait(lambda e: e.get("event") == event
                         and ("request_id" not in command or e.get("request_id") == command["request_id"]))

    def measure(self, request_id, tokens=None, output_tokens=256, installed=False):
        with self.log_lock:
            self.active_measurement = request_id
        began = time.monotonic()
        with self.event_condition:
            self.live_measurements[request_id] = {"started_monotonic_s": began, "deliveries": []}
        self.send({"op": "generate", "request_id": request_id,
                        "token_ids": tokens, "use_installed_replay": installed,
                        "max_new_tokens": output_tokens, "pause_after_new_tokens": 0})
        first = fourth = previous = None
        count, gaps, deliveries = 0, [], []
        previous_tokens = []
        while True:
            event = self.wait(lambda e: e.get("event") == "output" and e.get("request_id") == request_id,
                              request_id=request_id)
            new_count, now = event["generated_tokens"], event["host_received_monotonic_s"]
            previous_tokens = validate_stream_update(previous_tokens, event)
            if new_count > count:
                first = first or now
                if new_count >= 4 and fourth is None:
                    fourth = now
                if previous is not None:
                    gaps.append(now - previous)
                deliveries.append({"count": new_count, "relative_s": now - began})
                count, previous = new_count, now
                with self.event_condition:
                    self.client_visible[request_id] = list(event["cumulative_token_ids"])
                    self.live_measurements[request_id]["deliveries"] = list(deliveries)
            if event.get("finished"):
                ended = now
                break
        with self.log_lock:
            self.active_measurement = None
        generated = event["cumulative_token_ids"]
        if len(generated) != output_tokens or event.get("finish_reason") != "length" or first is None:
            raise ValueError("partial output cannot become a valid measurement")
        row = {"request_id": request_id, "latency_s": ended - began, "ttft_s": first - began,
               "started_monotonic_s": began, "first_token_monotonic_s": first,
               "completed_monotonic_s": ended,
               "first_four_tokens_s": None if fourth is None else fourth - began,
               "max_delivery_gap_s": max(gaps, default=0),
               "output_token_ids": generated, "output_sha256": digest(generated),
               "generated_tokens": len(generated), "finish_reason": event["finish_reason"],
               "prompt_sha256": digest(tokens) if tokens is not None else None,
               "delivery_events": deliveries, "timed_jit_warnings": [w for w in self.warnings
                                                                        if w["measurement"] == request_id]}
        self.records.append(row)
        return row

    def warm(self, tokens):
        self.measure(self.label + "-warmup", tokens, 64)
        self.cold_to_warmed_ready_s = time.monotonic() - self.started
        return self.cold_to_warmed_ready_s

    def freeze(self, request_id, prompt, output_tokens=512, threshold=256):
        self.send({"op": "generate", "request_id": request_id, "token_ids": prompt,
                        "max_new_tokens": output_tokens, "pause_after_new_tokens": threshold})
        paused = self.wait(lambda e: e.get("event") == "paused" and e.get("request_id") == request_id)
        notice = time.monotonic()
        with self.event_condition:
            observed_outputs = [e["host_received_monotonic_s"] for e in self.events
                if e.get("event") == "output" and e.get("request_id") == request_id
                and e.get("generated_tokens") == threshold]
        if not observed_outputs:
            raise ValueError("native source boundary token delivery was not independently observed")
        last_source_delivery = max(observed_outputs)
        self.command_event({"op": "pause_generation"}, "generation_paused")
        ack = time.monotonic()
        metadata = self.command_event({"op": "metadata", "request_id": request_id}, "metadata")["result"]
        frozen = validate_frozen_prefix(metadata, prompt, output_tokens, threshold)
        client, pending = backfill_client_prefix(frozen, paused["token_ids"])
        if not frozen["strict_requested_boundary_verified"] or not frozen["allocated_kv_block_count"]:
            raise ValueError("native freeze boundary/KV residency invalid")
        routes = validate_route_histogram(metadata, self.pool.config["num_hidden_layers"],
                                          self.pool.config["num_local_experts"],
                                          self.pool.config["num_experts_per_tok"])
        return {"request_id": request_id, "frozen": frozen, "client_prefix": client,
                "backfill_tokens": pending, "routes": routes,
                "source_last_token_monotonic_s": last_source_delivery,
                "notice_monotonic_s": notice, "notice_to_freeze_ack_s": ack - notice}

    def abort_resume(self, request_id):
        self.command_event({"op": "abort", "request_id": request_id}, "aborted")
        self.command_event({"op": "resume_generation"}, "generation_resumed")

    def snapshot_running(self, request_id, initial_prompt, carried_prefix, reference,
                         workload_id="formal-serving"):
        """Wall-clock pause: retain actual boundary, never round it to 256."""
        notice = time.monotonic()
        self.command_event({"op": "pause_generation"}, "generation_paused")
        ack = time.monotonic()
        metadata = self.command_event({"op": "metadata", "request_id": request_id}, "metadata")["result"]
        expected_prompt = initial_prompt + carried_prefix
        new_output = metadata.get("output_tokens", [])
        full_output = carried_prefix + new_output
        if (not metadata.get("found") or metadata.get("prompt_tokens") != expected_prompt
                or metadata.get("tokens") != initial_prompt + full_output
                or not 0 < len(full_output) < 512 or not metadata.get("allocated_kv_block_count")
                or full_output != reference["output_token_ids"][:len(full_output)]):
            raise ValueError("wall-clock source snapshot is missing, complete, or incorrect")
        with self.event_condition:
            visible = carried_prefix + list(self.client_visible.get(request_id, []))
            partial = dict(self.live_measurements.get(request_id, {}))
        frozen = {"tokens": initial_prompt + full_output, "generated_prefix": full_output,
                  "completed_tokens": len(full_output), "remaining_tokens": 512 - len(full_output),
                  "full_prefix_sha256": digest(initial_prompt + full_output),
                  "prompt_sha256": digest(initial_prompt),
                  "allocated_kv_block_count": metadata["allocated_kv_block_count"],
                  "computed_tokens": metadata.get("computed_tokens"),
                  "strict_requested_boundary_verified": None,
                  "requested_token_boundary": None, "freeze_mode": "fixed_wall_clock"}
        client, pending = backfill_client_prefix(frozen, visible)
        if pending:
            # The host control client has actually received these missing
            # tokens in the metadata response; record that receipt separately
            # from decoder-stream deliveries, never from the reference oracle.
            received = time.monotonic()
            partial["deliveries"] = list(partial.get("deliveries", [])) + [{
                "count": len(new_output),
                "relative_s": received - partial["started_monotonic_s"],
                "mode": "host_snapshot_backfill"}]
        routes = validate_route_histogram(metadata, 32, 40, 8)
        self.cancel_measurement(request_id)
        return {"workload_id": workload_id, "request_id": request_id, "reference": reference,
                "frozen": frozen, "client_prefix": client, "backfill_tokens": pending,
                "routes": routes, "notice_monotonic_s": notice,
                "notice_to_freeze_ack_s": ack - notice, "partial_delivery": partial}

    def close(self):
        if self.closed:
            return
        if self.conn is not None:
            try:
                self.send({"op": "shutdown"})
            except (OSError, EOFError):
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(["podman", "stop", "--time", "5", self.name], check=False,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.process.wait(timeout=15)
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.closed = True
        if self.conn is not None:
            self.conn.close()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1)
        self.listener.close()
        self.log_stream.close()

    def evidence(self):
        return {"label": self.label, "command": self.command, "physical_gpu_indices": self.group,
                "runtime": self.runtime, "ranks": self.ranks,
                "profile_runtime_signature": self.runtime_signature,
                "cold_to_engine_ready_s": self.cold_to_engine_ready_s,
                "cold_to_warmed_ready_s": getattr(self, "cold_to_warmed_ready_s", None),
                "worker_exit_code": self.process.returncode, "log_path": str(self.log_path)}


class GPUPool:
    def __init__(self, output, model, revision, gpu_uuids, timeout=600,
                 image="localhost/spotserve-python312-nixl:latest", compiler_cache=None,
                 diagnostic_env=None, max_num_seqs=1):
        self.repo = Path(__file__).resolve().parents[1]
        self.vllm = self.repo.parent / "vllm"
        self.model, self.revision = Path(model), revision
        self.max_num_seqs = max_num_seqs
        if max_num_seqs not in (1, 2):
            raise ValueError("controlled EP driver supports C=1/C=2")
        self.diagnostic_env = dict(diagnostic_env or {})
        self.config = json.loads((self.model / "config.json").read_text())
        if (self.config.get("architectures") != ["GraniteMoeForCausalLM"]
                or not 0 < self.config.get("num_experts_per_tok", 0) < self.config.get("num_local_experts", 0)):
            raise ValueError("refusing dense/non-sparse checkpoint")
        self.output, self.timeout, self.image = Path(output), timeout, image
        self.output.mkdir(parents=True, exist_ok=True)
        snapshots = {}
        for name, query in {
            "gpus": "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "compute_processes": "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        }.items():
            observation = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15)
            if observation.returncode:
                raise RuntimeError("GPU environment observation failed: " + observation.stderr)
            snapshots[name] = observation.stdout.splitlines()
        write_artifact(self.output / "environment_snapshot.json", {
            "observed_epoch_s": time.time(), "snapshots": snapshots,
            "existing_processes_not_stopped": True, "exclusive_environment": False,
            "snapshot_not_continuous_background_control": True})
        self.gpu_uuids, self.workers = gpu_uuids, []
        self.compiler_cache = Path(compiler_cache or self.output.parent / "compiler_cache").resolve()
        self.compiler_cache.mkdir(parents=True, exist_ok=True)
        self.control = tempfile.TemporaryDirectory(prefix="spotserve-route-run-")

    def launch(self, label, group, capture=False, startup_profiling=False,
               tp=None, dp=1, ep=False):
        active = {g for w in self.workers if not w.closed for g in w.group}
        if set(group) & active:
            raise ValueError("overlapping actual GPU workers are prohibited")
        return GPUWorker(self, label, group, capture, startup_profiling=startup_profiling,
                         tp=tp, dp=dp, ep=ep)

    def close(self):
        for worker in reversed(self.workers):
            worker.close()
        self.control.cleanup()
