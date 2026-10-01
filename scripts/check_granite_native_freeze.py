"""Three live C=1 native freeze checks; not migration or formal ablation."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import traceback

from scripts.profile_granite_moe_tp import canonical_gpu_uuid
from scripts.run_granite_moe_replay_pilot import (
    backfill_client_prefix, digest, inspect_ranks, validate_frozen_prefix, wait_event,
)


def progress(phase, **fields):
    print("NATIVE_FREEZE_PROGRESS=" + json.dumps({"phase": phase, **fields}), flush=True)


def validate_route_histogram(metadata, layers, experts, top_k):
    if (metadata.get("moe_route_histogram_available") is not True
            or metadata.get("moe_route_histogram_source") != "vllm_runtime_topk"
            or metadata.get("moe_route_histogram_kind") != "runtime_observed_topk"):
        raise ValueError("real live runtime routes are unavailable")
    histogram = metadata.get("per_request_expert_route_histogram", {})
    totals = {layer: 0 for layer in range(layers)}
    for key, count in histogram.items():
        try:
            layer_name, expert_name = key.split("/")
            if not layer_name.startswith("layer:") or not expert_name.startswith("expert:"):
                raise ValueError("invalid route key")
            layer, expert = int(layer_name[6:]), int(expert_name[7:])
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed runtime expert route") from exc
        if (layer not in totals or not 0 <= expert < experts
                or type(count) is not int or count <= 0):
            raise ValueError("runtime expert route exceeds checkpoint bounds")
        totals[layer] += count
    if (not histogram or len(set(totals.values())) != 1
            or not totals[0] or totals[0] % top_k):
        raise ValueError("incomplete per-layer runtime top-k routes")
    return {"histogram": histogram, "histogram_sha256": digest(histogram),
            "observed_routed_tokens": totals[0] // top_k,
            "top_k": top_k, "layers": layers, "experts_per_layer": experts,
            "origin": "live_frontend_routed_experts_chunks",
            "covers_all_frozen_prompt_tokens": None,
            "physical_dispatch_traffic_verified": False}


def run(args):
    from transformers import AutoTokenizer

    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    model = Path(args.model).resolve()
    config = json.loads((model / "config.json").read_text())
    if (config.get("architectures") != ["GraniteMoeForCausalLM"]
            or not 0 < config.get("num_experts_per_tok", 0) < config.get("num_local_experts", 0)):
        raise ValueError("refusing a dense or non-sparse checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    seed = tokenizer.encode("Request 0: Explain expert routing, KV cache and GPU scheduling. ")
    prompt = (seed * (4096 // len(seed) + 1))[:4096]
    result = {"scope": "three_live_native_freeze_checks_not_formal_ablation",
              "status": "running", "source_physical_gpu_indices": [0],
              "source_scheduling": "synchronous_controlled_C1_native_barrier",
              "model_revision": args.model_revision, "config_sha256": digest(config),
              "trials": [], "formal_experiment_eligible": False,
              "source_routing_capture_requested": args.capture_source_routes,
              "physical_gpu_revocation_simulated": False, "target_migration_tested": False}
    process, conn, thread, name = None, None, None, None
    with tempfile.TemporaryDirectory(prefix="spotserve-native-freeze-") as control:
        listener = Listener(str(Path(control) / "control.sock"), family="AF_UNIX", authkey=b"spotserve")
        listener._listener._socket.settimeout(args.timeout_s)
        try:
            name = f"spotserve-native-freeze-{Path(control).name}"
            command = ["podman", "run", "--rm", "--name", name, "--network", "none",
                       "--shm-size", "1g", "--device", "nvidia.com/gpu=0"]
            for source, target, mode in ((repo, repo, "ro"), (vllm, vllm, "ro"),
                                         (model, model, "ro"), (control, "/control", "rw"),
                                         ("/usr/local/cuda-13.0", "/usr/local/cuda", "ro")):
                command.extend(["--volume", f"{source}:{target}:{mode}"])
            environments = {
                "PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "0",
                "VLLM_HOST_IP": "127.0.0.1", "NCCL_SOCKET_IFNAME": "lo",
            }
            for key, value in environments.items():
                command.extend(["--env", f"{key}={value}"])
            command.extend([
                args.image, str(vllm / ".venv/bin/python"), "-u",
                str(repo / "tests/spotserve_test/cross_container_nixl_worker.py"),
                "--model", str(model), "--control-socket", "/control/control.sock",
                "--role", "source", "--node-id", "source-gpu0",
                "--side-channel-host", "127.0.0.1", "--side-channel-port", "54211",
                "--tensor-parallel-size", "1", "--max-model-len", "8704",
                "--max-new-tokens", "512", "--pause-after-new-tokens", "0",
                "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
                "--gpu-memory-utilization", "0.8", "--cpu-offload-gb", "0",
                "--kv-transfer-mode", "none",
                "--no-trust-remote-code", "--native-freeze-barrier",
            ])
            if not args.capture_source_routes:
                command.append("--disable-routed-experts-capture")
            result["command"] = command
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            def logs():
                for line in process.stdout:
                    print("[source] " + line, end="", flush=True)
            thread = threading.Thread(target=logs, daemon=True)
            thread.start()
            conn = listener.accept()
            wait_event(conn, lambda event: event.get("event") == "ready", args.timeout_s)
            conn.send({"op": "inspect"})
            inspection = wait_event(conn, lambda event: event.get("event") == "inspection", args.timeout_s)
            result["runtime_rank_checks"] = inspect_ranks(inspection["models"], 1)
            runtime = inspection["runtime"]
            if (runtime["async_scheduling"] is not False or runtime["max_num_seqs"] != 1
                    or runtime["tensor_parallel_size"] != 1 or runtime["dtype"] != "torch.bfloat16"
                    or "NativeFreezeScheduler" not in runtime["scheduler_cls"]
                    or runtime["routing_capture"] is not args.capture_source_routes
                    or [canonical_gpu_uuid(value) for value in runtime["actual_visible_gpu_uuids"]]
                    != [canonical_gpu_uuid(args.source_gpu_uuid)]):
                raise ValueError("actual source runtime does not match controlled native freeze")
            result["runtime"] = runtime
            def generate(request_id, limit, threshold):
                conn.send({"op": "generate", "request_id": request_id, "token_ids": prompt,
                           "max_new_tokens": limit, "pause_after_new_tokens": threshold})
            generate("warmup", 64, 0)
            wait_event(conn, lambda event: event.get("request_id") == "warmup" and event.get("finished"), args.timeout_s)
            progress("source_warmed")
            deadline = time.monotonic() + args.timeout_s
            while args.start_barrier is not None and not args.start_barrier.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("freeze start barrier timed out")
                time.sleep(0.25)
            generate("reference", 512, 0)
            reference = wait_event(conn, lambda event: event.get("request_id") == "reference"
                                   and event.get("finished"), args.timeout_s)
            reference_tokens = reference.get("cumulative_token_ids", reference["token_ids"])
            if len(reference_tokens) != 512 or reference.get("finish_reason") != "length":
                raise ValueError("uninterrupted reference is incomplete")
            result["uninterrupted_reference_sha256"] = digest(reference_tokens)
            for repeat in range(3):
                request_id = f"native-freeze-{repeat}"
                generate(request_id, 512, 256)
                paused = wait_event(conn, lambda event: event.get("event") == "paused"
                                    and event.get("request_id") == request_id, args.timeout_s)
                notice = time.monotonic()
                conn.send({"op": "pause_generation"})
                wait_event(conn, lambda event: event.get("event") == "generation_paused", args.timeout_s)
                ack_ms = (time.monotonic() - notice) * 1000
                conn.send({"op": "metadata", "request_id": request_id})
                metadata = wait_event(conn, lambda event: event.get("event") == "metadata"
                                      and event.get("request_id") == request_id, args.timeout_s)["result"]
                frozen = validate_frozen_prefix(metadata, prompt, 512, 256)
                visible, pending = backfill_client_prefix(frozen, paused["token_ids"])
                if (not frozen["strict_requested_boundary_verified"]
                        or visible != reference_tokens[:256]
                        or not metadata.get("allocated_kv_block_count", 0)):
                    raise ValueError("native boundary/client prefix/KV residency check failed")
                result["trials"].append({"repeat": repeat, "status": "passed", "frozen": frozen,
                                         "consumer_visible_tokens": len(paused["token_ids"]),
                                         "backfill_tokens": len(pending), "pause_ack_ms": ack_ms,
                                         "matches_uninterrupted_prefix": True})
                if args.capture_source_routes:
                    result["trials"][-1]["runtime_routes"] = validate_route_histogram(
                        metadata, config["num_hidden_layers"], config["num_local_experts"],
                        config["num_experts_per_tok"])
                progress("boundary_verified", repeat=repeat, generated_tokens=256, overshoot=0)
                conn.send({"op": "abort", "request_id": request_id})
                wait_event(conn, lambda event: event.get("event") == "aborted"
                           and event.get("request_id") == request_id, args.timeout_s)
                conn.send({"op": "resume_generation"})
                wait_event(conn, lambda event: event.get("event") == "generation_resumed", args.timeout_s)
            result["status"] = "passed"
        except Exception:
            result["status"] = "failed"
            result["traceback"] = traceback.format_exc()
        finally:
            if conn is not None:
                try:
                    conn.send({"op": "shutdown"})
                except (OSError, EOFError):
                    pass
            if process is not None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Only this function's uniquely named temporary container.
                    subprocess.run(["podman", "stop", "--time", "5", name], check=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    process.wait(timeout=15)
                result["worker_exit_code"] = process.returncode
                if process.returncode != 0:
                    result["status"] = "failed"
            if conn is not None:
                conn.close()
            listener.close()
            if thread is not None:
                thread.join(timeout=5)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-gpu-uuid", required=True)
    parser.add_argument("--start-barrier", type=Path)
    parser.add_argument("--capture-source-routes", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--image", default="localhost/spotserve-python312-nixl:latest")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception:
        report = {"status": "failed", "traceback": traceback.format_exc(),
                  "formal_experiment_eligible": False}
    print("NATIVE_FREEZE_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
