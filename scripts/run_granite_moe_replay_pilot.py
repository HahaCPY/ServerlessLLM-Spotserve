"""Replay one frozen live MoE prefix into manual TP=1 and TP=2 targets.

This is a sequential recovery-capability pilot, not two spot revocations, a
planner-selected plan, KV transfer, or a formal Original/MoE-aware ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing.connection import Listener
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import traceback


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_frozen_prefix(metadata, prompt, output_tokens, threshold) -> dict:
    if not metadata.get("found") or metadata.get("prompt_tokens") != prompt:
        raise ValueError("live metadata does not match the source request")
    output = metadata.get("output_tokens", [])
    completed = metadata.get("completed_tokens")
    if completed != len(output) or not threshold <= completed < output_tokens:
        raise ValueError("frozen generated prefix is missing or already complete")
    if metadata.get("tokens") != prompt + output:
        raise ValueError("frozen token ordering is inconsistent")
    return {
        "tokens": prompt + output, "generated_prefix": output,
        "completed_tokens": completed, "remaining_tokens": output_tokens - completed,
        "prompt_sha256": digest(prompt), "full_prefix_sha256": digest(prompt + output),
        "strict_requested_boundary_verified": completed == threshold,
        "freeze_overshoot_tokens": completed - threshold,
        "computed_tokens": metadata.get("computed_tokens"),
        "allocated_kv_block_count": metadata.get("allocated_kv_block_count"),
    }


def inspect_ranks(models, tp) -> list:
    if len(models) != tp or not all(isinstance(value, str) and
        value.lstrip().startswith("GraniteMoeForCausalLM") and
        "GraniteMoeMoE" in value and "FusedMoE" in value for value in models
    ):
        raise ValueError("actual runtime MoE/TP rank inspection failed")
    return [{"rank": rank, "inspection_sha256": hashlib.sha256(text.encode()).hexdigest()}
            for rank, text in enumerate(models)]


def backfill_client_prefix(frozen, visible_tokens):
    actual = frozen["generated_prefix"]
    if not isinstance(visible_tokens, list) or actual[:len(visible_tokens)] != visible_tokens:
        raise ValueError("consumer-visible prefix does not match frozen source tokens")
    pending = actual[len(visible_tokens):]
    return visible_tokens + pending, pending


def reference_output_hash(profile, prompt_hash, prompt_tokens, output_tokens) -> str:
    cells = [cell for cell in profile.get("cells", []) if (
        cell.get("prompt_tokens"), cell.get("concurrency"), cell.get("max_new_tokens")
    ) == (prompt_tokens, 1, output_tokens)]
    if (profile.get("status") != "passed" or len(cells) != 1
            or cells[0].get("status") != "passed"
            or len(cells[0].get("samples", [])) < 2):
        raise ValueError("missing uninterrupted source reference cell")
    hashes = set()
    for sample in cells[0].get("samples", []):
        row = sample["requests"][0]
        if (row.get("prompt_sha256") != prompt_hash
                or row.get("generated_tokens") != output_tokens
                or row.get("finish_reason") != "length"):
            raise ValueError("reference request does not match")
        hashes.add(row["output_sha256"])
    if len(hashes) != 1:
        raise ValueError("uninterrupted reference is absent or not deterministic")
    return hashes.pop()


def wait_event(conn, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if conn.poll(min(0.5, max(0, deadline - time.monotonic()))):
            event = conn.recv()
            if event.get("event") in {"fatal", "generation_error"}:
                raise RuntimeError(event.get("traceback", repr(event)))
            if predicate(event):
                return event
    raise TimeoutError("worker event deadline exceeded")


def run(args) -> dict:
    repo = Path(__file__).resolve().parents[1]
    root = repo.parent
    vllm = root / "vllm"
    model = Path(args.model).resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("architectures") != ["GraniteMoeForCausalLM"]:
        raise ValueError("refusing a non-MoE checkpoint")
    if not 0 < args.freeze_after < args.max_new_tokens:
        raise ValueError("freeze boundary must be inside the output")
    if args.prompt_tokens + args.max_new_tokens > args.max_model_len:
        raise ValueError("prompt plus output exceeds context")
    reference = json.loads(Path(args.reference_profile).read_text())
    if reference.get("config_sha256") != digest(config) or reference.get("tp") != 1:
        raise ValueError("reference must use the same TP=1 MoE checkpoint")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True,
                                             trust_remote_code=False)
    seed = tokenizer.encode("Request 0: Explain expert routing, KV cache and GPU scheduling. ")
    prompt = (seed * (args.prompt_tokens // len(seed) + 1))[:args.prompt_tokens]
    expected_hash = reference_output_hash(reference, digest(prompt), args.prompt_tokens,
                                          args.max_new_tokens)
    report = {
        "status": "running", "scope": "sequential_manual_tp_replay_capability_pilot",
        "model": str(model), "config_sha256": digest(config),
        "source_physical_gpu_indices": [0], "target_groups": {"1": [1], "2": [1, 3]},
        "prompt_tokens": args.prompt_tokens, "max_new_tokens": args.max_new_tokens,
        "requested_freeze_after_tokens": args.freeze_after,
        "reference_output_sha256": expected_hash, "cases": [],
        "container_image": args.image, "reference_profile": args.reference_profile,
        "kv_state_restore_used": False, "recovery_method": "token_replay",
        "planner_gpu_actuation_verified": None, "formal_experiment_eligible": False,
        "physical_gpu_revocation_simulated": False,
    }
    workers = []
    with tempfile.TemporaryDirectory(prefix="spotserve-granite-replay-") as control_dir:
        listener = Listener(str(Path(control_dir) / "control.sock"), family="AF_UNIX",
                            authkey=b"spotserve")
        listener._listener._socket.settimeout(args.timeout_s)

        def launch(label, role, tp, gpus):
            name = f"spotserve-granite-replay-{label}-{Path(control_dir).name}"
            command = ["podman", "run", "--rm", "--name", name,
                       "--network", "none", "--shm-size", "1g"]
            for gpu in gpus:
                command.extend(["--device", f"nvidia.com/gpu={gpu}"])
            for source, target, mode in (
                (repo, repo, "ro"), (vllm, vllm, "ro"),
                (model.parent, model.parent, "ro"),
                ("/usr/local/cuda-13.0", "/usr/local/cuda", "ro"),
                (control_dir, "/control", "rw"),
            ):
                command.extend(["--volume", f"{source}:{target}:{mode}"])
            environments = {
                "PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_HOST_IP": "127.0.0.1",
                "NCCL_SOCKET_IFNAME": "lo",
            }
            for key, value in environments.items():
                command.extend(["--env", f"{key}={value}"])
            command.extend([
                args.image, str(vllm / ".venv/bin/python"), "-u",
                str(repo / "tests/spotserve_test/cross_container_nixl_worker.py"),
                "--model", str(model), "--control-socket", "/control/control.sock",
                "--role", role, "--node-id", label,
                "--side-channel-host", "127.0.0.1", "--side-channel-port", "54210",
                "--tensor-parallel-size", str(tp), "--max-model-len", str(args.max_model_len),
                "--max-new-tokens", str(args.max_new_tokens), "--pause-after-new-tokens", "0",
                "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
                "--gpu-memory-utilization", "0.8", "--cpu-offload-gb", "0",
                "--kv-transfer-mode", "none", "--disable-routed-experts-capture",
                "--no-trust-remote-code",
            ])
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            worker = {"name": name, "process": process, "conn": None, "command": command}
            workers.append(worker)

            def drain_logs():
                for line in process.stdout:
                    print(f"[{label}] {line}", end="", flush=True)
            worker["thread"] = threading.Thread(target=drain_logs, daemon=True)
            worker["thread"].start()
            conn = listener.accept()
            worker["conn"] = conn
            wait_event(conn, lambda event: event.get("event") == "ready", args.timeout_s)
            ready = time.monotonic()
            conn.send({"op": "inspect"})
            inspection = wait_event(conn, lambda event: event.get("event") == "inspection",
                                    args.timeout_s)
            worker["rank_checks"] = inspect_ranks(inspection["models"], tp)
            runtime = inspection["runtime"]
            if (runtime["tensor_parallel_size"] != tp
                    or runtime["dtype"] != "torch.bfloat16"
                    or runtime["vllm_version"] != reference["vllm_version"]
                    or runtime["torch_version"] != reference["torch_version"]):
                raise ValueError("actual replay runtime differs from the reference")
            worker["runtime"] = runtime
            worker["cold_container_to_ready_s"] = ready - started
            return worker

        def stop(worker):
            if worker.get("stopped"):
                return
            conn, process = worker["conn"], worker["process"]
            if conn is not None:
                try:
                    conn.send({"op": "shutdown"})
                except (OSError, EOFError):
                    pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(["podman", "stop", "--time", "5", worker["name"]],
                               check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                process.wait(timeout=15)
            if conn is not None:
                conn.close()
            worker["thread"].join(timeout=5)
            worker["stopped"] = True
            if process.returncode != 0:
                raise RuntimeError(f"own worker {worker['name']} exited {process.returncode}")

        try:
            source = launch("source", "source", 1, [0])
            conn = source["conn"]
            conn.send({"op": "generate", "request_id": "warmup", "token_ids": prompt,
                       "max_new_tokens": 64, "pause_after_new_tokens": 0})
            wait_event(conn, lambda event: event.get("request_id") == "warmup"
                       and event.get("finished"), args.timeout_s)
            conn.send({"op": "generate", "request_id": "source-request", "token_ids": prompt,
                       "max_new_tokens": args.max_new_tokens,
                       "pause_after_new_tokens": args.freeze_after})
            paused = wait_event(conn, lambda event: event.get("event") == "paused",
                                args.timeout_s)
            notice = time.monotonic()
            conn.send({"op": "pause_generation"})
            wait_event(conn, lambda event: event.get("event") == "generation_paused",
                       args.timeout_s)
            report["notice_to_freeze_ack_s"] = time.monotonic() - notice
            conn.send({"op": "metadata", "request_id": "source-request"})
            metadata = wait_event(conn, lambda event: event.get("event") == "metadata",
                                  args.timeout_s)["result"]
            frozen = validate_frozen_prefix(metadata, prompt, args.max_new_tokens,
                                            args.freeze_after)
            client_prefix, pending = backfill_client_prefix(frozen, paused["token_ids"])
            report["frozen"] = {key: value for key, value in frozen.items() if key != "tokens"}
            report["consumer_visible_prefix_tokens"] = len(paused["token_ids"])
            report["client_prefix_backfill_tokens"] = len(pending)
            report["client_prefix_backfill_token_ids"] = pending
            report["source_rank_checks"] = source["rank_checks"]
            report["source_runtime"] = source["runtime"]
            print("MOE_REPLAY_PILOT_PROGRESS=" + json.dumps({
                "phase": "source_frozen", "completed_tokens": frozen["completed_tokens"],
                "freeze_overshoot_tokens": frozen["freeze_overshoot_tokens"],
            }), flush=True)
            for tp, gpus in ((1, [1]), (2, [1, 3])):
                case = {"status": "running", "tp": tp, "physical_gpu_indices": gpus,
                        "frozen_prefix_sha256": frozen["full_prefix_sha256"]}
                report["cases"].append(case)
                candidate_start = time.monotonic()
                target = launch(f"target-tp{tp}", "target", tp, gpus)
                tconn = target["conn"]
                request_id = f"replay-tp{tp}"
                replay_started = time.monotonic()
                tconn.send({"op": "generate", "request_id": request_id,
                            "token_ids": frozen["tokens"],
                            "max_new_tokens": frozen["remaining_tokens"],
                            "pause_after_new_tokens": 0})
                first, first_four, last_delivery = None, None, None
                last_generated, delivery_events, gaps = 0, 0, []
                while True:
                    output = wait_event(tconn, lambda event: event.get("event") == "output"
                                        and event.get("request_id") == request_id,
                                        args.timeout_s)
                    observed = output.get("generated_tokens", 0)
                    if observed > last_generated:
                        delivered = time.monotonic()
                        if first is None:
                            first = delivered
                        if first_four is None and observed >= 4:
                            first_four = delivered
                        if last_delivery is not None:
                            gaps.append(delivered - last_delivery)
                        last_delivery, last_generated = delivered, observed
                        delivery_events += 1
                    if output.get("finished"):
                        ended = time.monotonic()
                        break
                if first is None:
                    raise ValueError("replay finished without delivering any generated tokens")
                generated = output.get("cumulative_token_ids", output.get("token_ids", []))
                combined = client_prefix + generated
                passed = (len(combined) == args.max_new_tokens
                          and output.get("finish_reason") == "length"
                          and digest(combined) == expected_hash)
                case.update({
                    "status": "passed" if passed else "failed", "tp": tp,
                    "physical_gpu_indices": gpus, "runtime_rank_checks": target["rank_checks"],
                    "runtime": target["runtime"],
                    "runtime_tp_verified": True, "frozen_prefix_sha256": frozen["full_prefix_sha256"],
                    "cold_container_to_ready_s": target["cold_container_to_ready_s"],
                    "counterfactual_recovery_start_to_first_token_s": first - candidate_start,
                    "replay_command_to_first_token_s": first - replay_started,
                    "replay_command_to_first_four_tokens_s": (
                        None if first_four is None else first_four - replay_started
                    ),
                    "output_delivery_events": delivery_events,
                    "max_gap_between_output_delivery_events_s": max(gaps) if gaps else None,
                    "replay_command_to_complete_s": ended - replay_started,
                    "remaining_tokens": frozen["remaining_tokens"],
                    "generated_tokens": len(generated), "combined_output_tokens": len(combined),
                    "output_sha256": digest(combined), "matches_uninterrupted_reference": passed,
                    "command": target["command"],
                })
                stop(target)
                case["worker_exit_code"] = target["process"].returncode
                print("MOE_REPLAY_PILOT_PROGRESS=" + json.dumps({
                    "phase": "target_complete", "tp": tp, "passed": passed,
                }), flush=True)
            report["status"] = "passed" if all(case["status"] == "passed"
                                                 for case in report["cases"]) else "failed"
            report["token_replay_verified"] = report["status"] == "passed"
        except Exception:
            report["status"] = "failed"
            report["traceback"] = traceback.format_exc()
        finally:
            for worker in reversed(workers):
                try:
                    stop(worker)
                except Exception:
                    report.setdefault("cleanup_errors", []).append(traceback.format_exc())
            listener.close()
            if report.get("cleanup_errors"):
                report["status"] = "failed"
                report["token_replay_verified"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-profile", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--freeze-after", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--image", default="localhost/spotserve-python312-nixl:latest")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception:
        report = {"status": "failed", "traceback": traceback.format_exc(),
                  "formal_experiment_eligible": False}
    print("MOE_REPLAY_PILOT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
