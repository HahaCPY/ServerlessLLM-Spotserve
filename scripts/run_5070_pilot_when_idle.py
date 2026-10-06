#!/usr/bin/env python3
"""Run the four-GPU F1/F2 pilot one treatment at a time when GPUs are idle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_moe_spotserve_experiments import read_valid_result


PYTHON = Path("/work/containers/s112060021/Qwen3/vllm/.venv/bin/python")
MODEL = Path("/work/spotserve-models/Qwen1.5-MoE-A2.7B")
TRACE = REPO_ROOT / "examples/spotserve/spot_trace_qwen15_moe_a27b_batch_recovery.jsonl"
PROMPTS = (
    REPO_ROOT / "docs/spotserve-moe-aware-planner.md",
    REPO_ROOT / "docs/final_experiment_md/old_four_version_exp.md",
)
FALLBACK_ROUTE_PROFILE = (
    REPO_ROOT
    / "results/moe_spotserve_qwen_a27b_4k_v2/calibration/qwen-route-profile.json"
)
JOBS = (
    ("f1_rerouting", "rerouting"),
    ("f1_reparallelization", "reparallelization"),
    ("f1_original_spotserve", "spotserve"),
    ("f1_moe_spotserve", "moe_spotserve"),
    ("f2_original_reparallelization", "original_reparallelization"),
    ("f2_moe_reparallelization", "moe_reparallelization"),
    ("f2_original_migration", "original_migration"),
    ("f2_moe_migration", "moe_migration"),
)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_log(path: Path, message: str) -> None:
    line = f"{now()} {message}"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def gpu_snapshot() -> list[dict[str, int]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    rows = []
    for line in process.stdout.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        rows.append({
            "index": values[0],
            "memory_used_mib": values[1],
            "memory_free_mib": values[2],
            "utilization_percent": values[3],
        })
    if [row["index"] for row in rows] != [0, 1, 2, 3]:
        raise RuntimeError(f"expected GPUs 0-3, observed {rows}")
    return rows


def run_result(output_root: Path, label: str, mode: str) -> tuple[str, Path]:
    job_dir = output_root / label
    result_path = job_dir / mode / "run-1.json"
    if read_valid_result(result_path, mode) is not None:
        return "valid", result_path
    attempts_path = job_dir / "attempts.json"
    if attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text())
        if any(not bool(row.get("valid")) for row in attempts):
            return "failed", result_path
    return "pending", result_path


def experiment_command(
    output_root: Path, label: str, mode: str, route_profile: Path
) -> list[str]:
    return [
        str(PYTHON), "-u", "-m", "scripts.run_moe_spotserve_experiments",
        "--model", str(MODEL),
        "--trace", str(TRACE),
        "--output-dir", str(output_root / label),
        "--gpus", "0", "1", "2", "3",
        "--source-gpus", "0", "1",
        "--target-gpus", "2", "3",
        "--tensor-parallel-size", "2",
        "--request-count", "2",
        "--prompt-tokens", "4096",
        "--prompt-files", *(str(path) for path in PROMPTS),
        "--max-model-len", "4608",
        "--max-num-batched-tokens", "2048",
        "--max-new-tokens", "512",
        "--preempt-after-new-tokens", "256",
        "--preempt-max-overshoot", "16",
        "--gpu-memory-utilization", "0.94",
        "--cpu-offload-gb", "6",
        "--trace-speedup", "1000",
        "--token-delay-s", "0",
        "--timeout-s", "900",
        "--launch-idle-stable-s", "30",
        "--launch-idle-max-used-mib", "512",
        "--launch-idle-poll-s", "5",
        "--repeats", "1",
        "--max-attempts-per-run", "1",
        "--order-seed", "20261002",
        "--route-profile", str(route_profile),
        "--modes", mode,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stable-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument(
        "--replace-failed-job",
        action="append",
        default=[],
        help="Run one explicit replacement under a preserved replacement1 label.",
    )
    args = parser.parse_args()
    jobs = tuple(
        (
            f"{label}_replacement1"
            if label in set(args.replace_failed_job)
            else label,
            mode,
        )
        for label, mode in JOBS
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "background-state.json"
    log_path = output_root / "background-supervisor.log"
    stop_path = output_root / "STOP"
    lock_stream = (output_root / ".background-supervisor.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        append_log(log_path, "another supervisor already owns the lock")
        return 2

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state: dict[str, object] = {
        "status": "monitoring",
        "pid": os.getpid(),
        "started_at": now(),
        "output_root": str(output_root),
        "jobs": [{"label": label, "mode": mode} for label, mode in jobs],
    }
    write_json(state_path, state)
    append_log(log_path, f"supervisor started pid={os.getpid()}")

    for label, mode in jobs:
        while True:
            if stopping or stop_path.exists():
                state.update({"status": "stopped", "stopped_at": now()})
                write_json(state_path, state)
                append_log(log_path, "stop requested")
                return 0
            existing, result_path = run_result(output_root, label, mode)
            if existing == "valid":
                append_log(log_path, f"reuse valid result label={label}")
                break
            if existing == "failed":
                state.update({
                    "status": "blocked",
                    "blocked_at": now(),
                    "blocked_job": label,
                    "reason": "one-attempt run failed; no automatic retry",
                })
                write_json(state_path, state)
                append_log(log_path, f"blocked by failed result label={label}")
                return 1

            stable_started: float | None = None
            last_report = 0.0
            while True:
                if stopping or stop_path.exists():
                    break
                existing, result_path = run_result(output_root, label, mode)
                if existing != "pending":
                    break
                snapshot = gpu_snapshot()
                idle = all(
                    row["memory_used_mib"] <= args.maximum_used_mib
                    for row in snapshot
                )
                current = time.monotonic()
                if idle:
                    stable_started = stable_started or current
                else:
                    stable_started = None
                stable_for = 0.0 if stable_started is None else current - stable_started
                if current - last_report >= 30 or stable_for >= args.stable_seconds:
                    state.update({
                        "status": "monitoring",
                        "current_job": label,
                        "gpu_snapshot": snapshot,
                        "stable_for_seconds": stable_for,
                        "updated_at": now(),
                    })
                    write_json(state_path, state)
                    append_log(
                        log_path,
                        f"monitor label={label} idle={idle} stable_for={stable_for:.1f}",
                    )
                    last_report = current
                if stable_for >= args.stable_seconds:
                    break
                time.sleep(args.poll_seconds)
            if existing != "pending" or stopping or stop_path.exists():
                continue

            fresh_route_profile = (
                output_root / "f1_rerouting/rerouting/run-1.json"
            )
            route_profile = (
                fresh_route_profile
                if fresh_route_profile.is_file()
                else FALLBACK_ROUTE_PROFILE
            )
            command = experiment_command(
                output_root, label, mode, route_profile
            )
            child_log_path = output_root / label / "background-child.log"
            child_log_path.parent.mkdir(parents=True, exist_ok=True)
            state.update({
                "status": "running",
                "current_job": label,
                "command": command,
                "run_started_at": now(),
                "pre_run_gpu_snapshot": gpu_snapshot(),
            })
            write_json(state_path, state)
            append_log(log_path, f"starting label={label} mode={mode}")
            with child_log_path.open("a", encoding="utf-8") as child_log:
                process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            state.update({
                "last_exit_code": process.returncode,
                "run_finished_at": now(),
            })
            write_json(state_path, state)
            existing, result_path = run_result(output_root, label, mode)
            if process.returncode != 0 or existing != "valid":
                state.update({
                    "status": "blocked",
                    "blocked_at": now(),
                    "blocked_job": label,
                    "reason": "child process failed validation",
                    "result_path": str(result_path),
                })
                write_json(state_path, state)
                append_log(
                    log_path,
                    f"child failed label={label} rc={process.returncode}",
                )
                return 1
            append_log(log_path, f"valid result label={label}")
            break

    state.update({"status": "complete", "completed_at": now()})
    write_json(state_path, state)
    append_log(log_path, "all jobs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
