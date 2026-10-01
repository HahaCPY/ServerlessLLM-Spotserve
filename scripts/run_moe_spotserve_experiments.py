"""Run the controlled MoE x SpotServe experiment matrix.

The GPU work is delegated to the existing cross-container recovery harness.
This driver adds repetition, retry bookkeeping, immutable raw outputs, and a
single aggregate JSON file; it does not modify the serving implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "tests/spotserve_test/run_tiny_batch_recovery.py"
DEFAULT_MODES = (
    "rerouting",
    "reparallelization",
    "spotserve",
    "moe_spotserve",
    "original_migration",
    "moe_migration",
    "original_reparallelization",
    "moe_reparallelization",
)
MIGRATION_MODES = {
    "spotserve",
    "moe_spotserve",
    "original_migration",
    "moe_migration",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--source-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--target-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--preempt-after-new-tokens", type=int, default=256)
    parser.add_argument("--preempt-max-overshoot", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-attempts-per-run", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260912)
    parser.add_argument(
        "--route-profile",
        required=True,
        help="Audited Qwen live-routing canary result used by NIXL modes.",
    )
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES,
                        default=list(DEFAULT_MODES))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_valid_result(path: Path, mode: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    recovery = payload.get("recovery", {})
    requested = recovery.get("target_requested_tokens", {})
    outputs = recovery.get("target_outputs", {})
    outputs_complete = bool(requested) and set(outputs) == set(requested) and all(
        int(outputs[request_id].get("generated_tokens", 0) or 0)
        == int(expected)
        and outputs[request_id].get("finish_reason") == "length"
        for request_id, expected in requested.items()
    )
    sequence_checks = recovery.get("sequence_checks", {})
    sequences_equal = (
        bool(sequence_checks)
        and set(sequence_checks) == set(requested)
        and all(bool(row.get("equal")) for row in sequence_checks.values())
        and bool(recovery.get("all_sequences_equal_reference"))
    )
    completed = recovery.get("source_completed_tokens", {})
    boundary_valid = bool(completed) and all(
        args_value <= int(value) <= args_value + args_overshoot
        for value in completed.values()
        for args_value, args_overshoot in [(
            int(payload.get("preempt_after_new_tokens", 0)),
            int(payload.get("preempt_max_overshoot", 0)),
        )]
    )
    routing_valid = int(
        recovery.get("moe_route_histogram_available_count", 0) or 0
    ) == int(payload.get("request_count", 0) or 0)
    if (
        payload.get("status") != "passed"
        or payload.get("mode") != mode
        or payload.get("outcome") != "continued"
        or float(recovery.get("success_rate", 0.0) or 0.0) != 1.0
        or int(recovery.get("generated_tokens", 0) or 0) <= 0
        or not outputs_complete
        or not sequences_equal
        or not boundary_valid
        or not routing_valid
    ):
        return None
    return payload


def command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    result = [
        sys.executable,
        str(HARNESS),
        "--model", args.model,
        "--mode", mode,
        "--trace", args.trace,
        "--gpus", *map(str, args.gpus),
        "--source-gpus", *map(str, args.source_gpus),
        "--target-gpus", *map(str, args.target_gpus),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--request-count", str(args.request_count),
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-new-tokens", str(args.max_new_tokens),
        "--preempt-after-new-tokens", str(args.preempt_after_new_tokens),
        "--preempt-max-overshoot", str(args.preempt_max_overshoot),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--cpu-offload-gb", str(args.cpu_offload_gb),
        "--trace-speedup", str(args.trace_speedup),
        "--token-delay-s", str(args.token_delay_s),
        "--timeout-s", str(args.timeout_s),
        "--host-network",
        "--prestart-target",
        "--require-moe-routing",
        "--output", str(output),
    ]
    if mode in MIGRATION_MODES:
        result.extend(["--route-profile", args.route_profile])
    return result


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [
        float(row.get("recovery", {}).get(key))
        for row in rows
        if isinstance(row.get("recovery", {}).get(key), (int, float))
    ]
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    keys = (
        "recovery_s",
        "recovery_wall_s",
        "target_startup_s",
        "target_recovery_s",
        "avg_latency_s",
        "p99_latency_s",
        "downtime_s",
        "effective_throughput_tokens_s",
        "migration_operation_s",
        "reparallelization_s",
        "state_export_s",
        "state_restore_stage_s",
        "migration_planner_s",
        "reparallelization_planner_s",
        "success_rate",
        "recomputed_tokens",
        "restored_blocks",
        "generated_tokens",
    )
    return {
        mode: {
            "valid_runs": len(mode_rows),
            **{key: numeric_summary(mode_rows, key) for key in keys},
            "elapsed_s": {
                "mean": statistics.mean(float(row["elapsed_s"]) for row in mode_rows),
                "stdev": statistics.stdev(
                    float(row["elapsed_s"]) for row in mode_rows
                ) if len(mode_rows) > 1 else 0.0,
            },
        }
        for mode, mode_rows in by_mode.items()
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    valid_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []

    # Block by repeat so cache/thermal drift is distributed across policies
    # instead of running all three samples of one policy back-to-back.
    for repeat in range(1, args.repeats + 1):
        repeat_modes = list(args.modes)
        random.Random(args.order_seed + repeat).shuffle(repeat_modes)
        schedule.append({"repeat": repeat, "modes": repeat_modes})
        write_json(output_dir / "schedule.json", schedule)
        for mode in repeat_modes:
            mode_dir = output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            result_path = mode_dir / f"run-{repeat}.json"
            existing = read_valid_result(result_path, mode) if args.resume else None
            if existing is not None:
                valid_rows.append(existing)
                print(f"[matrix] resume mode={mode} run={repeat}", flush=True)
                continue
            for attempt in range(1, args.max_attempts_per_run + 1):
                attempt_path = mode_dir / f"run-{repeat}.attempt-{attempt}.json"
                log_path = mode_dir / f"run-{repeat}.attempt-{attempt}.log"
                started = time.monotonic()
                process = subprocess.run(
                    command(args, mode, attempt_path),
                    cwd=REPO_ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_path.write_text(process.stdout, encoding="utf-8")
                candidate = read_valid_result(attempt_path, mode)
                attempt_row = {
                    "mode": mode,
                    "run": repeat,
                    "attempt": attempt,
                    "returncode": process.returncode,
                    "elapsed_s": time.monotonic() - started,
                    "valid": candidate is not None,
                    "result": str(attempt_path),
                    "log": str(log_path),
                }
                attempts.append(attempt_row)
                write_json(output_dir / "attempts.json", attempts)
                print(
                    f"[matrix] mode={mode} run={repeat} attempt={attempt} "
                    f"valid={candidate is not None} rc={process.returncode}",
                    flush=True,
                )
                if candidate is None:
                    continue
                write_json(result_path, candidate)
                valid_rows.append(candidate)
                break
            else:
                raise RuntimeError(
                    f"no valid result for mode={mode} run={repeat}; "
                    f"see {mode_dir}"
                )
            write_json(output_dir / "raw_results.json", valid_rows)
            write_json(output_dir / "summary.json", summarize(valid_rows))

    report = {
        "setup": vars(args),
        "valid_result_count": len(valid_rows),
        "expected_result_count": len(args.modes) * args.repeats,
        "raw_results": valid_rows,
        "summary": summarize(valid_rows),
        "attempts": attempts,
        "schedule": schedule,
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps({
        "status": "passed",
        "valid_result_count": len(valid_rows),
        "report": str(output_dir / "report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
