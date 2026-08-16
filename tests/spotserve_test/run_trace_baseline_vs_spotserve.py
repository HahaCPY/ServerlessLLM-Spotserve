"""Run one full-recompute baseline and one NIXL restore run on one trace.

The comparison deliberately uses the same model, trace and request payload in
both runs.  The baseline aborts the source request and sends the complete
computed context to the target without a KV attach.  The SpotServe run uses
the source runtime state and NIXL restore path.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_four_container_fleet_churn_smoke import load_fleet_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.1)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def extract_summary(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise RuntimeError("runner did not emit a JSON summary")


def run_case(args: argparse.Namespace, mode: str, log_path: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_four_container_fleet_churn_smoke.py")),
        "--model",
        args.model,
        "--trace",
        args.trace,
        "--trace-speedup",
        str(args.trace_speedup),
        "--token-delay-s",
        str(args.token_delay_s),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--timeout-s",
        str(args.timeout_s),
        "--recovery-mode",
        mode,
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    started = time.monotonic()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{mode} case failed with exit={result.returncode}; see {log_path}"
        )
    summary = extract_summary(result.stdout)
    summary["harness_elapsed_s"] = round(time.monotonic() - started, 3)
    return summary


def main() -> None:
    args = parse_args()
    trace_events = load_fleet_trace(args.trace)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = run_case(
        args,
        "recompute",
        output_path.with_name(output_path.stem + ".baseline.log"),
    )
    spotserve = run_case(
        args,
        "nixl",
        output_path.with_name(output_path.stem + ".spotserve.log"),
    )
    baseline_migration = baseline.get("migration", {})
    spotserve_migration = spotserve.get("migration", {})
    baseline_phases = baseline_migration.get("phase_elapsed_s", {})
    spotserve_phases = spotserve_migration.get("phase_elapsed_s", {})
    comparison = {
        "trace_event_count": len(trace_events),
        "trace_states": sorted({event["event"] for event in trace_events}),
        "baseline_recomputed_tokens": baseline_migration.get(
            "recomputed_tokens", 0
        ),
        "spotserve_recomputed_tokens": spotserve_migration.get(
            "recomputed_tokens", 0
        ),
        "recomputed_tokens_saved": (
            baseline_migration.get("recomputed_tokens", 0)
            - spotserve_migration.get("recomputed_tokens", 0)
        ),
        "baseline_restored_blocks": baseline_migration.get("restored_blocks", 0),
        "spotserve_restored_blocks": spotserve_migration.get("restored_blocks", 0),
        "baseline_migration_elapsed_s": baseline_migration.get(
            "migration_elapsed_s"
        ),
        "spotserve_migration_elapsed_s": spotserve_migration.get(
            "migration_elapsed_s"
        ),
        "baseline_target_recovery_s": baseline_phases.get(
            "target_recovery_s"
        ),
        "spotserve_target_recovery_s": spotserve_phases.get(
            "target_recovery_s"
        ),
        "target_recovery_elapsed_delta_s": round(
            float(baseline_phases.get("target_recovery_s", 0.0))
            - float(spotserve_phases.get("target_recovery_s", 0.0)),
            3,
        ),
        "migration_elapsed_delta_s": round(
            float(baseline_migration.get("migration_elapsed_s", 0.0))
            - float(spotserve_migration.get("migration_elapsed_s", 0.0)),
            3,
        ),
        "baseline_restore_successes": baseline.get(
            "state_restore_successes_total", 0
        ),
        "spotserve_restore_successes": spotserve.get(
            "state_restore_successes_total", 0
        ),
        "baseline_fallbacks": baseline.get("state_restore_fallback_count", 0),
        "spotserve_fallbacks": spotserve.get("state_restore_fallback_count", 0),
        "both_requests_continued": bool(
            baseline_migration.get("target_continued_after_source_stop")
            and spotserve_migration.get("target_continued_after_source_stop")
        ),
    }
    report = {
        "status": "passed",
        "model": args.model,
        "trace": args.trace,
        "baseline": baseline,
        "spotserve": spotserve,
        "comparison": comparison,
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output": str(output_path), "comparison": comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
