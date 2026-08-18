"""Run the four recovery policies over several request-context sizes.

This driver intentionally invokes the single-case harness in a fresh process for
each cell.  That keeps GPU/container state isolated and makes a matrix cell a
reproducible experiment rather than a synthetic calculation.  The no-recovery
case is considered a successful experiment when the request fails as specified;
the harness reports that distinction through ``status`` and ``outcome``.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


MODES = ("no_recovery", "rerouting", "reparallelization", "modified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[64, 240, 480])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.05)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.08)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already completed cell JSON files at the output prefix.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def numeric_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def numeric_median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def run_cell(args: argparse.Namespace, mode: str, prompt_tokens: int,
             repeat: int, cell_path: Path) -> dict:
    harness = Path(__file__).with_name("run_four_version_recovery_smoke.py")
    command = [
        sys.executable,
        str(harness),
        "--model", args.model,
        "--mode", mode,
        "--trace", args.trace,
        "--gpus", *[str(gpu) for gpu in args.gpus],
        "--prompt-tokens", str(prompt_tokens),
        "--max-model-len", str(args.max_model_len),
        "--trace-speedup", str(args.trace_speedup),
        "--token-delay-s", str(args.token_delay_s),
        "--cpu-offload-gb", str(args.cpu_offload_gb),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--timeout-s", str(args.timeout_s),
        "--output", str(cell_path),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
    except OSError as exc:
        return {
            "status": "failed", "mode": mode, "prompt_tokens": prompt_tokens,
            "repeat": repeat, "elapsed_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    log_path = cell_path.with_suffix(".log")
    log_path.write_text(result.stdout, encoding="utf-8")
    common = {
        "mode": mode, "prompt_tokens": prompt_tokens, "repeat": repeat,
        "elapsed_s": round(time.monotonic() - started, 3),
        "returncode": result.returncode, "log": str(log_path),
    }
    if result.returncode != 0:
        return {"status": "failed", **common}
    try:
        report = json.loads(cell_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"status": "failed", **common, "error": str(exc)}
    return {
        "status": "passed", **common,
        "outcome": report.get("outcome"),
        "expected_outcome": report.get("expected_outcome"),
        "recovery": report.get("recovery", {}),
        "metrics": report.get("metrics", {}),
        "source_blocks": report.get("source_blocks", 0),
        "source_computed_tokens": report.get("source_computed_tokens", 0),
        "source_config": report.get("source_config", {}),
        "report": str(cell_path),
    }


def load_existing_cell(mode: str, prompt_tokens: int, repeat: int,
                       cell_path: Path) -> dict | None:
    """Reconstruct a matrix cell from a completed smoke report."""
    try:
        report = json.loads(cell_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if report.get("status") != "passed":
        return None
    return {
        "status": "passed",
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "elapsed_s": report.get("elapsed_s"),
        "returncode": 0,
        "log": str(cell_path.with_suffix(".log")),
        "outcome": report.get("outcome"),
        "expected_outcome": report.get("expected_outcome"),
        "recovery": report.get("recovery", {}),
        "metrics": report.get("metrics", {}),
        "source_blocks": report.get("source_blocks", 0),
        "source_computed_tokens": report.get("source_computed_tokens", 0),
        "source_config": report.get("source_config", {}),
        "report": str(cell_path),
    }


def summarize(cells: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    keys = (
        "recovery_s", "target_recovery_s", "target_continuation_s",
        "recomputed_tokens", "restored_blocks", "source_blocks",
        "source_computed_tokens", "p99_latency_s",
        "effective_throughput_tokens_s", "generated_tokens",
    )
    for mode in MODES:
        for prompt_tokens in sorted({int(c["prompt_tokens"]) for c in cells}):
            selected = [
                c for c in cells
                if c["mode"] == mode and int(c["prompt_tokens"]) == prompt_tokens
            ]
            passed = [c for c in selected if c["status"] == "passed"]
            summary = {
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "runs": len(selected),
                "passed": len(passed),
                "failed_runs": len(selected) - len(passed),
                "continued": sum(c.get("outcome") == "continued" for c in passed),
                "failed_request": sum(c.get("outcome") == "failed" for c in passed),
            }
            for key in keys:
                values = [
                    float(c.get("recovery", {}).get(key))
                    for c in passed
                    if isinstance(c.get("recovery", {}).get(key), (int, float))
                ]
                summary[f"{key}_mean"] = numeric_mean(values)
                summary[f"{key}_median"] = numeric_median(values)
            for key in ("engine_created", "placement_changed", "target_preexisting",
                        "restore_success", "target_continued"):
                values = [c.get("recovery", {}).get(key) for c in passed
                          if isinstance(c.get("recovery", {}).get(key), bool)]
                summary[f"{key}_count"] = sum(values)
            metric_values = [
                c.get("metrics", {}) for c in passed
                if isinstance(c.get("metrics"), dict)
            ]
            for key in (
                "recovery_time_s", "p99_latency_s",
                "effective_throughput_tokens_s", "success_rate",
            ):
                values = [
                    float(metric.get(key))
                    for metric in metric_values
                    if isinstance(metric.get(key), (int, float))
                ]
                summary[f"{key}_mean"] = numeric_mean(values)
                summary[f"{key}_median"] = numeric_median(values)
            summary["recovery_data"] = [
                metric.get("recovery_data", {}) for metric in metric_values
            ]
            summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise SystemExit("--gpus must contain four distinct GPU indices")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if any(prompt < 1 or prompt > args.max_model_len for prompt in args.prompt_tokens):
        raise SystemExit("every prompt length must fit --max-model-len")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    for prompt_tokens in args.prompt_tokens:
        for mode in args.modes:
            for repeat in range(1, args.repeats + 1):
                cell_path = output.with_name(
                    f"{output.stem}.{mode}.p{prompt_tokens}.r{repeat}.json"
                )
                if args.resume:
                    existing = load_existing_cell(
                        mode, prompt_tokens, repeat, cell_path
                    )
                    if existing is not None:
                        cells.append(existing)
                        print(json.dumps({
                            "event": "resume",
                            "mode": mode,
                            "prompt_tokens": prompt_tokens,
                            "repeat": repeat,
                            "report": str(cell_path),
                        }, sort_keys=True), flush=True)
                        continue
                print(json.dumps({"event": "start", "mode": mode,
                                  "prompt_tokens": prompt_tokens,
                                  "repeat": repeat}, sort_keys=True), flush=True)
                cell = run_cell(args, mode, prompt_tokens, repeat, cell_path)
                cells.append(cell)
                print(json.dumps({"event": "complete", **cell}, sort_keys=True),
                      flush=True)
    report = {
        "status": "passed" if all(c["status"] == "passed" for c in cells) else "partial",
        "model": args.model,
        "trace": args.trace,
        "gpus": args.gpus,
        "modes": args.modes,
        "prompt_tokens": args.prompt_tokens,
        "repeats": args.repeats,
        "max_model_len": args.max_model_len,
        "cells": cells,
        "by_mode_and_prompt": summarize(cells),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "summary", "output": str(output),
                      "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
