"""Run repeated baseline vs. modified (NIXL KV restore) comparisons.

Each matrix cell uses the same model, focused preemption trace, GPUs and
request payload.  The baseline disables KV restore and recomputes the full
source context; the modified path restores the exported KV state through
NIXL.  Results are kept per cell and summarized with mean/median values.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[64, 240, 480]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.05)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def median(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def run_cell(args: argparse.Namespace, prompt_tokens: int, repeat: int, path: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_trace_baseline_vs_spotserve.py")),
        "--model",
        args.model,
        "--trace",
        args.trace,
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--trace-speedup",
        str(args.trace_speedup),
        "--token-delay-s",
        str(args.token_delay_s),
        "--prompt-tokens",
        str(prompt_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--timeout-s",
        str(args.timeout_s),
        "--output",
        str(path),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    path.with_suffix(".runner.log").write_text(result.stdout, encoding="utf-8")
    elapsed = round(time.monotonic() - started, 3)
    if result.returncode != 0:
        return {
            "status": "failed",
            "prompt_tokens": prompt_tokens,
            "repeat": repeat,
            "elapsed_s": elapsed,
            "returncode": result.returncode,
            "log": str(path.with_suffix(".runner.log")),
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "prompt_tokens": prompt_tokens,
            "repeat": repeat,
            "elapsed_s": elapsed,
            "error": f"missing or invalid cell report: {exc}",
        }
    comparison = report["comparison"]
    return {
        "status": "passed",
        "prompt_tokens": prompt_tokens,
        "repeat": repeat,
        "elapsed_s": elapsed,
        "comparison": comparison,
    }


def summarize(cells: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for prompt_tokens in sorted({cell["prompt_tokens"] for cell in cells}):
        passed = [
            cell
            for cell in cells
            if cell["prompt_tokens"] == prompt_tokens and cell["status"] == "passed"
        ]
        if not passed:
            summaries.append(
                {"prompt_tokens": prompt_tokens, "passed": 0, "failed": 1}
            )
            continue
        metrics: dict[str, list[float]] = {}
        for cell in passed:
            for key, value in cell["comparison"].items():
                if isinstance(value, (int, float)):
                    metrics.setdefault(key, []).append(float(value))
        summary = {
            "prompt_tokens": prompt_tokens,
            "passed": len(passed),
            "failed": sum(
                1
                for cell in cells
                if cell["prompt_tokens"] == prompt_tokens
                and cell["status"] != "passed"
            ),
        }
        for key, values in metrics.items():
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_median"] = median(values)
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if any(prompt < 1 for prompt in args.prompt_tokens):
        raise SystemExit("--prompt-tokens values must be positive")
    if any(prompt > args.max_model_len for prompt in args.prompt_tokens):
        raise SystemExit("--max-model-len must cover every prompt length")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    for prompt_tokens in args.prompt_tokens:
        for repeat in range(1, args.repeats + 1):
            cell_path = output.with_name(
                f"{output.stem}.p{prompt_tokens}.r{repeat}.json"
            )
            print(
                json.dumps(
                    {
                        "event": "start",
                        "prompt_tokens": prompt_tokens,
                        "repeat": repeat,
                        "output": str(cell_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            cell = run_cell(args, prompt_tokens, repeat, cell_path)
            cells.append(cell)
            print(json.dumps({"event": "complete", **cell}, sort_keys=True), flush=True)
    report = {
        "status": "passed" if all(cell["status"] == "passed" for cell in cells) else "partial",
        "model": args.model,
        "trace": args.trace,
        "gpus": args.gpus,
        "prompt_tokens": args.prompt_tokens,
        "repeats": args.repeats,
        "max_model_len": args.max_model_len,
        "cells": cells,
        "by_prompt_tokens": summarize(cells),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "summary", "output": str(output), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
