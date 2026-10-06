#!/usr/bin/env python3
"""Run the complete fail-closed F1/F2 pipeline on an eight-GPU K8s pool.

This is the single user-facing entry point for the two-workspace deployment.
It does not create clusters, mutate Kubernetes objects, or delete checkpoint
files. Run it in the workspace-A head pod after all eight workers have joined
the same Ray cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts/run_k8s_moe_f1_f2.py"
POOL_VERIFIER = REPO_ROOT / "scripts/verify_k8s_two_workspace_pool.py"
DEFAULT_CONFIG = (
    REPO_ROOT / "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json"
)


class PipelineError(RuntimeError):
    """A pipeline gate failed and later GPU work must not start."""


def taipei_now() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_from_repo(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def ensure_safe_output(config_path: Path, output: Path) -> dict[str, Any]:
    """Reject result locations that could overlap the read-only checkpoint."""
    config = read_json(config_path)
    model_path = Path(str(config["model"]["path"])).resolve()
    output = output.resolve()
    if output == model_path or output in model_path.parents or model_path in output.parents:
        raise PipelineError(
            f"results path {output} must not overlap model path {model_path}"
        )
    return config


def driver_command(args: argparse.Namespace, phase: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(DRIVER),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
        "--endpoint",
        args.endpoint,
        "--ray-address",
        args.ray_address,
        "--ray-namespace",
        args.ray_namespace,
        "--phase",
        phase,
        "--repeats",
        str(args.repeats),
    ]


def run_stage(
    pipeline: dict[str, Any],
    summary_path: Path,
    log_dir: Path,
    name: str,
    command: Sequence[str],
) -> None:
    log_path = log_dir / f"{name}.log"
    stage = {
        "name": name,
        "status": "running",
        "started_at": taipei_now(),
        "command": list(command),
        "log": str(log_path),
    }
    pipeline["stages"].append(stage)
    write_json(summary_path, pipeline)
    print(f"[pipeline] {name}: {' '.join(command)}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            except KeyboardInterrupt:
                process.terminate()
                process.wait(timeout=30)
                raise
            return_code = process.wait()
    except BaseException as exc:
        stage.update(
            status="blocked",
            completed_at=taipei_now(),
            error=str(exc) or type(exc).__name__,
        )
        write_json(summary_path, pipeline)
        raise
    stage.update(
        status="passed" if return_code == 0 else "blocked",
        completed_at=taipei_now(),
        return_code=return_code,
    )
    write_json(summary_path, pipeline)
    if return_code != 0:
        raise PipelineError(
            f"stage {name} returned {return_code}; inspect {log_path}"
        )


def record_reused(
    pipeline: dict[str, Any], summary_path: Path, name: str, reason: str
) -> None:
    pipeline["stages"].append(
        {
            "name": name,
            "status": "reused",
            "reason": reason,
            "completed_at": taipei_now(),
        }
    )
    write_json(summary_path, pipeline)
    print(f"[pipeline] {name}: reuse ({reason})", flush=True)


def render_summary(output: Path, pipeline: dict[str, Any]) -> None:
    rows = []
    for stage in pipeline.get("stages", []):
        rows.append(
            f"| {stage['name']} | {stage['status']} | "
            f"`{stage.get('log', '—')}` | "
            f"{stage.get('reason', stage.get('error', ''))} |"
        )
    report = [
        "# K8s 雙 workspace F1 / F2 執行摘要",
        "",
        f"- 狀態：**{pipeline['status']}**",
        f"- 更新時間：`{taipei_now()}`",
        f"- 正式結果：`{output / 'results.md'}`",
        f"- 原始 ledger：`{output / 'run-ledger.json'}`",
        f"- 主狀態：`{output / 'state.json'}`",
        "",
        "| 階段 | 狀態 | Log | 說明 |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        "## 固定實驗定義",
        "",
        "- F1：Rerouting、Reparallelization、Original SpotServe、MoE-SpotServe。",
        "- F2：Original SpotServe、reparallelization-only、migration-only、full MoE-SpotServe。",
        "- Workload：Qwen1.5-MoE-A2.7B、8192 input tokens、2048 output tokens，目標在 512 output tokens 附近 preempt。",
        f"- 正式矩陣：每個 treatment {pipeline['formal_repeats']} 次；順序依固定種子打亂，完成的 run 可安全續跑。",
        "",
        "## 模型資料安全",
        "",
        "此 pipeline 不刪除 checkpoint 檔案。Kubernetes manifest 將 `/models` 以唯讀方式掛載；benchmark 中的 model delete 僅移除 ServerlessLLM 部署／actor，不會刪除 PVC 內的權重。",
        "",
    ]
    if pipeline.get("error"):
        report.extend(["## 阻擋原因", "", str(pipeline["error"]), ""])
    (output / "pipeline-summary.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-a-namespace", required=True)
    parser.add_argument("--workspace-b-namespace", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8343/v1/chat/completions"
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    parser.add_argument(
        "--repeats",
        type=positive_int,
        required=True,
        help="Number of formal runs for every F1/F2 configuration.",
    )
    parser.add_argument(
        "--force-smoke",
        action="store_true",
        help="Rerun a prior failed smoke only after a relevant change.",
    )
    parser.add_argument(
        "--force-core-probe",
        action="store_true",
        help="Rerun a prior failed KV core probe only after a relevant change.",
    )
    parser.add_argument(
        "--force-profile",
        action="store_true",
        help="Discard matching cached profiles and measure them again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the formal config and write a plan without contacting Ray.",
    )
    args = parser.parse_args()
    args.config = resolve_from_repo(args.config)
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.workspace_a_namespace == args.workspace_b_namespace:
        print("workspace A and B namespaces must differ", file=sys.stderr)
        return 2
    if not DRIVER.is_file() or not POOL_VERIFIER.is_file():
        print("experiment runtime files are missing from this image", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2
    try:
        config = ensure_safe_output(args.config, args.output_dir)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pipeline-summary.json"
    log_dir = args.output_dir / "pipeline-logs"
    pipeline: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": taipei_now(),
        "config": str(args.config),
        "model_path": str(config["model"]["path"]),
        "output_dir": str(args.output_dir),
        "workspace_a_namespace": args.workspace_a_namespace,
        "workspace_b_namespace": args.workspace_b_namespace,
        "formal_repeats": int(args.repeats),
        "stages": [],
        "checkpoint_files_deleted": False,
    }
    write_json(summary_path, pipeline)
    code = 2
    try:
        if args.dry_run:
            command = driver_command(args, "all") + ["--dry-run"]
            run_stage(pipeline, summary_path, log_dir, "config-dry-run", command)
            pipeline["status"] = "dry_run_passed"
            code = 0
            return code

        pool_output = args.output_dir / "preflight/two-workspace-pool.json"
        run_stage(
            pipeline,
            summary_path,
            log_dir,
            "pool-verification",
            [
                sys.executable,
                "-u",
                str(POOL_VERIFIER),
                "--workspace-a-namespace",
                args.workspace_a_namespace,
                "--workspace-b-namespace",
                args.workspace_b_namespace,
                "--ray-address",
                args.ray_address,
                "--ray-namespace",
                args.ray_namespace,
                "--output",
                str(pool_output),
            ],
        )

        smoke_path = args.output_dir / "smoke-results.json"
        smoke_status = (
            read_json(smoke_path).get("status") if smoke_path.is_file() else None
        )
        if smoke_status == "passed" and not args.force_smoke:
            record_reused(
                pipeline, summary_path, "smoke", "existing passed evidence"
            )
        else:
            if smoke_path.is_file() and not args.force_smoke:
                raise PipelineError(
                    f"prior smoke status is {smoke_status!r}; inspect {smoke_path} "
                    "and rerun with --force-smoke only after a relevant change"
                )
            command = driver_command(args, "smoke")
            if args.force_smoke:
                command.append("--force-smoke")
            run_stage(pipeline, summary_path, log_dir, "smoke", command)

        profile_command = driver_command(args, "profile")
        if args.force_profile:
            profile_command.append("--force-profile")
        run_stage(
            pipeline, summary_path, log_dir, "candidate-profiles", profile_command
        )

        core_path = args.output_dir / "core-probe-results.json"
        core_status = (
            read_json(core_path).get("status") if core_path.is_file() else None
        )
        if core_status == "live_kv_core_verified" and not args.force_core_probe:
            record_reused(
                pipeline,
                summary_path,
                "live-kv-core-probe",
                "existing live_kv_core_verified evidence",
            )
        else:
            if core_path.is_file() and not args.force_core_probe:
                raise PipelineError(
                    f"prior core probe status is {core_status!r}; inspect {core_path} "
                    "and rerun with --force-core-probe only after a relevant change"
                )
            command = driver_command(args, "core-probe")
            if args.force_core_probe:
                command.append("--force-core-probe")
            run_stage(
                pipeline, summary_path, log_dir, "live-kv-core-probe", command
            )

        run_stage(
            pipeline,
            summary_path,
            log_dir,
            "formal-f1-f2",
            driver_command(args, "formal"),
        )
        run_stage(
            pipeline,
            summary_path,
            log_dir,
            "final-report",
            driver_command(args, "report"),
        )
        state_path = args.output_dir / "state.json"
        pipeline["formal_state"] = (
            read_json(state_path) if state_path.is_file() else {}
        )
        pipeline["results_markdown"] = str(args.output_dir / "results.md")
        pipeline["run_ledger"] = str(args.output_dir / "run-ledger.json")
        pipeline["status"] = "passed"
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        pipeline["status"] = "blocked"
        pipeline["error"] = str(exc) or type(exc).__name__
        print(f"[pipeline] blocked: {pipeline['error']}", file=sys.stderr)
    finally:
        pipeline["completed_at"] = taipei_now()
        write_json(summary_path, pipeline)
        render_summary(args.output_dir, pipeline)
        print(f"[pipeline] summary: {args.output_dir / 'pipeline-summary.md'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
