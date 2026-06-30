import argparse
import asyncio
import csv
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, TextIO
from urllib import error, request


class TraceProcess(NamedTuple):
    process: subprocess.Popen
    log_file: TextIO
    log_path: Path


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Benchmark config is not JSON and PyYAML is not installed"
            ) from exc
        return yaml.safe_load(text)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return sorted(rows, key=lambda row: float(row.get("time", 0.0)))


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_reports(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[2]
    analyzer = load_module_from_path(
        "spotserve_benchmark_analyzer",
        repo_root / "scripts" / "analyze_spotserve_benchmark.py",
    )
    plotter = load_module_from_path(
        "spotserve_benchmark_plotter",
        repo_root / "scripts" / "plot_spotserve_benchmark.py",
    )

    summaries = []
    for run_dir in run_dirs:
        summary = analyzer.analyze_run(run_dir)
        plotter.render_report(run_dir)
        summaries.append(summary)
    return summaries


def write_combined_summary(output_root: Path, summaries: List[Dict[str, Any]]):
    if not summaries:
        return
    summary_path = output_root / "latest_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_root / "latest_summary.csv"
    fieldnames = list(summaries[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def post_json(
    endpoint: str, payload: Dict[str, Any], timeout_s: float
) -> Dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.time()
    try:
        with request.urlopen(http_request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body)
            success = "error" not in result
    except error.HTTPError as exc:
        result = {"error": exc.read().decode("utf-8")}
        success = False
    except Exception as exc:
        result = {"error": str(exc)}
        success = False

    return {
        "type": "request",
        "request_id": payload.get("request_id"),
        "model": payload.get("model"),
        "success": success,
        "latency_ms": (time.time() - started_at) * 1000,
        "response": result,
    }


def get_json(endpoint: str, timeout_s: float) -> Dict[str, Any]:
    with request.urlopen(endpoint, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def base_url_from_chat_endpoint(endpoint: str) -> str:
    suffix = "/v1/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)]
    return endpoint.rstrip("/")


def check_endpoint_ready(endpoint: str, model_names: List[str], timeout_s: float):
    base_url = base_url_from_chat_endpoint(endpoint)
    try:
        health = get_json(f"{base_url}/health", timeout_s)
    except Exception as exc:
        raise RuntimeError(
            f"Endpoint health check failed for {base_url}/health: {exc}"
        ) from exc
    if health.get("status") != "ok":
        raise RuntimeError(f"Endpoint is not healthy: {health}")

    try:
        models_response = get_json(f"{base_url}/v1/models", timeout_s)
    except Exception as exc:
        raise RuntimeError(
            f"Model list check failed for {base_url}/v1/models: {exc}"
        ) from exc

    deployed_models = {
        model.get("id") for model in models_response.get("models", [])
    }
    missing_models = sorted(set(model_names) - deployed_models)
    if missing_models:
        raise RuntimeError(
            "Benchmark models are not deployed: "
            f"{missing_models}. Deployed models: {sorted(deployed_models)}"
        )


async def send_workload(
    endpoint: str,
    model: str,
    workload: List[Dict[str, Any]],
    request_timeout_s: float,
) -> List[Dict[str, Any]]:
    started_at = time.monotonic()
    tasks = []

    async def send_at(row: Dict[str, Any]):
        arrival_time = float(row.get("time", 0.0))
        await asyncio.sleep(max(0.0, started_at + arrival_time - time.monotonic()))
        payload = {k: v for k, v in row.items() if k != "time"}
        payload["model"] = model
        sent_at = time.time()
        result = await asyncio.to_thread(
            post_json, endpoint, payload, request_timeout_s
        )
        result["arrival_time"] = arrival_time
        result["sent_at"] = sent_at
        result["completed_at"] = time.time()
        return result

    for row in workload:
        tasks.append(asyncio.create_task(send_at(row)))
    return await asyncio.gather(*tasks)


def start_trace_replayer(
    trace_path: Optional[str],
    speedup: float,
    log_path: Path,
    ray_address: str = "auto",
    ray_namespace: str = "sllm",
    controller_name: str = "controller",
) -> Optional[TraceProcess]:
    if not trace_path:
        return None
    if importlib.util.find_spec("ray") is None:
        message = (
            "Ray is not installed in the Python environment running this "
            "benchmark; skipping spot trace replay. Install ray, run the "
            "benchmark from an environment with ray, or pass --skip-trace."
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        print(f"[benchmark trace warning] {message}", file=sys.stderr)
        return None

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sllm.spot.preemption_simulator",
            "--trace",
            trace_path,
            "--speedup",
            str(speedup),
            "--ray-address",
            ray_address,
            "--ray-namespace",
            ray_namespace,
            "--controller-name",
            controller_name,
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return TraceProcess(process=process, log_file=log_file, log_path=log_path)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")


async def run_one(
    run_config: Dict[str, Any],
    endpoint: str,
    output_root: Path,
    speedup: float,
    request_timeout_s: float,
    skip_trace: bool = False,
    ray_address: str = "auto",
    ray_namespace: str = "sllm",
    controller_name: str = "controller",
) -> Path:
    run_id = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_"
        f"{run_config['name']}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_commit": git_commit(),
        "started_at": datetime.now().isoformat(),
        "endpoint": endpoint,
        **run_config,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    workload = load_jsonl(Path(run_config["workload"]))
    trace_process = None
    if not skip_trace:
        trace_process = start_trace_replayer(
            run_config.get("trace"),
            speedup,
            run_dir / "trace_replayer.log",
            ray_address=ray_address,
            ray_namespace=ray_namespace,
            controller_name=controller_name,
        )
    try:
        rows = await send_workload(
            endpoint, run_config["model"], workload, request_timeout_s
        )
    finally:
        if trace_process is not None:
            try:
                trace_process.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                trace_process.process.terminate()
                trace_process.process.wait(timeout=5)
            finally:
                trace_process.log_file.close()
            if trace_process.process.returncode not in (0, None):
                print(
                    "[benchmark trace warning] Trace replayer exited with "
                    f"code {trace_process.process.returncode}; see "
                    f"{trace_process.log_path}",
                    file=sys.stderr,
                )

    for row in rows:
        row["policy"] = run_config.get("policy", "none")
        row["benchmark_run"] = run_config["name"]
    write_jsonl(run_dir / "raw_requests.jsonl", rows)
    return run_dir


async def main_async(args):
    config = load_config(Path(args.config))
    output_root = Path(config.get("output_dir", "results/spotserve"))
    endpoint = args.endpoint or config["endpoint"]
    check_endpoint_ready(
        endpoint,
        [run_config["model"] for run_config in config["runs"]],
        args.request_timeout,
    )

    produced_runs = []
    for run_config in config["runs"]:
        produced_runs.append(
            await run_one(
                run_config,
                endpoint,
                output_root,
                args.speedup,
                args.request_timeout,
                skip_trace=args.skip_trace,
                ray_address=args.ray_address,
                ray_namespace=args.ray_namespace,
                controller_name=args.controller_name,
            )
        )

    summaries = []
    if not args.no_report:
        try:
            summaries = generate_reports(produced_runs)
            write_combined_summary(output_root, summaries)
        except Exception as exc:
            print(f"[benchmark report warning] {exc}", file=sys.stderr)

    print("Produced benchmark runs:")
    for run_dir in produced_runs:
        print(run_dir)
        if not args.no_report:
            print(f"  report: {run_dir / 'report.html'}")

    if summaries:
        print("\nBenchmark summary:")
        for summary in summaries:
            recovery_suffix = ""
            if int(summary.get("router_metrics_rows", 0) or 0) > 0:
                recovery_suffix = (
                    f", failed_attempts="
                    f"{summary.get('failed_attempts_total', 0)}, "
                    f"retries={summary.get('retry_count_total', 0)}, "
                    f"recovered_tokens="
                    f"{summary.get('recovered_tokens_total', 0)}, "
                    f"fallbacks="
                    f"{summary.get('recovery_fallback_count', 0)}"
                )
            print(
                "  "
                f"{summary['run_name']}: "
                f"successes={summary['successes']}/{summary['requests']}, "
                f"success_rate={summary['success_rate']:.2%}, "
                f"p95={summary['latency_p95_ms']:.2f}ms"
                f"{recovery_suffix}"
            )
        if all(summary["successes"] == 0 for summary in summaries):
            print(
                "\nWarning: every benchmark request failed. "
                "Check the deployed model states before trusting these results."
            )


def main():
    parser = argparse.ArgumentParser(description="Run SpotServe benchmarks")
    parser.add_argument(
        "--config",
        default="benchmarks/spotserve/benchmark_matrix.yaml",
    )
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Do not start the Ray trace replayer subprocess",
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray address passed to the trace replayer when trace replay is enabled",
    )
    parser.add_argument(
        "--ray-namespace",
        default="sllm",
        help="Ray namespace used to look up the controller actor",
    )
    parser.add_argument(
        "--controller-name",
        default="controller",
        help="Ray actor name used by the trace replayer",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only write raw benchmark files; skip summary and HTML report generation",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except RuntimeError as exc:
        print(f"[benchmark setup error] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
