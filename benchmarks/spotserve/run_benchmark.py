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
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, TextIO
from urllib import error, request


class TraceProcess(NamedTuple):
    process: subprocess.Popen
    log_file: TextIO
    log_path: Path


class TraceTask(NamedTuple):
    task: asyncio.Task
    log_path: Path


class BenchmarkSpotEvent(NamedTuple):
    time: float
    event: str
    node_id: Optional[str]
    model_name: Optional[str]
    instance_id: Optional[str]
    instance_index: Optional[int]
    instance_selector: Optional[str]


BENCHMARK_ONLY_WORKLOAD_KEYS = {
    "time",
    "benchmark_phase",
    "phase",
}

SUPPORTED_SPOT_EVENTS = {"preempt", "recover", "dead"}


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


def load_spot_trace_events(
    path: Path,
    default_model_name: Optional[str] = None,
) -> List[BenchmarkSpotEvent]:
    rows = load_jsonl(path)
    events = []
    for row in rows:
        event = str(row.get("event", ""))
        if event not in SUPPORTED_SPOT_EVENTS:
            raise ValueError(
                f"{path}: unsupported spot event {event}; expected one of "
                f"{sorted(SUPPORTED_SPOT_EVENTS)}"
            )
        instance_index = row.get("instance_index")
        if instance_index is not None:
            instance_index = int(instance_index)
        instance_selector = row.get("instance_selector")
        if (
            row.get("node_id") is None
            and row.get("instance_id") is None
            and instance_index is None
            and instance_selector is None
        ):
            raise ValueError(
                f"{path}: spot event must target node_id, instance_id, "
                "instance_index, or instance_selector"
            )
        model_name = row.get("model_name") or default_model_name
        if (
            instance_index is not None or instance_selector is not None
        ) and model_name is None:
            raise ValueError(
                f"{path}: instance-selected spot events require model_name"
            )
        events.append(
            BenchmarkSpotEvent(
                time=float(row.get("time", 0.0)),
                event=event,
                node_id=row.get("node_id"),
                model_name=model_name,
                instance_id=row.get("instance_id"),
                instance_index=instance_index,
                instance_selector=instance_selector,
            )
        )
    return events


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
    fieldnames = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def build_comparisons(
    comparisons: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_name = {summary.get("run_name"): summary for summary in summaries}
    rows = []
    default_fields = [
        "success_rate",
        "latency_p95_ms",
        "throughput_req_s",
        "replanning_events",
        "replanning_execution_applied",
        "replanning_execution_failed",
        "phase_replan_window_success_rate",
        "phase_replan_window_latency_p95_ms",
        "phase_post_replan_success_rate",
        "phase_post_replan_latency_p95_ms",
        "phase_post_replan_throughput_req_s",
    ]
    for comparison in comparisons:
        baseline_name = comparison.get("baseline")
        candidate_name = comparison.get("candidate")
        baseline = by_name.get(baseline_name)
        candidate = by_name.get(candidate_name)
        if baseline is None or candidate is None:
            continue
        row = {
            "name": comparison.get(
                "name", f"{candidate_name}_vs_{baseline_name}"
            ),
            "baseline": baseline_name,
            "candidate": candidate_name,
        }
        for field in comparison.get("fields", default_fields):
            if field not in baseline or field not in candidate:
                continue
            baseline_value = baseline.get(field, 0.0) or 0.0
            candidate_value = candidate.get(field, 0.0) or 0.0
            if not isinstance(baseline_value, (int, float)) or not isinstance(
                candidate_value, (int, float)
            ):
                continue
            row[f"{field}_baseline"] = baseline_value
            row[f"{field}_candidate"] = candidate_value
            row[f"{field}_delta"] = candidate_value - baseline_value
            row[f"{field}_ratio"] = (
                candidate_value / baseline_value if baseline_value else 0.0
            )
        rows.append(row)
    return rows


def write_comparisons(
    output_root: Path, comparisons: List[Dict[str, Any]]
) -> None:
    if not comparisons:
        return
    (output_root / "latest_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True),
        encoding="utf-8",
    )


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


def deploy_config_over_http(
    endpoint: str,
    config_path: str,
    timeout_s: float,
    router_metrics_path: Optional[str] = None,
) -> None:
    base_url = base_url_from_chat_endpoint(endpoint)
    payload = load_config(Path(config_path))
    if router_metrics_path:
        payload.setdefault("router_config", {})["metrics_path"] = str(
            router_metrics_path
        )
    result = post_json(f"{base_url}/register", payload, timeout_s)
    if not result.get("success"):
        raise RuntimeError(
            f"Deploy failed for {config_path}: {result.get('response')}"
        )


def clear_metrics_file(path_value: Any) -> None:
    if not path_value:
        return
    path = Path(str(path_value))
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        print(
            f"[benchmark cleanup warning] Could not remove metrics file "
            f"{path}: {exc}",
            file=sys.stderr,
        )


async def apply_scheduler_config(
    scheduler_config: Optional[Mapping[str, Any]],
    replace: bool,
    ray_address: str,
    ray_namespace: str,
) -> Optional[Dict[str, Any]]:
    if scheduler_config is None:
        return None
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray is required to update the live scheduler config."
        ) from exc

    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            namespace=ray_namespace,
            ignore_reinit_error=True,
        )
    scheduler = ray.get_actor("model_loading_scheduler")
    result_ref = scheduler.update_scheduler_config.remote(
        dict(scheduler_config),
        replace=replace,
    )
    return await asyncio.to_thread(ray.get, result_ref)


def delete_model_over_http(
    endpoint: str,
    model_name: str,
    timeout_s: float,
    fail_on_error: bool = False,
) -> None:
    base_url = base_url_from_chat_endpoint(endpoint)
    result = post_json(
        f"{base_url}/delete", {"model": model_name}, timeout_s
    )
    if fail_on_error and not result.get("success"):
        raise RuntimeError(
            f"Delete failed for {model_name}: {result.get('response')}"
        )
    if not result.get("success"):
        print(
            "[benchmark cleanup warning] Delete failed for "
            f"{model_name}: {result.get('response')}",
            file=sys.stderr,
        )


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


def is_ready_instance_state(state: Dict[str, Any]) -> bool:
    return state.get("pool") == "ready" and state.get("state") == "ready"


def is_preempting_instance_state(state: Dict[str, Any]) -> bool:
    return state.get("state") == "preempting"


def instance_concurrency(state: Dict[str, Any]) -> int:
    try:
        return int(state.get("concurrency", 0) or 0)
    except (TypeError, ValueError):
        return 0


async def get_model_instance_states(
    model_name: str,
    ray_address: str,
    ray_namespace: str,
) -> Dict[str, Any]:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray is required to inspect model instances. Run this benchmark "
            "inside the SpotServe head environment or remove "
            "min_ready_instances/instance_index/instance_selector from the "
            "benchmark config."
        ) from exc

    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            namespace=ray_namespace,
            ignore_reinit_error=True,
        )
    router = ray.get_actor(model_name, namespace="models")
    return await asyncio.to_thread(
        ray.get, router.get_instance_states.remote()
    )


async def wait_for_ready_instances(
    model_name: str,
    min_ready_instances: int,
    timeout_s: float,
    ray_address: str,
    ray_namespace: str,
) -> None:
    if min_ready_instances <= 0:
        return
    deadline = time.monotonic() + timeout_s
    latest_states: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest_states = await get_model_instance_states(
            model_name, ray_address, ray_namespace
        )
        ready_instances = [
            instance_id
            for instance_id, state in latest_states.items()
            if is_ready_instance_state(state)
        ]
        if len(ready_instances) >= min_ready_instances:
            return
        await asyncio.sleep(2.0)

    raise RuntimeError(
        f"Timed out waiting for {min_ready_instances} ready instances for "
        f"{model_name}; latest states={latest_states}"
    )


async def resolve_trace_instance_id(
    event: Any,
    ray_address: str,
    ray_namespace: str,
) -> Optional[str]:
    if event.instance_id is not None:
        return event.instance_id

    selector = getattr(event, "instance_selector", None)
    if getattr(event, "instance_index", None) is None and selector is None:
        return None

    states = await get_model_instance_states(
        event.model_name, ray_address, ray_namespace
    )
    if selector in ("active", "active_context", "busy"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if is_ready_instance_state(state)
        ]
        ready_instances = [
            (instance_id, state)
            for instance_id, state in ready_instances
            if instance_concurrency(state) > 0
        ]
        ready_instances = sorted(
            ready_instances,
            key=lambda item: (-instance_concurrency(item[1]), item[0]),
        )
    elif selector in ("preempting", "preempted"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if is_preempting_instance_state(state)
        ]
        ready_instances = sorted(ready_instances, key=lambda item: item[0])
    elif selector in (None, "ready"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if is_ready_instance_state(state)
        ]
        ready_instances = sorted(ready_instances, key=lambda item: item[0])
    else:
        raise RuntimeError(f"Unsupported instance_selector: {selector}")

    ready_instance_ids = [instance_id for instance_id, _ in ready_instances]
    index = int(event.instance_index or 0)
    if index < 0:
        index += len(ready_instance_ids)
    if index < 0 or index >= len(ready_instance_ids):
        raise RuntimeError(
            f"instance_index {event.instance_index} is out of range for "
            f"{event.model_name}; selector={selector}; "
            f"ready instances={ready_instance_ids}; states={states}"
        )
    return ready_instance_ids[index]


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
        payload = {
            k: v
            for k, v in row.items()
            if k not in BENCHMARK_ONLY_WORKLOAD_KEYS
        }
        payload["model"] = model
        sent_at = time.time()
        result = await asyncio.to_thread(
            post_json, endpoint, payload, request_timeout_s
        )
        result["arrival_time"] = arrival_time
        if row.get("benchmark_phase") or row.get("phase"):
            result["benchmark_phase"] = str(
                row.get("benchmark_phase") or row.get("phase")
            )
        result["sent_at"] = sent_at
        result["completed_at"] = time.time()
        return result

    for row in workload:
        tasks.append(asyncio.create_task(send_at(row)))
    return await asyncio.gather(*tasks)


def start_ray_trace_replayer(
    trace_path: Optional[str],
    speedup: float,
    log_path: Path,
    ray_address: str = "auto",
    ray_namespace: str = "sllm",
    controller_name: str = "controller",
    model_name: Optional[str] = None,
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
    command = [
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
    ]
    if model_name:
        command.extend(["--model-name", model_name])
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return TraceProcess(process=process, log_file=log_file, log_path=log_path)


async def replay_trace_over_http(
    trace_path: str,
    speedup: float,
    endpoint: str,
    log_path: Path,
    timeout_s: float,
    ray_address: str,
    ray_namespace: str,
    model_name: Optional[str] = None,
) -> None:
    if speedup <= 0:
        raise ValueError("speedup must be positive")

    events = load_spot_trace_events(
        Path(trace_path),
        default_model_name=model_name,
    )
    spot_endpoint = f"{base_url_from_chat_endpoint(endpoint)}/spot/event"
    replay_started_at = time.monotonic()
    last_event_time = 0.0

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            "HTTP trace replay started: "
            f"trace={trace_path}, events={len(events)}, endpoint={spot_endpoint}\n"
        )
        log_file.flush()
        for event in events:
            sleep_time = max(event.time - last_event_time, 0.0) / speedup
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            instance_id = await resolve_trace_instance_id(
                event,
                ray_address=ray_address,
                ray_namespace=ray_namespace,
            )
            payload = {
                "event": event.event,
                "node_id": event.node_id,
                "instance_id": instance_id,
                "model_name": event.model_name,
            }
            log_file.write(f"Replaying spot event: {payload}\n")
            log_file.flush()
            result = await asyncio.to_thread(
                post_json, spot_endpoint, payload, timeout_s
            )
            log_file.write(f"Spot event result: {json.dumps(result)}\n")
            log_file.flush()
            if not result.get("success"):
                raise RuntimeError(
                    f"Spot event replay failed for {payload}: "
                    f"{result.get('response')}"
                )
            last_event_time = event.time
        log_file.write(
            "HTTP trace replay finished: "
            f"elapsed_s={time.monotonic() - replay_started_at:.3f}\n"
        )


def start_http_trace_replayer(
    trace_path: Optional[str],
    speedup: float,
    endpoint: str,
    log_path: Path,
    timeout_s: float,
    ray_address: str,
    ray_namespace: str,
    model_name: Optional[str] = None,
) -> Optional[TraceTask]:
    if not trace_path:
        return None
    task = asyncio.create_task(
        replay_trace_over_http(
            trace_path=trace_path,
            speedup=speedup,
            endpoint=endpoint,
            log_path=log_path,
            timeout_s=timeout_s,
            ray_address=ray_address,
            ray_namespace=ray_namespace,
            model_name=model_name,
        )
    )
    return TraceTask(task=task, log_path=log_path)


async def wait_trace_replayer(trace_replayer) -> None:
    if trace_replayer is None:
        return
    if isinstance(trace_replayer, TraceProcess):
        try:
            trace_replayer.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            trace_replayer.process.terminate()
            trace_replayer.process.wait(timeout=5)
        finally:
            trace_replayer.log_file.close()
        if trace_replayer.process.returncode not in (0, None):
            print(
                "[benchmark trace warning] Trace replayer exited with "
                f"code {trace_replayer.process.returncode}; see "
                f"{trace_replayer.log_path}",
                file=sys.stderr,
            )
        return

    try:
        await trace_replayer.task
    except Exception as exc:
        print(
            "[benchmark trace warning] HTTP trace replayer failed: "
            f"{exc}; see {trace_replayer.log_path}",
            file=sys.stderr,
        )


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
    trace_transport: str = "http",
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

    clear_metrics_file(run_config.get("router_metrics_path"))
    scheduler_config = run_config.get("scheduler_config")
    if scheduler_config:
        clear_metrics_file(scheduler_config.get("metrics_path"))
    scheduler_update = await apply_scheduler_config(
        scheduler_config,
        bool(run_config.get("replace_scheduler_config", False)),
        ray_address,
        ray_namespace,
    )
    if scheduler_update is not None:
        (run_dir / "scheduler_update.json").write_text(
            json.dumps(scheduler_update, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    deleted_before_run = False
    for model_name in run_config.get("delete_models_before_run", []):
        delete_model_over_http(endpoint, model_name, request_timeout_s)
        deleted_before_run = True
    if deleted_before_run:
        await asyncio.sleep(float(run_config.get("delete_settle_s", 0.0) or 0.0))

    deploy_config = run_config.get("deploy_config")
    if deploy_config:
        deploy_config_over_http(
            endpoint,
            deploy_config,
            request_timeout_s,
            router_metrics_path=run_config.get("router_metrics_path"),
        )

    trace_replayer = None
    try:
        await wait_for_ready_instances(
            run_config["model"],
            int(run_config.get("min_ready_instances", 0) or 0),
            float(run_config.get("ready_timeout_s", 180.0) or 180.0),
            ray_address,
            ray_namespace,
        )

        workload = load_jsonl(Path(run_config["workload"]))
        if not skip_trace:
            if trace_transport == "ray":
                trace_replayer = start_ray_trace_replayer(
                    run_config.get("trace"),
                    speedup,
                    run_dir / "trace_replayer.log",
                    ray_address=ray_address,
                    ray_namespace=ray_namespace,
                    controller_name=controller_name,
                    model_name=run_config["model"],
                )
            else:
                trace_replayer = start_http_trace_replayer(
                    run_config.get("trace"),
                    speedup,
                    endpoint,
                    run_dir / "trace_replayer.log",
                    request_timeout_s,
                    ray_address,
                    ray_namespace,
                    model_name=run_config["model"],
                )
        rows = await send_workload(
            endpoint, run_config["model"], workload, request_timeout_s
        )
    finally:
        await wait_trace_replayer(trace_replayer)
        if run_config.get("delete_after_run"):
            delete_model_over_http(
                endpoint, run_config["model"], request_timeout_s
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
    static_models = [
        run_config["model"]
        for run_config in config["runs"]
        if not run_config.get("deploy_config")
    ]
    check_endpoint_ready(
        endpoint,
        static_models,
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
                trace_transport=args.trace_transport,
            )
        )

    summaries = []
    if not args.no_report:
        try:
            summaries = generate_reports(produced_runs)
            write_combined_summary(output_root, summaries)
            comparison_configs = config.get("comparisons", [])
            if config.get("comparison_fields"):
                comparison_configs = [
                    {
                        **comparison,
                        "fields": comparison.get(
                            "fields", config["comparison_fields"]
                        ),
                    }
                    for comparison in comparison_configs
                ]
            comparisons = build_comparisons(comparison_configs, summaries)
            write_comparisons(output_root, comparisons)
        except Exception as exc:
            print(f"[benchmark report warning] {exc}", file=sys.stderr)
            comparisons = []
    else:
        comparisons = []

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
            instance_suffix = ""
            if int(summary.get("instance_state_rows", 0) or 0) > 0:
                instance_suffix = (
                    f", instance_events="
                    f"{summary.get('instance_state_rows', 0)}, "
                    f"preempting="
                    f"{summary.get('instances_marked_preempting', 0)}, "
                    f"ready={summary.get('instances_marked_ready', 0)}, "
                    f"dead={summary.get('instances_marked_dead', 0)}"
                )
            replanning_suffix = ""
            if int(summary.get("replanning_events", 0) or 0) > 0:
                replanning_suffix = (
                    f", replans={summary.get('replanning_events', 0)}, "
                    f"applied="
                    f"{summary.get('replanning_execution_applied', 0)}, "
                    f"failed="
                    f"{summary.get('replanning_execution_failed', 0)}"
                )
            context_migration_suffix = ""
            if int(summary.get("context_migration_events", 0) or 0) > 0:
                context_migration_suffix = (
                    f", context_migrations="
                    f"{summary.get('context_migration_events', 0)}, "
                    f"reusable_blocks="
                    f"{summary.get('context_migration_reusable_context_blocks', 0)}, "
                    f"reuse_ratio="
                    f"{summary.get('context_migration_reuse_ratio', 0.0)}, "
                    f"kv_successes="
                    f"{summary.get('kv_cache_migration_successes', 0)}"
                )
            state_recovery_suffix = ""
            if (
                int(summary.get("state_recovery_events", 0) or 0) > 0
                or int(summary.get("state_restore_attempts_total", 0) or 0)
                > 0
            ):
                state_recovery_suffix = (
                    f", state_events="
                    f"{summary.get('state_recovery_events', 0)}, "
                    f"state_restores="
                    f"{summary.get('state_restore_successes_total', 0)}/"
                    f"{summary.get('state_restore_attempts_total', 0)}, "
                    f"state_tokens="
                    f"{summary.get('state_restored_tokens_total', 0)}"
                )
            print(
                "  "
                f"{summary['run_name']}: "
                f"successes={summary['successes']}/{summary['requests']}, "
                f"success_rate={summary['success_rate']:.2%}, "
                f"p95={summary['latency_p95_ms']:.2f}ms"
                f"{recovery_suffix}"
                f"{instance_suffix}"
                f"{replanning_suffix}"
                f"{context_migration_suffix}"
                f"{state_recovery_suffix}"
            )
        if comparisons:
            print("\nBenchmark comparisons:")
            for comparison in comparisons:
                print(
                    "  "
                    f"{comparison['name']}: "
                    f"baseline={comparison['baseline']}, "
                    f"candidate={comparison['candidate']}"
                )
                for key, value in comparison.items():
                    if not key.endswith("_delta"):
                        continue
                    field = key[: -len("_delta")]
                    baseline = comparison.get(f"{field}_baseline")
                    candidate = comparison.get(f"{field}_candidate")
                    print(
                        "    "
                        f"{field}: baseline={baseline}, "
                        f"candidate={candidate}, delta={value}"
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
        help="Do not replay spot trace events",
    )
    parser.add_argument(
        "--trace-transport",
        choices=["http", "ray"],
        default="http",
        help=(
            "How to replay spot traces. The default HTTP mode posts events "
            "to the SLLM API and avoids starting an extra Ray driver."
        ),
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
