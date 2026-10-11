#!/usr/bin/env python3
"""Run the formal single-pass F1/F2 matrix inside one CSCC Ray Job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--port", type=int, default=8343)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers != 8:
        parser.error("the formal protocol requires exactly 8 GPU workers")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def runtime_config(args: argparse.Namespace, output: Path) -> Path:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["path"] = str(Path(args.model))
    config["cluster"].update(
        {
            "endpoint": f"http://127.0.0.1:{args.port}/v1/chat/completions",
            "managed_ray_node_id_workers": True,
            "require_image_digest": False,
            "required_gpus": 8,
            "require_exact_gpu_count": True,
            "minimum_gpu_nodes": 8,
            "gpus_per_worker": 1,
        }
    )
    config["experiment"]["formal_repeats"] = 1
    destination = output / "cscc-formal-config.json"
    write_json(destination, config)
    return destination


def run_and_tee(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def hold_eight_gpu_nodes(ray: Any, worker_count: int, timeout_s: float):
    """Atomically reserve eight GPU nodes, then keep them alive without GPUs."""
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import (
        NodeAffinitySchedulingStrategy,
        PlacementGroupSchedulingStrategy,
    )

    class GPUReservation:
        def identify(self) -> dict[str, str]:
            import ray as actor_ray

            return {
                "node_id": str(actor_ray.get_runtime_context().get_node_id()),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            }

    class NodeKeeper:
        def ping(self) -> dict[str, str]:
            import ray as actor_ray

            return {
                "node_id": str(actor_ray.get_runtime_context().get_node_id()),
                "hostname": socket.gethostname(),
            }

    bundle = {"CPU": 0.1, "GPU": 1}
    group = placement_group(
        [bundle.copy() for _ in range(worker_count)],
        strategy="STRICT_SPREAD",
    )
    print(
        "CSCC_GPU_PROVISIONING="
        + json.dumps(
            {
                "phase": "placement_group_pending",
                "workers": worker_count,
                "bundle": bundle,
                "strategy": "STRICT_SPREAD",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ready_ref = group.ready()
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            snapshot = {
                "phase": "placement_group_timeout",
                "workers": worker_count,
                "cluster_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources(),
                "alive_nodes": [
                    {
                        "node_id": row.get("NodeID"),
                        "node_ip": row.get("NodeManagerAddress"),
                        "resources": row.get("Resources", {}),
                    }
                    for row in ray.nodes()
                    if row.get("Alive")
                ],
            }
            print(
                "CSCC_GPU_PROVISIONING="
                + json.dumps(snapshot, sort_keys=True),
                flush=True,
            )
            remove_placement_group(group)
            raise TimeoutError(
                "timed out waiting for one STRICT_SPREAD placement group "
                f"with {worker_count} GPU bundles"
            )
        ready, _ = ray.wait(
            [ready_ref], num_returns=1, timeout=min(15.0, remaining)
        )
        if ready:
            ray.get(ready_ref)
            break
        print(
            "CSCC_GPU_PROVISIONING="
            + json.dumps(
                {
                    "phase": "placement_group_pending",
                    "remaining_s": round(remaining, 1),
                    "cluster_gpu": ray.cluster_resources().get("GPU", 0),
                    "available_gpu": ray.available_resources().get("GPU", 0),
                    "alive_gpu_nodes": sum(
                        1
                        for row in ray.nodes()
                        if row.get("Alive")
                        and row.get("Resources", {}).get("GPU", 0) >= 1
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        "CSCC_GPU_PROVISIONING="
        + json.dumps(
            {
                "phase": "placement_group_ready",
                "cluster_gpu": ray.cluster_resources().get("GPU", 0),
                "alive_gpu_nodes": sum(
                    1
                    for row in ray.nodes()
                    if row.get("Alive")
                    and row.get("Resources", {}).get("GPU", 0) >= 1
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    reservation_type = ray.remote(num_cpus=0.1, num_gpus=1)(GPUReservation)
    reservations = [
        reservation_type.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_bundle_index=index,
                placement_group_capture_child_tasks=False,
            )
        ).remote()
        for index in range(worker_count)
    ]
    keepers = []
    try:
        rows = ray.get(
            [actor.identify.remote() for actor in reservations],
            timeout=60,
        )
        node_ids = [row["node_id"] for row in rows]
        if len(set(node_ids)) != worker_count:
            raise RuntimeError(
                f"expected {worker_count} distinct GPU Ray nodes, got {rows}"
            )
        keeper_type = ray.remote(num_cpus=0.1)(NodeKeeper)
        keepers = [
            keeper_type.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote()
            for node_id in node_ids
        ]
        keeper_rows = ray.get(
            [keeper.ping.remote() for keeper in keepers], timeout=60
        )
        if {row["node_id"] for row in keeper_rows} != set(node_ids):
            raise RuntimeError("node keepers were not pinned to all GPU nodes")
    finally:
        for actor in reservations:
            ray.kill(actor, no_restart=True)
        remove_placement_group(group)
    return rows, keepers


def wait_for_server(
    process: subprocess.Popen[Any], port: int, timeout_s: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-40:]
                )
            raise RuntimeError(
                f"ServerlessLLM exited with {process.returncode}:\n{tail}"
            )
        try:
            with request.urlopen(health_url, timeout=3) as response:
                payload = json.load(response)
            if payload.get("status") == "ok":
                return
            last_error = f"unexpected health response: {payload}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(
        f"ServerlessLLM did not become healthy in {timeout_s}s: {last_error}"
    )


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = runtime_config(args, output)
    driver_command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/run_k8s_moe_f1_f2.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--endpoint",
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        "--ray-address",
        "auto",
        "--ray-namespace",
        "sllm",
        "--repeats",
        "1",
        "--single-pass",
    ]
    env = os.environ.copy()
    env.update(
        {
            "SLLM_ALLOW_RAY_NODE_ID_WORKERS": "1",
            "SLLM_DIRECT_VLLM_NO_STORE": "1",
            "PYTHONPATH": (
                str(REPO_ROOT)
                + (
                    os.pathsep + env["PYTHONPATH"]
                    if env.get("PYTHONPATH")
                    else ""
                )
            ),
        }
    )
    if args.dry_run:
        return run_and_tee(
            driver_command + ["--dry-run"],
            output / "formal-driver-dry-run.log",
            env,
        )

    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError("this command must run inside the CSCC Ray image") from exc

    status = {
        "status": "provisioning_gpu_workers",
        "phase": "gpu_provisioning",
        "workers": args.workers,
        "model": args.model,
        "config": str(config_path),
        "unique_formal_gpu_runs": 6,
        "f1": list((
            "rerouting",
            "reparallelization",
            "original_spotserve",
            "moe_spotserve",
        )),
        "f2": list((
            "original_spotserve",
            "reparallelization_only",
            "migration_only",
            "full_moe_spotserve",
        )),
    }
    write_json(output / "cscc-wrapper-state.json", status)
    try:
        ray.init(address="auto", namespace="sllm", ignore_reinit_error=True)
        worker_rows, keepers = hold_eight_gpu_nodes(
            ray, args.workers, args.startup_timeout_s
        )
    except BaseException as exc:
        status.update(
            status="blocked",
            phase="gpu_provisioning",
            error=str(exc) or type(exc).__name__,
        )
        write_json(output / "cscc-wrapper-state.json", status)
        raise
    write_json(output / "held-gpu-workers.json", worker_rows)
    status.update(status="running", phase="server_start")
    write_json(output / "cscc-wrapper-state.json", status)
    server_log = output / "serverlessllm.log"
    server_process: subprocess.Popen[Any] | None = None
    try:
        try:
            ray.get_actor("controller", namespace="sllm")
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "a controller actor already exists in namespace sllm; use a "
                "fresh Ray session or stop the previous formal job first"
            )
        server_log.parent.mkdir(parents=True, exist_ok=True)
        server_stream = server_log.open("a", encoding="utf-8")
        server_process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "sllm.cli.clic",
                "start",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=server_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_server(
            server_process, args.port, args.startup_timeout_s, server_log
        )
        return_code = run_and_tee(
            driver_command,
            output / "formal-driver.log",
            env,
        )
        status["status"] = "passed" if return_code == 0 else "blocked"
        status["phase"] = "complete" if return_code == 0 else "formal_driver"
        status["driver_exit_code"] = return_code
        write_json(output / "cscc-wrapper-state.json", status)
        return return_code
    except BaseException as exc:
        status.update(
            status="blocked",
            phase="server_or_formal_driver",
            error=str(exc) or type(exc).__name__,
        )
        write_json(output / "cscc-wrapper-state.json", status)
        raise
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=10)
        if "server_stream" in locals():
            server_stream.close()
        for keeper in keepers:
            ray.kill(keeper, no_restart=True)


if __name__ == "__main__":
    raise SystemExit(main())
