"""Run a cross-container (same-host) NIXL export/restore smoke.

The two containers have separate network namespaces, hostnames, and GPU
assignments.  This exercises the networked handshake/transfer path without
claiming that the containers are two physical machines.
"""

import argparse
import json
import os
import shutil
import shlex
import socket
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener


IMAGE = "localhost/spotserve-python312-nixl:latest"
REPO_ROOT = "/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve"
VLLM_ROOT = "/work/containers/s112060021/Qwen3/vllm"
MODEL_ROOT = "/work/spotserve-models"
PYTHONPATH = (
    f"{VLLM_ROOT}/.venv/lib/python3.12/site-packages:"
    f"{VLLM_ROOT}:{REPO_ROOT}"
)


def run_podman(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["podman", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def dump_container_logs(names: list[str]) -> None:
    """Print startup logs when a worker dies before opening the control socket."""
    for name in names:
        result = run_podman(["logs", name], check=False)
        if result.stdout:
            print(f"--- {name} logs ---\n{result.stdout}", flush=True)


def send(conn, payload: dict) -> None:
    conn.send(payload)


def recv(conn) -> dict:
    message = conn.recv()
    if message.get("event") in {"fatal", "generation_error"}:
        raise RuntimeError(message.get("traceback", str(message)))
    return message


def wait_event(conn, expected: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not conn.poll(1.0):
            continue
        message = recv(conn)
        if message.get("event") == expected:
            return message
    raise TimeoutError(f"timed out waiting for {expected!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny"
    )
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--source-port", type=int, default=5600)
    parser.add_argument("--target-port", type=int, default=5700)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--image", default=IMAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        raise SystemExit(f"model config not found: {args.model}")

    control_dir = tempfile.mkdtemp(prefix="spotserve-cross-container-")
    # The container runs as root, but make the shared control directory
    # traversable even when the host uses a private /tmp umask.
    os.chmod(control_dir, 0o777)
    socket_path = os.path.join(control_dir, "control.sock")
    listener = Listener(socket_path, family="AF_UNIX", authkey=b"spotserve")
    os.chmod(socket_path, 0o666)
    network = f"spotserve-nixl-net-{os.getpid()}"
    container_names = [
        f"spotserve-nixl-source-{os.getpid()}",
        f"spotserve-nixl-target-{os.getpid()}",
    ]
    connections = {}
    processes = []
    started = time.monotonic()
    try:
        run_podman(["network", "create", network])
        common = [
            "run",
            "--detach",
            "--network",
            network,
            "--volume",
            f"{REPO_ROOT}:{REPO_ROOT}:ro",
            "--volume",
            f"{VLLM_ROOT}:{VLLM_ROOT}:ro",
            "--volume",
            f"{MODEL_ROOT}:{MODEL_ROOT}:ro",
            "--volume",
            "/usr/local/cuda-13.0:/usr/local/cuda:ro",
            "--volume",
            f"{control_dir}:/control:rw",
            "--env",
            f"PYTHONPATH={PYTHONPATH}",
            "--env",
            "PYTHONUNBUFFERED=1",
            args.image,
        ]
        worker_args = [
            "python",
            "-u",
            f"{REPO_ROOT}/tests/spotserve_test/cross_container_nixl_worker.py",
            "--model",
            args.model,
            "--control-socket",
            "/control/control.sock",
        ]

        def container_command(
            name: str,
            hostname: str,
            gpu: int,
            role: str,
            side_channel_host: str,
            side_channel_port: int,
        ) -> list[str]:
            role_args = [
                *worker_args,
                "--role",
                role,
                "--side-channel-host",
                side_channel_host,
                "--side-channel-port",
                str(side_channel_port),
            ]
            return [
                *common[:1],
                "--name",
                name,
                "--hostname",
                hostname,
                "--device",
                f"nvidia.com/gpu={gpu}",
                *common[1:],
                "bash",
                "-lc",
                "exec " + shlex.join(role_args),
            ]

        source_cmd = container_command(
            container_names[0], "source-node", args.source_gpu,
            "source", "source-node", args.source_port,
        )
        target_cmd = container_command(
            container_names[1], "target-node", args.target_gpu,
            "target", "target-node", args.target_port,
        )
        for command in (source_cmd, target_cmd):
            result = run_podman(command)
            processes.append(result.stdout.strip())

        # Listener.accept() otherwise waits forever if an engine fails during
        # CUDA/FlashInfer initialization.  Bound startup time and expose the
        # container log so the failure is actionable.
        listener._listener._socket.settimeout(args.timeout_s)
        for _ in range(2):
            conn = listener.accept()
            ready = wait_event(conn, "ready", args.timeout_s)
            connections[ready["role"]] = conn

        source = connections["source"]
        target = connections["target"]
        request_id = "cross-container-nixl-request"
        prompt_token_ids = [100 + index for index in range(64)]

        send(source, {"op": "generate", "request_id": request_id, "token_ids": prompt_token_ids})
        wait_event(source, "generate_started", args.timeout_s)
        wait_event(source, "paused", args.timeout_s)

        send(source, {"op": "metadata", "request_id": request_id})
        metadata = wait_event(source, "metadata", args.timeout_s)["result"]
        if not metadata.get("found") or not metadata.get("block_ids"):
            raise AssertionError(f"source has no active KV blocks: {metadata}")

        send(source, {"op": "export", "request_id": request_id})
        state = wait_event(source, "export", args.timeout_s)["result"]
        if not state.get("supports_restore"):
            raise AssertionError(f"source export failed: {state}")

        send(source, {"op": "abort", "request_id": request_id})
        wait_event(source, "aborted", args.timeout_s)

        send(target, {"op": "restore", "request_id": request_id, "state": state})
        staged = wait_event(target, "restore", args.timeout_s)["result"]
        if not staged.get("staged"):
            raise AssertionError(f"target restore was not staged: {staged}")

        computed_tokens = metadata["tokens"][: metadata["completed_tokens"]]
        send(target, {"op": "generate", "request_id": request_id, "token_ids": computed_tokens})
        wait_event(target, "generate_started", args.timeout_s)
        target_output = wait_event(target, "output", args.timeout_s)
        wait_event(target, "paused", args.timeout_s)

        print(
            json.dumps(
                {
                    "status": "passed",
                    "simulation": "cross_container_same_host",
                    "source_container": container_names[0],
                    "target_container": container_names[1],
                    "source_hostname": "source-node",
                    "target_hostname": "target-node",
                    "source_blocks": metadata["block_ids"],
                    "source_computed_tokens": metadata["completed_tokens"],
                    "staged_blocks": staged["expected_blocks"],
                    "target_generated_tokens": target_output["token_ids"],
                    "state_restore_attempts_total": 1,
                    "state_restore_successes_total": 1,
                    "state_restore_fallback_count": 0,
                    "elapsed_s": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            )
        )
    except (TimeoutError, socket.timeout):
        dump_container_logs(container_names)
        raise
    finally:
        for conn in connections.values():
            try:
                send(conn, {"op": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        listener.close()
        for name in container_names:
            result = run_podman(["rm", "--force", name], check=False)
            if result.returncode != 0 and result.stdout:
                print(result.stdout, end="")
        run_podman(["network", "rm", network], check=False)
        shutil.rmtree(control_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
