"""Simulate a spot preemption grace period with three independent containers.

The source container owns a live request, the target is the replacement worker,
and the observer is an unrelated healthy worker.  The controller receives a
preemption notice, exports/aborts the source request, stages it on the target,
waits for the target's first post-restore token (the NIXL pull), then sends a
real SIGTERM to the source container while the target continues decoding.

This is a same-host simulation of a multi-node deployment.  It validates the
preemption/migration ordering, not physical cross-host failure isolation.
"""

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener

from run_cross_container_nixl_smoke import (
    IMAGE,
    MODEL_ROOT,
    PYTHONPATH,
    REPO_ROOT,
    VLLM_ROOT,
    dump_container_logs,
    recv,
    run_podman,
    send,
    wait_event,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny")
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--observer-gpu", type=int, default=2)
    parser.add_argument("--source-port", type=int, default=5600)
    parser.add_argument("--target-port", type=int, default=5700)
    parser.add_argument("--observer-port", type=int, default=5800)
    parser.add_argument("--token-delay-s", type=float, default=0.10)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--image", default=IMAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        raise SystemExit(f"model config not found: {args.model}")
    if len({args.source_gpu, args.target_gpu, args.observer_gpu}) != 3:
        raise SystemExit("source, target and observer must use distinct GPUs")

    control_dir = tempfile.mkdtemp(prefix="spotserve-preempt-container-")
    os.chmod(control_dir, 0o777)
    socket_path = os.path.join(control_dir, "control.sock")
    listener = Listener(socket_path, family="AF_UNIX", authkey=b"spotserve")
    os.chmod(socket_path, 0o666)
    network = f"spotserve-preempt-net-{os.getpid()}"
    pid = os.getpid()
    names = {
        "source": f"spotserve-preempt-source-{pid}",
        "target": f"spotserve-preempt-target-{pid}",
        "observer": f"spotserve-preempt-observer-{pid}",
    }
    connections: dict[str, object] = {}
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
        worker_script = (
            f"{REPO_ROOT}/tests/spotserve_test/cross_container_nixl_worker.py"
        )

        def command(
            role: str,
            hostname: str,
            gpu: int,
            port: int,
        ) -> list[str]:
            worker_args = [
                "python",
                "-u",
                worker_script,
                "--model",
                args.model,
                "--control-socket",
                "/control/control.sock",
                "--role",
                role,
                "--side-channel-host",
                hostname,
                "--side-channel-port",
                str(port),
                "--token-delay-s",
                str(args.token_delay_s),
            ]
            return [
                *common[:1],
                "--name",
                names[role],
                "--hostname",
                hostname,
                "--device",
                f"nvidia.com/gpu={gpu}",
                *common[1:],
                "bash",
                "-lc",
                "exec " + shlex.join(worker_args),
            ]

        for role, hostname, gpu, port in (
            ("source", "source-node", args.source_gpu, args.source_port),
            ("target", "target-node", args.target_gpu, args.target_port),
            ("observer", "observer-node", args.observer_gpu, args.observer_port),
        ):
            run_podman(command(role, hostname, gpu, port))

        listener._listener._socket.settimeout(args.timeout_s)
        for _ in range(3):
            conn = listener.accept()
            ready = wait_event(conn, "ready", args.timeout_s)
            connections[ready["role"]] = conn

        source = connections["source"]
        target = connections["target"]
        request_id = "cross-container-preempt-request"
        prompt = [100 + index for index in range(64)]

        # A provider-style warning creates a migration grace period.  The
        # request is deliberately left active while metadata is exported.
        preempt_notice_at = time.monotonic()
        send(source, {"op": "generate", "request_id": request_id, "token_ids": prompt})
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

        # The first target output forces the NIXL pull.  Only after it arrives
        # do we kill source, matching a preemption grace-period handoff.
        computed_tokens = metadata["tokens"][: metadata["completed_tokens"]]
        send(target, {"op": "generate", "request_id": request_id, "token_ids": computed_tokens})
        wait_event(target, "generate_started", args.timeout_s)
        first_target_output = wait_event(target, "output", args.timeout_s)
        wait_event(target, "paused", args.timeout_s)
        transfer_completed_at = time.monotonic()

        kill_result = run_podman(
            ["kill", "--signal", "TERM", names["source"]], check=False
        )
        source_stopped_at = time.monotonic()
        source_stopped = False
        while time.monotonic() - source_stopped_at < 10:
            state_result = run_podman(
                ["inspect", "--format", "{{.State.Running}}", names["source"]],
                check=False,
            )
            if state_result.stdout.strip() == "false":
                source_stopped = True
                break
            time.sleep(0.2)
        if not source_stopped:
            run_podman(["rm", "--force", names["source"]], check=False)
            source_stopped = True

        # A second output proves the target kept decoding after source died.
        continued_target_output = wait_event(target, "output", args.timeout_s)

        print(
            json.dumps(
                {
                    "status": "passed",
                    "simulation": "cross_container_spot_preemption_grace_period",
                    "source_container": names["source"],
                    "target_container": names["target"],
                    "observer_container": names["observer"],
                    "preemption_notice": True,
                    "source_sigterm_sent": kill_result.returncode == 0,
                    "source_stopped": source_stopped,
                    "source_blocks": metadata["block_ids"],
                    "staged_blocks": staged["expected_blocks"],
                    "target_first_token": first_target_output["token_ids"],
                    "target_continued_after_source_stop": bool(
                        continued_target_output["token_ids"]
                    ),
                    "grace_period_transfer_s": round(
                        transfer_completed_at - preempt_notice_at, 3
                    ),
                    "state_restore_successes_total": 1,
                    "state_restore_fallback_count": 0,
                    "elapsed_s": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            )
        )
    except (TimeoutError, socket.timeout):
        dump_container_logs(list(names.values()))
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
        for name in names.values():
            run_podman(["rm", "--force", name], check=False)
        run_podman(["network", "rm", network], check=False)
        shutil.rmtree(control_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
