#!/usr/bin/env python3
"""Verify direct TCP transfer between two GPU workers in managed Ray."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import ray


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"connection closed with {remaining} bytes still expected"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _gpu_identity() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


@ray.remote
class TCPReceiver:
    def __init__(self) -> None:
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._error: BaseException | None = None

    def start(self, socket_timeout_s: float) -> dict[str, Any]:
        node_ip = ray.util.get_node_ip_address()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(socket_timeout_s)
        listener.bind((node_ip, 0))
        listener.listen(1)
        self._listener = listener

        def receive() -> None:
            try:
                started = time.monotonic()
                conn, peer = listener.accept()
                with conn:
                    conn.settimeout(socket_timeout_s)
                    expected_bytes = int.from_bytes(_recv_exact(conn, 8), "big")
                    received_bytes = 0
                    digest = hashlib.sha256()
                    while received_bytes < expected_bytes:
                        chunk = conn.recv(
                            min(1024 * 1024, expected_bytes - received_bytes)
                        )
                        if not chunk:
                            raise ConnectionError(
                                "sender closed before the declared payload arrived"
                            )
                        received_bytes += len(chunk)
                        digest.update(chunk)
                    elapsed_s = time.monotonic() - started
                    self._result = {
                        "bytes": received_bytes,
                        "sha256": digest.hexdigest(),
                        "elapsed_s": elapsed_s,
                        "throughput_mib_s": (
                            received_bytes / (1024 * 1024) / elapsed_s
                        ),
                        "peer": list(peer),
                    }
                    ack = json.dumps(self._result, sort_keys=True).encode("utf-8")
                    conn.sendall(len(ack).to_bytes(8, "big") + ack)
            except BaseException as exc:  # Preserve the worker-side cause.
                self._error = exc
            finally:
                listener.close()

        self._thread = threading.Thread(target=receive, daemon=True)
        self._thread.start()
        context = ray.get_runtime_context()
        return {
            "node_id": str(context.get_node_id()),
            "node_ip": node_ip,
            "port": listener.getsockname()[1],
            "hostname": socket.gethostname(),
            "gpu": _gpu_identity(),
        }

    def wait(self, timeout_s: float) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("receiver has not been started")
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("receiver did not finish before timeout")
        if self._error is not None:
            raise RuntimeError(f"receiver failed: {self._error!r}")
        if self._result is None:
            raise RuntimeError("receiver finished without a result")
        return self._result


@ray.remote
class TCPSender:
    def send(
        self,
        host: str,
        port: int,
        total_bytes: int,
        chunk_bytes: int,
        socket_timeout_s: float,
    ) -> dict[str, Any]:
        pattern = bytes(index % 251 for index in range(chunk_bytes))
        digest = hashlib.sha256()
        sent_bytes = 0
        started = time.monotonic()
        with socket.create_connection(
            (host, port), timeout=socket_timeout_s
        ) as conn:
            conn.settimeout(socket_timeout_s)
            conn.sendall(total_bytes.to_bytes(8, "big"))
            while sent_bytes < total_bytes:
                part = pattern[:min(chunk_bytes, total_bytes - sent_bytes)]
                conn.sendall(part)
                digest.update(part)
                sent_bytes += len(part)
            ack_size = int.from_bytes(_recv_exact(conn, 8), "big")
            receiver_ack = json.loads(_recv_exact(conn, ack_size))
        elapsed_s = time.monotonic() - started
        context = ray.get_runtime_context()
        return {
            "node_id": str(context.get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "gpu": _gpu_identity(),
            "bytes": sent_bytes,
            "sha256": digest.hexdigest(),
            "elapsed_s": elapsed_s,
            "throughput_mib_s": sent_bytes / (1024 * 1024) / elapsed_s,
            "receiver_ack": receiver_ack,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bytes <= 0 or args.chunk_bytes <= 0:
        raise ValueError("--bytes and --chunk-bytes must be positive")

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    receiver = TCPReceiver.options(
        num_cpus=0.1,
        num_gpus=1,
    ).remote()
    sender = TCPSender.options(
        num_cpus=0.1,
        num_gpus=1,
    ).remote()

    try:
        endpoint = ray.get(
            receiver.start.remote(args.timeout_s), timeout=args.timeout_s
        )
        sender_result = ray.get(
            sender.send.remote(
                endpoint["node_ip"],
                endpoint["port"],
                args.bytes,
                args.chunk_bytes,
                args.timeout_s,
            ),
            timeout=args.timeout_s,
        )
        receiver_result = ray.get(
            receiver.wait.remote(args.timeout_s), timeout=args.timeout_s
        )
    finally:
        ray.kill(sender, no_restart=True)
        ray.kill(receiver, no_restart=True)

    if sender_result["node_id"] == endpoint["node_id"]:
        raise RuntimeError("sender and receiver ran on the same Ray node")
    if sender_result["bytes"] != args.bytes:
        raise RuntimeError("sender byte count differs from requested payload")
    if receiver_result["bytes"] != args.bytes:
        raise RuntimeError("receiver byte count differs from requested payload")
    if sender_result["sha256"] != receiver_result["sha256"]:
        raise RuntimeError("sender and receiver SHA-256 digests differ")
    if sender_result["receiver_ack"]["sha256"] != receiver_result["sha256"]:
        raise RuntimeError("receiver acknowledgment does not match receiver result")

    result = {
        "schema_version": 1,
        "status": "passed",
        "measurement_kind": "direct_tcp_payload",
        "payload_bytes": args.bytes,
        "sender": sender_result,
        "receiver": {**endpoint, **receiver_result},
        "verified": [
            "two distinct Ray GPU nodes",
            "direct worker-to-worker TCP reachability",
            "payload byte count",
            "end-to-end SHA-256 integrity",
            "receiver acknowledgment",
        ],
        "not_verified": [
            "physical Kubernetes host identity",
            "NCCL collectives",
            "NIXL transfer",
            "KV cache attach",
            "graceful preemption",
            "F1/F2 performance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
