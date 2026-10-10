#!/usr/bin/env python3
"""Verify a two-worker NCCL all-reduce in managed Ray."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import ray


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
class NCCLWorker:
    def identity(self) -> dict[str, Any]:
        context = ray.get_runtime_context()
        return {
            "node_id": str(context.get_node_id()),
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "gpu": _gpu_identity(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        }

    def available_port(self) -> int:
        node_ip = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((node_ip, 0))
            return int(sock.getsockname()[1])

    def run(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        tensor_bytes: int,
        warmup_iterations: int,
        measured_iterations: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in NCCL worker")

        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        torch.cuda.set_device(0)
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_s),
        )

        try:
            element_size = torch.tensor([], dtype=torch.float32).element_size()
            numel = tensor_bytes // element_size
            if numel <= 0:
                raise ValueError("tensor payload is smaller than one float32 element")
            actual_tensor_bytes = numel * element_size
            tensor = torch.empty(numel, dtype=torch.float32, device="cuda")
            expected = sum(float(index + 1) for index in range(world_size))

            for _ in range(warmup_iterations):
                tensor.fill_(float(rank + 1))
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

            iteration_ms: list[float] = []
            for _ in range(measured_iterations):
                tensor.fill_(float(rank + 1))
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                end.record()
                end.synchronize()
                iteration_ms.append(float(start.elapsed_time(end)))

            observed = float(tensor[0].item())
            verified = observed == expected
            if not verified:
                raise RuntimeError(
                    f"rank {rank} observed {observed}, expected {expected}"
                )

            context = ray.get_runtime_context()
            mean_ms = sum(iteration_ms) / len(iteration_ms)
            raw_nccl_version = torch.cuda.nccl.version()
            nccl_version = (
                list(raw_nccl_version)
                if isinstance(raw_nccl_version, (tuple, list))
                else raw_nccl_version
            )
            return {
                "rank": rank,
                "node_id": str(context.get_node_id()),
                "node_ip": ray.util.get_node_ip_address(),
                "hostname": socket.gethostname(),
                "gpu": _gpu_identity(),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "nccl_version": nccl_version,
                "tensor_bytes": actual_tensor_bytes,
                "warmup_iterations": warmup_iterations,
                "measured_iterations": measured_iterations,
                "iteration_ms": iteration_ms,
                "mean_ms": mean_ms,
                "min_ms": min(iteration_ms),
                "max_ms": max(iteration_ms),
                "payload_mib_per_s": (
                    actual_tensor_bytes / (1024 * 1024) / (mean_ms / 1000)
                ),
                "observed_value": observed,
                "expected_value": expected,
                "verified": verified,
            }
        finally:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--tensor-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tensor_bytes <= 0:
        raise ValueError("--tensor-bytes must be positive")
    if args.warmup_iterations < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    workers = [
        NCCLWorker.options(num_cpus=0.5, num_gpus=1).remote()
        for _ in range(2)
    ]
    timeout = args.timeout_s

    try:
        identities = ray.get(
            [worker.identity.remote() for worker in workers], timeout=timeout
        )
        if identities[0]["node_id"] == identities[1]["node_id"]:
            raise RuntimeError("NCCL workers ran on the same Ray node")
        if identities[0]["gpu"] == identities[1]["gpu"]:
            raise RuntimeError("NCCL workers were assigned the same physical GPU")

        master_addr = identities[0]["node_ip"]
        master_port = ray.get(workers[0].available_port.remote(), timeout=timeout)
        refs = [
            worker.run.remote(
                rank,
                2,
                master_addr,
                master_port,
                args.tensor_bytes,
                args.warmup_iterations,
                args.iterations,
                timeout,
            )
            for rank, worker in enumerate(workers)
        ]
        rows = ray.get(refs, timeout=timeout + 30)
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)

    if not all(row["verified"] for row in rows):
        raise RuntimeError("at least one NCCL rank failed result verification")
    if len({row["node_id"] for row in rows}) != 2:
        raise RuntimeError("NCCL ranks did not execute on distinct Ray nodes")

    result = {
        "schema_version": 1,
        "status": "passed",
        "measurement_kind": "cross_ray_node_nccl_all_reduce",
        "world_size": 2,
        "tensor_bytes": rows[0]["tensor_bytes"],
        "ranks": sorted(rows, key=lambda row: row["rank"]),
        "verified": [
            "two distinct Ray GPU nodes",
            "two distinct physical GPU UUIDs",
            "NCCL process-group initialization",
            "cross-worker NCCL all-reduce",
            "all-reduce numerical result",
        ],
        "not_verified": [
            "physical Kubernetes host identity",
            "RDMA/RoCE transport selection",
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
