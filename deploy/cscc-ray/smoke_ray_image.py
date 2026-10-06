#!/usr/bin/env python3
"""Use one GPU actor per worker to verify a managed CSCC Ray image."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import ray
from packaging.version import Version


EXPECTED_BASE_VERSIONS = {
    "ray": "2.58.0",
    "torch": "2.9.0",
    "vllm": "0.11.2",
}


@ray.remote(num_cpus=0.1, num_gpus=1)
class GPUWorkerProbe:
    def collect(self, model_path: str | None) -> dict[str, Any]:
        import torch
        import vllm
        from nixl._api import nixl_agent

        import sllm
        import sllm_store

        del vllm, nixl_agent, sllm, sllm_store

        if not torch.cuda.is_available():
            raise RuntimeError("Ray assigned a GPU but CUDA is unavailable")
        context = ray.get_runtime_context()
        node_id = str(context.get_node_id())
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
        model = None
        if model_path:
            root = Path(model_path)
            model = {
                "path": str(root),
                "exists": root.is_dir(),
                "config_exists": (root / "config.json").is_file(),
            }
            if not model["exists"] or not model["config_exists"]:
                raise RuntimeError(f"model checkpoint is not visible: {root}")
        return {
            "node_id": node_id,
            "node_ip": ray.util.get_node_ip_address(),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": visible,
            "gpu": smi,
            "torch_cuda": torch.version.cuda,
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "versions": {
                name: metadata.version(name)
                for name in (
                    "ray",
                    "torch",
                    "vllm",
                    "nixl",
                    "serverless-llm",
                    "serverless-llm-store",
                )
            },
            "model": model,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm-cscc-image-smoke")
    parser.add_argument("--model-path")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_workers <= 0:
        raise ValueError("--expected-workers must be positive")
    ray.init(
        address=args.ray_address,
        namespace=args.ray_namespace,
        ignore_reinit_error=True,
    )
    actors = [GPUWorkerProbe.remote() for _ in range(args.expected_workers)]
    started = time.monotonic()
    try:
        rows = ray.get(
            [actor.collect.remote(args.model_path) for actor in actors],
            timeout=args.timeout_s,
        )
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)

    node_ids = {row["node_id"] for row in rows}
    if len(rows) != args.expected_workers:
        raise RuntimeError(
            f"expected {args.expected_workers} GPU probes, observed {len(rows)}"
        )
    if len(node_ids) != args.expected_workers:
        raise RuntimeError(
            "CSCC did not provide one GPU worker per Ray node: "
            f"{len(node_ids)} distinct nodes for {len(rows)} actors"
        )
    version_sets = {
        name: {row["versions"][name] for row in rows}
        for name in rows[0]["versions"]
    }
    inconsistent = {
        name: sorted(values) for name, values in version_sets.items()
        if len(values) != 1
    }
    if inconsistent:
        raise RuntimeError(f"package versions differ across workers: {inconsistent}")
    mismatched = {
        name: sorted(version_sets[name])
        for name, expected in EXPECTED_BASE_VERSIONS.items()
        if any(
            Version(observed).base_version != Version(expected).base_version
            for observed in version_sets[name]
        )
    }
    if mismatched:
        raise RuntimeError(
            "managed Ray image has unexpected package releases: "
            f"expected {EXPECTED_BASE_VERSIONS}, observed {mismatched}"
        )

    result = {
        "schema_version": 1,
        "status": "passed",
        "expected_workers": args.expected_workers,
        "observed_gpu_actors": len(rows),
        "distinct_ray_nodes": len(node_ids),
        "elapsed_s": time.monotonic() - started,
        "model_path": args.model_path,
        "version_sets": {
            name: sorted(values) for name, values in version_sets.items()
        },
        "workers": sorted(rows, key=lambda row: row["node_id"]),
        "not_verified": [
            "NIXL cross-worker transfer",
            "NCCL sparse all-to-all",
            "physical Kubernetes host identity",
            "graceful preemption",
            "F1/F2 performance",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
