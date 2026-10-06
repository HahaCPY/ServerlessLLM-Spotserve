#!/usr/bin/env python3
"""Fail-closed package and patch verification for the CSCC Ray image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import metadata
from packaging.version import Version
from pathlib import Path
from typing import Any


PATCH_MARKERS = (
    ("spotserve_moe.py", "def record_moe_routing"),
    ("v1/engine/async_llm.py", "def apply_expert_placement_plan"),
    ("v1/engine/async_llm.py", "def verify_expert_placement_plan"),
    ("v1/worker/worker_base.py", "def get_request_moe_metadata"),
    ("v1/worker/worker_base.py", "def apply_expert_placement_plan"),
    ("v1/worker/worker_base.py", "def verify_expert_placement_plan"),
    ("v1/engine/utils.py", "falling back to ray.nodes()"),
)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution is missing: {name}") from exc


def check_exact(name: str, expected: str) -> str:
    observed = package_version(name)
    if observed != expected:
        raise RuntimeError(f"{name} must be {expected}, observed {observed}")
    return observed


def check_base_version(name: str, expected: str) -> str:
    observed = package_version(name)
    if Version(observed).base_version != Version(expected).base_version:
        raise RuntimeError(
            f"{name} base version must be {expected}, observed {observed}"
        )
    return observed


def verify(require_gpu: bool = False) -> dict[str, Any]:
    import ray
    import sllm
    import sllm_store
    import torch
    import vllm
    from nixl._api import nixl_agent

    del ray, sllm, sllm_store, nixl_agent

    expected_ray = os.getenv("SPOTSERVE_EXPECTED_RAY_VERSION", "2.58.0")
    expected_torch = os.getenv("SPOTSERVE_EXPECTED_TORCH_VERSION", "2.9.0")
    expected_vllm = os.getenv("SPOTSERVE_EXPECTED_VLLM_VERSION", "0.11.2")
    versions = {
        "ray": check_exact("ray", expected_ray),
        # CUDA wheels use a local tag such as 2.9.0+cu128.  Requiring the same
        # base release keeps the check strict without rejecting that valid tag.
        "torch": check_base_version("torch", expected_torch),
        "vllm": check_exact("vllm", expected_vllm),
        "serverless-llm": package_version("serverless-llm"),
        "serverless-llm-store": package_version("serverless-llm-store"),
        "nixl": package_version("nixl"),
        "python": sys.version.split()[0],
        "torch_cuda": str(torch.version.cuda or ""),
    }

    vllm_root = Path(vllm.__file__).resolve().parent
    verified_markers = []
    for relative, marker in PATCH_MARKERS:
        target = vllm_root / relative
        if not target.is_file():
            raise RuntimeError(f"patched vLLM file is missing: {target}")
        if marker not in target.read_text(encoding="utf-8"):
            raise RuntimeError(f"vLLM patch marker {marker!r} missing from {target}")
        verified_markers.append(f"{relative}:{marker}")

    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError("--require-gpu was set but torch.cuda.is_available() is false")
    gpu: dict[str, Any] = {"cuda_available": cuda_available}
    if cuda_available:
        gpu.update(
            count=torch.cuda.device_count(),
            current_device=torch.cuda.current_device(),
            name=torch.cuda.get_device_name(torch.cuda.current_device()),
            capability=list(
                torch.cuda.get_device_capability(torch.cuda.current_device())
            ),
        )
        subprocess.run(["nvidia-smi", "-L"], check=True)

    return {
        "status": "passed",
        "versions": versions,
        "vllm_root": str(vllm_root),
        "patch_markers": verified_markers,
        "gpu": gpu,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify(require_gpu=args.require_gpu)
    result["scope"] = "build_time_no_gpu" if args.build_time else "runtime"
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
