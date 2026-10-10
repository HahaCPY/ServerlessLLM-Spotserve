#!/usr/bin/env python3
"""Create the minimal build context accepted by the CSCC Project Build UI."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import tarfile


REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "setup.py",
    "py.typed",
    "benchmarks/spotserve/run_benchmark.py",
    "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json",
    "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu_trace_plan.json",
    "scripts/analyze_spotserve_benchmark.py",
    "scripts/plot_spotserve_benchmark.py",
    "scripts/run_k8s_moe_f1_f2.py",
    "deploy/cscc-ray/requirements-serverless.txt",
    "deploy/cscc-ray/requirements-worker.txt",
    "deploy/cscc-ray/requirements-runtime.txt",
    "deploy/cscc-ray/verify_image.py",
    "deploy/cscc-ray/smoke_ray_image.py",
    "deploy/cscc-ray/run_formal_f1_f2.py",
)

REQUIRED_TREES = (
    "sllm",
    "sllm_store",
)

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}

SKIP_SUFFIXES = (".pyc", ".pyo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/spotserve-cscc-ray-build-context.tar.gz"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def is_allowed_file(path: Path) -> bool:
    return (
        path.is_file()
        and not any(part in SKIP_PARTS for part in path.parts)
        and not path.name.endswith(SKIP_SUFFIXES)
    )


def add_file(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source), arcname=archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output.resolve()
    dockerfile = root / "deploy/cscc-ray/Dockerfile"

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")
    if not dockerfile.is_file():
        raise FileNotFoundError(f"missing Dockerfile: {dockerfile}")

    files: dict[str, Path] = {"Dockerfile": dockerfile}
    for relative in REQUIRED_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing required file: {source}")
        files[relative] = source

    for relative_tree in REQUIRED_TREES:
        tree = root / relative_tree
        if not tree.is_dir():
            raise FileNotFoundError(f"missing required directory: {tree}")
        for source in sorted(tree.rglob("*")):
            if is_allowed_file(source):
                files[source.relative_to(root).as_posix()] = source

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for archive_name, source in sorted(files.items()):
            add_file(archive, source, archive_name)

        manifest = "\n".join(sorted(files)) + "\n"
        info = tarfile.TarInfo("BUILD_CONTEXT_MANIFEST.txt")
        encoded = manifest.encode("utf-8")
        info.size = len(encoded)
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mtime = 0
        archive.addfile(info, io.BytesIO(encoded))

    print(f"archive={output}")
    print(f"files={len(files) + 1}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
