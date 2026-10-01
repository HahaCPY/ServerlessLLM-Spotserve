"""Launch one isolated two-GPU native-MoE Elastic EP graceful-removal probe."""

import argparse
import json
from pathlib import Path
import subprocess
import uuid


def _check_gpu_processes(gpu_indices):
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    uuids = {
        int(row.split(",", 1)[0]): row.split(",", 1)[1].strip()
        for row in gpu_rows
    }
    selected = {uuids[index] for index in gpu_indices}
    process_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for row in process_rows:
        pid, gpu_uuid, _ = [item.strip() for item in row.split(",", 2)]
        if gpu_uuid not in selected:
            continue
        args = subprocess.run(
            ["ps", "-p", pid, "-o", "args="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if "sllm-store start" not in args:
            raise RuntimeError(
                f"GPU {gpu_uuid} already has a non-store compute process: {pid} {args}"
            )
    return {"gpu_uuids": uuids, "baseline_processes": process_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-indices", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(set(args.gpu_indices)) != 2:
        raise ValueError("Phase 1 requires two distinct physical GPUs")

    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    model = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = _check_gpu_processes(args.gpu_indices)
    container_name = f"spotserve-ep-phase1-{uuid.uuid4().hex[:12]}"
    image = "localhost/spotserve-python312-nixl:latest"
    command = [
        "podman",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "bridge",
        "--shm-size",
        "2g",
        "--workdir",
        str(repo),
    ]
    for index in args.gpu_indices:
        command += ["--device", f"nvidia.com/gpu={index}"]
    for source, target, mode in (
        (repo, repo, "ro"),
        (vllm, vllm, "ro"),
        (model, model, "ro"),
        (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"), "ro"),
        (output_dir, Path("/taskresults"), "rw"),
    ):
        command += ["--volume", f"{source}:{target}:{mode}"]
    env = {
        "PYTHONPATH": (
            f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}"
        ),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "TRITON_CACHE_DIR": "/taskresults/compiler-cache/triton",
        "CUDA_CACHE_PATH": "/taskresults/compiler-cache/cuda",
        "FLASHINFER_WORKSPACE_BASE": "/taskresults/compiler-cache/flashinfer",
    }
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [
        image,
        str(vllm / ".venv/bin/python"),
        "-u",
        "-m",
        "scripts.probe_moe_elastic_ep_phase1",
        "--model",
        str(model),
        "--output",
        "/taskresults/report.json",
    ]
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment_before": environment,
                "formal_migration_ablation": False,
                "container_name": container_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"MOE_ELASTIC_EP_PHASE1_CONTAINER={container_name}", flush=True)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        (output_dir / "container.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        (output_dir / "exit.json").write_text(
            json.dumps({"returncode": result.returncode}) + "\n",
            encoding="utf-8",
        )
        report = output_dir / "report.json"
        payload_status = None
        if report.is_file():
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload_status = payload.get("status")
            print(
                "MOE_ELASTIC_EP_PHASE1_SUMMARY="
                + json.dumps(
                    {
                        "status": payload.get("status"),
                        "scale_down_s": payload.get("scale_down_s"),
                        "survivor": payload.get("survivor"),
                        "failure": payload.get("failure", {}).get("reason"),
                    }
                ),
                flush=True,
            )
        else:
            print("MOE_ELASTIC_EP_PHASE1_NO_REPORT", flush=True)
        if result.returncode:
            raise RuntimeError("container_failed; inspect preserved container.log")
        if payload_status != "passed":
            raise RuntimeError("probe_failed; inspect report.json and container.log")
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        (output_dir / "container.log").write_text(partial, encoding="utf-8")
        subprocess.run(
            ["podman", "stop", "--time", "5", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
        raise


if __name__ == "__main__":
    main()
