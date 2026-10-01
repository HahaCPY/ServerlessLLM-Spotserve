"""Run owned Granite EP3/EP4 native canaries and preserve every result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid


SHAPES = (
    ("tp2-dp1-ep2-12", 2, 1, (1, 2)),
    ("tp2-dp1-ep2-21", 2, 1, (2, 1)),
    ("tp2-dp1-ep2-12-swap", 2, 1, (1, 2)),
    ("tp1-dp2-ep2-12", 1, 2, (1, 2)),
    ("tp2-dp1-ep2-13", 2, 1, (1, 3)),
    ("tp1-dp2-ep2-13", 1, 2, (1, 3)),
    ("tp1-dp3-ep3", 1, 3, (1, 2, 3)),
    ("tp4-dp1-ep4", 4, 1, (0, 1, 2, 3)),
    ("tp2-dp2-ep4", 2, 2, (0, 1, 2, 3)),
    ("tp1-dp4-ep4", 1, 4, (0, 1, 2, 3)),
)


def gpu_snapshot() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
               "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=True, timeout=15)
    rows = {}
    for line in result.stdout.splitlines():
        index, gpu_uuid, memory, utilization = [x.strip() for x in line.split(",")]
        rows[int(index)] = {"uuid": canonical_gpu_uuid(gpu_uuid),
                            "memory_used_mib": int(memory.split()[0]),
                            "utilization_percent": int(utilization.split()[0])}
    if set(rows) != {0, 1, 2, 3}:
        raise ValueError("four physical GPUs are required for shape screening")
    return rows


def validate(report: dict, tp: int, dp: int, group: tuple[int, ...],
             snapshot: dict) -> None:
    if report.get("status") != "passed" or report.get("observed_runtime") != {
        "tp": tp, "dp": dp, "ep": True,
        "all2all_backend": "allgather_reducescatter",
    }:
        raise ValueError("native runtime shape did not match the requested EP config")
    ranks = report.get("placement", [])
    if len(ranks) != tp * dp:
        raise ValueError("missing native EP rank placement evidence")
    expected = {snapshot[gpu]["uuid"] for gpu in group}
    actual = {canonical_gpu_uuid(row["actual_gpu_uuid"]) for row in ranks}
    if actual != expected:
        raise ValueError("native ranks used different physical GPUs")
    if any(row["model_class"] != "GraniteMoeForCausalLM"
           or len(row["expert_modules"]) != 32 for row in ranks):
        raise ValueError("Granite model or 32 real MoE layers missing")
    for layer in range(32):
        seen = set()
        ep_ranks = set()
        for rank in ranks:
            module = next(m for m in rank["expert_modules"] if m["layer"] == layer)
            if (not module["checkpoint_samples_match"]
                    or not module["weight_device"].startswith("cuda:")
                    or not module["parallel"]["use_ep"]
                    or module["parallel"]["ep_size"] != tp * dp):
                raise ValueError("native layer weight/EP evidence failed")
            ep_rank = module["parallel"]["ep_rank"]
            if ep_rank in ep_ranks:
                raise ValueError("duplicate EP rank in layer placement")
            ep_ranks.add(ep_rank)
            owned = set(module["global_expert_ids"])
            if seen & owned:
                raise ValueError("an expert is duplicated across EP ranks")
            seen |= owned
        if ep_ranks != set(range(tp * dp)) or seen != set(range(40)):
            raise ValueError("incomplete native expert coverage")
    request = report.get("request", {})
    if (len(request.get("output_token_ids", [])) != 32
            or request.get("finish_reason") != "length"):
        raise ValueError("normal canary request incomplete")


def run_shape(args: argparse.Namespace, output: Path, shape: tuple) -> dict:
    label, tp, dp, group = shape
    before = gpu_snapshot()
    if any(before[gpu]["memory_used_mib"] > 1500
           or before[gpu]["utilization_percent"] > 10 for gpu in group):
        raise RuntimeError("test GPUs have an active or high-memory workload")
    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    cache = (args.compiler_cache or output / "compiler_cache").resolve()
    cache.mkdir(exist_ok=True)
    name = "spotserve-ep-screen-" + uuid.uuid4().hex[:16]
    command = ["podman", "run", "--rm", "--name", name,
               "--network", "none", "--shm-size", "2g"]
    for gpu in group:
        command.extend(["--device", f"nvidia.com/gpu={gpu}"])
    for source, target, mode in (
        (repo, repo, "ro"), (vllm, vllm, "ro"),
        (args.model, args.model, "ro"),
        (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"), "ro"),
        (cache, Path("/taskcache"), "rw"),
    ):
        command.extend(["--volume", f"{source}:{target}:{mode}"])
    env = {
        "PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_HOST_IP": "127.0.0.1",
        "NCCL_SOCKET_IFNAME": "lo", "TRITON_CACHE_DIR": "/taskcache/triton",
        "CUDA_CACHE_PATH": "/taskcache/cuda",
        "FLASHINFER_WORKSPACE_BASE": "/taskcache/flashinfer",
    }
    if label.endswith("-swap"):
        env["CUDA_VISIBLE_DEVICES"] = "1,0"
    for key, value in env.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend([
        args.image, str(vllm / ".venv/bin/python"), "-u", "-m",
        "scripts.probe_moe_ep_shape_screen", "--model", str(args.model),
        "--tp", str(tp), "--dp", str(dp),
    ])
    row = {"label": label, "tp": tp, "dp": dp, "ep": True,
           "requested_gpus": group, "environment_before": before,
           "container_name": name, "command": command, "status": "running"}
    began = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=args.timeout_s)
        log = completed.stdout + completed.stderr
        (output / f"{label}.log").write_text(log, encoding="utf-8")
        row["exit_code"] = completed.returncode
        marker = "MOE_EP_SHAPE_JSON="
        payloads = [line[len(marker):] for line in log.splitlines()
                    if line.startswith(marker)]
        if len(payloads) != 1:
            raise ValueError("native shape screen produced no unique JSON report")
        row["report"] = json.loads(payloads[0])
        if completed.returncode:
            raise RuntimeError("native shape canary returned nonzero")
        validate(row["report"], tp, dp, group, before)
        row["status"] = "passed"
    except subprocess.TimeoutExpired as error:
        partial = (error.stdout or b"") + (error.stderr or b"")
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        (output / f"{label}.log").write_text(partial, encoding="utf-8")
        row["status"] = "failed"
        row["error"] = "native shape screen timed out"
    except Exception:
        row["status"] = "failed"
        row["traceback"] = traceback.format_exc()
    finally:
        subprocess.run(["podman", "stop", "--time", "5", name],
                       capture_output=True, text=True, timeout=30)
        row["elapsed_s"] = time.monotonic() - began
        row["environment_after"] = gpu_snapshot()
        write_artifact(output / f"{label}.json", row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", default="localhost/spotserve-python312-nixl:latest")
    parser.add_argument("--compiler-cache", type=Path)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--shapes", nargs="+", default=[row[0] for row in SHAPES])
    args = parser.parse_args()
    selected = [shape for shape in SHAPES if shape[0] in args.shapes]
    if len(selected) != len(args.shapes):
        raise ValueError("unknown or duplicate shape identifier")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for shape in selected:
        print("MOE_EP_SCREEN_PROGRESS=" + json.dumps({"phase": "begin", "shape": shape[0]}),
              flush=True)
        result = run_shape(args, output, shape)
        results.append({"label": result["label"], "status": result["status"],
                        "elapsed_s": result["elapsed_s"]})
        print("MOE_EP_SCREEN_PROGRESS=" + json.dumps({"phase": "done", **results[-1]}),
              flush=True)
    summary = {"status": "passed" if all(row["status"] == "passed" for row in results)
               else "partial", "shape_results": results,
               "scope": "native_candidate_screen_not_formal_spotserve_ablation"}
    write_artifact(output / "summary.json", summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
