#!/usr/bin/env python3
"""Run audited F1/F2 SpotServe MoE experiments on a multi-node Ray/K8s cluster.

The driver is intentionally fail closed.  It inventories every allocated GPU,
checks that the same checkpoint is visible on every GPU node, measures missing
parallel-shape profiles, runs an Original-vs-MoE-aware pilot, and only then
starts the configured F1/F2 matrix.  It must run in the ServerlessLLM head
pod (or another pod that can reach both the HTTP endpoint and the Ray cluster).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib import request
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RUNNER = REPO_ROOT / "benchmarks/spotserve/run_benchmark.py"

F1_TREATMENTS = (
    "rerouting",
    "reparallelization",
    "original_spotserve",
    "moe_spotserve",
)
F2_TREATMENTS = (
    "original_spotserve",
    "reparallelization_only",
    "migration_only",
    "full_moe_spotserve",
)
# Kept callable for diagnostics, never mixed into the pre-registered matrix.
F2_REPLAY_CONTROLS = (
    "original_spotserve_replay",
    "reparallelization_only_replay",
    "migration_only_replay",
    "full_moe_spotserve_replay",
)
DYNAMIC_F1 = {"reparallelization", "original_spotserve", "moe_spotserve"}
DYNAMIC_F2 = set(F2_TREATMENTS)


class ExperimentError(RuntimeError):
    """A failed evidence gate; formal runs must not continue."""


def taipei_now() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shape_signature(shape: Mapping[str, Any]) -> tuple[int, int, int, int, bool]:
    return (
        int(shape.get("tensor_parallel_size", 1) or 1),
        int(shape.get("pipeline_parallel_size", 1) or 1),
        int(shape.get("data_parallel_size", 1) or 1),
        int(shape.get("replica_count", 1) or 1),
        bool(shape.get("enable_expert_parallel", False)),
    )


def shape_label(shape: Mapping[str, Any]) -> str:
    tp, pp, dp, replicas, ep = shape_signature(shape)
    ep_size = tp * dp if ep else 1
    return f"tp{tp}-pp{pp}-dp{dp}-ep{ep_size}-r{replicas}"


def load_trace_plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    trace = spec.get("trace", {})
    relative = trace.get("plan")
    if not relative:
        raise ExperimentError("trace.plan is required")
    path = Path(str(relative))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ExperimentError(f"trace plan does not exist: {path}")
    plan = read_json(path)
    if not isinstance(plan, dict):
        raise ExperimentError("trace plan must be a JSON object")
    return plan


def trace_target_slot(target: str, maximum_gpus: int) -> int:
    prefix = "pool-slot-"
    if not str(target).startswith(prefix):
        raise ExperimentError(f"unsupported trace target selector: {target}")
    try:
        slot = int(str(target)[len(prefix):])
    except ValueError as exc:
        raise ExperimentError(
            f"invalid trace target selector: {target}"
        ) from exc
    if slot < 0 or slot >= maximum_gpus:
        raise ExperimentError(
            f"trace target {target} exceeds the {maximum_gpus}-GPU pool"
        )
    return slot


def validate_trace_plan(
    spec: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if int(plan.get("schema_version", 0)) != 1:
        raise ExperimentError("trace plan schema_version must be 1")
    units = plan.get("units", {})
    if units.get("capacity") != "gpu" or units.get("time") != "second":
        raise ExperimentError("trace units must be GPU and second")
    cluster_gpus = int(spec["cluster"]["required_gpus"])
    if int(plan.get("maximum_gpus", 0)) != cluster_gpus:
        raise ExperimentError(
            "trace maximum_gpus must equal cluster.required_gpus"
        )
    initial_available = int(plan.get("initial_available_gpus", 0))
    if not 0 < initial_available <= cluster_gpus:
        raise ExperimentError(
            "trace initial_available_gpus must be inside the allocated pool"
        )
    unavailable_targets = list(
        plan.get("initially_unavailable_targets", [])
    )
    unavailable_slots = {
        trace_target_slot(str(target), cluster_gpus)
        for target in unavailable_targets
    }
    if len(unavailable_slots) != cluster_gpus - initial_available:
        raise ExperimentError(
            "initially_unavailable_targets does not match initial capacity"
        )
    events = list(plan.get("events", []))
    if not events or not any(row.get("event") == "add" for row in events):
        raise ExperimentError("trace plan must contain add events")
    if not any(row.get("event") == "preempt" for row in events):
        raise ExperimentError("trace plan must contain preempt events")
    trace_end_s = float(
        plan.get("constraints", {}).get("trace_end_s", 0.0)
    )
    transitions: list[tuple[float, int, int]] = []
    last_notice_s = -1.0
    for event_index, event in enumerate(events):
        event_type = str(event.get("event", ""))
        if event_type not in {"add", "preempt"}:
            raise ExperimentError(
                f"unsupported formal trace event: {event_type}"
            )
        time_s = float(event.get("time_s", -1.0))
        if time_s < 0 or time_s < last_notice_s or time_s > trace_end_s:
            raise ExperimentError("trace event times must be ordered and bounded")
        last_notice_s = time_s
        targets = [str(value) for value in event.get("targets", [])]
        if len(targets) != int(event.get("gpu_count", 0)) or not targets:
            raise ExperimentError(
                "trace gpu_count must equal the number of GPU targets"
            )
        slots = [trace_target_slot(value, cluster_gpus) for value in targets]
        if len(set(slots)) != len(slots):
            raise ExperimentError("one trace event cannot target a GPU twice")
        if event_type == "preempt":
            grace = float(event.get("grace_period_s", -1.0))
            if grace <= 0:
                raise ExperimentError(
                    "formal preempt events require a positive grace period"
                )
            transition_time = time_s + grace
            delta = -1
        else:
            transition_time = time_s
            delta = 1
        for slot in slots:
            transitions.append((transition_time, event_index, delta * (slot + 1)))

    available = {
        slot for slot in range(cluster_gpus) if slot not in unavailable_slots
    }
    capacity_path = [len(available)]
    grouped_times = sorted({row[0] for row in transitions})
    for transition_time in grouped_times:
        for _, _, encoded in sorted(
            (row for row in transitions if row[0] == transition_time),
            key=lambda row: row[1],
        ):
            delta = 1 if encoded > 0 else -1
            slot = abs(encoded) - 1
            if delta > 0:
                if slot in available:
                    raise ExperimentError(
                        f"trace adds already-available pool-slot-{slot}"
                    )
                available.add(slot)
            else:
                if slot not in available:
                    raise ExperimentError(
                        f"trace preempts unavailable pool-slot-{slot}"
                    )
                available.remove(slot)
        if len(available) > cluster_gpus:
            raise ExperimentError("trace exceeds the allocated GPU maximum")
        capacity_path.append(len(available))
    expected_path = [int(value) for value in plan.get(
        "expected_capacity_path", []
    )]
    if capacity_path != expected_path:
        raise ExperimentError(
            f"trace capacity path {capacity_path} != {expected_path}"
        )


def validate_spec(spec: Mapping[str, Any]) -> None:
    if int(spec.get("schema_version", 0)) != 1:
        raise ExperimentError("config schema_version must be 1")
    model = spec.get("model", {})
    if not model.get("path"):
        raise ExperimentError("model.path is required")
    cluster = spec.get("cluster", {})
    if int(cluster.get("required_gpus", 0)) != 8:
        raise ExperimentError("cluster.required_gpus must be exactly 8")
    if int(cluster.get("minimum_gpu_nodes", 0)) < 2:
        raise ExperimentError("cluster.minimum_gpu_nodes must be at least 2")

    workload = spec.get("workload", {})
    prompt_tokens = int(workload.get("prompt_tokens", 0))
    output_tokens = int(workload.get("output_tokens", 0))
    max_model_len = int(workload.get("max_model_len", 0))
    preempt_after = int(workload.get("preempt_after_output_tokens", 0))
    if prompt_tokens < 4096:
        raise ExperimentError("workload.prompt_tokens must be >= 4096")
    if output_tokens < 512:
        raise ExperimentError("workload.output_tokens must be >= 512")
    if prompt_tokens + output_tokens > max_model_len:
        raise ExperimentError("prompt_tokens + output_tokens exceeds max_model_len")
    if not 0 < preempt_after < output_tokens:
        raise ExperimentError(
            "preempt_after_output_tokens must be inside the decode window"
        )
    calibration_tokens = int(workload.get("calibration_output_tokens", 0))
    if not 0 < calibration_tokens < preempt_after:
        raise ExperimentError(
            "calibration_output_tokens must be between 1 and the preemption target"
        )
    progress_min = int(workload.get("preemption_progress_min_tokens", 0))
    progress_max = int(workload.get("preemption_progress_max_tokens", 0))
    if not 0 <= progress_min <= preempt_after <= progress_max < output_tokens:
        raise ExperimentError(
            "preemption progress window must contain the target and remain inside decode"
        )
    if float(workload.get("preemption_time_s", 0.0)) <= float(
        workload.get("measured_start_s", 0.0)
    ):
        raise ExperimentError(
            "fallback preemption_time_s must be later than measured_start_s"
        )

    initial = spec.get("initial_parallel", {})
    if not initial.get("enable_expert_parallel"):
        raise ExperimentError("initial_parallel must enable expert parallelism")
    if int(initial.get("tensor_parallel_size", 0)) != 1:
        raise ExperimentError(
            "distributed Ray-DP ownership currently supports TP=1 only"
        )
    initial_gpus = (
        int(initial.get("tensor_parallel_size", 1))
        * int(initial.get("pipeline_parallel_size", 1))
        * int(initial.get("data_parallel_size", 1))
    )
    if initial_gpus != int(initial.get("num_gpus", 0)):
        raise ExperimentError("initial_parallel num_gpus does not match TP*PP*DP")
    initial_instances = int(initial.get("sllm_instances", 0))
    if initial_instances <= 0:
        raise ExperimentError("initial_parallel.sllm_instances must be positive")
    initial_deployment_gpus = initial_gpus * initial_instances
    cluster_gpus = int(cluster["required_gpus"])
    if initial_deployment_gpus > cluster_gpus:
        raise ExperimentError(
            "initial instances consume more than cluster.required_gpus"
        )
    minimum_headroom = int(
        cluster.get("minimum_migration_headroom_gpus", 0) or 0
    )
    actual_headroom = cluster_gpus - initial_deployment_gpus
    if actual_headroom < minimum_headroom:
        raise ExperimentError(
            "initial deployment leaves insufficient migration headroom: "
            f"{actual_headroom} < {minimum_headroom}"
        )

    candidates = list(spec.get("candidate_shapes", []))
    signatures = {shape_signature(shape) for shape in candidates}
    if len(signatures) < 2:
        raise ExperimentError(
            "planner requires at least two distinct parallel shapes"
        )
    for shape in candidates:
        tp, pp, dp, replicas, ep = shape_signature(shape)
        if tp != 1:
            raise ExperimentError(
                f"{shape_label(shape)}: cross-node Ray-DP candidate must use TP=1"
            )
        if not ep or tp * dp <= 1:
            raise ExperimentError(
                f"{shape_label(shape)}: every candidate must use EP>1"
            )
        expected = tp * pp * dp * replicas
        if expected != int(shape.get("num_gpus", 0)):
            raise ExperimentError(
                f"{shape_label(shape)}: num_gpus must equal TP*PP*DP*replicas"
            )
        if replicas != 1:
            raise ExperimentError(
                f"{shape_label(shape)}: distributed candidate replica_count must be 1"
            )

    experiment = spec.get("experiment", {})
    if int(experiment.get("formal_repeats", 0)) < 1:
        raise ExperimentError("formal_repeats must be positive")
    if int(experiment.get("profile_repeats", 0)) < 2:
        raise ExperimentError("profile_repeats must be at least 2")
    validate_trace_plan(spec, load_trace_plan(spec))
    trace_plan = load_trace_plan(spec)
    formal_bursts = int(
        trace_plan.get("constraints", {}).get("formal_burst_count", 0)
    )
    if formal_bursts <= 0 or int(workload.get("request_count", 0)) != (
        formal_bursts * int(workload.get("burst_size", 0))
    ):
        raise ExperimentError(
            "formal request_count must equal trace burst_count * burst_size"
        )
    trace_headroom = (
        int(trace_plan["initial_available_gpus"])
        - initial_deployment_gpus
    )
    if trace_headroom < minimum_headroom:
        raise ExperimentError(
            "trace initial capacity leaves insufficient migration headroom: "
            f"{trace_headroom} < {minimum_headroom}"
        )


def git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(
            list(args), cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        commit = command("git", "rev-parse", "HEAD")
        status = command("git", "status", "--short")
    except Exception:
        commit, status = "unavailable", "unavailable"
    return {"commit": commit, "status": status}


def _manifest_for_model(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config missing: {config_path}")
    patterns = (
        "config.json",
        "tokenizer*.json",
        "tokenizer*.model",
        "special_tokens_map.json",
        "*.safetensors",
        "*.bin",
        "*.index.json",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    files = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique
    ]
    config = read_json(config_path)
    return {
        "model_path": str(root),
        "config": config,
        "files": files,
        "manifest_sha256": digest_value(files),
        "total_weight_bytes": sum(
            row["size"]
            for row in files
            if row["path"].endswith((".safetensors", ".bin"))
        ),
    }


def probe_cluster(
    spec: Mapping[str, Any], ray_address: str, ray_namespace: str
) -> dict[str, Any]:
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ModuleNotFoundError as exc:
        raise ExperimentError("Ray is required for cluster preflight") from exc

    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            namespace=ray_namespace,
            ignore_reinit_error=True,
        )
    ray_nodes = [node for node in ray.nodes() if node.get("Alive")]
    if spec["cluster"].get("require_single_control_head", False):
        heads = [node for node in ray_nodes
                 if node.get("Resources", {}).get("control_node", 0) > 0]
        if len(heads) != 1 or heads[0].get("Resources", {}).get("GPU", 0) > 0:
            raise ExperimentError("require exactly one control_node head with zero GPUs")
    gpu_nodes = [
        node
        for node in ray_nodes
        if float((node.get("Resources") or {}).get("GPU", 0) or 0) > 0
    ]
    if not gpu_nodes:
        raise ExperimentError("Ray reports no live GPU nodes")

    class GPUProbe:
        def __init__(
            self, model_path: str, include_manifest: bool,
            image_digest_env: str,
        ):
            self.model_path = model_path
            self.include_manifest = include_manifest
            self.image_digest_env = image_digest_env

        def collect(self) -> dict[str, Any]:
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
            visible_device = visible_devices.split(",", 1)[0].strip()
            if not visible_device:
                raise RuntimeError("Ray GPU actor has no CUDA_VISIBLE_DEVICES")
            query = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    visible_device,
                    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
                    "utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().splitlines()
            if len(query) != 1:
                raise RuntimeError(
                    f"one Ray GPU actor must see exactly one GPU, observed {query}"
                )
            values = [value.strip() for value in query[0].split(",")]
            row: dict[str, Any] = {
                "hostname": socket.gethostname(),
                "cuda_visible_devices": visible_devices,
                "image_digest": os.getenv(self.image_digest_env, ""),
                "gpu": {
                    "index": values[0],
                    "uuid": values[1],
                    "name": values[2],
                    "memory_total_mib": int(values[3]),
                    "memory_used_mib": int(values[4]),
                    "memory_free_mib": int(values[5]),
                    "utilization_percent": int(values[6]),
                    "driver_version": values[7],
                },
            }
            try:
                import torch

                row["torch_version"] = torch.__version__
                row["torch_cuda_version"] = torch.version.cuda
            except Exception as exc:
                row["torch_error"] = str(exc)
            try:
                import vllm

                row["vllm_version"] = vllm.__version__
            except Exception as exc:
                row["vllm_error"] = str(exc)
            if self.include_manifest:
                row["model_manifest"] = _manifest_for_model(self.model_path)
            return row

    RemoteGPUProbe = ray.remote(num_cpus=0.1, num_gpus=1)(GPUProbe)
    actors = []
    actor_nodes = []
    for node in gpu_nodes:
        gpu_count = int(float((node.get("Resources") or {}).get("GPU", 0)))
        for local_index in range(gpu_count):
            actor = RemoteGPUProbe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(
                str(spec["model"]["path"]),
                local_index == 0,
                str(spec["cluster"].get("image_digest_env", "IMAGE_DIGEST")),
            )
            actors.append(actor)
            actor_nodes.append(node)
    try:
        observations = ray.get(
            [actor.collect.remote() for actor in actors],
            timeout=float(spec["cluster"].get("probe_timeout_s", 900)),
        )
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)

    nodes: list[dict[str, Any]] = []
    for node in gpu_nodes:
        resources = node.get("Resources") or {}
        physical_markers = sorted(
            key
            for key in resources
            if key.startswith("spotserve_physical_host_")
        )
        worker_ids = sorted(
            key.removeprefix("worker_id_")
            for key, value in resources.items()
            if key.startswith("worker_id_") and float(value or 0) > 0
        )
        nodes.append({
            "ray_node_id": node["NodeID"],
            "address": node.get("NodeManagerAddress"),
            "gpu_count": int(float(resources.get("GPU", 0) or 0)),
            "worker_ids": worker_ids,
            "physical_host_markers": physical_markers,
        })
    for observation, node in zip(observations, actor_nodes):
        observation["ray_node_id"] = node["NodeID"]
        observation["address"] = node.get("NodeManagerAddress")

    profile = {
        "schema_version": 1,
        "created_at": taipei_now(),
        "ray_address": ray_address,
        "ray_namespace": ray_namespace,
        "nodes": nodes,
        "gpus": observations,
        "git": git_state(),
        "image_digest": os.getenv(
            str(spec["cluster"].get("image_digest_env", "IMAGE_DIGEST")), ""
        ),
    }
    validate_cluster_profile(spec, profile)
    profile["hardware_fingerprint"] = digest_value({
        # Deliberately exclude UUID, utilization, free memory, Ray node IDs,
        # pod IPs, and timestamps.  Those are inventory/audit fields, not the
        # stable hardware/software identity used to reuse offline profiles.
        "gpu_types": sorted({
            (
                str(row["gpu"].get("name", "")),
                int(row["gpu"].get("memory_total_mib", 0) or 0),
                str(row["gpu"].get("driver_version", "")),
            )
            for row in observations
        }),
        "software": sorted({
            (
                str(row.get("torch_version", "")),
                str(row.get("torch_cuda_version", "")),
                str(row.get("vllm_version", "")),
                str(row.get("image_digest", "")),
            )
            for row in observations
        }),
        "model_manifests": sorted({
            row.get("model_manifest", {}).get("manifest_sha256")
            for row in observations
            if row.get("model_manifest")
        }),
        "image_digest": profile["image_digest"],
    })
    return profile


def validate_cluster_profile(
    spec: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    cluster = spec["cluster"]
    gpus = list(profile.get("gpus", []))
    if len(gpus) < int(cluster["required_gpus"]):
        raise ExperimentError(
            f"need {cluster['required_gpus']} GPUs, observed {len(gpus)}"
        )
    if cluster.get("require_exact_gpu_count", True) and len(gpus) != int(
        cluster["required_gpus"]
    ):
        raise ExperimentError(
            f"expected exactly {cluster['required_gpus']} isolated GPUs, "
            f"observed {len(gpus)}"
        )
    uuids = [row.get("gpu", {}).get("uuid") for row in gpus]
    if len(set(uuids)) != len(uuids) or any(not value for value in uuids):
        raise ExperimentError("GPU UUID inventory is missing or duplicated")
    if len(profile.get("nodes", [])) < int(cluster["minimum_gpu_nodes"]):
        raise ExperimentError("not enough distinct Ray GPU nodes")
    if cluster.get("gpus_per_worker") is not None:
        expected = int(cluster["gpus_per_worker"])
        for node in profile.get("nodes", []):
            if int(node.get("gpu_count", 0)) != expected or len(node.get("worker_ids", [])) != 1:
                raise ExperimentError("worker contract requires one unique worker ID and one GPU per pod")
        ids = [worker for node in profile.get("nodes", []) for worker in node["worker_ids"]]
        if len(set(ids)) != len(ids):
            raise ExperimentError("worker IDs must be unique across Ray pods")
    if cluster.get("require_physical_host_markers", True):
        markers = {
            marker
            for node in profile.get("nodes", [])
            for marker in node.get("physical_host_markers", [])
        }
        if len(markers) < int(cluster.get("minimum_physical_hosts", 1)):
            raise ExperimentError(
                "physical-host markers are missing; pod count is not host count"
            )
    maximum_used = int(cluster.get("maximum_initial_gpu_memory_used_mib", 1024))
    busy = [
        row["gpu"]["uuid"]
        for row in gpus
        if int(row["gpu"].get("memory_used_mib", 0)) > maximum_used
    ]
    if busy:
        raise ExperimentError(f"allocated GPUs are already busy: {busy}")
    if cluster.get("require_homogeneous_gpu_model", True):
        models = {str(row["gpu"].get("name", "")) for row in gpus}
        if len(models) != 1:
            raise ExperimentError(
                "mixed GPU models would confound the planner comparison: "
                f"{sorted(models)}"
            )
        memory_sizes = {
            int(row["gpu"].get("memory_total_mib", 0) or 0) for row in gpus
        }
        if 0 in memory_sizes or len(memory_sizes) != 1:
            raise ExperimentError(
                "GPU memory capacities differ or are unavailable: "
                f"{sorted(memory_sizes)}"
            )
    version_fields = (
        "driver_version", "torch_version", "torch_cuda_version", "vllm_version"
    )
    for field in version_fields:
        if field == "driver_version":
            values = {str(row["gpu"].get(field, "")) for row in gpus}
        else:
            values = {str(row.get(field, "")) for row in gpus}
        if "" in values or len(values) != 1:
            raise ExperimentError(
                f"cross-node {field} is missing or inconsistent: {sorted(values)}"
            )
    if cluster.get("require_image_digest", True):
        image_digests = {str(row.get("image_digest", "")) for row in gpus}
        head_digest = str(profile.get("image_digest", ""))
        if "" in image_digests or not head_digest:
            raise ExperimentError(
                "IMAGE_DIGEST must be exported in the head and every GPU worker"
            )
        if image_digests != {head_digest}:
            raise ExperimentError(
                "head and GPU workers do not use one immutable image digest"
            )
    manifests = [
        row["model_manifest"]
        for row in gpus
        if isinstance(row.get("model_manifest"), Mapping)
    ]
    if len(manifests) != len(profile.get("nodes", [])):
        raise ExperimentError("model manifest was not collected on every GPU node")
    if len({row["manifest_sha256"] for row in manifests}) != 1:
        raise ExperimentError("model checkpoint differs between GPU nodes")
    config = manifests[0].get("config", {})
    expert_count = max(
        int(config.get(key, 0) or 0)
        for key in ("num_experts", "num_local_experts", "n_routed_experts")
    )
    architectures = [str(value) for value in config.get("architectures", [])]
    if expert_count <= 1 or not any("moe" in value.lower() for value in architectures):
        raise ExperimentError("configured checkpoint is not a verified MoE model")
    expected_architectures = set(spec["model"].get("architectures", []))
    if expected_architectures and not expected_architectures.intersection(
        architectures
    ):
        raise ExperimentError(
            f"unexpected model architecture: {architectures}"
        )
    maximum_positions = int(config.get("max_position_embeddings", 0) or 0)
    requested_model_len = int(spec["workload"]["max_model_len"])
    if maximum_positions <= 0 or requested_model_len > maximum_positions:
        raise ExperimentError(
            f"max_model_len={requested_model_len} exceeds checkpoint "
            f"max_position_embeddings={maximum_positions}"
        )
    for shape in [spec["initial_parallel"], *spec["candidate_shapes"]]:
        ep_size = (
            int(shape["tensor_parallel_size"])
            * int(shape["data_parallel_size"])
        )
        if expert_count % ep_size:
            raise ExperimentError(
                f"{shape_label(shape)} cannot evenly partition "
                f"{expert_count} experts across EP={ep_size}"
            )
    total_weight_bytes = int(manifests[0].get("total_weight_bytes", 0) or 0)
    minimum_gpu_bytes = min(
        int(row["gpu"]["memory_total_mib"]) * 1024 * 1024 for row in gpus
    )
    memory_budget = minimum_gpu_bytes * float(
        spec["model"].get("gpu_memory_utilization", 0.9)
    )
    for shape in [spec["initial_parallel"], *spec["candidate_shapes"]]:
        ep_size = int(shape["tensor_parallel_size"]) * int(
            shape["data_parallel_size"]
        )
        if total_weight_bytes and total_weight_bytes / ep_size > memory_budget:
            raise ExperimentError(
                f"{shape_label(shape)} fails the conservative per-rank "
                "checkpoint-size/GPU-memory feasibility gate"
            )


def probe_network(
    profile: Mapping[str, Any], spec: Mapping[str, Any], ray_address: str,
    ray_namespace: str,
) -> dict[str, Any]:
    """Measure direct pod-to-pod TCP; this is not claimed as GPUDirect/NCCL."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            namespace=ray_namespace,
            ignore_reinit_error=True,
        )
    nodes = list(profile["nodes"])
    byte_count = int(spec["cluster"].get("network_probe_bytes", 32 * 1024 * 1024))

    class Sink:
        def __init__(self, address: str):
            self.address = address
            self.listener: socket.socket | None = None
            self.thread: threading.Thread | None = None
            self.result: dict[str, Any] = {}

        def open(self, expected_bytes: int) -> dict[str, Any]:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("0.0.0.0", 0))
            self.listener.listen(1)
            port = self.listener.getsockname()[1]

            def receive() -> None:
                assert self.listener is not None
                connection, _ = self.listener.accept()
                received = 0
                started = time.monotonic()
                with connection:
                    while received < expected_bytes:
                        chunk = connection.recv(min(1024 * 1024, expected_bytes - received))
                        if not chunk:
                            break
                        received += len(chunk)
                self.result = {
                    "bytes": received,
                    "receive_s": time.monotonic() - started,
                }
                self.listener.close()

            self.thread = threading.Thread(target=receive, daemon=True)
            self.thread.start()
            return {"address": self.address, "port": port}

        def wait(self) -> dict[str, Any]:
            if self.thread is None:
                raise RuntimeError("sink was not opened")
            self.thread.join(timeout=300)
            if self.thread.is_alive():
                raise TimeoutError("network sink timed out")
            return self.result

    def send(address: str, port: int, expected_bytes: int) -> dict[str, Any]:
        chunk = b"s" * min(1024 * 1024, expected_bytes)
        sent = 0
        started = time.monotonic()
        with socket.create_connection((address, port), timeout=30) as stream:
            while sent < expected_bytes:
                payload = chunk[: min(len(chunk), expected_bytes - sent)]
                stream.sendall(payload)
                sent += len(payload)
        elapsed = time.monotonic() - started
        return {
            "bytes": sent,
            "send_s": elapsed,
            "throughput_gbps": sent * 8 / elapsed / 1e9,
        }

    RemoteSink = ray.remote(num_cpus=0.1)(Sink)
    remote_send = ray.remote(num_cpus=0.1)(send)
    rows = []
    for source in nodes:
        for target in nodes:
            if source["ray_node_id"] >= target["ray_node_id"]:
                continue
            sink = RemoteSink.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=target["ray_node_id"], soft=False
                )
            ).remote(str(target["address"]))
            try:
                endpoint = ray.get(sink.open.remote(byte_count), timeout=30)
                sent = ray.get(
                    remote_send.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=source["ray_node_id"], soft=False
                        )
                    ).remote(endpoint["address"], endpoint["port"], byte_count),
                    timeout=300,
                )
                received = ray.get(sink.wait.remote(), timeout=300)
            finally:
                ray.kill(sink, no_restart=True)
            if sent["bytes"] != byte_count or received.get("bytes") != byte_count:
                raise ExperimentError("cross-node TCP profile transferred partial data")
            rows.append({
                "source_node": source["ray_node_id"],
                "target_node": target["ray_node_id"],
                **sent,
                **received,
            })
    if not rows:
        raise ExperimentError("network profile needs at least two GPU nodes")
    return {
        "scope": "pod_to_pod_tcp_not_gpudirect_or_nccl",
        "created_at": taipei_now(),
        "probe_bytes": byte_count,
        "pairs": rows,
        "minimum_throughput_gbps": min(row["throughput_gbps"] for row in rows),
    }


def worker_ids(profile: Mapping[str, Any]) -> list[str]:
    values = sorted({
        str(worker_id)
        for node in profile.get("nodes", [])
        for worker_id in node.get("worker_ids", [])
    })
    if len(values) < 2:
        raise ExperimentError("Ray nodes do not advertise multiple worker_id_* resources")
    return values


def exact_chat_prompt(
    model_path: str, target_tokens: int, nonce: str = "request-000"
) -> tuple[str, int]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ExperimentError("transformers is required to freeze the workload") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )

    def count(repetitions: int) -> tuple[str, int]:
        content = (
            f"Unique request nonce: {nonce}. "
            "Analyze this retained KV context and continue deterministically."
            + " kv" * repetitions
        )
        messages = [{"role": "user", "content": content}]
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        else:
            ids = tokenizer.encode(content, add_special_tokens=True)
        return content, len(ids)

    low, high = 0, target_tokens * 2
    while low <= high:
        middle = (low + high) // 2
        _, observed = count(middle)
        if observed < target_tokens:
            low = middle + 1
        elif observed > target_tokens:
            high = middle - 1
        else:
            content, observed = count(middle)
            return content, observed
    for repetitions in range(max(0, high - 64), low + 65):
        content, observed = count(repetitions)
        if observed == target_tokens:
            return content, observed
    raise ExperimentError(
        f"could not construct an exact {target_tokens}-token chat prompt"
    )


def exact_chat_prompts(
    model_path: str, target_tokens: int, count: int
) -> list[str]:
    prompts = []
    for index in range(count):
        prompt, observed = exact_chat_prompt(
            model_path, target_tokens, nonce=f"request-{index:04d}"
        )
        if observed != target_tokens:
            raise ExperimentError(
                f"prompt {index} has {observed}, expected {target_tokens} tokens"
            )
        prompts.append(prompt)
    if len(set(prompts)) != len(prompts):
        raise ExperimentError("workload prompts are not unique")
    return prompts


def build_workload(
    spec: Mapping[str, Any], prompts: list[str], profile: bool = False,
    output_tokens: int | None = None, phase_name: str = "measured",
) -> list[dict[str, Any]]:
    workload = spec["workload"]
    warmups = int(workload.get("warmup_requests", 4))
    measured = int(
        workload.get("profile_request_count", 8)
        if profile
        else workload.get("request_count", 24)
    )
    burst = int(workload.get("burst_size", 8))
    measured_start = float(workload.get("measured_start_s", 20.0))
    burst_gap = float(workload.get("burst_gap_s", 5.0))
    formal_burst_times = (
        []
        if profile or workload.get("core_probe_mode", False)
        else [
            float(value)
            for value in workload.get("formal_burst_start_times_s", [])
        ]
    )
    if formal_burst_times and len(formal_burst_times) * burst != measured:
        raise ExperimentError(
            "resolved formal burst times do not cover every measured request"
        )
    if len(prompts) < warmups + measured:
        raise ExperimentError(
            f"need {warmups + measured} unique prompts, got {len(prompts)}"
        )
    measured_output_tokens = int(
        output_tokens if output_tokens is not None else workload["output_tokens"]
    )
    rows: list[dict[str, Any]] = []
    for index in range(warmups):
        rows.append({
            "time": index * 0.1,
            "benchmark_phase": "warmup",
            "request_id": f"warmup-{index:03d}",
            "messages": [{"role": "user", "content": prompts[index]}],
            "max_tokens": int(workload.get("warmup_output_tokens", 32)),
            "temperature": 0.0,
            "ignore_eos": True,
        })
    for index in range(measured):
        arrival_time = (
            formal_burst_times[index // burst]
            if formal_burst_times
            else measured_start + (index // burst) * burst_gap
        )
        rows.append({
            "time": arrival_time,
            "benchmark_phase": phase_name,
            "request_id": f"{phase_name}-{index:03d}",
            "messages": [{
                "role": "user", "content": prompts[warmups + index]
            }],
            "max_tokens": measured_output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "_spotserve_return_token_ids": True,
        })
    return rows


def local_dp_size(shape: Mapping[str, Any], hardware: Mapping[str, Any]) -> int:
    if shape.get("data_parallel_size_local") is not None:
        return int(shape["data_parallel_size_local"])
    dp_size = int(shape["data_parallel_size"])
    maximum_node_gpus = max(
        (int(node.get("gpu_count", 0) or 0) for node in hardware.get("nodes", [])),
        default=0,
    )
    # Let vLLM pack the complete DP group on one node when possible.  Only
    # request span mode when the allocated topology cannot fit that group.
    return dp_size if maximum_node_gpus >= dp_size else 1


def ray_dp_pack_strategy(
    shape: Mapping[str, Any], hardware: Mapping[str, Any]
) -> str:
    if int(shape["tensor_parallel_size"]) * int(shape.get("pipeline_parallel_size", 1)) == 1:
        # DP ranks span pods, but each rank's TP*PP world fits in one pod.
        # vLLM's "span" mode is for a single TP/PP world spanning nodes.
        return "fill"
    return (
        "fill"
        if local_dp_size(shape, hardware) == int(shape["data_parallel_size"])
        else "span"
    )


def base_deploy_config(
    spec: Mapping[str, Any], hardware: Mapping[str, Any], shape: Mapping[str, Any],
    model_name: str, instances: int, metrics_path: Path,
) -> dict[str, Any]:
    workload = spec["workload"]
    backend = {
        "pretrained_model_name_or_path": str(spec["model"]["path"]),
        "load_format": str(spec["model"].get("load_format", "auto")),
        "torch_dtype": str(spec["model"].get("dtype", "bfloat16")),
        "gpu_memory_utilization": float(
            spec["model"].get("gpu_memory_utilization", 0.90)
        ),
        "max_model_len": int(workload["max_model_len"]),
        "max_num_seqs": int(workload.get("max_num_seqs", 16)),
        "max_num_batched_tokens": int(
            workload.get("max_num_batched_tokens", workload["max_model_len"])
        ),
        "tensor_parallel_size": int(shape["tensor_parallel_size"]),
        "pipeline_parallel_size": int(shape.get("pipeline_parallel_size", 1)),
        "data_parallel_size": int(shape["data_parallel_size"]),
        "data_parallel_size_local": local_dp_size(shape, hardware),
        "data_parallel_backend": "ray",
        "enable_expert_parallel": True,
        "planned_effective_expert_parallel_size": int(
            shape["tensor_parallel_size"]
        ) * int(shape["data_parallel_size"]),
        "expert_parallel_size_source": "derived_from_tp_dp",
        "enable_moe_route_instrumentation": True,
        "all2all_backend": str(spec["model"].get(
            "all2all_backend", "allgather_reducescatter"
        )),
        "expert_placement_strategy": "linear",
        "enable_prefix_caching": bool(
            spec["model"].get("enable_prefix_caching", False)
        ),
        "enforce_eager": bool(spec["model"].get("enforce_eager", True)),
        "trust_remote_code": True,
        "ray_dp_pack_strategy": ray_dp_pack_strategy(shape, hardware),
        "spotserve_vllm_ray_dp_owns_gpus": True,
        "spotserve_distributed_gpu_allocation": True,
        "spotserve_target_worker_nodes": worker_ids(hardware),
        "spotserve_expected_prompt_tokens": int(workload["prompt_tokens"]),
        "spotserve_preemption_progress_min_tokens": int(
            workload["preemption_progress_min_tokens"]
        ),
        "spotserve_preemption_progress_max_tokens": int(
            workload["preemption_progress_max_tokens"]
        ),
        "trace_debug": True,
    }
    return {
        "model": model_name,
        "backend": "vllm",
        "router_num_cpus": 0,
        "num_cpus": 1,
        "num_gpus": int(shape["num_gpus"]),
        "auto_scaling_config": {
            "metric": "concurrency",
            "target": int(workload.get("max_num_seqs", 16)),
            "min_instances": instances,
            "max_instances": instances,
            "keep_alive": 3600,
        },
        "backend_config": backend,
        "router_config": {
            "count_preempting_toward_capacity": True,
            "enable_reparallelization": False,
            "enable_context_migration": False,
            "enable_kv_cache_migration": False,
            "metrics_path": str(metrics_path),
        },
    }


def _kv_bytes_per_request(model_config: Mapping[str, Any], prompt_tokens: int,
                          dtype_bytes: int) -> int:
    layers = int(model_config.get("num_hidden_layers", 0) or 0)
    hidden = int(model_config.get("hidden_size", 0) or 0)
    attention_heads = int(model_config.get("num_attention_heads", 0) or 0)
    kv_heads = int(
        model_config.get("num_key_value_heads", attention_heads) or attention_heads
    )
    if not layers or not hidden or not attention_heads or not kv_heads:
        raise ExperimentError("model config lacks KV geometry")
    head_dim = hidden // attention_heads
    return 2 * layers * kv_heads * head_dim * dtype_bytes * prompt_tokens


def _routed_expert_weight_bytes(
    model_config: Mapping[str, Any], dtype_bytes: int
) -> int:
    layers = int(model_config.get("num_hidden_layers", 0) or 0)
    hidden = int(model_config.get("hidden_size", 0) or 0)
    experts = max(
        int(model_config.get(key, 0) or 0)
        for key in ("num_experts", "num_local_experts", "n_routed_experts")
    )
    intermediate = int(
        model_config.get(
            "moe_intermediate_size",
            model_config.get("expert_intermediate_size", 0),
        )
        or 0
    )
    if not layers or not hidden or not experts or not intermediate:
        raise ExperimentError("model config lacks routed-expert weight geometry")
    # Qwen MoE routed experts use gated MLPs: gate/up/down projections.
    return 3 * layers * experts * hidden * intermediate * dtype_bytes


def build_capability(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    network: Mapping[str, Any], profiles: list[Mapping[str, Any]],
) -> dict[str, Any]:
    manifests = [
        row["model_manifest"]
        for row in hardware["gpus"]
        if row.get("model_manifest")
    ]
    model_config = manifests[0]["config"]
    dtype_bytes = 2 if spec["model"].get("dtype") in {
        "float16", "bfloat16", "half"
    } else 4
    kv_bytes = _kv_bytes_per_request(
        model_config, int(spec["workload"]["prompt_tokens"]), dtype_bytes
    )
    bandwidth = float(network["minimum_throughput_gbps"])
    transfer_ms = kv_bytes * 8 / (bandwidth * 1e9) * 1000
    initial_ep = int(spec["initial_parallel"]["data_parallel_size"])
    checkpoint_weight_bytes = int(
        manifests[0].get("total_weight_bytes", 0)
    )
    estimated_expert_weight_bytes = _routed_expert_weight_bytes(
        model_config, dtype_bytes
    )
    if checkpoint_weight_bytes:
        estimated_expert_weight_bytes = min(
            estimated_expert_weight_bytes, checkpoint_weight_bytes
        )
    supported = []
    for profile in profiles:
        shape = profile["shape"]
        target_ep = int(shape["data_parallel_size"])
        movement_fraction = abs(initial_ep - target_ep) / max(initial_ep, target_ep)
        moved_expert_bytes = estimated_expert_weight_bytes * movement_fraction
        movement_ms = moved_expert_bytes * 8 / (bandwidth * 1e9) * 1000
        supported.append({
            **shape,
            "batch_size": int(spec["workload"].get("max_num_seqs", 16)),
            "latency_estimate_ms": float(profile["latency_avg_ms"]),
            "throughput_estimate_req_s": float(profile["throughput_req_s"]),
            "load_time_estimate_ms": float(profile["deployment_ready_latency_ms"]),
            "migration_cost_estimate_ms": transfer_ms,
            "expert_weight_movement_cost_estimate_ms": movement_ms,
            "expert_weight_movement_bytes_estimate": int(
                moved_expert_bytes
            ),
            "reason": "measured_exact_workload_k8s_ep_profile",
            "profile_artifact": str(profile["artifact"]),
            "profile_sha256": str(profile["artifact_sha256"]),
        })
    return {
        "source": "measured_k8s_offline_profiles",
        "hardware_fingerprint": hardware["hardware_fingerprint"],
        "network_profile_sha256": digest_value(network),
        "network_cost_measurement_kind": "pod_to_pod_tcp_proxy_not_nixl",
        "kv_bytes_per_migrated_request": kv_bytes,
        "routed_expert_weight_bytes_estimate": estimated_expert_weight_bytes,
        "checkpoint_weight_bytes": checkpoint_weight_bytes,
        "supports_tp": True,
        "supports_dp": True,
        "supports_ep": True,
        "supported_configs": supported,
    }


def execute_benchmark(
    matrix_path: Path, endpoint: str, ray_address: str, ray_namespace: str,
    log_path: Path, request_timeout_s: float, trace_event_timeout_s: float,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-u",
        str(BENCHMARK_RUNNER),
        "--config",
        str(matrix_path),
        "--endpoint",
        endpoint,
        "--request-timeout",
        str(request_timeout_s),
        "--trace-event-timeout",
        str(trace_event_timeout_s),
        "--ray-address",
        ray_address,
        "--ray-namespace",
        ray_namespace,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream into the artifact so engine startup can be diagnosed while the
    # child is still running, rather than hiding its log until it exits.
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise ExperimentError(
            f"benchmark failed with exit {completed.returncode}; see {log_path}"
        )
    matrix = read_json(matrix_path)
    summary_path = Path(matrix["output_dir"]) / "latest_summary.json"
    if not summary_path.is_file():
        raise ExperimentError(f"benchmark did not write {summary_path}")
    return read_json(summary_path)


def _completion_tokens(row: Mapping[str, Any]) -> int:
    response = row.get("response", {})
    usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
    return int(usage.get("completion_tokens", 0) or 0)


def validate_generated_tokens(
    summary: Mapping[str, Any], expected: int, allow_failures: bool = False,
    phase_name: str = "measured", expected_requests: int | None = None,
) -> None:
    run_dir = Path(str(summary["run_dir"]))
    rows = [
        json.loads(line)
        for line in (run_dir / "raw_requests.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    measured = [row for row in rows if row.get("benchmark_phase") == phase_name]
    successful = [row for row in measured if row.get("success")]
    invalid_success = any(
        _completion_tokens(row) != expected for row in successful
    )
    invalid_failures = not allow_failures and len(successful) != len(measured)
    invalid_count = (
        expected_requests is not None
        and len(measured) != int(expected_requests)
    )
    if (
        not measured
        or not successful
        or invalid_success
        or invalid_failures
        or invalid_count
    ):
        raise ExperimentError(
            f"{phase_name} requests did not contain exactly "
            f"{expected_requests if expected_requests is not None else 'the expected'} "
            f"requests generating {expected} tokens"
        )


def ensure_candidate_profiles(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    network: Mapping[str, Any], prompts: list[str], output: Path, endpoint: str,
    ray_address: str, ray_namespace: str, force: bool, cache_only: bool = False,
) -> list[dict[str, Any]]:
    generated = output / "generated" / "profiles"
    full_workload_path = generated / "profile-workload-full.jsonl"
    short_workload_path = generated / "profile-workload-short.jsonl"
    calibration_tokens = int(spec["workload"]["calibration_output_tokens"])
    write_jsonl(
        full_workload_path,
        build_workload(spec, prompts, profile=True, phase_name="measured"),
    )
    write_jsonl(
        short_workload_path,
        build_workload(
            spec,
            prompts,
            profile=True,
            output_tokens=calibration_tokens,
            phase_name="calibration_short",
        ),
    )
    passing: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repeats = int(spec["experiment"]["profile_repeats"])
    for shape in spec["candidate_shapes"]:
        label = shape_label(shape)
        artifact_path = output / "profiles" / f"{label}.json"
        profile_input_hash = digest_value({
            "shape": shape,
            "hardware": hardware["hardware_fingerprint"],
            "worker_placement": hardware.get("nodes", []),
            "network_probe_protocol_bytes": int(spec["cluster"].get("network_probe_bytes", 33554432)),
            "model": spec["model"],
            "workload": spec["workload"],
            "prompt_hashes": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in prompts
            ],
        })
        if artifact_path.is_file() and not force:
            prior = read_json(artifact_path)
            if prior.get("profile_input_hash") == profile_input_hash:
                if prior.get("status") == "passed":
                    passing.append(prior)
                    continue
                if prior.get("status") == "failed":
                    failures.append(prior)
                    continue

        if cache_only:
            raise ExperimentError(
                f"{label}: core-probe needs a matching measured profile; "
                "prepare it once with --phase profile first"
            )

        model_name = f"profile-{label}"
        metrics_path = (output / "profiles" / f"{label}-router.jsonl").resolve()
        deploy = base_deploy_config(
            spec, hardware, shape, model_name, 1, metrics_path
        )
        deploy_path = generated / f"{label}-deploy.json"
        write_json(deploy_path, deploy)
        profile_output = (output / "profile-runs" / label).resolve()
        runs = []
        for repeat in range(1, repeats + 1):
            for length_name, workload_path in (
                ("short", short_workload_path),
                ("full", full_workload_path),
            ):
                runs.append({
                    "name": f"{model_name}-{length_name}-r{repeat}",
                    "model": model_name,
                    "backend": "vllm",
                    "policy": "profile",
                    "deploy_config": str(deploy_path),
                    "delete_models_before_run": [model_name],
                    "delete_settle_s": 5,
                    "delete_after_run": True,
                    "fail_on_stale_actor_cleanup": True,
                    "min_ready_instances": 1,
                    "ready_timeout_s": float(
                        spec["experiment"]["ready_timeout_s"]
                    ),
                    "request_timeout_s": float(
                        spec["experiment"]["request_timeout_s"]
                    ),
                    "workload": str(workload_path),
                    "router_metrics_path": str(metrics_path),
                    "exclude_phases_from_overall": ["warmup"],
                    "require_runtime_ep_audit": True,
                    "runtime_ep_audit_expected_size": int(
                        shape["tensor_parallel_size"]
                    ) * int(shape["data_parallel_size"]),
                })
        matrix_path = generated / f"{label}-matrix.json"
        write_json(matrix_path, {
            "endpoint": endpoint,
            "output_dir": str(profile_output),
            "runs": runs,
        })
        artifact: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "created_at": taipei_now(),
            "profile_input_hash": profile_input_hash,
            "shape": dict(shape),
            "matrix": str(matrix_path),
        }
        try:
            summaries = execute_benchmark(
                matrix_path, endpoint, ray_address, ray_namespace,
                output / "logs" / f"profile-{label}.log",
                float(spec["experiment"]["request_timeout_s"]),
                float(spec["experiment"]["trace_event_timeout_s"]),
            )
            if len(summaries) != repeats * 2:
                raise ExperimentError(f"{label}: incomplete profile repeats")
            short_summaries = [
                row for row in summaries if "-short-r" in str(row.get("name", ""))
            ]
            full_summaries = [
                row for row in summaries if "-full-r" in str(row.get("name", ""))
            ]
            if len(short_summaries) != repeats or len(full_summaries) != repeats:
                raise ExperimentError(f"{label}: profile length pairing is incomplete")
            for summary in full_summaries:
                if float(summary.get("success_rate", 0.0)) != 1.0:
                    raise ExperimentError(f"{label}: profile request failed")
                validate_generated_tokens(
                    summary,
                    int(spec["workload"]["output_tokens"]),
                    expected_requests=int(
                        spec["workload"]["profile_request_count"]
                    ),
                )
            for summary in short_summaries:
                if float(summary.get("success_rate", 0.0)) != 1.0:
                    raise ExperimentError(f"{label}: short profile request failed")
                validate_generated_tokens(
                    summary,
                    calibration_tokens,
                    phase_name="calibration_short",
                    expected_requests=int(
                        spec["workload"]["profile_request_count"]
                    ),
                )
            full_latency = statistics.mean(
                float(row["latency_avg_ms"]) for row in full_summaries
            )
            short_latency = statistics.mean(
                float(row["latency_avg_ms"]) for row in short_summaries
            )
            decode_ms_per_token = (
                (full_latency - short_latency)
                / (int(spec["workload"]["output_tokens"]) - calibration_tokens)
            )
            if decode_ms_per_token <= 0:
                raise ExperimentError(
                    f"{label}: measured decode slope is not positive"
                )
            estimated_target_ms = short_latency + decode_ms_per_token * (
                int(spec["workload"]["preempt_after_output_tokens"])
                - calibration_tokens
            )
            artifact.update({
                "status": "passed",
                "samples": full_summaries,
                "calibration_samples": short_summaries,
                "calibration_output_tokens": calibration_tokens,
                "decode_ms_per_token": decode_ms_per_token,
                "estimated_time_to_preemption_target_ms": estimated_target_ms,
                "latency_avg_ms": full_latency,
                "latency_p95_ms": statistics.mean(
                    float(row["latency_p95_ms"]) for row in full_summaries
                ),
                "throughput_req_s": statistics.mean(
                    float(row["throughput_req_s"]) for row in full_summaries
                ),
                "deployment_ready_latency_ms": statistics.mean(
                    float(row["deployment_ready_latency_ms"])
                    for row in full_summaries
                ),
                "runtime_ep_audit_verified": all(
                    bool(row.get("runtime_ep_audit_verified", False))
                    for row in summaries
                ),
            })
            if not artifact["runtime_ep_audit_verified"]:
                raise ExperimentError(f"{label}: runtime EP readback failed")
        except Exception as exc:
            artifact.update({"status": "failed", "error": str(exc)})
        artifact["artifact"] = str(artifact_path)
        artifact["artifact_sha256"] = digest_value({
            key: value
            for key, value in artifact.items()
            if key != "artifact_sha256"
        })
        # A file cannot contain its own byte-for-byte digest without
        # recursion; this is the canonical evidence-payload digest.
        write_json(artifact_path, artifact)
        if artifact["status"] == "passed":
            passing.append(artifact)
        else:
            failures.append(artifact)
    if len({shape_signature(row["shape"]) for row in passing}) < 2:
        raise ExperimentError(
            "fewer than two distinct EP shapes passed profiling; "
            f"failures={[(row['shape'], row.get('error')) for row in failures]}"
        )
    return passing


def derive_preemption_schedule(
    spec: Mapping[str, Any], profiles: list[Mapping[str, Any]]
) -> dict[str, Any]:
    initial_signature = shape_signature(spec["initial_parallel"])
    matching = [
        profile for profile in profiles
        if shape_signature(profile["shape"]) == initial_signature
    ]
    if len(matching) != 1:
        raise ExperimentError(
            "exactly one initial-shape profile is required to schedule preemption"
        )
    estimate_ms = float(
        matching[0].get("estimated_time_to_preemption_target_ms", 0.0) or 0.0
    )
    if estimate_ms <= 0:
        raise ExperimentError("initial-shape profile lacks decode calibration")
    decode_ms_per_token = float(
        matching[0].get("decode_ms_per_token", 0.0) or 0.0
    )
    if decode_ms_per_token <= 0:
        raise ExperimentError("initial-shape profile lacks a positive decode slope")
    plan = load_trace_plan(spec)
    validate_trace_plan(spec, plan)
    preempt_after_tokens = int(
        spec["workload"]["preempt_after_output_tokens"]
    )
    constraints = dict(plan.get("constraints", {}))
    if int(constraints.get("target_generated_tokens_at_preempt", -1)) != (
        preempt_after_tokens
    ):
        raise ExperimentError(
            "trace target_generated_tokens_at_preempt must match workload"
        )
    preemption_notice_times = sorted({
        float(event["time_s"])
        for event in plan["events"]
        if event.get("event") == "preempt"
    })
    if not preemption_notice_times:
        raise ExperimentError("trace plan has no preemption notice")
    estimated_time_to_target_s = estimate_ms / 1000.0
    anchor_arrival_times = [
        notice_s - estimated_time_to_target_s
        for notice_s in preemption_notice_times
    ]
    minimum_arrival_s = float(spec["workload"]["measured_start_s"])
    if any(value < minimum_arrival_s for value in anchor_arrival_times):
        raise ExperimentError(
            "measured model is too slow for this fixed trace: an anchor "
            f"would need to start before {minimum_arrival_s}s; "
            f"notices={preemption_notice_times}, "
            f"estimated_time_to_{preempt_after_tokens}_tokens_s="
            f"{estimated_time_to_target_s:.3f}"
        )
    background_bursts = [
        float(value)
        for value in constraints.get("background_burst_times_s", [])
    ]
    formal_burst_times = sorted(background_bursts + anchor_arrival_times)
    expected_bursts = int(constraints.get("formal_burst_count", 0))
    if len(formal_burst_times) != expected_bursts:
        raise ExperimentError(
            "trace-aligned workload does not contain the declared burst count"
        )

    capacity_events: dict[float, int] = {}
    for event in plan["events"]:
        time_s = float(event["time_s"])
        count = int(event["gpu_count"])
        if event["event"] == "add":
            capacity_events[time_s] = capacity_events.get(time_s, 0) + count
        else:
            deadline_s = time_s + float(event["grace_period_s"])
            capacity_events[deadline_s] = (
                capacity_events.get(deadline_s, 0) - count
            )
    available_gpus = int(plan["initial_available_gpus"])
    capacity_timeline = [{
        "event": "initial",
        "time_s": 0.0,
        "available_gpus": available_gpus,
    }]
    for time_s in sorted(capacity_events):
        available_gpus += capacity_events[time_s]
        capacity_timeline.append({
            "event": "capacity_change",
            "time_s": time_s,
            "available_gpus": available_gpus,
        })
    add_times = [
        float(event["time_s"])
        for event in plan["events"]
        if event.get("event") == "add"
    ]
    return {
        "source": "fixed_gpu_availability_trace_with_profile_aligned_arrivals",
        "trace_plan": str(spec["trace"]["plan"]),
        "trace_plan_sha256": sha256_file(
            (REPO_ROOT / str(spec["trace"]["plan"])).resolve()
        ),
        "capacity_unit": "gpu",
        "time_unit": "second",
        "initial_available_gpus": int(plan["initial_available_gpus"]),
        "maximum_gpus": int(plan["maximum_gpus"]),
        "profile_shape": shape_label(matching[0]["shape"]),
        "calibration_output_tokens": int(
            matching[0]["calibration_output_tokens"]
        ),
        "target_output_tokens": preempt_after_tokens,
        "decode_ms_per_token": decode_ms_per_token,
        "estimated_time_to_target_ms": estimate_ms,
        "effective_preemption_time_s": preemption_notice_times[0],
        "effective_add_time_s": min(add_times),
        "preemption_notice_times_s": preemption_notice_times,
        "calibrated_anchor_arrival_times_s": anchor_arrival_times,
        "formal_burst_start_times_s": formal_burst_times,
        "trace_end_s": float(constraints["trace_end_s"]),
        "configured_output_tokens": int(spec["workload"]["output_tokens"]),
        "timing_formula": {
            "anchor_arrival_s": "fixed_preemption_notice_s - estimated_time_to_preemption_target_ms / 1000",
            "preemption_deadline_s": "fixed_preemption_notice_s + grace_period_s"
        },
        "capacity_timeline": capacity_timeline,
        "fallback_configured_preemption_time_s": float(
            spec["workload"]["preemption_time_s"]
        ),
    }


def apply_trace_schedule(
    spec: dict[str, Any], schedule: Mapping[str, Any]
) -> None:
    spec["workload"]["effective_preemption_time_s"] = float(
        schedule["effective_preemption_time_s"]
    )
    if schedule.get("effective_add_time_s") is not None:
        spec["workload"]["effective_add_time_s"] = float(
            schedule["effective_add_time_s"]
        )
    if schedule.get("formal_burst_start_times_s") is not None:
        spec["workload"]["formal_burst_start_times_s"] = [
            float(value)
            for value in schedule["formal_burst_start_times_s"]
        ]


def treatment_flags(experiment: str, treatment: str) -> dict[str, Any]:
    if experiment == "f1":
        table = {
            "rerouting": ("naive_retry", 2, False, False, False, False),
            "reparallelization": (
                "generated_token_replay", 2, True, False, False, False
            ),
            "original_spotserve": (
                "stateful_recovery", 2, True, True, False, False
            ),
            "moe_spotserve": (
                "stateful_recovery", 2, True, True, True, True
            ),
        }
    else:
        base = {
            "original_spotserve": (False, False),
            "reparallelization_only": (True, False),
            "migration_only": (False, True),
            "full_moe_spotserve": (True, True),
        }
        replay = treatment.endswith("_replay")
        base_treatment = treatment.removesuffix("_replay")
        moe_replan, moe_migration = base[base_treatment]
        table = {
            treatment: (
                "generated_token_replay" if replay else "stateful_recovery",
                2,
                True,
                True,
                moe_replan,
                moe_migration,
            )
        }
    policy, retries, replan, migration, moe_replan, moe_migration = table[treatment]
    return {
        "recovery_policy": policy,
        "max_retries": retries,
        "enable_reparallelization": replan,
        "enable_context_migration": migration,
        "enable_kv_cache_migration": migration,
        "moe_reparallelization": moe_replan,
        "moe_migration": moe_migration,
    }


def formal_deploy_config(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    capability: Mapping[str, Any], experiment: str, treatment: str,
    model_name: str, metrics_path: Path,
) -> dict[str, Any]:
    initial = spec["initial_parallel"]
    config = base_deploy_config(
        spec,
        hardware,
        initial,
        model_name,
        int(initial["sllm_instances"]),
        metrics_path,
    )
    backend = config["backend_config"]
    backend["spotserve_backend_capability"] = dict(capability)
    backend["kv_transfer_config"] = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
    }
    flags = treatment_flags(experiment, treatment)
    moe_replan = flags.pop("moe_reparallelization")
    moe_migration = flags.pop("moe_migration")
    true_kv_restore = flags["recovery_policy"] == "stateful_recovery"
    router = config["router_config"]
    router.update(flags)
    router["require_native_kv_restore"] = true_kv_restore
    router["enable_stateful_target_planner"] = True
    router["reparallelization_config"] = {
        "selection_policy": "spotserve_algorithm1",
        "queue_model": "none",
        "enable_workload_cost_model": True,
        "workload_window_s": 180,
        "workload_window_max_requests": 256,
        "batch_size": int(spec["workload"]["burst_size"]),
        "min_tensor_parallel_size": 1,
        "max_tensor_parallel_size": 1,
        "min_pipeline_parallel_size": 1,
        "max_pipeline_parallel_size": 1,
        "min_data_parallel_size": min(
            int(row["data_parallel_size"])
            for row in capability["supported_configs"]
        ),
        "max_data_parallel_size": max(
            int(row["data_parallel_size"])
            for row in capability["supported_configs"]
        ),
        "min_replica_count": 1,
        "max_replica_count": 1,
        "disable_logical_expert_placement": not moe_replan,
        "enable_live_expert_remap": moe_replan,
        "expert_weight_movement_penalty_weight": 1.0 if moe_replan else 0.0,
        "throughput_score_weight": 100.0,
        "latency_penalty_weight": 1.0,
        "load_time_penalty_weight": 1.0,
        "migration_cost_penalty_weight": 1.0,
        "queue_penalty_weight": 1.0,
        "emit_candidate_component_costs": True,
        # A vLLM/NIXL KV export is a lease over source-engine allocations.
        # Stateful treatments must keep that source alive until the target has
        # attached; replay/control treatments may use the fixed-pool
        # break-before-make path.
        "allow_stop_before_recreate": not true_kv_restore,
        "migrate_before_create": true_kv_restore,
        "transition_mode": (
            "make_before_break" if true_kv_restore else "break_before_make"
        ),
        "migration_requires_live_source": true_kv_restore,
        "require_free_gpu_overlap": true_kv_restore,
        "drain_timeout_s": 600,
    }
    router["context_migration_config"] = {
        "emit_candidate_component_costs": True,
        "require_target_runtime_reuse": True,
        "target_capacity": int(spec["workload"].get("max_num_seqs", 16)),
        "target_warmup_cost": 1.0,
        "planner_config": {
            "base_migration_cost": 0.0,
            "token_transfer_cost": 1.0,
            "context_block_transfer_cost": 4.0,
            "cross_node_penalty": 10.0,
            "enable_moe_expert_locality": moe_migration,
            "expert_dispatch_weight": 10.0 if moe_migration else 0.0,
            "queue_penalty_weight": 1.0,
            "unmatched_penalty": 1_000_000.0,
        },
    }
    router["stateful_recovery_config"] = {
        "enable_moe_expert_locality": moe_migration,
        "expert_dispatch_weight": 10.0 if moe_migration else 0.0,
    }
    router["treatment_audit"] = {
        "experiment": experiment,
        "treatment": treatment,
        "moe_reparallelization": moe_replan,
        "moe_migration": moe_migration,
        "kv_state_restore": true_kv_restore,
        "transition_mode": (
            "make_before_break" if true_kv_restore else "break_before_make"
        ),
        "source_context_ownership": "vllm_engine",
        "persistent_context_daemon": False,
        "required_overlap_headroom_gpus": (
            max(
                int(row["num_gpus"])
                for row in capability["supported_configs"]
            )
            if true_kv_restore
            else 0
        ),
        "expert_parallel_required": True,
    }
    return config


def resolve_trace_worker_id(
    target: str, hardware: Mapping[str, Any], maximum_gpus: int
) -> str:
    discovered = worker_ids(hardware)
    if len(discovered) != maximum_gpus:
        raise ExperimentError(
            f"trace requires exactly {maximum_gpus} discovered GPU workers"
        )
    return discovered[trace_target_slot(target, maximum_gpus)]


def initial_unavailable_worker_ids(
    spec: Mapping[str, Any], hardware: Mapping[str, Any]
) -> list[str]:
    plan = load_trace_plan(spec)
    maximum_gpus = int(plan["maximum_gpus"])
    return [
        resolve_trace_worker_id(str(target), hardware, maximum_gpus)
        for target in plan.get("initially_unavailable_targets", [])
    ]


def trace_managed_worker_ids(
    spec: Mapping[str, Any], hardware: Mapping[str, Any]
) -> list[str]:
    plan = load_trace_plan(spec)
    maximum_gpus = int(plan["maximum_gpus"])
    targets = {
        str(target)
        for event in plan.get("events", [])
        for target in event.get("targets", [])
    } | {
        str(target)
        for target in plan.get("initially_unavailable_targets", [])
    }
    return sorted({
        resolve_trace_worker_id(target, hardware, maximum_gpus)
        for target in targets
    })


def build_trace(spec: Mapping[str, Any], hardware: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    plan = load_trace_plan(spec)
    validate_trace_plan(spec, plan)
    if hardware is None:
        raise ExperimentError("formal worker trace requires hardware or preemption_worker_id")
    if spec["workload"].get("core_probe_mode", False):
        target = worker_ids(hardware)[1]
        return [{
            "time": float(
                spec["workload"].get(
                    "effective_preemption_time_s",
                    spec["workload"]["preemption_time_s"],
                )
            ),
            "event": "preempt",
            "node_id": str(target),
            "grace_period_s": float(
                spec["experiment"].get("grace_period_s", 30.0)
            ),
            "gpu_count": 1,
            "capacity_unit": "gpu",
        }]

    maximum_gpus = int(plan["maximum_gpus"])
    rows: list[dict[str, Any]] = []
    for event in plan["events"]:
        for target in event["targets"]:
            row: dict[str, Any] = {
                "time": float(event["time_s"]),
                "event": str(event["event"]),
                "node_id": resolve_trace_worker_id(
                    str(target), hardware, maximum_gpus
                ),
                "gpu_count": 1,
                "capacity_unit": "gpu",
                "trace_event_name": str(event["name"]),
            }
            if event["event"] == "add":
                row["node_info"] = dict(event.get("node_info", {}))
            if event["event"] == "preempt":
                row["grace_period_s"] = float(event["grace_period_s"])
            rows.append(row)
    return rows


def write_one_run_artifacts(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    capability: Mapping[str, Any], prompts: list[str], output: Path,
    experiment: str, treatment: str, repeat: int, endpoint: str,
) -> tuple[Path, Path]:
    run_key = f"{experiment}-{treatment}-r{repeat}"
    generated = output / "generated" / "formal" / run_key
    workload_path = output / "generated" / f"formal-workload-r{repeat}.jsonl"
    trace_path = output / "generated" / "formal-trace.jsonl"
    # Always regenerate from the current config.  The content hashes stored in
    # the matrix then prove no stale workload/trace survived a config edit.
    paired_prompts = list(prompts)
    random.Random(int(spec["experiment"]["order_seed"]) + repeat).shuffle(paired_prompts)
    write_jsonl(workload_path, build_workload(spec, paired_prompts, profile=False))
    write_jsonl(trace_path, build_trace(spec, hardware))
    model_name = f"qwen-moe-{run_key}"
    metrics_path = (output / "router-metrics" / f"{run_key}.jsonl").resolve()
    deploy = formal_deploy_config(
        spec, hardware, capability, experiment, treatment, model_name,
        metrics_path,
    )
    deploy_path = generated / "deploy.json"
    write_json(deploy_path, deploy)
    result_dir = (output / "runs" / run_key).resolve()
    matrix = {
        "endpoint": endpoint,
        "output_dir": str(result_dir),
        "runs": [{
            "name": run_key,
            "model": model_name,
            "backend": "vllm",
            "policy": deploy["router_config"]["recovery_policy"],
            "deploy_config": str(deploy_path),
            "delete_models_before_run": [model_name],
            "delete_settle_s": 10,
            "delete_after_run": True,
            "fail_on_stale_actor_cleanup": True,
            "min_ready_instances": int(spec["initial_parallel"]["sllm_instances"]),
            "ready_timeout_s": float(spec["experiment"]["ready_timeout_s"]),
            "request_timeout_s": float(spec["experiment"]["request_timeout_s"]),
            "trace_event_timeout_s": float(
                spec["experiment"]["trace_event_timeout_s"]
            ),
            "workload": str(workload_path),
            "trace": str(trace_path),
            "initial_unavailable_worker_nodes": (
                []
                if spec["workload"].get("core_probe_mode", False)
                else initial_unavailable_worker_ids(spec, hardware)
            ),
            "restore_worker_nodes_after_run": (
                []
                if spec["workload"].get("core_probe_mode", False)
                else trace_managed_worker_ids(spec, hardware)
            ),
            "router_metrics_path": str(metrics_path),
            "exclude_phases_from_overall": ["warmup"],
            "formal_experiment": experiment,
            "formal_treatment": treatment,
            "formal_repeat": repeat,
            "require_runtime_ep_audit": True,
            "runtime_ep_audit_expected_size": int(
                spec["initial_parallel"]["tensor_parallel_size"]
            ) * int(spec["initial_parallel"]["data_parallel_size"]),
            "workload_sha256": sha256_file(workload_path),
            "trace_sha256": sha256_file(trace_path),
            "deploy_config_sha256": sha256_file(deploy_path),
        }],
    }
    matrix_path = generated / "matrix.json"
    write_json(matrix_path, matrix)
    return matrix_path, deploy_path


def validate_run_summary(
    spec: Mapping[str, Any], summary: Mapping[str, Any], experiment: str,
    treatment: str,
) -> None:
    if not bool(summary.get("runtime_ep_audit_verified", False)):
        raise ExperimentError(
            f"{experiment}/{treatment}: runtime EP rank/size readback failed"
        )
    if int(summary.get("trace_replay_success", 0)) != 1:
        raise ExperimentError(f"{experiment}/{treatment}: trace replay failed")
    if int(summary.get("instances_marked_preempting", 0)) < 1:
        raise ExperimentError(f"{experiment}/{treatment}: preemption was not observed")
    progress_gates = {
        "progress snapshot": int(
            summary.get("preemption_progress_observed_events", 0)
        ) >= 1,
        "in-flight request": int(
            summary.get("preemption_inflight_request_count", 0)
        ) >= 1,
        "target decode window": int(
            summary.get("preemption_requests_in_progress_window", 0)
        ) >= 1,
        "before decode completion": int(
            summary.get("preemption_generated_tokens_max", 0)
        ) < int(spec["workload"]["output_tokens"]),
    }
    failed_progress = [
        name for name, passed in progress_gates.items() if not passed
    ]
    if failed_progress:
        raise ExperimentError(
            f"{experiment}/{treatment}: invalid preemption progress: "
            f"{failed_progress}"
        )
    validate_generated_tokens(
        summary,
        int(spec["workload"]["output_tokens"]),
        allow_failures=False,
        expected_requests=int(spec["workload"]["request_count"]),
    )
    dynamic = treatment in (DYNAMIC_F1 if experiment == "f1" else DYNAMIC_F2)
    if dynamic:
        expected = int(summary.get("replanning_preemption_events", 0))
        expected_add = int(summary.get("replanning_add_events", 0))
        gates = {
            "planner event": expected >= 1,
            "event/planner IDs": (
                int(summary.get("replanning_unique_preemption_event_ids", 0))
                == expected
                == int(summary.get("replanning_unique_planner_invocation_ids", 0))
            ),
            "planner audit": int(
                summary.get("replanning_planner_audit_complete_events", 0)
            ) == expected,
            "distinct shapes": int(
                summary.get("replanning_min_distinct_candidate_shapes", 0)
            ) >= 2,
            "EP selected": bool(
                summary.get("replanning_all_selected_plans_use_ep", False)
            ),
            "planner executed": int(
                summary.get("replanning_execution_failed", 0)
            ) == 0,
            "add planner event": expected_add >= 1,
            "add event/planner IDs": (
                int(summary.get("replanning_unique_add_event_ids", 0))
                == expected_add
                == int(
                    summary.get(
                        "replanning_unique_add_planner_invocation_ids", 0
                    )
                )
            ),
            "add planner audit": int(
                summary.get(
                    "replanning_add_planner_audit_complete_events", 0
                )
            ) == expected_add,
            "add execution": int(
                summary.get("replanning_add_execution_failed", 0)
            ) == 0,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            raise ExperimentError(
                f"{experiment}/{treatment}: planner gates failed: {failed}"
            )
    stateful = treatment_flags(experiment, treatment)[
        "recovery_policy"
    ] == "stateful_recovery"
    restored_blocks = int(summary.get("true_kv_restored_blocks_total", 0))
    if stateful and restored_blocks <= 0:
        raise ExperimentError(
            f"{experiment}/{treatment}: stateful recovery did not restore "
            "real KV blocks"
        )
    if not stateful and restored_blocks != 0:
        raise ExperimentError(
            f"{experiment}/{treatment}: replay/control unexpectedly restored "
            "real KV blocks"
        )
    moe_migration_treatments = {
        "moe_spotserve",
        "migration_only",
        "full_moe_spotserve",
        "migration_only_replay",
        "full_moe_spotserve_replay",
    }
    if treatment in moe_migration_treatments:
        sources = str(summary.get("context_migration_moe_route_histogram_sources", ""))
        if "vllm_runtime_topk" not in sources:
            raise ExperimentError("MoE-aware run lacks runtime top-k routing evidence")


def plan_signature_from_summary(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    raw = summary.get("replanning_latest_plan")
    if not raw:
        return ()
    plan = json.loads(raw) if isinstance(raw, str) else raw
    return shape_signature(plan)


def migration_targets_from_summary(summary: Mapping[str, Any]) -> str:
    return str(summary.get("context_migration_latest_normalized_plan", ""))


def expert_placement_from_summary(summary: Mapping[str, Any]) -> str:
    return str(
        summary.get("replanning_latest_normalized_expert_placement", "")
    )


def pilot_contribution_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_treatment = {str(row["treatment"]): row["summary"] for row in rows}
    original = by_treatment["original_spotserve"]
    candidate = by_treatment["moe_spotserve"]
    original_shape = plan_signature_from_summary(original)
    candidate_shape = plan_signature_from_summary(candidate)
    original_targets = migration_targets_from_summary(original)
    candidate_targets = migration_targets_from_summary(candidate)
    original_placement = expert_placement_from_summary(original)
    candidate_placement = expert_placement_from_summary(candidate)
    shape_diverged = bool(original_shape and candidate_shape and original_shape != candidate_shape)
    migration_diverged = bool(
        original_targets and candidate_targets and original_targets != candidate_targets
    )
    placement_diverged = bool(
        candidate_placement and candidate_placement != original_placement
    )
    decision_divergence = (
        shape_diverged or migration_diverged or placement_diverged
    )
    return {
        # A non-divergent pilot is evidence, not an execution error.  Formal
        # paired runs continue and may legitimately conclude not_supported.
        "passed": True,
        "decision_divergence_observed": decision_divergence,
        "reason": (
            "MoE-specific cost changed a selected shape, migration target, or expert placement"
            if decision_divergence
            else "Pilot decisions matched after normalization; formal runs continue and the final result may be not_supported"
        ),
        "original_shape": original_shape,
        "moe_shape": candidate_shape,
        "original_targets": original_targets,
        "moe_targets": candidate_targets,
        "original_expert_placement": original_placement,
        "moe_expert_placement": candidate_placement,
        "shape_diverged": shape_diverged,
        "migration_diverged": migration_diverged,
        "expert_placement_diverged": placement_diverged,
    }


def run_one_formal(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    capability: Mapping[str, Any], prompts: list[str], output: Path,
    endpoint: str, ray_address: str, ray_namespace: str, experiment: str,
    treatment: str, repeat: int,
) -> dict[str, Any]:
    matrix_path, deploy_path = write_one_run_artifacts(
        spec, hardware, capability, prompts, output, experiment, treatment,
        repeat, endpoint,
    )
    summaries = execute_benchmark(
        matrix_path, endpoint, ray_address, ray_namespace,
        output / "logs" / f"{experiment}-{treatment}-r{repeat}.log",
        float(spec["experiment"]["request_timeout_s"]),
        float(spec["experiment"]["trace_event_timeout_s"]),
    )
    if len(summaries) != 1:
        raise ExperimentError("one-run matrix produced an unexpected summary count")
    summary = summaries[0]
    validate_run_summary(spec, summary, experiment, treatment)
    return {
        "status": "valid",
        "experiment": experiment,
        "treatment": treatment,
        "repeat": repeat,
        "requested_formal_repeats": int(
            spec["experiment"]["formal_repeats"]
        ),
        "completed_at": taipei_now(),
        "matrix": str(matrix_path),
        "deploy_config": str(deploy_path),
        "summary": summary,
    }


def mean_sd(rows: list[Mapping[str, Any]], key: str) -> str:
    values = [float(row["summary"].get(key, 0.0) or 0.0) for row in rows]
    if not values:
        return "—"
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f} ± {deviation:.3f}"


def contribution_result(
    ledger: list[Mapping[str, Any]], expected_repeats: int = 3
) -> dict[str, Any]:
    formal = [row for row in ledger if row.get("stage") == "formal"]
    original = [
        row for row in formal
        if row["experiment"] == "f1" and row["treatment"] == "original_spotserve"
    ]
    candidate = [
        row for row in formal
        if row["experiment"] == "f1" and row["treatment"] == "moe_spotserve"
    ]
    if len(original) != expected_repeats or len(candidate) != expected_repeats:
        return {"status": "pending", "reason": "formal paired runs are incomplete"}
    original_p95 = statistics.mean(
        float(row["summary"]["latency_p95_ms"]) for row in original
    )
    candidate_p95 = statistics.mean(
        float(row["summary"]["latency_p95_ms"]) for row in candidate
    )
    original_throughput = statistics.mean(
        float(row["summary"]["throughput_req_s"]) for row in original
    )
    candidate_throughput = statistics.mean(
        float(row["summary"]["throughput_req_s"]) for row in candidate
    )
    paired = list(zip(
        sorted(original, key=lambda row: row["repeat"]),
        sorted(candidate, key=lambda row: row["repeat"]),
    ))
    if any(
        int(left["repeat"]) != int(right["repeat"])
        for left, right in paired
    ):
        return {"status": "pending", "reason": "paired repeat IDs do not match"}
    shape_divergence_by_repeat = [
        plan_signature_from_summary(left["summary"])
        != plan_signature_from_summary(right["summary"])
        for left, right in paired
    ]
    migration_divergence_by_repeat = [
        migration_targets_from_summary(left["summary"])
        != migration_targets_from_summary(right["summary"])
        for left, right in paired
    ]
    expert_placement_divergence_by_repeat = [
        bool(expert_placement_from_summary(right["summary"]))
        and expert_placement_from_summary(right["summary"])
        != expert_placement_from_summary(left["summary"])
        for left, right in paired
    ]
    shape_divergence = any(shape_divergence_by_repeat)
    migration_divergence = any(migration_divergence_by_repeat)
    expert_placement_divergence = any(
        expert_placement_divergence_by_repeat
    )
    decision_divergence = (
        shape_divergence
        or migration_divergence
        or expert_placement_divergence
    )
    decision_divergence_by_repeat = [
        shape or migration or placement
        for shape, migration, placement in zip(
            shape_divergence_by_repeat,
            migration_divergence_by_repeat,
            expert_placement_divergence_by_repeat,
        )
    ]
    p95_deltas_ms = [
        float(right["summary"]["latency_p95_ms"])
        - float(left["summary"]["latency_p95_ms"])
        for left, right in paired
    ]
    throughput_deltas_req_s = [
        float(right["summary"]["throughput_req_s"])
        - float(left["summary"]["throughput_req_s"])
        for left, right in paired
    ]

    def student_t_critical_95(sample_count: int) -> float:
        """Two-sided 95% Student-t critical without a SciPy dependency."""
        degrees_of_freedom = sample_count - 1
        if degrees_of_freedom <= 0:
            return 0.0
        exact_small_df = (
            12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
            2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
            2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
            2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
        )
        if degrees_of_freedom <= len(exact_small_df):
            return exact_small_df[degrees_of_freedom - 1]
        # Cornish-Fisher expansion around the standard-normal 97.5th
        # percentile for larger degrees of freedom.
        z = 1.959963984540054
        df = float(degrees_of_freedom)
        return (
            z
            + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
            / (384 * df**3)
        )

    def paired_interval(values: list[float]) -> dict[str, Any]:
        average = statistics.mean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        # A confidence interval is not estimable from one paired repeat.
        critical = student_t_critical_95(len(values))
        half_width = critical * deviation / (len(values) ** 0.5)
        return {
            "values": values,
            "mean": average,
            "sample_sd": deviation,
            "ci95_low": average - half_width,
            "ci95_high": average + half_width,
            "ci95_available": len(values) >= 2,
        }

    p95_delta_stats = paired_interval(p95_deltas_ms)
    throughput_delta_stats = paired_interval(throughput_deltas_req_s)
    p95_direction_consistent = all(delta < 0 for delta in p95_deltas_ms)
    throughput_direction_consistent = all(
        delta > 0 for delta in throughput_deltas_req_s
    )
    performance_direction_consistent = (
        p95_direction_consistent or throughput_direction_consistent
    )
    supported = decision_divergence and performance_direction_consistent
    return {
        "status": "supported" if supported else "not_supported",
        "reason": (
            "MoE-aware planning changed a normalized decision and every paired repeat improved in the same direction for at least one primary metric"
            if supported
            else "the paired formal evidence does not show both normalized decision divergence and direction-consistent improvement"
        ),
        "original_p95_ms": original_p95,
        "moe_p95_ms": candidate_p95,
        "p95_change_percent": (
            (candidate_p95 / original_p95 - 1) * 100 if original_p95 else 0.0
        ),
        "original_throughput_req_s": original_throughput,
        "moe_throughput_req_s": candidate_throughput,
        "throughput_change_percent": (
            (candidate_throughput / original_throughput - 1) * 100
            if original_throughput else 0.0
        ),
        "shape_divergence_observed": shape_divergence,
        "migration_target_divergence_observed": migration_divergence,
        "expert_placement_divergence_observed": (
            expert_placement_divergence
        ),
        "decision_divergence_observed": decision_divergence,
        "decision_divergence_by_repeat": decision_divergence_by_repeat,
        "decision_divergence_repeat_count": sum(
            decision_divergence_by_repeat
        ),
        "p95_paired_delta_ms": p95_delta_stats,
        "throughput_paired_delta_req_s": throughput_delta_stats,
        "p95_improvement_direction_consistent": p95_direction_consistent,
        "throughput_improvement_direction_consistent": (
            throughput_direction_consistent
        ),
        "performance_improvement_direction_consistent": (
            performance_direction_consistent
        ),
    }


def render_report(
    spec: Mapping[str, Any], output: Path, hardware: Mapping[str, Any] | None,
    profiles: list[Mapping[str, Any]], pilot_gate: Mapping[str, Any] | None,
    ledger: list[Mapping[str, Any]], status: str, error: str = "",
) -> None:
    trace_plan = load_trace_plan(spec)
    preemption_time_s = float(
        spec["workload"].get(
            "effective_preemption_time_s",
            spec["workload"]["preemption_time_s"],
        )
    )
    add_times_s = [
        float(event["time_s"])
        for event in trace_plan["events"]
        if event.get("event") == "add"
    ]
    preemption_times_s = [
        float(event["time_s"])
        for event in trace_plan["events"]
        if event.get("event") == "preempt"
    ]
    lines = [
        "# K8s 8-GPU Qwen MoE — F1 / F2 Results",
        "",
        f"- Status: **{status}**",
        f"- Generated: `{taipei_now()}`",
        f"- Model: `{spec['model']['path']}`",
        f"- Formal repeats per configuration: "
        f"`{spec['experiment']['formal_repeats']}` (required CLI input).",
        f"- Workload: {spec['workload']['prompt_tokens']} input + "
        f"{spec['workload']['output_tokens']} output tokens",
        f"- Preemption: `{preemption_time_s} s` after workload start; "
        f"target decode progress `{spec['workload']['preempt_after_output_tokens']}` tokens",
        f"- Add times: `{add_times_s} s`; preemption notice times: "
        f"`{preemption_times_s} s`; grace period "
        f"`{spec['experiment']['grace_period_s']} s`.",
        f"- Capacity trace: `{' -> '.join(str(value) for value in trace_plan['expected_capacity_path'])} GPU`; hard maximum "
        f"`{trace_plan['maximum_gpus']} GPU`.",
        "- Initial unavailable GPUs are marked in the scheduler before model "
        "deployment; physical Ray workers remain allocated.",
        "- Prefix caching: disabled; every request has a unique exact-token prompt.",
        "- Network-cost caveat: planner bandwidth is a pod-to-pod TCP proxy; "
        "runtime KV movement uses NixlConnector.",
    ]
    if error:
        lines += [f"- Blocking error: `{error}`"]
    if hardware:
        lines += [
            "",
            "## Hardware and checkpoint gate",
            "",
            "| GPU node | Worker IDs | GPU count | Physical-host marker |",
            "| --- | --- | ---: | --- |",
        ]
        for node in hardware.get("nodes", []):
            lines.append(
                f"| `{node['address']}` | `{','.join(node['worker_ids'])}` | "
                f"{node['gpu_count']} | `{','.join(node['physical_host_markers'])}` |"
            )
        lines += [
            "",
            f"Hardware fingerprint: `{hardware.get('hardware_fingerprint', '')}`.",
        ]
    lines += [
        "",
        "## Offline EP candidate profiles",
        "",
        "| Shape | Status | Mean latency (ms) | Throughput (req/s) | Ready (ms) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        lines.append(
            f"| `{shape_label(profile['shape'])}` | {profile.get('status')} | "
            f"{float(profile.get('latency_avg_ms', 0)):.3f} | "
            f"{float(profile.get('throughput_req_s', 0)):.3f} | "
            f"{float(profile.get('deployment_ready_latency_ms', 0)):.3f} |"
        )
    lines += [
        "",
        "These are distinct TP/PP/DP/EP shapes; GPU IDs are deliberately not part of the shape signature.",
        "",
        "## Contribution-readiness pilot",
        "",
    ]
    if pilot_gate:
        lines += [
            f"- Passed: **{bool(pilot_gate.get('passed'))}**",
            f"- Normalized decision divergence observed: "
            f"**{bool(pilot_gate.get('decision_divergence_observed'))}**",
            f"- Reason: {pilot_gate.get('reason')}",
            f"- Original shape: `{pilot_gate.get('original_shape')}`",
            f"- MoE-aware shape: `{pilot_gate.get('moe_shape')}`",
            f"- Original migration targets: `{pilot_gate.get('original_targets')}`",
            f"- MoE-aware migration targets: `{pilot_gate.get('moe_targets')}`",
            f"- Original expert placement: `{pilot_gate.get('original_expert_placement')}`",
            f"- MoE-aware expert placement: `{pilot_gate.get('moe_expert_placement')}`",
        ]
    else:
        lines.append("Not run.")

    formal = [row for row in ledger if row.get("stage") == "formal"]
    for experiment, treatments in (("f1", F1_TREATMENTS), ("f2", F2_TREATMENTS)):
        lines += [
            "",
            f"## {experiment.upper()} aggregate",
            "",
            "| Treatment | Recovery | Valid n | Success | Mean latency (ms) | P95 (ms) | Throughput (req/s) | KV blocks restored | Preempt output token | Add planner events | Add applied | EP readback | Replan (ms) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for treatment in treatments:
            rows = [
                row for row in formal
                if row["experiment"] == experiment
                and row["treatment"] == treatment
            ]
            recovery_policy = treatment_flags(experiment, treatment)[
                "recovery_policy"
            ]
            lines.append(
                f"| {treatment} | `{recovery_policy}` | {len(rows)} | "
                f"{mean_sd(rows, 'success_rate')} | "
                f"{mean_sd(rows, 'latency_avg_ms')} | {mean_sd(rows, 'latency_p95_ms')} | "
                f"{mean_sd(rows, 'throughput_req_s')} | "
                f"{mean_sd(rows, 'true_kv_restored_blocks_total')} | "
                f"{mean_sd(rows, 'preemption_generated_tokens_avg')} | "
                f"{mean_sd(rows, 'replanning_add_events')} | "
                f"{mean_sd(rows, 'replanning_add_execution_applied')} | "
                f"{mean_sd(rows, 'runtime_ep_audit_verified')} | "
                f"{mean_sd(rows, 'replanning_avg_execution_duration_ms')} |"
            )
    contribution = contribution_result(
        ledger, int(spec["experiment"]["formal_repeats"])
    )
    lines += [
        "",
        "## SpotServe vs MoE-aware SpotServe contribution check",
        "",
        f"- Result: **{contribution['status']}**",
        f"- Interpretation: {contribution['reason']}",
    ]
    if contribution["status"] != "pending":
        p95_delta = contribution["p95_paired_delta_ms"]
        throughput_delta = contribution["throughput_paired_delta_req_s"]
        lines += [
            f"- P95 change: {contribution['p95_change_percent']:.2f}%",
            f"- Throughput change: {contribution['throughput_change_percent']:.2f}%",
            f"- Parallel-shape divergence: {contribution['shape_divergence_observed']}",
            f"- Migration-target divergence: {contribution['migration_target_divergence_observed']}",
            f"- Expert-placement divergence: {contribution['expert_placement_divergence_observed']}",
            f"- Decision divergence by repeat: `{contribution['decision_divergence_by_repeat']}`",
            f"- Paired P95 delta (MoE - Original): {p95_delta['mean']:.3f} ms "
            f"(SD {p95_delta['sample_sd']:.3f}; CI available: "
            f"{p95_delta['ci95_available']})",
            f"- Paired throughput delta (MoE - Original): "
            f"{throughput_delta['mean']:.3f} req/s "
            f"(SD {throughput_delta['sample_sd']:.3f}; CI available: "
            f"{throughput_delta['ci95_available']})",
            f"- Direction-consistent improvement: "
            f"{contribution['performance_improvement_direction_consistent']}",
        ]
    lines += [
        "",
        "A `not_supported` result is retained as a valid negative result; the script never rewrites the workload or candidate set after seeing formal performance.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")


def smoke_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A small startup probe, not an alternative formal workload/profile."""
    if shape_signature(spec["initial_parallel"]) != (1, 1, 2, 1, True):
        raise ExperimentError("smoke requires initial TP1/PP1/DP2/EP2")
    if int(spec["initial_parallel"]["sllm_instances"]) != 2:
        raise ExperimentError("smoke requires two initial EP2 instances")
    result = json.loads(json.dumps(spec))
    result["workload"].update({
        "prompt_tokens": 512,
        "output_tokens": 64,
        "max_model_len": 2048,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
        "warmup_requests": 1,
        "warmup_output_tokens": 16,
        "request_count": 4,
        "measured_start_s": 1.0,
        "burst_size": 2,
        "burst_gap_s": 1.0,
        "preemption_progress_min_tokens": 0,
        "preemption_progress_max_tokens": 63,
    })
    return result


def check_existing_endpoint(endpoint: str, timeout_s: float = 10.0) -> dict[str, Any]:
    """Read-only check; never start the user's HTTP service or Ray head."""
    suffix = "/v1/chat/completions"
    if not endpoint.endswith(suffix):
        raise ExperimentError(f"endpoint must end with {suffix}")
    base = endpoint[:-len(suffix)]
    try:
        with request.urlopen(f"{base}/health", timeout=timeout_s) as response:
            health = json.load(response)
        if health.get("status") != "ok":
            raise ExperimentError(f"endpoint health is not ok: {health}")
        with request.urlopen(f"{base}/v1/models", timeout=timeout_s) as response:
            models = json.load(response)
        if not isinstance(models, Mapping):
            raise ExperimentError("models endpoint did not return a JSON object")
    except Exception as exc:
        raise ExperimentError(
            "existing ServerlessLLM endpoint is unavailable; start/check the "
            f"HTTP service in the head pod before running smoke: {exc}"
        ) from exc
    return {"health": health, "models_endpoint_reachable": True}


def validate_smoke_summary(
    spec: Mapping[str, Any], hardware: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(summary["run_dir"])
    metadata = read_json(run_dir / "run_metadata.json")
    audit = metadata.get("runtime_ep_audit", {})
    if not summary.get("runtime_ep_audit_verified") or not audit.get("verified"):
        raise ExperimentError("smoke runtime EP readback failed")
    if audit.get("ready_instance_count") != 2 or audit.get("audited_instance_count") != 2:
        raise ExperimentError("smoke must audit exactly two READY instances")
    audits = audit.get("instances", [])
    if len(audits) != 2 or any(
        not row.get("verified") or row.get("observed_ep_sizes") != [2]
        or row.get("observed_ep_ranks") != [0, 1] for row in audits
    ):
        raise ExperimentError("smoke requires EP rank 0/1 readback on both instances")
    states = read_json(run_dir / "instance_states.json")
    ready = {key: row for key, row in states.items()
             if row.get("pool") == "ready" and row.get("state") == "ready"}
    if len(ready) != 2 or set(ready) != {row["instance_id"] for row in audits}:
        raise ExperimentError("smoke READY membership and runtime audit do not match")
    pool = set(worker_ids(hardware))
    reserved: set[str] = set()
    membership = {}
    for instance_id, state in ready.items():
        members = [str(value) for value in state.get("member_node_ids", [])]
        if len(members) != 2 or len(set(members)) != 2:
            raise ExperimentError("smoke EP2 instance lacks two unique member workers")
        if not set(members).issubset(pool) or str(state.get("node_id")) not in members:
            raise ExperimentError("smoke membership lies outside the discovered GPU pool")
        if reserved.intersection(members):
            raise ExperimentError("smoke EP2 instances have overlapping worker reservations")
        reserved.update(members)
        membership[instance_id] = members
    validate_generated_tokens(
        summary, int(spec["workload"]["output_tokens"]), phase_name="smoke",
        expected_requests=int(spec["workload"]["request_count"]),
    )
    if float(summary.get("success_rate", 0)) != 1.0:
        raise ExperimentError("smoke requests failed")
    return {
        "member_workers": membership,
        "reserved_worker_count": len(reserved),
        "pool_worker_count": len(pool),
        "unreserved_worker_count": len(pool - reserved),
        "ep_rank_size_readback_verified": True,
        "measured_requests": int(spec["workload"]["request_count"]),
        "output_tokens_per_measured_request": int(spec["workload"]["output_tokens"]),
    }


def run_smoke(
    args: argparse.Namespace, spec: Mapping[str, Any], output: Path, endpoint: str,
) -> int:
    latest = output / "smoke-results.json"
    if latest.exists() and read_json(latest).get("status") != "dry_run_passed":
        if not args.force_smoke:
            print("Smoke already attempted; inspect its evidence first. Use --force-smoke "
                  "only after a relevant change or a new diagnostic hypothesis.", file=sys.stderr)
            return 2
    attempt = output / "smoke-attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "running", "started_at": taipei_now(), "attempt_dir": str(attempt),
        "scope": "startup_and_short_inference_only", "endpoint": endpoint,
        "not_verified": ["physical_rank_to_pod_mapping", "preemption",
                         "cross_pod_kv_restore", "persistent_context", "F1_F2_performance"],
    }
    code = 2
    try:
        small = smoke_spec(spec)
        write_json(attempt / "smoke-config.json", small)
        result["config_sha256"] = digest_value(small)
        result["protocol"] = {"initial_instances": 2, "ep_size": 2,
                              "prompt_tokens": 512, "measured_output_tokens": 64,
                              "warmup_requests": 1, "measured_requests": 4}
        if args.dry_run:
            result["status"] = "dry_run_passed"
            code = 0
        else:
            print("[smoke] Checking existing HTTP endpoint; no services will be started.", flush=True)
            result["endpoint_check"] = check_existing_endpoint(endpoint)
            print("[smoke] Inventorying the idle GPU pool and checkpoint on every worker.", flush=True)
            hardware = probe_cluster(spec, args.ray_address, args.ray_namespace)
            write_json(attempt / "hardware.json", hardware)
            prompts = exact_chat_prompts(str(small["model"]["path"]), 512, 5)
            workload_path = attempt / "workload.jsonl"
            write_jsonl(workload_path, build_workload(small, prompts, phase_name="smoke"))
            name = f"smoke-ep2-{digest_value(str(attempt))[:12]}"
            metrics_path = attempt / "router-metrics.jsonl"
            deploy_path = attempt / "deploy.json"
            write_json(deploy_path, base_deploy_config(
                small, hardware, small["initial_parallel"], name, 2, metrics_path))
            matrix_path = attempt / "matrix.json"
            write_json(matrix_path, {"endpoint": endpoint, "output_dir": str(attempt / "runs"),
                "runs": [{"name": name, "model": name, "backend": "vllm", "policy": "smoke",
                    "deploy_config": str(deploy_path), "workload": str(workload_path),
                    "min_ready_instances": 2, "capture_instance_states": True,
                    "require_runtime_ep_audit": True, "runtime_ep_audit_expected_size": 2,
                    "ready_timeout_s": float(spec["experiment"]["ready_timeout_s"]),
                    "request_timeout_s": float(spec["experiment"]["request_timeout_s"]),
                    "router_metrics_path": str(metrics_path),
                    "exclude_phases_from_overall": ["warmup"],
                    "delete_after_run": True, "fail_on_stale_actor_cleanup": True}]})
            print(f"[smoke] Starting two EP2 instances; live log: {attempt / 'benchmark.log'}", flush=True)
            summaries = execute_benchmark(
                matrix_path, endpoint, args.ray_address, args.ray_namespace,
                attempt / "benchmark.log", float(spec["experiment"]["request_timeout_s"]),
                float(spec["experiment"]["trace_event_timeout_s"]),
            )
            if len(summaries) != 1:
                raise ExperimentError("smoke requires exactly one benchmark summary")
            result["summary"] = summaries[0]
            result["checks"] = validate_smoke_summary(small, hardware, summaries[0])
            result["status"] = "passed"
            code = 0
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="blocked", error=str(exc) or type(exc).__name__)
    finally:
        result["completed_at"] = taipei_now()
        write_json(attempt / "smoke-results.json", result)
        write_json(latest, result)
        report = "\n".join([
            "# K8s two-EP2 smoke verification", "",
            f"- Status: **{result['status']}**", f"- Scope: `{result['scope']}`",
            f"- Artifacts: `{attempt}`", f"- Error: {result.get('error', 'none')}", "",
            "## Observed checks", "", "```json",
            json.dumps(result.get("checks", {}), indent=2, sort_keys=True), "```", "",
            "## Not verified", "", *[f"- {item}" for item in result["not_verified"]], "",
            "This is not a performance comparison or evidence of stateful SpotServe recovery.", "",
        ])
        (attempt / "smoke-results.md").write_text(report, encoding="utf-8")
        (output / "smoke-results.md").write_text(report, encoding="utf-8")
        print(f"[smoke] {result['status']}: {latest}", flush=True)
    return code


def run_core_probe(
    args: argparse.Namespace, spec: dict[str, Any], output: Path, endpoint: str,
) -> int:
    """One baseline/recovery pair; never profiles or starts the F1/F2 matrix."""
    latest = output / "core-probe-results.json"
    if latest.is_file() and not args.dry_run and not args.force_core_probe:
        print("[core-probe] prior attempt exists; inspect evidence before an explicit rerun",
              file=sys.stderr)
        return 2
    attempt = output / "core-probe-attempts" / str(time.time_ns())
    result = {
        "status": "blocked", "scope": "live_KV_and_dynamic_target_only",
        "full_spotserve_verified": False, "attempt_dir": str(attempt),
        "profile_policy": "matching_measured_cache_only", "runs": {},
    }
    code = 2
    try:
        write_json(attempt / "validated-config.json", spec)
        if args.dry_run:
            result["status"] = "dry_run_passed"
            result["not_verified"] = ["HTTP", "Ray", "KV_transfer", "dynamic_target"]
            code = 0
        else:
            # Prerequisites are checked before touching GPUs. A missing cache
            # cannot accidentally launch a lengthy profiling campaign.
            hardware_path = output / "preflight" / "hardware.json"
            network_path = output / "preflight" / "network.json"
            if not hardware_path.is_file() or not network_path.is_file():
                raise ExperimentError("run --phase profile once in this output directory first")
            result["endpoint"] = check_existing_endpoint(endpoint)
            hardware = probe_cluster(spec, args.ray_address, args.ray_namespace)
            network = read_json(network_path)
            write_json(attempt / "hardware.json", hardware)
            prompts = exact_chat_prompts(
                str(spec["model"]["path"]), int(spec["workload"]["prompt_tokens"]),
                int(spec["workload"]["warmup_requests"]) + max(
                    int(spec["workload"]["request_count"]),
                    int(spec["workload"]["profile_request_count"])),
            )
            profiles = ensure_candidate_profiles(
                spec, hardware, network, prompts, output, endpoint,
                args.ray_address, args.ray_namespace, False, cache_only=True,
            )
            schedule = derive_preemption_schedule(spec, profiles)
            spec = json.loads(json.dumps(spec))
            apply_trace_schedule(spec, schedule)
            # The core probe verifies one recovery mechanism only. It must not
            # replay the 22-minute formal capacity trace.
            spec["workload"]["core_probe_mode"] = True
            capability = build_capability(spec, hardware, network, profiles)
            write_json(attempt / "preemption-schedule.json", schedule)
            for case in ("baseline", "recovery"):
                matrix_path, deploy_path = write_one_run_artifacts(
                    spec, hardware, capability, prompts, attempt / case,
                    "f1", "original_spotserve", 0, endpoint,
                )
                matrix, deploy = read_json(matrix_path), read_json(deploy_path)
                run = matrix["runs"][0]
                model = f"kv-core-{attempt.name}-{case}"
                deploy["model"] = model
                if case == "baseline":
                    deploy["router_config"].update(
                        recovery_policy="naive_retry", max_retries=0,
                        enable_reparallelization=False, enable_context_migration=False,
                        enable_kv_cache_migration=False, require_native_kv_restore=False,
                        enable_stateful_target_planner=False,
                    )
                    run.pop("trace")
                    run.pop("trace_sha256")
                    run["policy"] = "naive_retry"
                run.update(
                    model=model, delete_models_before_run=[],
                    capture_instance_states=True,
                    capture_instance_states_after_workload=True,
                )
                write_json(deploy_path, deploy)
                run["deploy_config_sha256"] = sha256_file(deploy_path)
                write_json(matrix_path, matrix)
                result["runs"][case] = {
                    "matrix": str(matrix_path), "log": str(attempt / case / "benchmark.log"),
                    "metrics": run["router_metrics_path"],
                }
                print(f"[core-probe] {case}: {result['runs'][case]['log']}", flush=True)
                summaries = execute_benchmark(
                    matrix_path, endpoint, args.ray_address, args.ray_namespace,
                    attempt / case / "benchmark.log",
                    float(spec["experiment"]["request_timeout_s"]),
                    float(spec["experiment"]["trace_event_timeout_s"]),
                )
                if len(summaries) != 1:
                    raise ExperimentError("core-probe expected one summary per case")
                result["runs"][case]["run_dir"] = summaries[0]["run_dir"]
                if not summaries[0].get("runtime_ep_audit_verified"):
                    raise ExperimentError(f"{case}: initial EP rank/size readback failed")
                if case == "baseline":
                    validate_generated_tokens(
                        summaries[0], int(spec["workload"]["output_tokens"]),
                        expected_requests=int(spec["workload"]["request_count"]),
                    )

            def rows(path):
                return [json.loads(line) for line in Path(path).read_text().splitlines()
                        if line.strip()]

            from sllm.spot.core_validation import audit_live_kv_core

            baseline = Path(result["runs"]["baseline"]["run_dir"])
            recovery = Path(result["runs"]["recovery"]["run_dir"])
            result.update(audit_live_kv_core(
                rows(baseline / "raw_requests.jsonl"), rows(recovery / "raw_requests.jsonl"),
                rows(result["runs"]["recovery"]["metrics"]),
                read_json(recovery / "instance_states.json"),
                read_json(recovery / "final_instance_states.json"),
                build_trace(spec, hardware)[0]["node_id"],
                int(spec["workload"]["output_tokens"]),
            ))
            code = 0 if result["status"] == "live_kv_core_verified" else 2
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = str(exc) or type(exc).__name__
    finally:
        write_json(attempt / "core-probe-results.json", result)
        # Dry runs must not replace evidence from an earlier real attempt.
        if not args.dry_run or not latest.exists():
            write_json(latest, result)
        report = (
            "# Live-KV / dynamic-target core probe\n\n"
            f"Status: `{result['status']}`. Full SpotServe verified: **false**.\n\n"
            "This is one paired correctness probe, not an F1/F2 performance result.\n\n"
            f"```json\n{json.dumps(result, indent=2, sort_keys=True)}\n```\n"
        )
        (attempt / "core-probe-results.md").write_text(report, encoding="utf-8")
        if not args.dry_run or not (output / "core-probe-results.md").exists():
            (output / "core-probe-results.md").write_text(report, encoding="utf-8")
        print(f"[core-probe] {result['status']}: {attempt}", flush=True)
    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model-path", default=None, help="Smoke-only checkpoint path visible at the same location in head and workers")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    parser.add_argument(
        "--repeats",
        type=positive_int,
        required=True,
        help="Number of formal runs for every F1/F2 configuration.",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "preflight", "smoke", "core-probe", "profile", "pilot", "formal", "report"],
        default="all",
    )
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--force-smoke", action="store_true", help="Explicit diagnostic rerun; keeps the prior attempt artifacts")
    parser.add_argument("--force-core-probe", action="store_true", help="Explicit diagnostic rerun after inspecting the prior core probe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model_path and args.phase != "smoke":
        parser.error("--model-path is smoke-only; for formal runs use a separate config and output directory")
    if args.force_smoke and args.phase != "smoke":
        parser.error("--force-smoke requires --phase smoke")
    if args.force_core_probe and args.phase != "core-probe":
        parser.error("--force-core-probe requires --phase core-probe")
    if args.force_profile and args.phase == "core-probe":
        parser.error("core-probe only reuses frozen profiles; run --phase profile separately")
    return args


def main() -> int:
    args = parse_args()
    spec_path = (REPO_ROOT / args.config).resolve() if not Path(
        args.config
    ).is_absolute() else Path(args.config)
    spec = read_json(spec_path)
    # The checked-in value is only a schema placeholder. Every invocation
    # must explicitly choose the formal repeat count.
    spec["experiment"]["formal_repeats"] = int(args.repeats)
    if args.model_path:
        spec["model"]["path"] = args.model_path
    validate_spec(spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint or str(spec["cluster"]["endpoint"])
    if args.phase == "smoke":
        return run_smoke(args, spec, output, endpoint)
    if args.phase == "core-probe":
        return run_core_probe(args, spec, output, endpoint)
    state_path = output / "state.json"
    ledger_path = output / "run-ledger.json"
    state = read_json(state_path) if state_path.exists() else {
        "status": "initialized", "created_at": taipei_now(),
        "config": str(spec_path), "requested_formal_repeats": int(args.repeats),
    }
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    recorded_repeats = state.get("requested_formal_repeats")
    if recorded_repeats is not None and int(recorded_repeats) != int(
        args.repeats
    ):
        print(
            "[k8s-moe-f1-f2] blocked: this output directory was created "
            f"with --repeats {recorded_repeats}, not {args.repeats}; use "
            "the original value to resume or choose a new output directory",
            file=sys.stderr,
        )
        return 2
    state["requested_formal_repeats"] = int(args.repeats)
    hardware: dict[str, Any] | None = None
    profiles: list[dict[str, Any]] = []
    pilot_gate: dict[str, Any] | None = None
    error = ""
    try:
        if args.phase == "report":
            hardware_path = output / "preflight" / "hardware.json"
            if hardware_path.is_file():
                hardware = read_json(hardware_path)
            if (output / "profiles").is_dir():
                for path in sorted((output / "profiles").glob("tp*.json")):
                    profile = read_json(path)
                    if profile.get("status") == "passed":
                        profiles.append(profile)
            pilot_path = output / "pilot" / "contribution-readiness.json"
            if pilot_path.is_file():
                pilot_gate = read_json(pilot_path)
            render_report(
                spec,
                output,
                hardware,
                profiles,
                pilot_gate,
                ledger,
                str(state.get("status", "unknown")),
                str(state.get("error", "")),
            )
            return 0
        if args.dry_run:
            generated = output / "generated"
            write_json(generated / "validated-config.json", spec)
            state.update({
                "status": "dry_run_passed",
                "config_sha256": sha256_file(spec_path),
                "distinct_candidate_shapes": len({
                    shape_signature(shape) for shape in spec["candidate_shapes"]
                }),
                "all_candidates_use_ep": all(
                    shape_signature(shape)[-1] for shape in spec["candidate_shapes"]
                ),
            })
            write_json(state_path, state)
            render_report(spec, output, None, [], None, ledger, state["status"])
            return 0

        hardware_path = output / "preflight" / "hardware.json"
        network_path = output / "preflight" / "network.json"
        if (
            args.phase in {"all", "preflight"}
            or not hardware_path.exists()
            or not network_path.exists()
        ):
            hardware = probe_cluster(spec, args.ray_address, args.ray_namespace)
            write_json(hardware_path, hardware)
            network = probe_network(
                hardware, spec, args.ray_address, args.ray_namespace
            )
            write_json(network_path, network)
        else:
            hardware = read_json(hardware_path)
            validate_cluster_profile(spec, hardware)
            network = read_json(network_path)
        state.update({
            "status": "preflight_passed",
            "hardware": str(hardware_path),
            "network": str(network_path),
        })
        write_json(state_path, state)
        if args.phase == "preflight":
            render_report(spec, output, hardware, [], None, ledger, state["status"])
            return 0

        prompt_count = int(spec["workload"]["warmup_requests"]) + max(
            int(spec["workload"]["request_count"]),
            int(spec["workload"]["profile_request_count"]),
        )
        prompts = exact_chat_prompts(
            str(spec["model"]["path"]),
            int(spec["workload"]["prompt_tokens"]),
            prompt_count,
        )
        workload_audit = {
            "prompt_tokens": int(spec["workload"]["prompt_tokens"]),
            "prompt_count": len(prompts),
            "unique_prompt_count": len(set(prompts)),
            "prompt_sha256": [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in prompts
            ],
            "output_tokens": int(spec["workload"]["output_tokens"]),
            "max_model_len": int(spec["workload"]["max_model_len"]),
            "prefix_caching_enabled": bool(
                spec["model"].get("enable_prefix_caching", False)
            ),
        }
        write_json(output / "preflight" / "workload.json", workload_audit)
        profiles = ensure_candidate_profiles(
            spec, hardware, network, prompts, output, endpoint, args.ray_address,
            args.ray_namespace, args.force_profile,
        )
        preemption_schedule = derive_preemption_schedule(spec, profiles)
        apply_trace_schedule(spec, preemption_schedule)
        write_json(
            output / "profiles" / "preemption-schedule.json",
            preemption_schedule,
        )
        capability = build_capability(spec, hardware, network, profiles)
        write_json(output / "profiles" / "planner-capability.json", capability)
        state.update({
            "status": "profiles_passed",
            "passing_candidate_shapes": [shape_label(row["shape"]) for row in profiles],
        })
        write_json(state_path, state)
        if args.phase == "profile":
            render_report(spec, output, hardware, profiles, None, ledger, state["status"])
            return 0

        pilot_rows = []
        for treatment in ("original_spotserve", "moe_spotserve"):
            matching_pilots = [
                row for row in ledger
                if row.get("stage") == "pilot"
                and row.get("status") == "valid"
                and row.get("experiment") == "f1"
                and row.get("treatment") == treatment
                and int(row.get("repeat", -1)) == 0
            ]
            if matching_pilots:
                # Preserve the evidence ledger.  If an older runner already
                # left duplicate pilots, use the latest valid entry without
                # launching another GPU run.
                pilot_rows.append(matching_pilots[-1])
            else:
                row = run_one_formal(
                    spec, hardware, capability, prompts, output, endpoint,
                    args.ray_address, args.ray_namespace, "f1", treatment, 0,
                )
                row["stage"] = "pilot"
                ledger.append(row)
                pilot_rows.append(row)
                write_json(ledger_path, ledger)
        pilot_gate = pilot_contribution_gate(pilot_rows)
        write_json(output / "pilot" / "contribution-readiness.json", pilot_gate)
        state["status"] = "pilot_completed"
        write_json(state_path, state)
        if args.phase == "pilot":
            render_report(
                spec, output, hardware, profiles, pilot_gate, ledger,
                state["status"],
            )
            return 0

        completed = {
            (row.get("experiment"), row.get("treatment"), row.get("repeat"))
            for row in ledger
            if row.get("stage") == "formal" and row.get("status") == "valid"
        }
        seed = int(spec["experiment"].get("order_seed", 20261002))
        for experiment, treatments in (("f1", F1_TREATMENTS), ("f2", F2_TREATMENTS)):
            for repeat in range(
                1, int(spec["experiment"]["formal_repeats"]) + 1
            ):
                ordered = list(treatments)
                random.Random(seed + repeat + (0 if experiment == "f1" else 100)).shuffle(ordered)
                for treatment in ordered:
                    key = (experiment, treatment, repeat)
                    if key in completed:
                        continue
                    row = run_one_formal(
                        spec, hardware, capability, prompts, output, endpoint,
                        args.ray_address, args.ray_namespace, experiment,
                        treatment, repeat,
                    )
                    row["stage"] = "formal"
                    ledger.append(row)
                    completed.add(key)
                    write_json(ledger_path, ledger)
                    render_report(
                        spec, output, hardware, profiles, pilot_gate, ledger,
                        "formal_running",
                    )
        state["status"] = "passed"
        state["contribution"] = contribution_result(
            ledger, int(spec["experiment"]["formal_repeats"])
        )
        write_json(state_path, state)
        render_report(
            spec, output, hardware, profiles, pilot_gate, ledger,
            state["status"],
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc)
        state.update({"status": "blocked", "error": error, "updated_at": taipei_now()})
        write_json(state_path, state)
        render_report(
            spec, output, hardware, profiles, pilot_gate, ledger,
            state["status"], error,
        )
        print(f"[k8s-moe-f1-f2] blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
