"""Run one real four-policy batch recovery experiment.

This is the GPU2-independent fallback described in the recovery comparison
report.  It deliberately uses one target after preemption so all three
recoverable policies pay the same engine-startup shape.  The batch is kept in
one vLLM engine (max_num_seqs=8), which makes context recomputation scale with
the number of requests instead of paying eight independent startups.  Setting
TP to 2 reuses this driver for the Qwen1.5-MoE-A2.7B experiment.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import shlex
import socket
import statistics
import time
from multiprocessing.connection import Listener
from pathlib import Path

from run_cross_container_nixl_smoke import (
    IMAGE,
    MODEL_ROOT,
    PYTHONPATH,
    REPO_ROOT,
    VLLM_ROOT,
    dump_container_logs,
    recv,
    run_podman,
)
from run_four_container_fleet_churn_smoke import load_fleet_trace, trace_slot
from sllm.spot.reparallelization import (
    ParallelPlan,
    plan_dynamic_reparallelization,
)


MODE_PROPERTIES = {
    "no_recovery": {},
    "rerouting": {},
    "reparallelization": {"reparallelization": True},
    "spotserve": {"migration": True, "reparallelization": True},
    "moe_spotserve": {
        "migration": True,
        "reparallelization": True,
        "moe_migration": True,
        "moe_reparallelization": True,
    },
    "original_migration": {"migration": True},
    "moe_migration": {"migration": True, "moe_migration": True},
    "original_reparallelization": {"reparallelization": True},
    "moe_reparallelization": {
        "reparallelization": True,
        "moe_reparallelization": True,
    },
    # Preserve the existing command line while mapping it to the complete
    # MoE-aware SpotServe path used by older recovery reports.
    "modified": {
        "migration": True,
        "reparallelization": True,
        "moe_migration": True,
        "moe_reparallelization": True,
    },
}
MODES = tuple(MODE_PROPERTIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--source-gpus", type=int, nargs="+", default=None,
        help=(
            "Explicit source GPU group.  This may overlap target-gpus because "
            "the source is stopped before the target starts."
        ),
    )
    parser.add_argument(
        "--target-gpus", type=int, nargs="+", default=None,
        help="Explicit target GPU group; defaults to the second TP group.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="TP size for source and target (1 for Tiny, 2 for Qwen MoE).",
    )
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=480)
    parser.add_argument(
        "--prompt-files",
        nargs="+",
        default=[
            str(Path(REPO_ROOT) / "docs/spotserve-moe-aware-planner.md"),
            str(Path(REPO_ROOT) / "docs/four_version_recovery_comparison.md"),
        ],
        help="Text workloads tokenized locally; files cycle across requests.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help="Optional scheduler chunk size; context length is still max-model-len.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--preempt-after-new-tokens", type=int, default=16)
    parser.add_argument("--preempt-max-overshoot", type=int, default=16)
    parser.add_argument(
        "--require-moe-routing",
        action="store_true",
        help="Reject a run unless every request has real vLLM top-k data.",
    )
    parser.add_argument(
        "--route-profile",
        default=None,
        help=(
            "Audited live-routing result used only when vLLM's NIXL "
            "connector makes in-run routed-expert capture unavailable."
        ),
    )
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.08)
    parser.add_argument(
        "--cpu-offload-gb", type=float, default=0.0,
        help="CPU weight offload used when long Qwen batches exceed one GPU's VRAM.",
    )
    parser.add_argument("--timeout-s", type=float, default=420.0)
    parser.add_argument(
        "--host-network", action="store_true",
        help="Use host networking so NIXL side-channel ports are directly reachable.",
    )
    parser.add_argument(
        "--nixl-prestart-target", action="store_true",
        help=(
            "Start a fresh NIXL target before the source request. Disabled by "
            "default so target startup is part of recovery/no-warmup timing."
        ),
    )
    parser.add_argument(
        "--prestart-target", action="store_true",
        help="Create the fresh target before source work for every policy.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", default=IMAGE)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile.  With a two-request batch, P99 must be the
    # slower request rather than the faster request selected by floor().
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return round(ordered[index], 3)


def build_workload_prompts(
    args: argparse.Namespace, request_ids: list[str]
) -> tuple[dict[str, list[int]], dict[str, dict]]:
    """Build exact-length, natural-text prompts with auditable token hashes."""
    from transformers import AutoTokenizer

    prompt_files = [Path(value).resolve() for value in args.prompt_files]
    missing = [str(path) for path in prompt_files if not path.is_file()]
    if missing:
        raise SystemExit(f"prompt workload file not found: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    prompts: dict[str, list[int]] = {}
    audit: dict[str, dict] = {}
    for index, request_id in enumerate(request_ids):
        path = prompt_files[index % len(prompt_files)]
        source_text = path.read_text(encoding="utf-8")
        header = (
            f"Workload {index + 1}: Carefully analyze the following engineering "
            "document and continue with a precise technical discussion.\n\n"
        )
        body = header + source_text
        token_ids = list(tokenizer.encode(body, add_special_tokens=True))
        continuation = list(tokenizer.encode("\n\n" + source_text, add_special_tokens=False))
        if not continuation:
            raise SystemExit(f"prompt workload tokenized to no content: {path}")
        while len(token_ids) < args.prompt_tokens:
            token_ids.extend(continuation)
        token_ids = token_ids[: args.prompt_tokens]
        digest = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        prompts[request_id] = token_ids
        audit[request_id] = {
            "source_file": str(path.relative_to(Path(REPO_ROOT))),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "token_ids_sha256": digest,
            "token_count": len(token_ids),
        }
    return prompts, audit


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_route_profiles(path: str | None) -> tuple[dict[str, dict], str | None]:
    if not path:
        return {}, None
    profile_path = Path(path).resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = payload.get("recovery", {}).get("moe_route_profiles", {})
    by_prompt_hash = {
        str(row.get("prompt_token_ids_sha256")): dict(row)
        for row in profiles.values()
        if isinstance(row, dict)
        and row.get("prompt_token_ids_sha256")
        and row.get("per_request_expert_route_histogram")
    }
    if not by_prompt_hash:
        raise SystemExit(f"route profile contains no live histograms: {path}")
    return by_prompt_hash, hashlib.sha256(profile_path.read_bytes()).hexdigest()


class EventInbox:
    """Keep unrelated control messages while waiting for a batch event."""

    def __init__(self) -> None:
        self.pending: collections.deque[dict] = collections.deque()

    def pop(self, conn, predicate, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for index, message in enumerate(self.pending):
                if predicate(message):
                    del self.pending[index]
                    return message
            remaining = max(0.05, min(1.0, deadline - time.monotonic()))
            if not conn.poll(remaining):
                continue
            message = recv(conn)
            if predicate(message):
                return message
            self.pending.append(message)
        raise TimeoutError("timed out waiting for batch control event")


def main() -> None:
    args = parse_args()
    mode_properties = MODE_PROPERTIES[args.mode]
    uses_migration = bool(mode_properties.get("migration"))
    uses_reparallelization = bool(mode_properties.get("reparallelization"))
    uses_moe_migration = bool(mode_properties.get("moe_migration"))
    uses_moe_reparallelization = bool(
        mode_properties.get("moe_reparallelization")
    )
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise SystemExit("--gpus must contain four distinct GPU indices")
    if args.tensor_parallel_size not in (1, 2):
        raise SystemExit("--tensor-parallel-size must be 1 or 2")
    if args.tensor_parallel_size * 2 > len(args.gpus):
        raise SystemExit("not enough GPU slots for source and target TP groups")
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.prompt_tokens + args.max_new_tokens > args.max_model_len:
        raise SystemExit("prompt plus output must fit --max-model-len")
    if not (0 < args.preempt_after_new_tokens < args.max_new_tokens):
        raise SystemExit("preempt threshold must be between 1 and max-new-tokens")
    if args.preempt_max_overshoot < 0:
        raise SystemExit("--preempt-max-overshoot must not be negative")
    if not Path(args.model, "config.json").is_file():
        raise SystemExit(f"model config not found: {args.model}")
    trace_events = load_fleet_trace(args.trace)
    tensor_parallel_size = int(args.tensor_parallel_size)
    source_gpus = list(args.source_gpus or args.gpus[:tensor_parallel_size])
    target_gpus = list(args.target_gpus or (
        [args.gpus[1]]
        if tensor_parallel_size == 1
        else args.gpus[2:2 + tensor_parallel_size]
    ))
    if len(source_gpus) != tensor_parallel_size or len(target_gpus) != tensor_parallel_size:
        raise SystemExit("source/target GPU groups must match tensor-parallel-size")
    if len(set(source_gpus)) != len(source_gpus) or len(set(target_gpus)) != len(target_gpus):
        raise SystemExit("source/target GPU groups must not repeat a GPU")
    if not set(source_gpus + target_gpus).issubset(set(args.gpus)):
        raise SystemExit("source/target GPUs must be listed in --gpus")
    source_gpu_set = set(source_gpus)
    if not any(
        event["event"] == "remove"
        and trace_slot(node) in source_gpu_set
        for event in trace_events
        for node in event["nodes"]
    ):
        raise SystemExit("trace must remove the source GPU")

    control_dir = os.path.abspath(f"/tmp/tiny-batch-recovery-{os.getpid()}")
    os.makedirs(control_dir, mode=0o777, exist_ok=False)
    os.chmod(control_dir, 0o777)
    socket_path = os.path.join(control_dir, "control.sock")
    listener = Listener(socket_path, family="AF_UNIX", authkey=b"spotserve")
    os.chmod(socket_path, 0o666)
    listener._listener._socket.settimeout(args.timeout_s)
    network = f"spotserve-tiny-batch-net-{os.getpid()}"
    workers: dict[str, dict] = {}
    container_names: list[str] = []
    inboxes: dict[int, EventInbox] = {}
    started = time.monotonic()

    network_name = "host" if args.host_network else network
    common = [
        "run", "--detach", "--log-driver", "k8s-file",
        "--network", network_name,
        "--volume", f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "--volume", f"{VLLM_ROOT}:{VLLM_ROOT}:ro",
        "--volume", f"{MODEL_ROOT}:{MODEL_ROOT}:ro",
        "--volume", "/home/undergrad2026/s112060021/.cache/vllm:/root/.cache/vllm:rw",
        "--volume", "/tmp/torchinductor_s112060021:/tmp/torchinductor_s112060021:rw",
        "--volume", "/usr/local/cuda-13.0:/usr/local/cuda:ro",
        "--volume", f"{control_dir}:/control:rw",
        "--env", f"PYTHONPATH={PYTHONPATH}",
        "--env", "VLLM_CACHE_ROOT=/root/.cache/vllm",
        "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_s112060021",
        "--env", "PYTHONUNBUFFERED=1", args.image,
    ]
    worker_script = f"{REPO_ROOT}/tests/spotserve_test/cross_container_nixl_worker.py"

    def worker_command(spec: dict) -> list[str]:
        name = f"spotserve-tiny-batch-{spec['label']}-{os.getpid()}"
        spec["name"] = name
        container_names.append(name)
        role_args = [
            "python", "-u", worker_script,
            "--model", args.model,
            "--control-socket", "/control/control.sock",
            "--role", spec["role"],
            "--node-id", spec["node_id"],
            "--side-channel-host",
            "127.0.0.1" if args.host_network else spec["node_id"],
            "--side-channel-port", str(spec["port"]),
            "--token-delay-s", str(args.token_delay_s),
            "--max-new-tokens", str(args.max_new_tokens),
            "--pause-after-new-tokens", str(args.preempt_after_new_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--cpu-offload-gb", str(max(float(args.cpu_offload_gb), 0.0)),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-num-seqs", str(args.request_count),
            "--max-num-batched-tokens", str(
                args.max_num_batched_tokens or args.max_model_len
            ),
            "--max-model-len", str(args.max_model_len),
            "--kv-transfer-mode", "nixl" if uses_migration else "none",
        ]
        devices: list[str] = []
        for gpu in spec["gpus"]:
            devices += ["--device", f"nvidia.com/gpu={gpu}"]
        return [
            *common[:1], "--name", name, "--hostname", spec["node_id"],
            *devices, *common[1:], "bash", "-lc", "exec " + shlex.join(role_args),
        ]

    def launch(spec: dict) -> None:
        run_podman(worker_command(spec))

    def register(specs: list[dict]) -> None:
        expected = {spec["node_id"]: spec for spec in specs}
        register_deadline = time.monotonic() + args.timeout_s
        listener._listener._socket.settimeout(5.0)
        while expected:
            try:
                conn = listener.accept()
            except socket.timeout:
                for spec in expected.values():
                    inspected = run_podman([
                        "inspect", "--format",
                        "{{.State.Running}} {{.State.ExitCode}}", spec["name"],
                    ], check=False)
                    if inspected.stdout.strip().startswith("false"):
                        dump_container_logs([spec["name"]])
                        raise RuntimeError(
                            f"worker exited before ready: {spec['name']} "
                            f"({inspected.stdout.strip()})"
                        )
                if time.monotonic() >= register_deadline:
                    raise TimeoutError("worker did not register before timeout")
                continue
            inboxes[id(conn)] = EventInbox()
            while True:
                if not conn.poll(args.timeout_s):
                    raise TimeoutError("worker did not send ready")
                ready = recv(conn)
                if ready.get("event") == "ready":
                    break
            node_id = ready.get("node_id")
            if node_id not in expected:
                raise RuntimeError(f"unexpected worker node {node_id}")
            spec = expected.pop(node_id)
            workers[spec["label"]] = {**spec, "conn": conn, **ready}

    def inbox(worker: dict) -> EventInbox:
        return inboxes[id(worker["conn"])]

    def send(worker: dict, payload: dict) -> None:
        worker["conn"].send(payload)

    def wait(worker: dict, predicate) -> dict:
        return inbox(worker).pop(worker["conn"], predicate, args.timeout_s)

    def stop(label: str) -> None:
        worker = workers.pop(label)
        try:
            send(worker, {"op": "shutdown"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            worker["conn"].close()
        except OSError:
            pass
        run_podman(["rm", "--force", worker["name"]], check=False)

    def plan_for(
        active_slots: set[int], moe_aware: bool
    ) -> tuple[dict, ParallelPlan]:
        from sllm.backends.vllm_capability import get_vllm_capability

        model_name = Path(args.model).name
        capability = get_vllm_capability({
            "model": model_name,
            "num_gpus": len(args.gpus),
            "backend_config": {
                "pretrained_model_name_or_path": args.model,
                "tensor_parallel_size": tensor_parallel_size,
            },
        }).to_dict()
        capability["supported_configs"] = [
            config for config in capability["supported_configs"]
            if int(config.get("expert_parallel_size", 1) or 1) == 1
            and int(config.get("tensor_parallel_size", 1) or 1)
            >= tensor_parallel_size
        ]
        nodes = {
            f"node-{gpu}": {
                "ray_node_id": f"node-{gpu}", "address": f"node-{gpu}",
                "free_gpu": int(gpu in active_slots), "total_gpu": 1,
                "state": "ready" if gpu in active_slots else "dead",
            }
            for gpu in args.gpus
        }
        planner_model_name = model_name if moe_aware else "original-transformer"
        planner_model_config = {
            "model": planner_model_name,
            "backend": "vllm",
            "num_gpus": len(args.gpus),
            "backend_config": {
                "tensor_parallel_size": tensor_parallel_size,
            },
            "backend_capability": capability,
        }
        if moe_aware:
            planner_model_config["backend_config"][
                "pretrained_model_name_or_path"
            ] = args.model
        decision = plan_dynamic_reparallelization(
            model_name=planner_model_name,
            worker_nodes=nodes,
            model_config=planner_model_config,
            planner_config={
                "model_gpu_requirement": tensor_parallel_size,
                "target_replica_gpus": tensor_parallel_size,
                "min_tensor_parallel_size": tensor_parallel_size,
                "max_tensor_parallel_size": max(
                    tensor_parallel_size, len(active_slots)
                ),
                "max_pipeline_parallel_size": 1,
                "min_data_parallel_size": 1,
            },
            event="remove", node_id="remove", backend="vllm",
        )
        selected = decision.get("parallel_plan")
        if not selected:
            raise AssertionError(f"planner returned no target: {decision}")
        return decision, ParallelPlan.from_dict(selected)

    def plan_context_migration(
        exported: dict[str, dict], moe_aware: bool
    ) -> dict:
        """Run the existing context migration planner on measured KV state.

        Both sides use the same planner and fixed target.  The only ablated
        option is expert-locality scoring.  If runtime routing histograms are
        unavailable, the decision records that fact instead of substituting
        synthetic hotness data.
        """
        from sllm.spot.context_migration import plan_low_cost_migration_from_dict
        from sllm.spot.moe_placement import build_logical_expert_placement_plan

        target_parallel_plan = {
            "model_name": Path(args.model).name,
            "backend": "vllm",
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "replica_count": 1,
            "sllm_replica_count": 1,
            "enable_expert_parallel": False,
            "effective_expert_parallel_size": 1,
            "num_gpus": tensor_parallel_size,
            "target_nodes": [f"node-{gpu}" for gpu in target_gpus],
            "placement_epoch": 1,
        }
        placement = build_logical_expert_placement_plan(
            model_name=Path(args.model).name,
            target_parallel_plan=target_parallel_plan,
            model_config={
                "model": Path(args.model).name,
                "backend": "vllm",
                "backend_config": {
                    "pretrained_model_name_or_path": args.model,
                    "tensor_parallel_size": tensor_parallel_size,
                },
            },
            placement_epoch=1,
        )
        target_metadata = placement.to_dict() if placement is not None else {}
        sources = []
        for request_id, state in exported.items():
            metadata = dict(state.get("metadata", {}) or {})
            block_counts = metadata.get("kv_block_count_by_group", []) or []
            sources.append({
                "request_id": request_id,
                "instance_id": "source",
                "node_id": ",".join(str(gpu) for gpu in source_gpus),
                "num_tokens": int(metadata.get("computed_tokens", 0) or 0),
                "context_blocks": max(
                    [int(value or 0) for value in block_counts] or [0]
                ),
                "cache_block_size": int(
                    metadata.get("cache_block_size", 0) or 0
                ),
                "cache_dtype": str(metadata.get("cache_dtype", "") or ""),
                "cache_layout": str(metadata.get("cache_layout", "") or ""),
                "metadata": metadata,
            })
        return plan_low_cost_migration_from_dict({
            "sources": sources,
            "targets": [{
                "instance_id": "target",
                "node_id": ",".join(str(gpu) for gpu in target_gpus),
                "capacity": max(len(sources), 1),
                "concurrency": 0,
                "metadata": target_metadata,
            }],
            "planner_config": {
                "enable_moe_expert_locality": moe_aware,
                "expert_dispatch_weight": 10.0 if moe_aware else 0.0,
                "queue_penalty_weight": 0.0,
                "unmatched_penalty": 1000000.0,
            },
        }).to_dict()

    def apply_trace() -> set[int]:
        active: set[int] = set()
        previous_ms = 0.0
        for event in trace_events:
            delay = max(float(event["time_ms"]) - previous_ms, 0.0) / 1000.0
            delay /= max(args.trace_speedup, 1e-9)
            if delay:
                time.sleep(delay)
            previous_ms = float(event["time_ms"])
            if event["event"] == "DONE":
                break
            for node in event["nodes"]:
                gpu = trace_slot(node)
                if event["event"] == "add":
                    active.add(gpu)
                elif event["event"] == "remove":
                    active.discard(gpu)
            if event["event"] == "remove" and source_gpu_set.intersection(
                {trace_slot(node) for node in event["nodes"]}
            ):
                return active
        raise AssertionError("trace did not remove source")

    request_ids = [f"batch-{os.getpid()}-{i}" for i in range(args.request_count)]
    prompts, prompt_audit = build_workload_prompts(args, request_ids)
    calibrated_routes, route_profile_sha256 = load_route_profiles(
        args.route_profile
    )
    metadata: dict[str, dict] = {}
    source_tokens: dict[str, list[int]] = {}
    source_completed_tokens: dict[str, int] = {}
    remaining_by_request: dict[str, int] = {}
    paused_output_tokens: dict[str, list[int]] = {}
    reference_outputs: dict[str, list[int]] = {}
    results: dict = {}
    source: dict | None = None
    try:
        run_podman(["network", "create", network])
        source_spec = {
            "label": "source", "node_id": "tiny-batch-source",
            "role": "source", "gpus": source_gpus, "port": 5600,
        }
        target_spec = {
            "label": "target", "node_id": "tiny-batch-target",
            "role": "observer", "gpus": target_gpus, "port": 5700,
        }
        target_boot_started = None
        target_boot_s = None
        # NIXL requires the consumer before the producer.  The formal matrix
        # applies that same fresh-target-ready condition to every policy so
        # startup asymmetry cannot masquerade as a migration improvement.
        prestart_target = bool(
            args.prestart_target or args.nixl_prestart_target
        )
        if prestart_target:
            target_boot_started = time.monotonic()
            launch(target_spec)
            # Build the two fresh engines serially.  Concurrent TP2
            # FlashInfer/JIT initialization can contend on the shared cache
            # and leave both EngineCore processes waiting indefinitely.
            register([target_spec])
            target_boot_s = time.monotonic() - target_boot_started
            launch(source_spec)
            register([source_spec])
        else:
            launch(source_spec)
            register([source_spec])
        source = workers["source"]

        # Generate an uninterrupted deterministic reference on the same
        # source engine. Prefix caching is disabled in the worker, so these
        # requests validate token identity without donating reusable KV state
        # to the measured requests.
        reference_ids = {
            request_id: f"reference-{request_id}" for request_id in request_ids
        }
        for request_id in request_ids:
            send(source, {
                "op": "generate",
                "request_id": reference_ids[request_id],
                "token_ids": prompts[request_id],
                "max_new_tokens": args.max_new_tokens,
                "pause_after_new_tokens": 0,
            })
        for request_id in request_ids:
            reference_id = reference_ids[request_id]
            wait(source, lambda msg, reference_id=reference_id:
                 msg.get("event") == "generate_started"
                 and msg.get("request_id") == reference_id)
        for request_id in request_ids:
            reference_id = reference_ids[request_id]
            completed_reference = wait(
                source,
                lambda msg, reference_id=reference_id:
                msg.get("event") == "output"
                and msg.get("request_id") == reference_id
                and bool(msg.get("finished")),
            )
            reference_outputs[request_id] = list(
                completed_reference.get("cumulative_token_ids", []) or []
            )
            if (
                len(reference_outputs[request_id]) != args.max_new_tokens
                or completed_reference.get("finish_reason") != "length"
            ):
                raise AssertionError(
                    f"uninterrupted reference incomplete: {request_id}"
                )

        for request_id in request_ids:
            send(source, {
                "op": "generate", "request_id": request_id,
                "token_ids": prompts[request_id],
                "max_new_tokens": args.max_new_tokens,
                "pause_after_new_tokens": args.preempt_after_new_tokens,
            })
        for request_id in request_ids:
            wait(source, lambda msg, request_id=request_id:
                 msg.get("event") == "generate_started"
                 and msg.get("request_id") == request_id)
        for request_id in request_ids:
            paused = wait(source, lambda msg, request_id=request_id:
                          msg.get("event") == "paused"
                          and msg.get("request_id") == request_id)
            if paused.get("finished"):
                raise AssertionError(f"request finished before preemption: {request_id}")
            paused_output_tokens[request_id] = list(paused.get("token_ids", []))
        send(source, {"op": "pause_generation"})
        wait(source, lambda msg: msg.get("event") == "generation_paused")
        # Freeze first, then take an authoritative EngineCore boundary for
        # every policy. This prevents frontend scheduling jitter from moving
        # the preemption point between ablations.
        for request_id in request_ids:
            send(source, {"op": "metadata", "request_id": request_id})
        for request_id in request_ids:
            metadata[request_id] = wait(
                source,
                lambda msg, request_id=request_id:
                msg.get("event") == "metadata"
                and msg.get("request_id") == request_id,
            )["result"]
            if not metadata[request_id].get("found", False):
                raise AssertionError(
                    f"source metadata unavailable: {metadata[request_id]}"
                )
            source_tokens[request_id] = list(
                metadata[request_id].get("tokens", []) or []
            )
            source_completed_tokens[request_id] = int(
                metadata[request_id].get("completed_tokens", 0) or 0
            )
            lower = args.preempt_after_new_tokens
            upper = lower + args.preempt_max_overshoot
            if not lower <= source_completed_tokens[request_id] <= upper:
                raise AssertionError(
                    "preemption boundary outside accepted window for "
                    f"{request_id}: completed="
                    f"{source_completed_tokens[request_id]}, window={lower}..{upper}"
                )
            if len(source_tokens[request_id]) != (
                args.prompt_tokens + source_completed_tokens[request_id]
            ):
                raise AssertionError(
                    f"inconsistent source token snapshot: {request_id}"
                )
            if not metadata[request_id].get("moe_route_histogram_available"):
                calibrated = calibrated_routes.get(
                    prompt_audit[request_id]["token_ids_sha256"]
                )
                if calibrated is not None:
                    metadata[request_id].update({
                        "moe_route_histogram_available": True,
                        "moe_route_histogram_source": (
                            "vllm_runtime_topk_calibration"
                        ),
                        "moe_route_histogram_kind": "runtime_observed_topk",
                        "per_request_expert_route_histogram": dict(
                            calibrated["per_request_expert_route_histogram"]
                        ),
                        "moe_route_profile_sha256": route_profile_sha256,
                        "moe_route_profile_completed_tokens": int(
                            calibrated.get("captured_completed_tokens", 0) or 0
                        ),
                    })
            if args.require_moe_routing and not (
                metadata[request_id].get("moe_route_histogram_available")
                and metadata[request_id].get(
                    "per_request_expert_route_histogram"
                )
                and metadata[request_id].get("moe_route_histogram_source")
                in {
                    "vllm_runtime_topk",
                    "vllm_runtime_topk_calibration",
                }
                and metadata[request_id].get("moe_route_histogram_kind")
                == "runtime_observed_topk"
            ):
                raise AssertionError(
                    f"real MoE routing unavailable: {request_id}"
                )
        remaining_by_request = {
            request_id: max(
                args.max_new_tokens - source_completed_tokens[request_id], 1
            )
            for request_id in request_ids
        }

        active = apply_trace()
        preempt_started = time.monotonic()
        print(f"[tiny-batch] mode={args.mode} active={sorted(active)}", flush=True)
        if args.mode == "no_recovery":
            # NIXL export creates a connector-owned lease; the producer must
            # remain alive until the consumer has fetched the referenced KV
            # blocks.  Token-replay policies can release the source before
            # target startup, while migration policies release it only after
            # target completion below.
            if not uses_migration:
                stop("source")
            results = {
                "status": "passed", "outcome": "failed",
                "recovery": {
                    "recovery_s": round(time.monotonic() - preempt_started, 3),
                    "p99_latency_s": None,
                    "effective_throughput_tokens_s": 0.0,
                    "success_rate": 0.0, "recomputed_tokens": 0,
                    "restored_blocks": 0, "generated_tokens": 0,
                    "target_preexisting": False, "engine_created": False,
                    "placement_changed": False,
                },
            }
        else:
            target_groups: list[list[int]] = [target_gpus]
            planner_decision = None
            migration_planner_decision = None
            plan = None
            planner_started = time.monotonic()
            if uses_reparallelization:
                planner_decision, plan = plan_for(
                    active, uses_moe_reparallelization
                )
                planned_slots = [trace_slot(node) for node in plan.target_nodes]
                target_groups = [
                    planned_slots[index:index + tensor_parallel_size]
                    for index in range(0, len(planned_slots), tensor_parallel_size)
                ]
                if target_groups != [target_gpus]:
                    raise AssertionError(
                        f"expected one target group on GPUs {target_gpus}, "
                        f"planner selected {target_groups}"
                    )
            planner_s = time.monotonic() - planner_started

            exported: dict[str, dict] = {}
            export_s = 0.0
            migration_planner_s = 0.0
            migration_operation_started = None
            if uses_migration:
                migration_operation_started = time.monotonic()
                export_started = time.monotonic()
                for request_id in request_ids:
                    send(source, {"op": "export", "request_id": request_id})
                for request_id in request_ids:
                    exported[request_id] = wait(
                        source, lambda msg, request_id=request_id:
                        msg.get("event") == "export"
                        and msg.get("request_id") == request_id
                    )["result"]
                    if not exported[request_id].get("supports_restore"):
                        raise AssertionError(f"export failed: {exported[request_id]}")
                    # Keep the full visible token sequence.  computed_tokens
                    # marks how much KV is transferable; a usually one-token
                    # sampled tail may legitimately be recomputed on target.
                    exported_metadata = dict(
                        exported[request_id].get("metadata", {}) or {}
                    )
                    exported_metadata.update({
                        key: metadata[request_id][key]
                        for key in (
                            "moe_route_histogram_available",
                            "moe_route_histogram_source",
                            "moe_route_histogram_kind",
                            "per_request_expert_route_histogram",
                            "moe_route_profile_sha256",
                            "moe_route_profile_completed_tokens",
                        )
                        if key in metadata[request_id]
                    })
                    exported[request_id]["metadata"] = exported_metadata
                    exported_tokens = list(exported_metadata.get("tokens", []) or [])
                    computed_tokens = max(
                        int(exported_metadata.get("computed_tokens", 0) or 0), 0
                    )
                    if computed_tokens < 1 or computed_tokens > len(exported_tokens):
                        raise AssertionError(
                            "invalid exported token boundary for "
                            f"{request_id}: computed={computed_tokens}, "
                            f"available={len(exported_tokens)}"
                        )
                    source_tokens[request_id] = exported_tokens
                    source_completed_tokens[request_id] = max(
                        int(exported_metadata.get("completed_tokens", 0) or 0), 0
                    )
                    if source_completed_tokens[request_id] != int(
                        metadata[request_id].get("completed_tokens", 0) or 0
                    ):
                        raise AssertionError(
                            f"export moved frozen boundary: {request_id}"
                        )
                    remaining_by_request[request_id] = max(
                        args.max_new_tokens - source_completed_tokens[request_id], 1
                    )
                export_s = time.monotonic() - export_started
                migration_planner_started = time.monotonic()
                migration_planner_decision = plan_context_migration(
                    exported, uses_moe_migration
                )
                migration_planner_s = time.monotonic() - migration_planner_started
                # Finalize each source request while keeping its NIXL
                # producer engine alive.  This mirrors the project's proven
                # single-request smoke and lets the consumer fetch the
                # exported lease before the source container is released.
                for request_id in request_ids:
                    send(source, {"op": "abort", "request_id": request_id})
                for request_id in request_ids:
                    wait(
                        source,
                        lambda msg, request_id=request_id:
                        msg.get("event") == "aborted"
                        and msg.get("request_id") == request_id,
                    )

            if not uses_migration:
                stop("source")
            if "target" not in workers:
                target_boot_started = time.monotonic()
                launch(target_spec)
                register([target_spec])
                target_boot_s = time.monotonic() - target_boot_started
            target = workers["target"]
            restore_stage_s = 0.0
            if uses_migration:
                restore_started = time.monotonic()
                for request_id in request_ids:
                    send(target, {
                        "op": "restore", "request_id": request_id,
                        "state": exported[request_id],
                    })
                restored: dict[str, dict] = {}
                for request_id in request_ids:
                    restored[request_id] = wait(
                        target, lambda msg, request_id=request_id:
                        msg.get("event") == "restore"
                        and msg.get("request_id") == request_id
                    )["result"]
                    if not restored[request_id].get("staged"):
                        raise AssertionError(f"restore failed: {restored[request_id]}")
                restore_stage_s = time.monotonic() - restore_started
            else:
                restored = {}

            target_started = time.monotonic()
            for request_id in request_ids:
                send(target, {
                    "op": "generate", "request_id": request_id,
                    "token_ids": source_tokens[request_id],
                    "max_new_tokens": remaining_by_request[request_id],
                    "pause_after_new_tokens": 0,
                })
            for request_id in request_ids:
                wait(target, lambda msg, request_id=request_id:
                     msg.get("event") == "generate_started"
                     and msg.get("request_id") == request_id)
            first_started = time.monotonic()
            first_latencies: dict[str, float] = {}
            continued: dict[str, dict] = {}
            for request_id in request_ids:
                first_output = wait(
                    target,
                    lambda msg, request_id=request_id:
                    msg.get("event") == "output"
                    and msg.get("request_id") == request_id,
                )
                first_latencies[request_id] = time.monotonic() - first_started
                if first_output.get("finished"):
                    continued[request_id] = first_output
            downtime_s = time.monotonic() - preempt_started
            migration_operation_s = (
                time.monotonic() - migration_operation_started
                if migration_operation_started is not None else 0.0
            )
            # Wait for the final output of every request, not merely the first
            # token.  This makes recovery_s the requested
            # time-to-all-complete and makes throughput a real batch metric.
            for request_id in request_ids:
                if request_id in continued:
                    continue
                continued[request_id] = wait(
                    target, lambda msg, request_id=request_id:
                    msg.get("event") == "output"
                    and msg.get("request_id") == request_id
                    and bool(msg.get("finished"))
                )
            if uses_migration and "source" in workers:
                stop("source")
            recovery_wall_s = time.monotonic() - preempt_started
            recovery_s = recovery_wall_s
            generated = sum(
                int(row.get("generated_tokens", 0) or 0)
                for row in continued.values()
            )
            incomplete = {
                request_id: {
                    "expected": remaining_by_request[request_id],
                    "generated": int(
                        continued[request_id].get("generated_tokens", 0) or 0
                    ),
                    "finish_reason": continued[request_id].get("finish_reason"),
                }
                for request_id in request_ids
                if (
                    int(continued[request_id].get("generated_tokens", 0) or 0)
                    != remaining_by_request[request_id]
                    or continued[request_id].get("finish_reason") != "length"
                )
            }
            if incomplete:
                raise AssertionError(f"target generation incomplete: {incomplete}")
            sequence_checks: dict[str, dict] = {}
            for request_id in request_ids:
                target_suffix = list(
                    continued[request_id].get("cumulative_token_ids", []) or []
                )
                actual_output = (
                    source_tokens[request_id][args.prompt_tokens:] + target_suffix
                )
                reference_output = reference_outputs[request_id]
                equal = actual_output == reference_output
                sequence_checks[request_id] = {
                    "equal": equal,
                    "reference_token_count": len(reference_output),
                    "actual_token_count": len(actual_output),
                    "reference_sha256": token_hash(reference_output),
                    "actual_sha256": token_hash(actual_output),
                }
            if not all(row["equal"] for row in sequence_checks.values()):
                raise AssertionError(
                    f"continued sequence differs from reference: {sequence_checks}"
                )
            restored_blocks = sum(
                int(row.get("expected_blocks", 0) or 0)
                for row in restored.values()
            )
            if uses_migration:
                recomputed = sum(
                    max(
                        len(source_tokens[request_id])
                        - int(
                            exported[request_id]
                            .get("metadata", {})
                            .get("computed_tokens", 0)
                            or 0
                        ),
                        0,
                    )
                    for request_id in request_ids
                )
            else:
                recomputed = sum(
                    len(source_tokens[request_id]) for request_id in request_ids
                )
            route_audit = {
                request_id: {
                    "available": bool(
                        metadata[request_id].get(
                            "moe_route_histogram_available"
                        )
                    ),
                    "source": metadata[request_id].get(
                        "moe_route_histogram_source", "unavailable"
                    ),
                    "kind": metadata[request_id].get(
                        "moe_route_histogram_kind", "unavailable"
                    ),
                    "unique_layer_experts": len(
                        metadata[request_id].get(
                            "per_request_expert_route_histogram", {}
                        )
                        or {}
                    ),
                    "routed_topk_assignments": sum(
                        int(value or 0)
                        for value in (
                            metadata[request_id].get(
                                "per_request_expert_route_histogram", {}
                            )
                            or {}
                        ).values()
                    ),
                    "profile_sha256": metadata[request_id].get(
                        "moe_route_profile_sha256"
                    ),
                    "profile_completed_tokens": metadata[request_id].get(
                        "moe_route_profile_completed_tokens"
                    ),
                }
                for request_id in request_ids
            }
            results = {
                "status": "passed", "outcome": "continued",
                "planner_decision": planner_decision,
                "migration_planner_decision": migration_planner_decision,
                "recovery": {
                    "recovery_s": round(recovery_s, 3),
                    "recovery_wall_s": round(recovery_wall_s, 3),
                    "target_startup_s": round(target_boot_s or 0.0, 3),
                    "target_recovery_s": round(time.monotonic() - target_started, 3),
                    "avg_latency_s": round(
                        statistics.mean(first_latencies.values()), 3
                    ),
                    "p99_latency_s": percentile(list(first_latencies.values()), 0.99),
                    "downtime_s": round(downtime_s, 3),
                    "state_export_s": round(export_s, 6),
                    "state_restore_stage_s": round(restore_stage_s, 6),
                    "migration_planner_s": round(migration_planner_s, 6),
                    "migration_operation_s": round(migration_operation_s, 6),
                    "reparallelization_planner_s": round(planner_s, 6),
                    "reparallelization_s": round(
                        planner_s + (target_boot_s or 0.0), 6
                    ) if uses_reparallelization else 0.0,
                    "effective_throughput_tokens_s": round(
                        generated / max(recovery_s, 1e-9), 3
                    ),
                    "success_rate": 1.0,
                    "recomputed_tokens": recomputed,
                    "restored_blocks": restored_blocks,
                    "generated_tokens": generated,
                    "target_preexisting": prestart_target,
                    "engine_created": True,
                    "placement_changed": True,
                    "target_gpus": target_gpus,
                    "target_replicas": len(target_groups),
                    "configuration_applied": uses_reparallelization,
                    "restore_success": uses_migration,
                    "moe_migration_optimization": uses_moe_migration,
                    "moe_reparallelization_optimization": (
                        uses_moe_reparallelization
                    ),
                    "request_count": args.request_count,
                    "source_prefix_tokens": {
                        request_id: len(source_tokens[request_id])
                        for request_id in request_ids
                    },
                    "source_completed_tokens": source_completed_tokens,
                    "target_requested_tokens": remaining_by_request,
                    "target_outputs": {
                        request_id: {
                            "generated_tokens": int(
                                row.get("generated_tokens", 0) or 0
                            ),
                            "finish_reason": row.get("finish_reason"),
                            "stop_reason": row.get("stop_reason"),
                        }
                        for request_id, row in continued.items()
                    },
                    "sequence_checks": sequence_checks,
                    "all_sequences_equal_reference": all(
                        row["equal"] for row in sequence_checks.values()
                    ),
                    "moe_route_audit": route_audit,
                    "moe_route_profiles": {
                        request_id: {
                            "prompt_token_ids_sha256": prompt_audit[
                                request_id
                            ]["token_ids_sha256"],
                            "captured_completed_tokens": (
                                metadata[request_id].get(
                                    "moe_route_profile_completed_tokens",
                                    source_completed_tokens[request_id],
                                )
                            ),
                            "source": metadata[request_id].get(
                                "moe_route_histogram_source", "unavailable"
                            ),
                            "kind": metadata[request_id].get(
                                "moe_route_histogram_kind", "unavailable"
                            ),
                            "per_request_expert_route_histogram": dict(
                                metadata[request_id].get(
                                    "per_request_expert_route_histogram", {}
                                )
                                or {}
                            ),
                        }
                        for request_id in request_ids
                    },
                    "moe_route_histogram_available_count": sum(
                        int(row["available"]) for row in route_audit.values()
                    ),
                },
            }
            stop("target")
        results.update({
            "mode": args.mode,
            "tensor_parallel_size": tensor_parallel_size,
            "cpu_offload_gb": args.cpu_offload_gb,
            "request_count": args.request_count,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "preempt_after_new_tokens": args.preempt_after_new_tokens,
            "preempt_max_overshoot": args.preempt_max_overshoot,
            "prompt_audit": prompt_audit,
            "uninterrupted_reference": {
                request_id: {
                    "token_count": len(tokens),
                    "token_ids_sha256": token_hash(tokens),
                }
                for request_id, tokens in reference_outputs.items()
            },
            "route_profile_input": args.route_profile,
            "route_profile_input_sha256": route_profile_sha256,
            "trace": args.trace,
            "trace_speedup": args.trace_speedup,
            "gpus": args.gpus,
            "elapsed_s": round(time.monotonic() - started, 3),
        })
    except (TimeoutError, socket.timeout):
        dump_container_logs(container_names)
        raise
    finally:
        for label in list(workers):
            stop(label)
        listener.close()
        run_podman(["network", "rm", network], check=False)
        shutil.rmtree(control_dir, ignore_errors=True)
    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True))
    recovery = results.get("recovery", {})
    print(json.dumps({
        "status": results.get("status"),
        "mode": results.get("mode"),
        "outcome": results.get("outcome"),
        "elapsed_s": results.get("elapsed_s"),
        "recovery_s": recovery.get("recovery_s"),
        "target_recovery_s": recovery.get("target_recovery_s"),
        "generated_tokens": recovery.get("generated_tokens"),
        "recomputed_tokens": recovery.get("recomputed_tokens"),
        "restored_blocks": recovery.get("restored_blocks"),
        "all_sequences_equal_reference": recovery.get(
            "all_sequences_equal_reference"
        ),
        "moe_route_histogram_available_count": recovery.get(
            "moe_route_histogram_available_count"
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
