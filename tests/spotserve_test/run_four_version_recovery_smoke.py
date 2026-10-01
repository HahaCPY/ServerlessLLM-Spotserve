"""Compare four explicitly separated preemption recovery policies.

The four modes are:

* ``no_recovery``: the source is preempted and the request is expected to fail;
* ``rerouting``: an already-ready replica receives the full context, without
  changing its placement or creating an engine;
* ``reparallelization``: the source is stopped, any existing recovery target
  is stopped, and a new legal engine is created on the GPUs still active in the
  four-slot trace;
* ``modified``: the SpotServe NIXL export/restore path attaches source KV state
  to the already-ready target replica.

This is a same-host, separate-container experiment over four explicit GPU
slots.  Tiny uses TP1 source/target placements; the Qwen1.5-MoE-A2.7B
checkpoint uses TP2 source/target placements.  The workload is real vLLM
generation and the modified path uses the real NIXL connector; the first
three modes deliberately do not call export or restore.
"""

import argparse
import json
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
    run_podman,
    send,
    wait_event,
)
from run_four_container_fleet_churn_smoke import load_fleet_trace
from run_four_container_fleet_churn_smoke import trace_slot
from sllm.spot.reparallelization import (
    ParallelPlan,
    plan_dynamic_reparallelization,
)


MODES = ("no_recovery", "rerouting", "reparallelization", "modified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--prompt-tokens", type=int, default=480)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--trace-speedup", type=float, default=1000.0)
    parser.add_argument("--token-delay-s", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--preempt-after-new-tokens",
        type=int,
        default=1,
        help=(
            "Number of generated output tokens to produce before the source "
            "is paused/preempted."
        ),
    )
    parser.add_argument(
        "--dynamic-planner",
        action="store_true",
        help=(
            "For Modified, invoke the live ParallelPlan planner after the "
            "trace event and create the selected target configuration before "
            "restoring KV state."
        ),
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="Optional per-worker CPU weight offload for larger checkpoints.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.08,
        help="Fraction of each worker GPU memory reserved by vLLM.",
    )
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def infer_tensor_parallel_size(model_path: str) -> int:
    """Infer the minimum TP from the checkpoint instead of hard-coding Tiny."""
    try:
        config = json.loads(Path(model_path, "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        config = {}
    # Qwen1.5-MoE-A2.7B has 60 experts and a 26.67 GiB checkpoint; it needs
    # TP2 on the 16 GiB cards used by this host.  Tiny has four experts and
    # fits on one card.
    experts = max(
        int(config.get("num_experts", 0) or 0),
        int(config.get("num_local_experts", 0) or 0),
    )
    return 2 if experts >= 30 or "A2.7B" in model_path else 1


def p99(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(0, int(0.99 * (len(ordered) - 1)))
    return round(ordered[rank], 3)


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise SystemExit("--gpus must contain four distinct GPU indices")
    if args.prompt_tokens < 1 or args.prompt_tokens > args.max_model_len:
        raise SystemExit("--prompt-tokens must fit within --max-model-len")
    if args.preempt_after_new_tokens < 0:
        raise SystemExit("--preempt-after-new-tokens must be non-negative")
    if (
        args.preempt_after_new_tokens > 0
        and args.preempt_after_new_tokens >= args.max_new_tokens
    ):
        raise SystemExit(
            "--preempt-after-new-tokens must be smaller than "
            "--max-new-tokens so the request is still live at preemption"
        )
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        raise SystemExit(f"model config not found: {args.model}")
    tensor_parallel_size = infer_tensor_parallel_size(args.model)
    if tensor_parallel_size > len(args.gpus) // 2:
        raise SystemExit(
            f"{args.model} needs TP={tensor_parallel_size}, but only four GPU slots "
            "were provided"
        )
    model_is_large = tensor_parallel_size > 1
    # Tiny uses one source card and one pre-existing target card.  The target
    # is deliberately node-3 because the Tiny capacity trace removes
    # node-0/1/2 together at the first pressure drop, leaving node-3 ready.
    # The Qwen2.7B-class checkpoint uses two-card source and target replicas.
    source_gpus = list(args.gpus[:tensor_parallel_size])
    target_gpus = (
        list(args.gpus[2:4])
        if model_is_large
        else [args.gpus[3]]
    )
    trace_events = load_fleet_trace(args.trace)
    if not any(
        event["event"] == "remove"
        and any(
            trace_slot(node) in set(source_gpus)
            for node in event["nodes"]
        )
        for event in trace_events
    ):
        raise SystemExit("trace must remove at least one source GPU")

    control_dir = os.path.abspath(
        os.path.join("/tmp", f"spotserve-four-version-{os.getpid()}")
    )
    os.makedirs(control_dir, mode=0o777, exist_ok=False)
    os.chmod(control_dir, 0o777)
    socket_path = os.path.join(control_dir, "control.sock")
    listener = Listener(socket_path, family="AF_UNIX", authkey=b"spotserve")
    os.chmod(socket_path, 0o666)
    listener._listener._socket.settimeout(args.timeout_s)
    network = f"spotserve-four-version-net-{os.getpid()}"
    container_names: list[str] = []
    workers: dict[str, dict] = {}
    pending: dict[str, tuple[dict, object]] = {}
    started = time.monotonic()

    common = [
        "run",
        "--detach",
        "--network",
        network,
        "--volume",
        f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "--volume",
        f"{VLLM_ROOT}:{VLLM_ROOT}:ro",
        # Keep vLLM/Triton compilation artifacts across the isolated cells.
        # Each cell still creates fresh workers and follows the trace, but it
        # should not pay the same kernel compilation cost 36 times.
        "--volume",
        "/home/undergrad2026/s112060021/.cache/vllm:/root/.cache/vllm:rw",
        "--volume",
        "/tmp/torchinductor_s112060021:/tmp/torchinductor_s112060021:rw",
        "--volume",
        f"{MODEL_ROOT}:{MODEL_ROOT}:ro",
        "--volume",
        "/usr/local/cuda-13.0:/usr/local/cuda:ro",
        "--volume",
        f"{control_dir}:/control:rw",
        "--env",
        f"PYTHONPATH={PYTHONPATH}",
        "--env",
        "VLLM_CACHE_ROOT=/root/.cache/vllm",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_s112060021",
        "--env",
        "PYTHONUNBUFFERED=1",
        args.image,
    ]
    worker_script = f"{REPO_ROOT}/tests/spotserve_test/cross_container_nixl_worker.py"

    def command(
        label: str,
        node_id: str,
        role: str,
        gpus: list[int],
        tensor_parallel_size: int,
        port: int,
    ) -> list[str]:
        name = f"spotserve-four-version-{label}-{os.getpid()}"
        container_names.append(name)
        role_args = [
            "python",
            "-u",
            worker_script,
            "--model",
            args.model,
            "--control-socket",
            "/control/control.sock",
            "--role",
            role,
            "--node-id",
            node_id,
            "--side-channel-host",
            node_id,
            "--side-channel-port",
            str(port),
            "--token-delay-s",
            str(args.token_delay_s),
            "--max-new-tokens",
            str(max(int(args.max_new_tokens), 1)),
            "--pause-after-new-tokens",
            str(max(int(args.preempt_after_new_tokens), 0)),
            "--cpu-offload-gb",
            str(max(float(args.cpu_offload_gb), 0.0)),
            "--gpu-memory-utilization",
            str(min(max(float(args.gpu_memory_utilization), 0.01), 0.99)),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--max-model-len",
            str(args.max_model_len),
            # Only the Modified policy uses the NIXL connector.  The three
            # baselines explicitly disallow KV transfer; starting them without
            # a connector avoids charging baseline startup with NIXL
            # initialization that their recovery policy never uses.
            "--kv-transfer-mode",
            # The spare is deliberately not a NIXL participant.  It is
            # removed/re-added by the trace before source preemption; keeping
            # a connector on that churn-only worker can tear down the source
            # connector's peer bookkeeping and make the paused request look
            # inactive.  Source and the ready recovery target still use the
            # real NIXL path.
            "nixl"
            if args.mode == "modified" and label != "spare"
            else "none",
        ]
        device_args: list[str] = []
        for gpu in gpus:
            device_args.extend(["--device", f"nvidia.com/gpu={gpu}"])
        return [
            *common[:1],
            "--name",
            name,
            "--hostname",
            node_id,
            *device_args,
            *common[1:],
            "bash",
            "-lc",
            "exec " + shlex.join(role_args),
        ]

    def launch(spec: dict) -> None:
        spec["name"] = f"spotserve-four-version-{spec['label']}-{os.getpid()}"
        run_podman(
            command(
                spec["label"],
                spec["node_id"],
                spec["role"],
                spec["gpus"],
                spec["tp"],
                spec["port"],
            )
        )

    def register(specs: list[dict]) -> None:
        expected = {spec["node_id"]: spec for spec in specs}
        while expected:
            cached = pending.pop(next(iter(expected)), None)
            if cached is not None:
                ready, conn = cached
            else:
                conn = listener.accept()
                ready = wait_event(conn, "ready", args.timeout_s)
            node_id = ready.get("node_id")
            if node_id not in expected:
                pending[node_id] = (ready, conn)
                continue
            spec = expected.pop(node_id)
            ready["conn"] = conn
            workers[spec["label"]] = {**spec, **ready}

    def stop(label: str) -> None:
        worker = workers.pop(label)
        name = worker["name"]
        run_podman(["kill", "--signal", "TERM", name], check=False)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = run_podman(
                ["inspect", "--format", "{{.State.Running}}", name], check=False
            )
            if state.returncode != 0 or state.stdout.strip() == "false":
                break
            time.sleep(0.2)
        run_podman(["rm", "--force", name], check=False)
        try:
            worker["conn"].close()
        except OSError:
            pass

    def target_generate(
        target: dict,
        request_id: str,
        token_ids: list[int],
        remaining_new_tokens: int | None = None,
    ) -> dict:
        phase_started = time.monotonic()
        print(
            f"[four-version] target generate label={target['label']} "
            f"tokens={len(token_ids)}",
            flush=True,
        )
        send(
            target["conn"],
            {
                "op": "generate",
                "request_id": request_id,
                "token_ids": token_ids,
                "max_new_tokens": max(
                    int(
                        remaining_new_tokens
                        if remaining_new_tokens is not None
                        else args.max_new_tokens
                    ),
                    1,
                ),
                # The target is allowed to produce one output before the
                # harness resumes it and measures continuation.  The source
                # pause threshold is controlled separately below.
                "pause_after_new_tokens": 1,
            },
        )
        wait_event(target["conn"], "generate_started", args.timeout_s)
        print("[four-version] target generate_started", flush=True)
        first = wait_event(target["conn"], "output", args.timeout_s)
        print("[four-version] target first output", flush=True)
        wait_event(target["conn"], "paused", args.timeout_s)
        first_latency_s = time.monotonic() - phase_started
        send(target["conn"], {"op": "resume", "request_id": request_id})
        wait_event(target["conn"], "resumed", args.timeout_s)
        print("[four-version] target resumed", flush=True)
        continued_started = time.monotonic()
        output_latencies = [first_latency_s]
        continued = wait_event(target["conn"], "output", args.timeout_s)
        output_latencies.append(time.monotonic() - phase_started)
        output_count = len(continued.get("token_ids", []))
        continuation_s = round(time.monotonic() - continued_started, 3)
        if not continued.get("token_ids") and output_count <= 0:
            raise AssertionError("target did not continue after source preemption")
        total_s = max(time.monotonic() - phase_started, 1e-9)
        return {
            "first_output_tokens": len(first.get("token_ids", [])),
            "target_recovery_s": round(first_latency_s, 3),
            "target_continuation_s": continuation_s,
            "p99_latency_s": p99(output_latencies),
            "effective_throughput_tokens_s": round(output_count / total_s, 3),
            "generated_tokens": output_count,
            "continued": True,
        }

    source_spec = {
        "label": "source",
        "node_id": "four-version-source",
        "role": "source",
        "gpus": source_gpus,
        "tp": tensor_parallel_size,
        "port": 5600,
    }
    reroute_spec = {
        "label": "reroute_replica",
        "node_id": "four-version-reroute-replica",
        "role": "observer",
        "gpus": target_gpus,
        "tp": tensor_parallel_size,
        "port": 5700,
    }
    # A controlled Modified run may keep a compatible target READY, but the
    # dynamic Modified path intentionally creates the target only after the
    # planner sees the post-preemption capacity.  Rerouting never receives a
    # prewarmed backup engine: it starts a same-shape replacement after the
    # source is removed, so its baseline includes target startup cost.
    request_id = f"four-version-request-{os.getpid()}"
    prompt = [100 + index for index in range(args.prompt_tokens)]
    metadata: dict = {}
    source_computed: list[int] = []
    exported: dict | None = None
    planner_decision: dict | None = None
    source_generated_before_preemption = 0
    trace_history: list[dict] = []
    outcome = "failed"
    recovery: dict = {}

    def dynamic_plan(active_slots: set[int], event: str) -> dict:
        """Run the same capacity-aware planner used by the SpotServe router.

        The four visible cards are represented as one-GPU logical nodes.  The
        resulting target node list is then converted back into explicit GPU
        devices for the container deployment below.  This keeps planner
        selection and the actual target configuration in one test path.
        """
        model_name = Path(args.model).name
        worker_nodes = {
            f"node-{gpu}": {
                "ray_node_id": f"node-{gpu}",
                "address": f"node-{gpu}",
                "free_gpu": 1 if gpu in active_slots else 0,
                "total_gpu": 1,
                "state": "ready" if gpu in active_slots else "dead",
            }
            for gpu in args.gpus
        }
        # NIXL KV layout must remain compatible across the migration.  The
        # backend advertises EP variants, but this request starts on EP=1;
        # filter the capability allowlist to the verified EP=1 shapes so the
        # planner cannot select an incompatible target just because it scores
        # higher on raw GPU utilisation.
        from sllm.backends.vllm_capability import get_vllm_capability

        capability = get_vllm_capability(
            {
                "model": model_name,
                "num_gpus": len(args.gpus),
                "backend_config": {
                    "pretrained_model_name_or_path": args.model,
                    "tensor_parallel_size": tensor_parallel_size,
                },
            }
        ).to_dict()
        capability["supported_configs"] = [
            config
            for config in capability["supported_configs"]
            if int(config.get("expert_parallel_size", 1) or 1) == 1
        ]
        return plan_dynamic_reparallelization(
            model_name=model_name,
            worker_nodes=worker_nodes,
            model_config={
                "model": model_name,
                "backend": "vllm",
                "num_gpus": len(args.gpus),
                "backend_config": {
                    "pretrained_model_name_or_path": args.model,
                    "tensor_parallel_size": tensor_parallel_size,
                },
                "backend_capability": capability,
            },
            planner_config={
                "model_gpu_requirement": tensor_parallel_size,
                "target_replica_gpus": tensor_parallel_size,
                "min_tensor_parallel_size": 1,
                "max_tensor_parallel_size": len(active_slots),
                "max_pipeline_parallel_size": 1,
                "min_data_parallel_size": 1,
            },
            event=event,
            node_id=event,
            backend="vllm",
        )

    def plan_target_groups(
        decision: dict,
    ) -> tuple[ParallelPlan, list[list[int]]]:
        selected = decision.get("parallel_plan")
        if not selected:
            raise AssertionError(
                f"planner returned no usable target: {json.dumps(decision)}"
            )
        plan = ParallelPlan.from_dict(selected)
        target_gpus = [trace_slot(node) for node in plan.target_nodes]
        if len(target_gpus) < plan.num_gpus:
            raise AssertionError(
                f"planner target {plan.target_nodes} has fewer devices than "
                f"selected plan requires: {plan.to_dict()}"
            )
        replica_gpu_count = plan.tensor_parallel_size * plan.pipeline_parallel_size
        groups = [
            target_gpus[index : index + replica_gpu_count]
            for index in range(
                0, replica_gpu_count * plan.num_replicas, replica_gpu_count
            )
        ]
        if len(groups) != plan.num_replicas or any(
            len(group) != replica_gpu_count for group in groups
        ):
            raise AssertionError(
                f"planner target nodes cannot realize all replicas: "
                f"{plan.to_dict()}"
            )
        return plan, groups

    def follow_trace_to_preemption() -> set[int]:
        """Apply the bounded four-slot trace until source preemption.

        Add/remove events are applied to all four logical GPU slots.  For
        rerouting and Modified, the target engine was started before the
        trace and must remain READY until the source-removal event.
        """
        source_set = set(source_gpus)
        target_set = set(target_gpus)
        # Start with no logical capacity and let the trace add the source and
        # target slots.  The containers are provisioned by the harness before
        # the trace because vLLM startup is separate from capacity admission;
        # the planner, however, must see the trace's actual active set.
        active_slots: set[int] = set()
        previous_time_ms = 0.0
        for event in trace_events:
            delay_s = (
                max(float(event["time_ms"]) - previous_time_ms, 0.0)
                / 1000.0
                / args.trace_speedup
            )
            if delay_s:
                time.sleep(delay_s)
            previous_time_ms = float(event["time_ms"])
            action = event["event"]
            if action == "DONE":
                break
            changed_slots: set[int] = set()
            for node in event["nodes"]:
                gpu = trace_slot(node)
                if gpu not in set(args.gpus):
                    raise AssertionError(
                        f"trace GPU {gpu} is not present in --gpus: {event}"
                    )
                if action == "add":
                    active_slots.add(gpu)
                    changed_slots.add(gpu)
                elif action == "remove":
                    active_slots.discard(gpu)
                    changed_slots.add(gpu)
            if action not in {"add", "remove"}:
                raise AssertionError(
                    f"four-version trace only supports add/remove/DONE: {event}"
                )
            trace_history.append(
                {
                    "time_ms": event["time_ms"],
                    "event": action,
                    "nodes": list(event["nodes"]),
                    "active_slots": sorted(active_slots),
                }
            )
            if action == "remove" and target_set & changed_slots and not (
                source_set & changed_slots
            ):
                raise AssertionError(
                    "trace removed the pre-existing recovery target before "
                    "source preemption"
                )
            if source_set & changed_slots and action == "remove":
                return active_slots
        raise AssertionError("trace must remove at least one source GPU")

    try:
        run_podman(["network", "create", network])
        launch(source_spec)
        startup_specs = [source_spec]
        if args.mode == "modified" and not args.dynamic_planner:
            launch(reroute_spec)
            startup_specs.append(reroute_spec)
        register(startup_specs)
        print(
            f"[four-version] startup ready labels={sorted(workers)} "
            f"tp={tensor_parallel_size}",
            flush=True,
        )
        source = workers["source"]
        send(
            source["conn"],
            {
                "op": "generate",
                "request_id": request_id,
                "token_ids": prompt,
                "max_new_tokens": max(int(args.max_new_tokens), 1),
                "pause_after_new_tokens": max(
                    int(args.preempt_after_new_tokens), 0
                ),
            },
        )
        wait_event(source["conn"], "generate_started", args.timeout_s)
        if args.preempt_after_new_tokens > 0:
            source_pause = wait_event(source["conn"], "paused", args.timeout_s)
            if source_pause.get("finished"):
                raise AssertionError(
                    "source request finished before preemption; increase "
                    "--max-new-tokens or lower "
                    "--preempt-after-new-tokens"
                )
            source_generated_before_preemption = int(
                source_pause.get("generated_tokens", 0)
                or len(source_pause.get("token_ids", []))
            )
            print(
                "[four-version] source paused after "
                f"{source_generated_before_preemption} generated tokens",
                flush=True,
            )
        else:
            print(
                "[four-version] source remains live while trace advances",
                flush=True,
            )

        active_slots = follow_trace_to_preemption()
        preempt_started = time.monotonic()
        print(
            f"[four-version] source preempted active={sorted(active_slots)}",
            flush=True,
        )

        if args.mode != "no_recovery":
            send(source["conn"], {"op": "metadata", "request_id": request_id})
            metadata = wait_event(source["conn"], "metadata", args.timeout_s)["result"]
            computed_tokens = int(
                metadata.get("computed_tokens", len(metadata.get("tokens", prompt)))
                or 0
            )
            source_computed = list(metadata.get("tokens", prompt))[:computed_tokens]
            if args.preempt_after_new_tokens == 0:
                source_generated_before_preemption = max(
                    0, len(source_computed) - len(prompt)
                )
            print(
                f"[four-version] source metadata computed={computed_tokens}",
                flush=True,
            )
        remaining_new_tokens = max(
            int(args.max_new_tokens) - source_generated_before_preemption,
            1,
        )

        if args.mode == "no_recovery":
            stop("source")
            recovery = {
                "request_outcome": "failed",
                "failure_reason": "preempted_worker_invalid",
                "target_continued": False,
                "recomputed_tokens": 0,
                "restored_blocks": 0,
                "target_preexisting": False,
                "engine_created": False,
                "placement_changed": False,
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": None,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
                "p99_latency_s": None,
                "effective_throughput_tokens_s": 0.0,
                "generated_tokens": 0,
            }
            outcome = "failed"
        elif args.mode == "rerouting":
            stop("source")
            print(
                "[four-version] source stopped; creating same-shape "
                "rerouting target",
                flush=True,
            )
            available_gpus = [gpu for gpu in args.gpus if gpu in active_slots]
            if len(available_gpus) < tensor_parallel_size:
                raise AssertionError(
                    f"trace leaves {available_gpus}, cannot create TP="
                    f"{tensor_parallel_size} rerouting target"
                )
            new_target_gpus = available_gpus[:tensor_parallel_size]
            new_target_spec = {
                "label": "reroute_target",
                "node_id": "four-version-reroute-target",
                "role": "observer",
                "gpus": new_target_gpus,
                "tp": tensor_parallel_size,
                "port": 5700,
            }
            launch(new_target_spec)
            register([new_target_spec])
            target = workers["reroute_target"]
            target_result = target_generate(
                target,
                request_id,
                source_computed,
                remaining_new_tokens,
            )
            recovery = {
                "request_outcome": "continued",
                "target_continued": target_result["continued"],
                "recomputed_tokens": len(source_computed),
                "restored_blocks": 0,
                "target_preexisting": False,
                "engine_created": True,
                "placement_changed": True,
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": tensor_parallel_size,
                "new_target_gpus": new_target_gpus,
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "continued"
        elif args.mode == "reparallelization":
            stop("source")
            if "reroute_replica" in workers:
                stop("reroute_replica")
            available_gpus = [gpu for gpu in args.gpus if gpu in active_slots]
            if len(available_gpus) < tensor_parallel_size:
                raise AssertionError(
                    f"trace leaves {available_gpus}, cannot create TP="
                    f"{tensor_parallel_size} target"
                )
            new_target_gpus = available_gpus[:tensor_parallel_size]
            new_target_spec = {
                "label": "reparallelized_target",
                "node_id": "four-version-reparallelized-target",
                "role": "observer",
                "gpus": new_target_gpus,
                "tp": tensor_parallel_size,
                "port": 5900,
            }
            launch(new_target_spec)
            register([new_target_spec])
            target_result = target_generate(
                workers["reparallelized_target"],
                request_id,
                source_computed,
                remaining_new_tokens,
            )
            recovery = {
                "request_outcome": "continued",
                "target_continued": target_result["continued"],
                "recomputed_tokens": len(source_computed),
                "restored_blocks": 0,
                "target_preexisting": False,
                "engine_created": True,
                "placement_changed": True,
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": tensor_parallel_size,
                "new_target_gpus": new_target_gpus,
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "continued"
        else:
            if args.dynamic_planner:
                planner_decision = dynamic_plan(active_slots, "remove")
                plan, planned_target_groups = plan_target_groups(
                    planner_decision
                )
                print(
                    "[four-version] planner selected "
                    f"TP={plan.tensor_parallel_size} "
                    f"target_groups={planned_target_groups} "
                    f"DP={plan.data_parallel_size}",
                    flush=True,
                )
                # Export while the source request is still live.  The target
                # engine is then created from the selected ParallelPlan, so
                # this path tests configuration re-selection and KV restore
                # together rather than using a fixed READY target.
                print(
                    "[four-version] exporting source state before dynamic "
                    "target startup",
                    flush=True,
                )
            else:
                plan = None
                planned_target_gpus = []
                target = workers["reroute_replica"]
                print("[four-version] exporting source state", flush=True)
            send(source["conn"], {"op": "export", "request_id": request_id})
            exported = wait_event(source["conn"], "export", args.timeout_s)["result"]
            if not exported.get("supports_restore"):
                raise AssertionError(f"source export failed: {exported}")
            if args.dynamic_planner:
                dynamic_target_specs = []
                for replica, replica_gpus in enumerate(planned_target_groups):
                    dynamic_target_specs.append(
                        {
                            "label": f"planned_target_{replica}",
                            "node_id": f"four-version-planned-target-{replica}",
                            "role": "observer",
                            "gpus": replica_gpus,
                            "tp": plan.tensor_parallel_size,
                            "port": 5700 + replica,
                        }
                    )
                for dynamic_target_spec in dynamic_target_specs:
                    launch(dynamic_target_spec)
                register(dynamic_target_specs)
                target = workers["planned_target_0"]
            send(source["conn"], {"op": "abort", "request_id": request_id})
            wait_event(source["conn"], "aborted", args.timeout_s)
            send(
                target["conn"],
                {"op": "restore", "request_id": request_id, "state": exported},
            )
            staged = wait_event(target["conn"], "restore", args.timeout_s)["result"]
            if not staged.get("staged"):
                raise AssertionError(f"target restore failed: {staged}")
            target_result = target_generate(
                target,
                request_id,
                source_computed,
                remaining_new_tokens,
            )
            stop("source")
            recovery = {
                "request_outcome": "continued",
                "target_continued": target_result["continued"],
                "recomputed_tokens": 0,
                "restored_blocks": int(staged.get("expected_blocks", 0) or 0),
                "target_preexisting": not args.dynamic_planner,
                "engine_created": bool(args.dynamic_planner),
                "placement_changed": bool(args.dynamic_planner),
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": (
                    plan.tensor_parallel_size
                    if plan is not None
                    else tensor_parallel_size
                ),
                "restore_success": True,
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            if planner_decision is not None:
                recovery["planner_action"] = planner_decision.get("action")
                recovery["planner_selected_plan"] = planner_decision.get(
                    "parallel_plan"
                )
                recovery["target_replicas"] = plan.num_replicas
                recovery["target_groups"] = planned_target_groups
                recovery["configuration_applied"] = True
            outcome = "continued"

        source_blocks = len(metadata.get("block_ids", []))
        report = {
            "status": "passed",
            "mode": args.mode,
            "model": args.model,
            "trace": args.trace,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "preempt_after_new_tokens": args.preempt_after_new_tokens,
            "source_generated_before_preemption": (
                source_generated_before_preemption
            ),
            "dynamic_planner": bool(args.dynamic_planner),
            "planner_decision": planner_decision,
            "trace_events_applied": trace_history,
            "source_computed_tokens": len(source_computed),
            "source_blocks": source_blocks,
            "source_config": {
                "gpus": source_gpus,
                "tensor_parallel_size": tensor_parallel_size,
            },
            "outcome": outcome,
            "expected_outcome": "failed" if args.mode == "no_recovery" else "continued",
            "recovery": recovery,
            "metrics": {
                "recovery_time_s": recovery.get("recovery_s"),
                "p99_latency_s": recovery.get("p99_latency_s"),
                "effective_throughput_tokens_s": recovery.get(
                    "effective_throughput_tokens_s", 0.0
                ),
                "success_rate": 1.0 if outcome == "continued" else 0.0,
                "recovery_data": {
                    "recomputed_tokens": recovery.get("recomputed_tokens", 0),
                    "restored_blocks": recovery.get("restored_blocks", 0),
                    "generated_tokens": recovery.get("generated_tokens", 0),
                    "target_preexisting": recovery.get("target_preexisting", False),
                    "engine_created": recovery.get("engine_created", False),
                    "placement_changed": recovery.get("placement_changed", False),
                },
            },
            "physical_cross_node": False,
            "trace_event_count": len(trace_events),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, sort_keys=True))
    except (TimeoutError, socket.timeout):
        dump_container_logs(container_names)
        raise
    finally:
        for worker in list(workers.values()):
            try:
                send(worker["conn"], {"op": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                worker["conn"].close()
            except OSError:
                pass
            run_podman(["rm", "--force", worker["name"]], check=False)
        listener.close()
        run_podman(["network", "rm", network], check=False)
        shutil.rmtree(control_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
