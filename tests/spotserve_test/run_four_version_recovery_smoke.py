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

    def target_generate(target: dict, request_id: str, token_ids: list[int]) -> dict:
        phase_started = time.monotonic()
        print(
            f"[four-version] target generate label={target['label']} "
            f"tokens={len(token_ids)}",
            flush=True,
        )
        send(
            target["conn"],
            {"op": "generate", "request_id": request_id, "token_ids": token_ids},
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
    # Spare workers are useful for Tiny's churn trace, but Qwen's TP2 source
    # and target replicas already cover all four cards.  They are not needed
    # to define the recovery policy, so the trace can mark those slots ready
    # without starting extra engines.
    request_id = f"four-version-request-{os.getpid()}"
    prompt = [100 + index for index in range(args.prompt_tokens)]
    metadata: dict = {}
    source_computed: list[int] = []
    exported: dict | None = None
    outcome = "failed"
    recovery: dict = {}

    def follow_trace_to_preemption() -> set[int]:
        """Apply the bounded four-slot trace until source preemption.

        Add/remove events are applied to all four logical GPU slots.  For
        rerouting and Modified, the target engine was started before the
        trace and must remain READY until the source-removal event.
        """
        source_set = set(source_gpus)
        target_set = set(target_gpus)
        # Source and (when required by the policy) the recovery target are
        # already provisioned before the trace starts.  The trace controls
        # which slots remain available after each event; starting from all
        # four slots keeps the four-version comparison's READY-target
        # contract independent of whether a capacity trace adds a target
        # node at an earlier timestamp.
        active_slots = set(args.gpus)
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
        if args.mode in {"rerouting", "modified"}:
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
            {"op": "generate", "request_id": request_id, "token_ids": prompt},
        )
        wait_event(source["conn"], "generate_started", args.timeout_s)
        wait_event(source["conn"], "paused", args.timeout_s)
        print("[four-version] source paused", flush=True)

        if args.mode != "no_recovery":
            send(source["conn"], {"op": "metadata", "request_id": request_id})
            metadata = wait_event(source["conn"], "metadata", args.timeout_s)["result"]
            computed_tokens = int(
                metadata.get("computed_tokens", len(metadata.get("tokens", prompt)))
                or 0
            )
            source_computed = list(metadata.get("tokens", prompt))[:computed_tokens]
            print(
                f"[four-version] source metadata computed={computed_tokens}",
                flush=True,
            )
        active_slots = follow_trace_to_preemption()
        print(f"[four-version] source preempted active={sorted(active_slots)}", flush=True)
        preempt_started = time.monotonic()

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
            print("[four-version] source stopped; rerouting", flush=True)
            target = workers["reroute_replica"]
            target_result = target_generate(target, request_id, source_computed)
            recovery = {
                "request_outcome": "continued",
                "target_continued": target_result["continued"],
                "recomputed_tokens": len(source_computed),
                "restored_blocks": 0,
                "target_preexisting": True,
                "engine_created": False,
                "placement_changed": False,
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": tensor_parallel_size,
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
                workers["reparallelized_target"], request_id, source_computed
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
            target = workers["reroute_replica"]
            print("[four-version] exporting source state", flush=True)
            send(source["conn"], {"op": "export", "request_id": request_id})
            exported = wait_event(source["conn"], "export", args.timeout_s)["result"]
            if not exported.get("supports_restore"):
                raise AssertionError(f"source export failed: {exported}")
            send(source["conn"], {"op": "abort", "request_id": request_id})
            wait_event(source["conn"], "aborted", args.timeout_s)
            send(
                target["conn"],
                {"op": "restore", "request_id": request_id, "state": exported},
            )
            staged = wait_event(target["conn"], "restore", args.timeout_s)["result"]
            if not staged.get("staged"):
                raise AssertionError(f"target restore failed: {staged}")
            target_result = target_generate(target, request_id, source_computed)
            stop("source")
            recovery = {
                "request_outcome": "continued",
                "target_continued": target_result["continued"],
                "recomputed_tokens": 0,
                "restored_blocks": int(staged.get("expected_blocks", 0) or 0),
                "target_preexisting": True,
                "engine_created": False,
                "placement_changed": False,
                "old_tensor_parallel_size": tensor_parallel_size,
                "new_tensor_parallel_size": tensor_parallel_size,
                "restore_success": True,
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "continued"

        source_blocks = len(metadata.get("block_ids", []))
        report = {
            "status": "passed",
            "mode": args.mode,
            "model": args.model,
            "trace": args.trace,
            "prompt_tokens": args.prompt_tokens,
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
