"""Exercise random worker additions/preemptions with at most four GPU nodes.

The test starts three workers, leaving one GPU slot free.  A seeded controller
then emits random ``add`` and ``preempt`` events.  A preempted worker's GPU
slot may be reused by a newly created container, so the number of live
containers never exceeds four.  At least one event is forced to preempt the
active source request; that event performs a real NIXL KV handoff before the
source receives SIGTERM and the target continues decoding.

This is a same-host, multi-container simulation.  It validates fleet churn and
the preemption grace-period ordering, not physical cross-host isolation.
"""

import argparse
import json
import os
import random
import shlex
import shutil
import socket
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--gpu-groups",
        nargs="+",
        help=(
            "Optional per-container GPU groups, e.g. 0,1 2,3. Each group is "
            "one container and must match --tensor-parallel-size."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--recovery-mode",
        choices=("nixl", "recompute"),
        default="nixl",
        help=(
            "How to continue the active request after source removal: "
            "NIXL KV restore or full-context recompute."
        ),
    )
    parser.add_argument(
        "--trace",
        help=(
            "Optional JSONL fleet trace. Events use time_ms, event=(add|remove|"
            "preempt|DONE), and nodes such as slot-0."
        ),
    )
    parser.add_argument(
        "--trace-speedup",
        type=float,
        default=1000.0,
        help="Divide trace delays by this factor (default: 1000).",
    )
    parser.add_argument("--source-port", type=int, default=5600)
    parser.add_argument("--base-port", type=int, default=5700)
    parser.add_argument("--token-delay-s", type=float, default=0.10)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=64,
        help="Number of source context tokens used for the migration request.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM maximum sequence length for long-context tests.",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="Pass vLLM CPU weight offload to each container.",
    )
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--image", default=IMAGE)
    return parser.parse_args()


def load_fleet_trace(path: str) -> list[dict]:
    """Load the small add/remove trace format used by this container smoke.

    The production SpotServe trace reader intentionally accepts only spot
    lifecycle events.  This runner additionally models capacity churn, so it
    keeps the add/remove/DONE format separate and explicit.
    """
    events: list[dict] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        action = str(event.get("event", ""))
        if action not in {"add", "remove", "preempt", "DONE"}:
            raise ValueError(
                f"{path}:{line_number}: unsupported fleet event {action!r}"
            )
        time_ms = float(event.get("time_ms", event.get("time", 0.0)))
        if time_ms < 0:
            raise ValueError(f"{path}:{line_number}: negative time_ms")
        nodes = event.get("nodes", [])
        if not isinstance(nodes, list):
            raise ValueError(f"{path}:{line_number}: nodes must be a list")
        events.append({"time_ms": time_ms, "event": action, "nodes": nodes})
    return sorted(events, key=lambda event: event["time_ms"])


def trace_slot(node: object) -> int:
    """Convert slot-2/node-2/2 trace names to a numeric GPU slot."""
    text = str(node)
    if text.startswith("slot-"):
        text = text.removeprefix("slot-")
    elif text.startswith("node-"):
        text = text.removeprefix("node-")
    return int(text)


def main() -> None:
    args = parse_args()
    if args.tensor_parallel_size < 1:
        raise SystemExit("--tensor-parallel-size must be positive")
    if args.gpu_groups:
        try:
            gpu_groups = [
                [int(gpu.strip()) for gpu in group.split(",") if gpu.strip()]
                for group in args.gpu_groups
            ]
        except ValueError as exc:
            raise SystemExit("--gpu-groups must use comma-separated integers") from exc
        if not gpu_groups or any(
            len(group) != args.tensor_parallel_size for group in gpu_groups
        ):
            raise SystemExit(
                "each --gpu-groups entry must contain exactly "
                "--tensor-parallel-size GPUs"
            )
        flattened_gpus = [gpu for group in gpu_groups for gpu in group]
        if len(set(flattened_gpus)) != len(flattened_gpus):
            raise SystemExit("--gpu-groups must not reuse a GPU")
    else:
        if not 2 <= len(args.gpus) <= 4 or len(set(args.gpus)) != len(args.gpus):
            raise SystemExit("--gpus must contain two to four distinct GPU indices")
        if args.tensor_parallel_size != 1:
            raise SystemExit("--tensor-parallel-size > 1 requires --gpu-groups")
        gpu_groups = [[gpu] for gpu in args.gpus]
    if len(gpu_groups) > 4:
        raise SystemExit("at most four container GPU groups are supported")
    if args.trace_speedup <= 0:
        raise SystemExit("--trace-speedup must be positive")
    if args.prompt_tokens < 1:
        raise SystemExit("--prompt-tokens must be positive")
    if args.max_model_len is not None and args.max_model_len < args.prompt_tokens:
        raise SystemExit("--max-model-len must cover --prompt-tokens")
    trace_events = load_fleet_trace(args.trace) if args.trace else None
    if args.events < 2 and trace_events is None:
        raise SystemExit("--events must be at least 2")
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        raise SystemExit(f"model config not found: {args.model}")

    rng = random.Random(args.seed)
    control_dir = os.path.abspath(
        os.path.join("/tmp", f"spotserve-four-fleet-{os.getpid()}")
    )
    os.makedirs(control_dir, mode=0o777, exist_ok=False)
    os.chmod(control_dir, 0o777)
    socket_path = os.path.join(control_dir, "control.sock")
    listener = Listener(socket_path, family="AF_UNIX", authkey=b"spotserve")
    os.chmod(socket_path, 0o666)
    network = f"spotserve-four-fleet-net-{os.getpid()}"
    started = time.monotonic()

    workers: dict[int, dict] = {}
    slot_count = len(gpu_groups)
    generation: dict[int, int] = {slot: 0 for slot in range(slot_count)}
    connections: dict[str, object] = {}
    pending_ready: dict[str, tuple[dict, object]] = {}
    event_log: list[dict] = []
    source_slot = 0
    source_request_id = "four-container-fleet-request"
    source_migrated = False
    source_started = False
    migration_summary: dict = {}
    pinned_target_slot = 1

    common = [
        "run",
        "--detach",
        "--network",
        network,
        "--volume",
        f"{REPO_ROOT}:{REPO_ROOT}:ro",
        "--volume",
        f"{VLLM_ROOT}:{VLLM_ROOT}:ro",
        "--volume",
        f"{MODEL_ROOT}:{MODEL_ROOT}:ro",
        "--volume",
        "/usr/local/cuda-13.0:/usr/local/cuda:ro",
        "--volume",
        f"{control_dir}:/control:rw",
        "--env",
        f"PYTHONPATH={PYTHONPATH}",
        "--env",
        "PYTHONUNBUFFERED=1",
        args.image,
    ]
    worker_script = (
        f"{REPO_ROOT}/tests/spotserve_test/cross_container_nixl_worker.py"
    )

    def worker_name(slot: int) -> str:
        return f"spotserve-fleet-s{slot}-g{generation[slot]}-{os.getpid()}"

    def node_id(slot: int) -> str:
        return f"fleet-slot-{slot}-gen-{generation[slot]}"

    def command(slot: int, role: str) -> list[str]:
        host = node_id(slot)
        port = args.source_port if slot == source_slot else args.base_port + slot
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
            host,
            "--side-channel-host",
            host,
            "--side-channel-port",
            str(port),
            "--token-delay-s",
            str(args.token_delay_s),
            "--cpu-offload-gb",
            str(args.cpu_offload_gb),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-model-len",
            str(
                args.max_model_len
                if args.max_model_len is not None
                else max(args.prompt_tokens + 256, 256)
            ),
        ]
        device_args: list[str] = []
        for gpu in gpu_groups[slot]:
            device_args.extend(["--device", f"nvidia.com/gpu={gpu}"])
        return [
            *common[:1],
            "--name",
            worker_name(slot),
            "--hostname",
            host,
            *device_args,
            *common[1:],
            "bash",
            "-lc",
            "exec " + shlex.join(role_args),
        ]

    def accept_worker(slot: int) -> dict:
        expected_node_id = node_id(slot)
        cached = pending_ready.pop(expected_node_id, None)
        if cached is not None:
            ready, conn = cached
            ready["conn"] = conn
            return ready
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            conn = listener.accept()
            ready = wait_event(conn, "ready", args.timeout_s)
            if ready.get("node_id") == expected_node_id:
                ready["conn"] = conn
                return ready
            # No other worker should connect during a single add operation,
            # but retain it so an initial batch can register out of order.
            pending_ready[ready["node_id"]] = (ready, conn)
        raise TimeoutError(f"timed out waiting for {expected_node_id}")

    def launch_worker(slot: int, role: str) -> None:
        run_podman(command(slot, role))

    def register_worker(slot: int, role: str) -> None:
        ready = accept_worker(slot)
        connections[ready["node_id"]] = ready["conn"]
        workers[slot] = {
            "slot": slot,
            "role": role,
            "node_id": ready["node_id"],
            "name": worker_name(slot),
            "gpus": gpu_groups[slot],
            "conn": ready["conn"],
        }

    def start_worker(slot: int, role: str) -> None:
        launch_worker(slot, role)
        register_worker(slot, role)

    def stop_worker(slot: int, reason: str) -> None:
        worker = workers.pop(slot)
        conn = worker["conn"]
        connections.pop(worker["node_id"], None)
        result = run_podman(
            ["kill", "--signal", "TERM", worker["name"]], check=False
        )
        stopped = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = run_podman(
                ["inspect", "--format", "{{.State.Running}}", worker["name"]],
                check=False,
            )
            if state.stdout.strip() == "false" or state.returncode != 0:
                stopped = True
                break
            time.sleep(0.2)
        if not stopped:
            run_podman(["rm", "--force", worker["name"]], check=False)
            stopped = True
        try:
            conn.close()
        except OSError:
            pass
        event_log.append(
            {
                "action": "preempt",
                "slot": slot,
                "node_id": worker["node_id"],
                "reason": reason,
                "sigterm_sent": result.returncode == 0,
                "stopped": stopped,
            }
        )

    def add_worker(slot: int, role: str, reason: str) -> None:
        generation[slot] += 1
        start_worker(slot, role)
        event_log.append(
            {
                "action": "add",
                "slot": slot,
                "node_id": workers[slot]["node_id"],
                "gpus": gpu_groups[slot],
                "reason": reason,
            }
        )

    def migrate_source() -> None:
        nonlocal source_migrated, migration_summary
        source = workers[source_slot]
        target = workers[pinned_target_slot]
        if target["role"] != "observer":
            raise AssertionError("pinned migration target is not available")
        prompt = [100 + index for index in range(args.prompt_tokens)]
        preempt_notice_at = time.monotonic()
        phase_times: dict[str, float] = {}
        phase_started = preempt_notice_at
        send(source["conn"], {
            "op": "generate",
            "request_id": source_request_id,
            "token_ids": prompt,
        })
        wait_event(source["conn"], "generate_started", args.timeout_s)
        wait_event(source["conn"], "paused", args.timeout_s)
        send(source["conn"], {"op": "metadata", "request_id": source_request_id})
        metadata = wait_event(source["conn"], "metadata", args.timeout_s)["result"]
        if not metadata.get("block_ids"):
            raise AssertionError(f"source has no KV blocks: {metadata}")
        completed_tokens = int(metadata.get("completed_tokens", 0) or 0)
        computed_tokens = int(
            metadata.get("computed_tokens", len(metadata["tokens"])) or 0
        )
        computed = metadata["tokens"][:computed_tokens]
        phase_times["source_prepare_s"] = round(
            time.monotonic() - phase_started, 3
        )
        exported = None
        if args.recovery_mode == "nixl":
            phase_started = time.monotonic()
            send(source["conn"], {"op": "export", "request_id": source_request_id})
            exported = wait_event(source["conn"], "export", args.timeout_s)["result"]
            if not exported.get("supports_restore"):
                raise AssertionError(f"source export failed: {exported}")
            phase_times["source_export_s"] = round(
                time.monotonic() - phase_started, 3
            )
        else:
            phase_times["source_export_s"] = 0.0
        phase_started = time.monotonic()
        send(source["conn"], {"op": "abort", "request_id": source_request_id})
        wait_event(source["conn"], "aborted", args.timeout_s)
        phase_times["source_abort_s"] = round(time.monotonic() - phase_started, 3)
        staged = {"expected_blocks": 0, "staged": False}
        phase_started = time.monotonic()
        if args.recovery_mode == "nixl":
            send(target["conn"], {
                "op": "restore",
                "request_id": source_request_id,
                "state": exported,
            })
            staged = wait_event(target["conn"], "restore", args.timeout_s)["result"]
            if not staged.get("staged"):
                raise AssertionError(f"target restore was not staged: {staged}")
        # In recompute mode this is deliberately the complete source context,
        # with no target-side KV attach.  It is the baseline comparison path.
        send(target["conn"], {
            "op": "generate",
            "request_id": source_request_id,
            "token_ids": computed,
        })
        wait_event(target["conn"], "generate_started", args.timeout_s)
        wait_event(target["conn"], "output", args.timeout_s)
        wait_event(target["conn"], "paused", args.timeout_s)
        transfer_completed = time.monotonic()
        phase_times["target_recovery_s"] = round(
            transfer_completed - phase_started, 3
        )
        stop_worker(source_slot, "active_source_preemption")
        phase_started = time.monotonic()
        continued = wait_event(target["conn"], "output", args.timeout_s)
        if not continued.get("token_ids"):
            raise AssertionError("target did not continue after source stop")
        phase_times["target_continuation_s"] = round(
            time.monotonic() - phase_started, 3
        )
        source_migrated = True
        migration_summary = {
            "recovery_mode": args.recovery_mode,
            "source_blocks": len(metadata["block_ids"]),
            "source_completed_tokens": completed_tokens,
            "source_computed_tokens": computed_tokens,
            "restored_blocks": int(staged.get("expected_blocks", 0) or 0),
            "recomputed_tokens": computed_tokens
            if args.recovery_mode == "recompute"
            else 0,
            "target_continued_after_source_stop": True,
            "migration_elapsed_s": round(time.monotonic() - preempt_notice_at, 3),
            "transfer_grace_s": round(
                transfer_completed - preempt_notice_at, 3
            ),
            "phase_elapsed_s": phase_times,
        }
        event_log.append(
            {
                "action": "migrate_active_request",
                "recovery_mode": args.recovery_mode,
                "source_slot": source_slot,
                "target_slot": pinned_target_slot,
                "source_blocks": metadata["block_ids"],
                "staged_blocks": staged["expected_blocks"],
                "source_completed_tokens": completed_tokens,
                "source_computed_tokens": computed_tokens,
                "recomputed_tokens": migration_summary["recomputed_tokens"],
                "target_continued_after_source_stop": True,
                "migration_elapsed_s": migration_summary["migration_elapsed_s"],
                "transfer_grace_s": round(
                    transfer_completed - preempt_notice_at, 3
                ),
            }
        )

    try:
        listener._listener._socket.settimeout(args.timeout_s)
        run_podman(["network", "create", network])
        # Start up to three workers and leave one GPU slot available when the
        # host supplied four GPUs.  With fewer GPUs, the trace still exercises
        # remove/add by reusing a freed slot.
        # Launch the initial engines together; each one has a lengthy CUDA
        # startup, so sequential startup would unnecessarily multiply time.
        initial_slots = list(range(min(3, slot_count)))
        for slot in initial_slots:
            launch_worker(slot, "source" if slot == source_slot else "observer")
        for slot in initial_slots:
            register_worker(slot, "source" if slot == source_slot else "observer")
        source_started = True

        if trace_events is not None:
            previous_time_ms = 0.0
            for event_index, trace_event in enumerate(trace_events):
                delay_s = (
                    max(trace_event["time_ms"] - previous_time_ms, 0.0)
                    / 1000.0
                    / args.trace_speedup
                )
                if delay_s:
                    time.sleep(delay_s)
                previous_time_ms = trace_event["time_ms"]
                action = trace_event["event"]
                if action == "DONE":
                    event_log.append(
                        {
                            "action": "DONE",
                            "trace_time_ms": trace_event["time_ms"],
                            "live_container_count": len(workers),
                        }
                    )
                    break
                slots = [trace_slot(node) for node in trace_event["nodes"]]
                if any(slot < 0 or slot >= slot_count for slot in slots):
                    raise AssertionError(
                        f"trace slot outside --gpus: {trace_event}"
                    )
                for slot in slots:
                    if action == "add":
                        if slot in workers:
                            event_log.append(
                                {
                                    "action": "add_observed",
                                    "slot": slot,
                                    "trace_time_ms": trace_event["time_ms"],
                                    "live_container_count": len(workers),
                                }
                            )
                            continue
                        role = "source" if slot == source_slot else "observer"
                        add_worker(slot, role, "trace_add")
                    elif action == "remove":
                        if slot not in workers:
                            raise AssertionError(
                                f"trace remove targets inactive slot {slot}"
                            )
                        if slot == pinned_target_slot:
                            raise AssertionError(
                                "trace cannot remove the live migration target"
                            )
                        if workers[slot]["role"] == "source":
                            if not source_migrated:
                                # The fleet trace intentionally has only
                                # add/remove/DONE states.  The first removal
                                # of the active source is therefore the spot
                                # preemption boundary and performs the real
                                # export/restore handoff.
                                migrate_source()
                            else:
                                stop_worker(slot, "trace_remove_source")
                        else:
                            stop_worker(slot, "trace_remove")
                    elif action == "preempt":
                        if slot != source_slot or slot not in workers:
                            raise AssertionError(
                                "trace preempt must target the active source slot"
                            )
                        migrate_source()
                    event_log[-1]["trace_time_ms"] = trace_event["time_ms"]
                    event_log[-1]["event_index"] = event_index
                    event_log[-1]["live_container_count"] = len(workers)
        else:
            for event_index in range(args.events):
                live_slots = sorted(workers)
                free_slots = [slot for slot in range(slot_count) if slot not in workers]
                if not source_migrated and event_index == args.events - 1:
                    action = "preempt_source"
                elif free_slots and rng.random() < 0.45:
                    action = "add"
                else:
                    action = "preempt"

                if action == "add":
                    slot = rng.choice(free_slots)
                    add_worker(slot, "observer", "random_capacity_add")
                elif action == "preempt_source":
                    if not source_started:
                        raise AssertionError("source was not started")
                    migrate_source()
                    # Replenish the source slot with a new producer container.
                    add_worker(source_slot, "source", "replacement_after_source_preempt")
                else:
                    candidates = [
                        slot
                        for slot in live_slots
                        if slot != pinned_target_slot and workers[slot]["role"] != "source"
                    ]
                    if not candidates:
                        # Keep the active migration target alive and make progress
                        # with an add if the fleet is temporarily full.
                        if free_slots:
                            add_worker(rng.choice(free_slots), "observer", "capacity_fill")
                            continue
                        raise AssertionError("no safe observer available to preempt")
                    slot = rng.choice(candidates)
                    stop_worker(slot, "random_preemption")

                # Record each event with the live fleet size; this makes the
                # maximum-four-container invariant directly auditable.
                event_log[-1]["event_index"] = event_index
                event_log[-1]["live_container_count"] = len(workers)

        if not source_migrated:
            raise AssertionError("random event sequence did not migrate source")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "simulation": "four_gpu_container_fleet_churn_trace"
                    if trace_events is not None
                    else "four_gpu_container_fleet_churn",
                    "seed": args.seed,
                    "trace": args.trace,
                    "gpu_groups": gpu_groups,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "prompt_tokens": args.prompt_tokens,
                    "max_live_containers": max(
                        [entry.get("live_container_count", 0) for entry in event_log]
                        + [len(initial_slots)]
                    ),
                    "final_live_containers": len(workers),
                    "events": event_log,
                    "recovery_mode": args.recovery_mode,
                    "migration": migration_summary,
                    "state_restore_successes_total": 1
                    if args.recovery_mode == "nixl"
                    else 0,
                    "state_restore_fallback_count": 0,
                    "physical_cross_node": False,
                    "elapsed_s": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            )
        )
    except (TimeoutError, socket.timeout):
        dump_container_logs([worker["name"] for worker in workers.values()])
        raise
    finally:
        for worker in list(workers.values()):
            conn = worker["conn"]
            try:
                send(conn, {"op": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
            run_podman(["rm", "--force", worker["name"]], check=False)
        listener.close()
        run_podman(["network", "rm", network], check=False)
        shutil.rmtree(control_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
