"""Compare four explicitly separated preemption recovery policies.

The four modes are:

* ``no_recovery``: the source is preempted and the request is expected to fail;
* ``rerouting``: an already-ready replica receives the full context, without
  changing its placement or creating an engine;
* ``reparallelization``: the source is stopped, the spare worker is stopped,
  and a new TP2 engine is created on the freed source GPU plus the spare GPU;
* ``modified``: the SpotServe NIXL export/restore path attaches source KV state
  to the already-ready target replica.

This is a same-host, separate-container experiment.  The workload is real
vLLM generation and the modified path uses the real NIXL connector; the first
three modes deliberately do not call export or restore.
"""

import argparse
import json
import os
import shutil
import shlex
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
from run_four_container_fleet_churn_smoke import load_fleet_trace
from run_four_container_fleet_churn_smoke import trace_slot


MODES = ("no_recovery", "rerouting", "reparallelization", "modified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_ROOT}/Qwen2-MoE-Tiny")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 3])
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


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 3 or len(set(args.gpus)) != 3:
        raise SystemExit("--gpus must contain three distinct GPU indices")
    if args.prompt_tokens < 1 or args.prompt_tokens > args.max_model_len:
        raise SystemExit("--prompt-tokens must fit within --max-model-len")
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        raise SystemExit(f"model config not found: {args.model}")
    trace_events = load_fleet_trace(args.trace)
    if not any(
        event["event"] == "remove"
        and any(trace_slot(node) == 0 for node in event["nodes"])
        for event in trace_events
    ):
        raise SystemExit("trace must remove node-0 to trigger source preemption")

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
            # Baselines never invoke export or restore. Keeping the connector
            # initialized lets metadata expose the same runtime context while
            # the policy itself remains strictly no-transfer/reroute/rebuild.
            "--kv-transfer-mode",
            "nixl",
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
        send(
            target["conn"],
            {"op": "generate", "request_id": request_id, "token_ids": token_ids},
        )
        wait_event(target["conn"], "generate_started", args.timeout_s)
        first = wait_event(target["conn"], "output", args.timeout_s)
        wait_event(target["conn"], "paused", args.timeout_s)
        recovery_s = round(time.monotonic() - phase_started, 3)
        send(target["conn"], {"op": "resume", "request_id": request_id})
        wait_event(target["conn"], "resumed", args.timeout_s)
        continued_started = time.monotonic()
        continued = wait_event(target["conn"], "output", args.timeout_s)
        continuation_s = round(time.monotonic() - continued_started, 3)
        if not continued.get("token_ids"):
            raise AssertionError("target did not continue after source preemption")
        return {
            "first_output_tokens": len(first.get("token_ids", [])),
            "target_recovery_s": recovery_s,
            "target_continuation_s": continuation_s,
            "continued": True,
        }

    source_spec = {
        "label": "source",
        "node_id": "four-version-source",
        "role": "source",
        "gpus": [args.gpus[0]],
        "tp": 1,
        "port": 5600,
    }
    reroute_spec = {
        "label": "reroute_replica",
        "node_id": "four-version-reroute-replica",
        "role": "observer",
        "gpus": [args.gpus[1]],
        "tp": 1,
        "port": 5700,
    }
    spare_spec = {
        "label": "spare",
        "node_id": "four-version-spare",
        "role": "observer",
        "gpus": [args.gpus[2]],
        "tp": 1,
        "port": 5800,
    }
    request_id = f"four-version-request-{os.getpid()}"
    prompt = [100 + index for index in range(args.prompt_tokens)]
    metadata: dict = {}
    source_computed: list[int] = []
    exported: dict | None = None
    outcome = "failed"
    recovery: dict = {}

    def follow_trace_to_preemption() -> None:
        """Apply add/remove churn until the trace removes the source.

        The four policy branches need the reroute target to remain READY, but
        they can tolerate the spare being removed and re-added before the
        recovery point.  This keeps the trace's capacity changes real without
        changing the semantics of the four policies themselves.
        """
        label_by_gpu = {
            args.gpus[0]: "source",
            args.gpus[1]: "reroute_replica",
            args.gpus[2]: "spare",
        }
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
            for node in event["nodes"]:
                gpu = trace_slot(node)
                if gpu not in label_by_gpu:
                    raise AssertionError(
                        f"trace GPU {gpu} is not present in --gpus: {event}"
                    )
                label = label_by_gpu[gpu]
                if action == "add":
                    if label in workers:
                        continue
                    if label != "spare":
                        raise AssertionError(
                            f"trace cannot add inactive policy worker {label}"
                        )
                    launch(spare_spec)
                    register([spare_spec])
                elif action == "remove":
                    if label == "source":
                        return
                    if label == "reroute_replica":
                        raise AssertionError(
                            "trace cannot remove the pre-existing recovery target"
                        )
                    if label in workers:
                        stop(label)
                else:
                    raise AssertionError(
                        f"four-version trace only supports add/remove/DONE: {event}"
                    )
        raise AssertionError("trace must remove source node-0")

    try:
        run_podman(["network", "create", network])
        launch(source_spec)
        launch(reroute_spec)
        launch(spare_spec)
        register([source_spec, reroute_spec, spare_spec])
        source = workers["source"]
        send(
            source["conn"],
            {"op": "generate", "request_id": request_id, "token_ids": prompt},
        )
        wait_event(source["conn"], "generate_started", args.timeout_s)
        wait_event(source["conn"], "paused", args.timeout_s)

        if args.mode != "no_recovery":
            send(source["conn"], {"op": "metadata", "request_id": request_id})
            metadata = wait_event(source["conn"], "metadata", args.timeout_s)["result"]
            computed_tokens = int(
                metadata.get("computed_tokens", len(metadata.get("tokens", prompt)))
                or 0
            )
            source_computed = list(metadata.get("tokens", prompt))[:computed_tokens]
        follow_trace_to_preemption()
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
                "old_tensor_parallel_size": 1,
                "new_tensor_parallel_size": None,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "failed"
        elif args.mode == "rerouting":
            stop("source")
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
                "old_tensor_parallel_size": 1,
                "new_tensor_parallel_size": 1,
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "continued"
        elif args.mode == "reparallelization":
            stop("source")
            stop("spare")
            new_target_spec = {
                "label": "reparallelized_target",
                "node_id": "four-version-reparallelized-target",
                "role": "observer",
                "gpus": [args.gpus[0], args.gpus[2]],
                "tp": 2,
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
                "old_tensor_parallel_size": 1,
                "new_tensor_parallel_size": 2,
                "new_target_gpus": [args.gpus[0], args.gpus[2]],
                **target_result,
                "recovery_s": round(time.monotonic() - preempt_started, 3),
            }
            outcome = "continued"
        else:
            target = workers["reroute_replica"]
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
                "old_tensor_parallel_size": 1,
                "new_tensor_parallel_size": 1,
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
            "source_config": {"gpus": [args.gpus[0]], "tensor_parallel_size": 1},
            "outcome": outcome,
            "expected_outcome": "failed" if args.mode == "no_recovery" else "continued",
            "recovery": recovery,
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
