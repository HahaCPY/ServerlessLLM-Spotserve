"""EP feasibility plus existing cold recovery timings. No new optimization."""

import argparse
import json
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time
import traceback
import uuid

from scripts.moe_gpu_runtime import GPUPool, write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid, digest
from scripts.run_moe_tp_microbenchmark import snapshot


MAJOR_WORKER_SPANS = ("worker_constructor", "worker_init_device", "worker_load_model",
                      "engine_memory_profiling", "kv_cache_initialization", "engine_kernel_warmup")


def diagnostic_log_events(line):
    marker = "MOE_DIAGNOSTIC_EVENT="
    decoder = json.JSONDecoder()
    rows = []
    for part in line.split(marker)[1:]:
        try:
            value, _ = decoder.raw_decode(part.lstrip())
        except json.JSONDecodeError:
            continue  # Retain the raw log; telemetry must not abort inference.
        rows.append(value)
    return rows


def validate_placement(report, expected_uuids):
    if report["status"] != "passed" or len(report["placement"]) != 2:
        raise ValueError("two successfully inspected ranks are required")
    ranks = report["placement"]
    if {canonical_gpu_uuid(r["actual_gpu_uuid"]) for r in ranks} != set(expected_uuids):
        raise ValueError("actual GPU identities differ")
    for rank in ranks:
        if rank["model_class"] != "GraniteMoeForCausalLM" or len(rank["expert_modules"]) != 32:
            raise ValueError("actual Granite expert modules missing")
        for module in rank["expert_modules"]:
            if not module["checkpoint_samples_match"] or not module["weight_device"].startswith("cuda:"):
                raise ValueError("GPU expert weights fail checkpoint verification")
            ep = report["ep_flag"]
            if (module["parallel"]["use_ep"] is not ep
                    or module["parallel"]["ep_size"] != (2 if ep else 1)
                    or module["parallel"]["tp_size"] != (1 if ep else 2)
                    or module["w13_shape"][0] != (20 if ep else 40)
                    or module["intermediate_partition"] != (512 if ep else 256)):
                raise ValueError("flag did not produce expected real expert sharding")
    if report["ep_flag"]:
        by_layer = [{m["layer"]: m for m in r["expert_modules"]} for r in ranks]
        for layer in range(32):
            a, b = (set(group[layer]["global_expert_ids"]) for group in by_layer)
            if a & b or a | b != set(range(40)):
                raise ValueError("EP ownership is not a disjoint complete partition")
    return True


def span_durations(events):
    result = {}
    for row in events:
        if row["event"] == "span_end":
            result[row["name"]] = result.get(row["name"], 0.0) + row["duration_s"]
    return result


def cold_breakdown(evidence, diagnostics, frontend_events, host_started, warm_start, warm_end):
    spans = span_durations(diagnostics["startup"]["events"])
    starts = [e for e in frontend_events if e["event"] == "span_start"
              and e["name"] == "recovery_engine_constructor"]
    ends = [e for e in frontend_events if e["event"] == "span_end"
            and e["name"] == "recovery_engine_constructor"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("missing actual engine constructor timestamps")
    frontend_pid = starts[0].get("pid")
    frontend = {e["event"]: e["monotonic_s"] for e in frontend_events
                if e.get("pid") == frontend_pid and
                e["event"] in {"recovery_frontend_entry", "recovery_control_connected"}}
    engine_start, engine_end = starts[0]["monotonic_s"], ends[0]["monotonic_s"]
    major = {name: spans[name] for name in MAJOR_WORKER_SPANS}
    major["container_python_startup"] = frontend["recovery_frontend_entry"] - host_started
    major["frontend_imports_and_configuration"] = engine_start - frontend["recovery_frontend_entry"]
    residual = engine_end - engine_start - sum(major[name] for name in MAJOR_WORKER_SPANS)
    if residual < -0.01:
        raise ValueError("overlapping spans cannot be added as serial cold stages")
    major["engine_bootstrap_and_other"] = max(0.0, residual)
    major["control_connection_and_runtime_verification"] = warm_start - engine_end
    major["application_warmup_4352_input_64_output"] = warm_end - warm_start
    total = warm_end - host_started
    if any(value < 0 for value in major.values()):
        raise ValueError("cross-process timestamp mismatch produced negative stages")
    if abs(sum(major.values()) - total) > 0.02:
        raise ValueError("cold stage accounting does not match total")
    return {"total_s": total, "major_nonoverlapping_stages_s": major,
            "nested_worker_spans_s": spans,
            "weight_loading_share_percent": 100 * spans["checkpoint_and_weight_loading"] / total,
            "entire_model_loading_upper_bound_percent": 100 * spans["worker_load_model"] / total,
            "startup_cpu_totals": diagnostics["startup"]["cpu_totals"],
            "existing_worker_evidence": evidence,
            "clock_provenance": {"frontend_pid": frontend_pid,
                "host_started": host_started, "frontend_entry": frontend["recovery_frontend_entry"],
                "engine_start": engine_start, "engine_end": engine_end,
                "warm_start": warm_start, "warm_end": warm_end},
            "note": "nested spans/iterator/callback CPU times are not independent physical disk or PCIe times"}


def probe(args, out, label, tp, dp, ep, gpu_uuids):
    before = snapshot()
    group = [2, 3]
    expected = [gpu_uuids[g] for g in group]
    if any(canonical_gpu_uuid(r.split(",")[0].strip()) in expected for r in before["processes"]):
        raise RuntimeError("occupied test GPU; other process will not be stopped")
    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    name = "spotserve-arch-" + uuid.uuid4().hex[:16]
    command = ["podman", "run", "--rm", "--name", name, "--network", "none", "--shm-size", "1g"]
    for gpu in group:
        command += ["--device", f"nvidia.com/gpu={gpu}"]
    for source, target, mode in ((repo, repo, "ro"), (vllm, vllm, "ro"),
                                 (args.model, args.model, "ro"),
                                 ("/usr/local/cuda-13.0", "/usr/local/cuda", "ro"),
                                 (args.compiler_cache, "/taskcache", "rw")):
        command += ["--volume", f"{source}:{target}:{mode}"]
    env = {"PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "0", "NCCL_SOCKET_IFNAME": "lo",
           "TRITON_CACHE_DIR": "/taskcache/triton", "CUDA_CACHE_PATH": "/taskcache/cuda",
           "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET",
           "MOE_DIAG_CPROFILE": "1" if not ep else "0"}
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [args.image, str(vllm / ".venv/bin/python"), "-u", "-m", "scripts.probe_granite_ep",
                "--model", args.model, "--tp", str(tp), "--dp", str(dp)]
    if ep:
        command.append("--ep")
    result = {"label": label, "command": command, "environment_before": before, "status": "running"}
    process = None
    messages = queue.Queue()
    try:
        with (out / (label + ".log")).open("x") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            def drain():
                for line in process.stdout:
                    messages.put(line)
                messages.put(None)
            thread = threading.Thread(target=drain, daemon=True)
            thread.start()
            deadline = time.monotonic() + args.timeout_s
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("EP diagnostic timeout")
                try:
                    line = messages.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if line.startswith("MOE_EP_PROBE_JSON="):
                    result["report"] = json.loads(line.split("=", 1)[1])
                elif "MOE_DIAGNOSTIC_EVENT=" in line:
                    for event in diagnostic_log_events(line):
                        if event["event"] == "ep_cell_complete" or (
                                event["event"] == "span_end" and event["name"] in MAJOR_WORKER_SPANS):
                            print("MOE_ARCH_PROGRESS=" + json.dumps({"label": label, **event}), flush=True)
            result["exit_code"] = process.wait(timeout=20)
        if result["exit_code"] or "report" not in result:
            raise RuntimeError("EP probe did not complete successfully")
        validate_placement(result["report"], expected)
        result["placement_verified"] = True
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(["podman", "stop", "--time", "5", name], capture_output=True, timeout=30)
            process.wait(timeout=20)
        result["environment_after"] = snapshot()
        write_artifact(out / (label + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compiler-cache", required=True)
    parser.add_argument("--image", default="localhost/spotserve-python312-nixl:latest")
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--cold-repeats", type=int, default=3)
    parser.add_argument("--phase", choices=("all", "ep", "cold"), default="all")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    Path(args.compiler_cache).mkdir(parents=True, exist_ok=True)
    initial = snapshot()
    uuids = {int(r.split(",")[0]): canonical_gpu_uuid(r.split(",")[1].strip()) for r in initial["gpus"]}
    if set(uuids) != {0, 1, 2, 3} or initial["processes"]:
        raise ValueError("four available GPUs required; existing jobs will not be stopped")
    data = json.loads(Path(args.calibration).read_text())
    protocol = {"revision": args.model_revision, "model": args.model, "phase": args.phase,
                "ep_groups": [2, 3], "cold_group": [1], "cold_repeats": args.cold_repeats,
                "calibration_sha256": digest(data), "no_new_optimization": True,
                "no_formal_comparison": True, "no_vllm_source_edits": True,
                "cold_instrumentation": "timestamp only", "ep_control_cpu_profiler": True}
    write_artifact(out / "protocol.json", protocol)
    write_artifact(out / "environment_initial.json", initial)
    result = {"status": "running", "protocol": protocol, "ep_probes": [], "cold_runs": []}
    try:
        probes = (() if args.phase == "cold" else
                  (("tp2-control", 2, 1, False), ("tp2-ep2", 2, 1, True), ("dp2-ep2", 1, 2, True)))
        for label, tp, dp, ep in probes:
            tested = probe(args, out, label, tp, dp, ep, uuids)
            result["ep_probes"].append(tested)
            if tested["status"] != "passed":
                print("MOE_ARCH_PROGRESS=" + json.dumps({"label": label, "status": "failed",
                      "later_ep_probes_stopped": True}), flush=True)
                break
        # Independently requested cold breakdown still runs if EP is unsupported.
        if args.phase == "ep":
            result["status"] = "passed" if all(p["status"] == "passed" for p in result["ep_probes"]) else "failed"
            return 0 if result["status"] == "passed" else 1
        pool = GPUPool(out / "cold", args.model, args.model_revision, uuids,
                       timeout=args.timeout_s, image=args.image, compiler_cache=args.compiler_cache)
        try:
            capture = data["candidates"]["gpu1-tp1"]["runtime"]["routing_capture"]
            warm_tokens = data["source_cases"]["train-code"]["frozen"]["tokens"]
            if len(warm_tokens) != 4352:
                raise ValueError("historical recovery warmup workload differs")
            for repeat in range(args.cold_repeats):
                if any(canonical_gpu_uuid(r.split(",")[0].strip()) == uuids[1]
                       for r in snapshot()["processes"]):
                    raise RuntimeError("cold test GPU occupied; existing job not stopped")
                worker = pool.launch(f"cold-r{repeat}", [1], capture=capture, startup_profiling=True)
                warm_start = time.monotonic()
                worker.warm(warm_tokens)
                warm_end = time.monotonic()
                diagnostic = worker.inspection["startup_diagnostics"][0]
                frontend_events = [json.loads(line.split("=", 1)[1]) for line in worker.log_lines
                                   if line.startswith("MOE_DIAGNOSTIC_EVENT=")]
                worker.close()
                cold = cold_breakdown(worker.evidence(), diagnostic, frontend_events,
                                      worker.started, warm_start, warm_end)
                cold["diagnostics"] = diagnostic
                cold["warmup_sample"] = worker.records[-1]
                result["cold_runs"].append(cold)
                write_artifact(out / f"cold-r{repeat}.json", cold)
                print("MOE_ARCH_PROGRESS=" + json.dumps({"label": f"cold-r{repeat}",
                      "status": "passed", "total_s": cold["total_s"],
                      "weight_loading_share_percent": cold["weight_loading_share_percent"]}), flush=True)
        finally:
            pool.close()
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
    finally:
        result["environment_final"] = snapshot()
        write_artifact(out / "architecture_diagnostics.json", result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
