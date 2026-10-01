"""Continuous fixed-arrival workload with two precommitted wall-clock revocations."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import traceback

from scripts.moe_gpu_runtime import GenerationPreempted, GPUPool, write_artifact
from scripts.moe_recovery_actor import execute_recovery
from scripts.run_granite_moe_replay_pilot import digest
from scripts.run_moe_calibrated_experiments import GPU_UUIDS, progress


def trace_from_pilot(pilot):
    maximum = max(row["notice_to_first_token_s"] for row in pilot["runs"])
    second = 5.0 + math.ceil(maximum) + 37.5
    return {"first_revoke_s": 5.0, "first_revoked_gpus": [0],
            "second_revoke_s": second, "second_revoked_gpus": [1, 2, 3],
            "second_return_s": second + 5.0,
            "request_arrivals_s": [float(i) for i in range(12)],
            "first_gpu_never_returns_during_run": True,
            "event_notice_lateness_limit_s": 0.5,
            "timing_rule": "5_then_5_plus_ceil_pilot_max_first_token_plus_37.5",
            "not_adjusted_per_version_or_run": True}


def serializable_live(state):
    return {key: value for key, value in state.items() if key not in {"future", "delivery_executor"}}


def run_stream(pool, data, trace, label, moe_aware):
    row = {"status": "running", "label": label, "moe_aware": moe_aware,
           "requests": [], "preemptions": [], "trace": trace}
    worker = pool.launch(label + "-source", [0], capture=True)
    prompt = data["source_cases"]["formal-serving"]["prompt_token_ids"]
    worker.warm(prompt)
    reference = worker.measure(label + "-reference", prompt, 512)
    epoch = time.monotonic()
    event_index, live = 0, None
    arrivals = trace["request_arrivals_s"]
    timings = [trace["first_revoke_s"], trace["second_revoke_s"]]
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    try:
        for request_index, arrival in enumerate(arrivals):
            while time.monotonic() - epoch < arrival:
                time.sleep(0.01)
            rid = f"{label}-q{request_index}"
            carried, deliveries = [], []
            segment_start = time.monotonic()
            future = executor.submit(worker.measure, rid, prompt, 512)
            while True:
                now = time.monotonic()
                if event_index < 2 and now - epoch >= timings[event_index]:
                    lateness = now - epoch - timings[event_index]
                    if lateness > trace["event_notice_lateness_limit_s"]:
                        raise ValueError("fixed trace event was handled too late")
                    case = worker.snapshot_running(rid, prompt, carried, reference)
                    case["trace_event_index"] = event_index
                    if live is not None:
                        previous = row["preemptions"][-1]
                        previous["interrupted_again_before_request_finished"] = True
                        previous["partial_delivery"] = case["partial_delivery"]
                        partial_events = case["partial_delivery"].get("deliveries", [])
                        if partial_events:
                            first = (case["partial_delivery"]["started_monotonic_s"]
                                     + partial_events[0]["relative_s"])
                            previous["notice_to_first_token_s"] = first - previous["source_case"]["notice_monotonic_s"]
                    try:
                        future.result(timeout=5)
                    except GenerationPreempted:
                        pass
                    if live is not None:
                        live["delivery_executor"].shutdown(wait=True)
                    deliveries.extend({"monotonic_s": case["partial_delivery"]["started_monotonic_s"]
                                       + item["relative_s"], "count": len(carried) + item["count"]}
                                      for item in case["partial_delivery"].get("deliveries", []))
                    if event_index == 1:
                        # All possible targets are unavailable for this fixed
                        # five-second interval, independently of the chosen TP.
                        worker.close()
                        while time.monotonic() - epoch < trace["second_return_s"]:
                            time.sleep(0.01)
                    live, worker = execute_recovery(pool, data, worker, case, "overall", moe_aware,
                                                    f"{label}-p{event_index}", live=True)
                    future, rid = live["future"], live["request_id"]
                    carried = list(case["client_prefix"])
                    event = {"trace_event_index": event_index,
                             "planned_notice_s": timings[event_index], "notice_lateness_s": lateness,
                             "actual_notice_s": case["notice_monotonic_s"] - epoch,
                             "source_case": case, "execution": serializable_live(live),
                             "actual_target": worker.evidence()}
                    row["preemptions"].append(event)
                    progress("wall_clock_preemption", label=label, event_index=event_index,
                             generated_tokens=case["frozen"]["completed_tokens"],
                             selected_candidate=live["decision"]["selected_candidate"])
                    event_index += 1
                if future.done():
                    output = future.result()
                    combined = carried + output["output_token_ids"]
                    if combined != reference["output_token_ids"] or len(combined) != 512:
                        raise ValueError("continuous client sequence differs from reference")
                    if output["timed_jit_warnings"]:
                        raise ValueError("JIT occurred in timed stream delivery")
                    deliveries.extend({"monotonic_s": output["started_monotonic_s"] + item["relative_s"],
                                       "count": len(carried) + item["count"]} for item in output["delivery_events"])
                    counts = [item["count"] for item in deliveries]
                    if not counts or counts[-1] != 512 or any(b <= a for a, b in zip(counts, counts[1:])):
                        raise ValueError("continuous client delivery counts are duplicated, revised, or incomplete")
                    if live is not None:
                        event = row["preemptions"][-1]
                        event["remaining_service_output"] = output
                        notice = event["source_case"]["notice_monotonic_s"]
                        event["notice_to_first_token_s"] = output["first_token_monotonic_s"] - notice
                        event["notice_to_request_complete_s"] = output["completed_monotonic_s"] - notice
                        live["delivery_executor"].shutdown(wait=True)
                        live = None
                    end = output["completed_monotonic_s"]
                    gaps = [b["monotonic_s"] - a["monotonic_s"] for a, b in zip(deliveries, deliveries[1:])]
                    row["requests"].append({"request_index": request_index, "arrival_s": arrival,
                        "start_s": segment_start - epoch, "complete_s": end - epoch,
                        "latency_s": end - epoch - arrival, "max_delivery_gap_s": max(gaps, default=0),
                        "output_sha256": digest(combined), "generated_tokens": 512,
                        "matches_reference": True, "delivery_events": deliveries})
                    progress("continuous_request_complete", label=label, request_index=request_index)
                    break
                time.sleep(0.01)
        if event_index != 2:
            raise ValueError("workload finished before both precommitted preemptions")
        row["status"] = "passed"
        row["stream_complete_s"] = row["requests"][-1]["complete_s"]
        row["mean_request_latency_s"] = sum(r["latency_s"] for r in row["requests"]) / len(row["requests"])
        row["output_tokens_per_s"] = 512 * len(row["requests"]) / row["stream_complete_s"]
        row["max_delivery_gap_s"] = max(r["max_delivery_gap_s"] for r in row["requests"])
        return row
    finally:
        if future is not None and not future.done():
            worker.cancel_measurement(rid)
        worker.close()
        if live is not None:
            live["delivery_executor"].shutdown(wait=True)
        executor.shutdown(wait=True)


def run_churn_matrix(args, formal=False):
    data = json.loads(Path(args.calibration).read_text())
    pilot = json.loads(Path(args.pilot_report).read_text())
    shape = json.loads(Path(args.shape_report).read_text())
    if (pilot["status"] != "passed" or shape["status"] != "passed"
            or pilot["calibration_sha256"] != digest(data) or shape["calibration_sha256"] != digest(data)
            or not all(c["runtime"]["routing_capture"] for c in data["candidates"].values())):
        raise ValueError("matching actuation/shape/live-target-routing gates must pass")
    if formal:
        gate = json.loads(Path(args.overall_pilot_report).read_text())
        if gate["status"] != "passed" or gate["calibration_sha256"] != digest(data):
            raise ValueError("matching two-event stream pilot must pass")
        trace = gate["protocol"]["trace"]
    else:
        trace = trace_from_pilot(pilot)
    out = Path(args.output_dir)
    protocol = {"scope": "formal_overall_fixed_wall_churn" if formal else "overall_churn_pilot",
                "calibration_sha256": digest(data), "trace": trace, "versions": ["original", "moe_aware"],
                "repeats_per_version": 3 if formal else 1,
                "availability_simulation": "logical_GPU_slots_and_own_container_termination_not_hardware_power_off",
                "legal_target_groups": [[1], [2], [3], [1, 3]],
                "only_selected_group_created_after_each_notice": True,
                "initialization_excluded_from_stream_clock": True}
    write_artifact(out / "protocol.json", protocol)
    result = {"status": "running", "protocol": protocol, "calibration_sha256": digest(data),
              "runs": [], "formal": formal}
    pool = GPUPool(out / "worker_logs", args.model, args.model_revision, GPU_UUIDS, args.timeout_s,
                   compiler_cache=args.compiler_cache)
    try:
        for repeat in range(3 if formal else 1):
            versions = ["moe_aware", "original"] if repeat == 1 else ["original", "moe_aware"]
            for version in versions:
                label = f"overall-r{repeat}-{version}"
                row = run_stream(pool, data, trace, label, version == "moe_aware")
                row["version"], row["repeat"], row["formal_run"] = version, repeat, formal
                result["runs"].append(row)
                write_artifact(out / (label + ".json"), row)
                progress("churn_run_complete", label=label, stream_complete_s=row["stream_complete_s"], formal=formal)
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt):
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
    finally:
        pool.close()
        write_artifact(out / ("formal_overall.json" if formal else "overall_pilot.json"), result)
    progress("churn_matrix_finished", status=result["status"], formal=formal, error=result.get("traceback"))
    return result
