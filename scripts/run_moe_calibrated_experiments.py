"""Audited calibration and real local-GPU recovery experiments; no dense fallback."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback

from scripts.moe_gpu_runtime import GPUPool, write_artifact
from scripts.run_granite_moe_replay_pilot import digest
from sllm.spot.moe_route_cost import validate_held_out_predictions


GPU_UUIDS = {
    0: "GPU-ebfc3b1b-e996-c058-6dc6-df5448c830cb",
    1: "GPU-33f01ad1-6b61-a118-4bb0-a34b2c6841e5",
    2: "GPU-f8166666-3067-b764-e524-ecc41b05ea38",
    3: "GPU-a90a5521-c5fc-807b-be8c-4479014ed079",
}
WORKLOADS = [
    ("train-code", "Write a Python function for sorting and explain its tests. "),
    ("train-math", "Solve a system of linear equations and explain each algebraic step. "),
    ("train-story", "Tell a story about a lighthouse keeper and the changing weather. "),
    ("train-history", "Explain the development of printing and its cultural effects. "),
    ("train-language", "Translate a short greeting into English and explain its grammar. "),
    ("train-reasoning", "Compare two scheduling policies using a worked example. "),
    ("validation-science", "Explain how a plant transforms sunlight into chemical energy. "),
    ("validation-systems", "Explain a database transaction, consistency, and recovery. "),
    ("formal-serving", "Explain serving a sparse language model on changing GPU resources. "),
]


def progress(phase, **fields):
    print("CALIBRATED_MOE_PROGRESS=" + json.dumps({"phase": phase, **fields}), flush=True)


def protocol(args):
    return {
        "scope": "calibration_before_formal_ablation", "revision": args.model_revision,
        "model": args.model, "checkpoint_class": "GraniteMoeForCausalLM",
        "workloads": WORKLOADS, "synthetic_repeated_token_prompts": True,
        "source_gpu": 0, "migration_targets": [[1], [2], [3]],
        "reparallelization_candidates": [[1], [1, 3]],
        "initial_prompt_tokens": 4096, "source_output_tokens": 512,
        "native_freeze_generated_tokens": 256, "remaining_output_tokens": 256,
        "concurrency": 1, "calibration_repeats_per_workload": 3,
        "independent_cold_target_constructions": 3,
        "route_model": "fixed_k3_inverse_distance_nearest_neighbour_normalized_runtime_histogram",
        "generic_model": "per_candidate_training_workload_mean",
        "validation_ids": [key for key, _ in WORKLOADS if key.startswith("validation-")],
        "formal_ids_excluded_from_calibration_and_validation": ["formal-serving"],
        "background_trace": "idle_no_submitted_background_requests",
        "cache_policy": "prefix_cache_off_KV_new_each_request_shared_task_triton_cache",
        "recovery": "host_token_handoff_then_GPU_prefill_and_remaining_decode",
        "cost_partition": "warmed_engine_ready_plus_host_handoff_plus_full_remaining_service",
        "physical_expert_weight_placement": "not_implemented_not_claimed",
        "existing_idle_sllm_store_kept": True, "four_gpu_exclusive": False,
        "formal_repeats_each_version_each_scenario": 3,
        "formal_versions": ["original", "moe_aware"], "no_forced_different_selection": True,
        "target_routing_capture": getattr(args, "capture_target_routes", False),
        "compiler_cache_directory": getattr(args, "compiler_cache", None),
    }


def calibrate(args):
    from transformers import AutoTokenizer

    out = Path(args.output_dir)
    plan = protocol(args)
    write_artifact(out / "protocol.json", plan)
    pool = GPUPool(out / "worker_logs", args.model, args.model_revision, GPU_UUIDS, args.timeout_s,
                   compiler_cache=args.compiler_cache)
    report = {"scope": "independent_prior_calibration_not_formal_runs", "status": "running",
              "protocol_sha256": digest(plan), "protocol": plan, "source_cases": {},
              "candidates": {}, "formal_experiment_eligible": False, "formal_runs_completed": 0}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                                 trust_remote_code=False)
        prior = json.loads(Path(args.source_calibration).read_text()) if args.source_calibration else None
        if prior is not None and (prior["status"] != "passed"
                                  or prior["protocol"]["revision"] != args.model_revision):
            raise ValueError("prior live source cases/checkpoint evidence invalid")
        source = None if prior is not None else pool.launch("calibration-source", [0], capture=True)
        prompts = {}
        for key, text in WORKLOADS:
            seed = tokenizer.encode(text)
            prompts[key] = (seed * (4096 // len(seed) + 1))[:4096]
        if source is not None:
            source.warm(prompts[WORKLOADS[0][0]])
        for key, _ in WORKLOADS:
            if prior is not None:
                case = prior["source_cases"][key]
                write_artifact(out / "source_cases" / (key + ".json"), case)
                report["source_cases"][key] = case
                continue
            reference = source.measure(key + "-reference", prompts[key], 512)
            freeze = source.freeze(key + "-freeze", prompts[key])
            if freeze["client_prefix"] != reference["output_token_ids"][:256]:
                raise ValueError("source native frozen prefix differs from uninterrupted reference")
            case = {"workload_id": key, "prompt_token_ids": prompts[key], "reference": reference,
                    **freeze}
            write_artifact(out / "source_cases" / (key + ".json"), case)
            report["source_cases"][key] = case
            source.abort_resume(key + "-freeze")
            progress("source_case_verified", workload_id=key,
                     route_sha256=freeze["routes"]["histogram_sha256"])
        warm_tokens = report["source_cases"][WORKLOADS[0][0]]["frozen"]["tokens"]

        def candidate(group, round_id):
            cid = "gpu" + "_".join(map(str, group)) + "-tp" + str(len(group))
            worker = pool.launch(cid + "-cold" + str(round_id), group,
                                 capture=args.capture_target_routes)
            worker.warm(warm_tokens)
            result = {"candidate_id": cid, "physical_gpu_indices": group, "tp": len(group),
                      "runtime": worker.profile_runtime, "runtime_signature": worker.runtime_signature,
                      "ready_samples": [], "cases": {}}
            keys = [key for key, _ in WORKLOADS if not key.startswith("formal-")]
            if round_id:
                keys = [WORKLOADS[0][0]]
            for key in keys:
                case = report["source_cases"][key]
                if not round_id:
                    worker.measure(f"{cid}-route-warm-{key}", case["frozen"]["tokens"], 64)
                rows = []
                for repeat in range(3 if not round_id else 1):
                    rid = f"{cid}-{round_id}-{key}-{repeat}"
                    began = time.monotonic()
                    ack = worker.command_event({"op": "install_replay_tokens", "request_id": rid,
                                                "token_ids": case["frozen"]["tokens"]},
                                               "replay_tokens_installed")
                    install_ms = 1000 * (time.monotonic() - began)
                    if (ack["frozen_prefix_sha256"] != case["frozen"]["full_prefix_sha256"]
                            or ack["token_count"] != 4352 or ack["gpu_kv_restored"]):
                        raise ValueError("actual target host token handoff mismatch")
                    row = worker.measure(rid, output_tokens=256, installed=True)
                    row["host_handoff_ms"] = install_ms
                    row["handoff_ack"] = ack
                    combined = case["client_prefix"] + row["output_token_ids"]
                    row["matches_uninterrupted_reference"] = combined == case["reference"]["output_token_ids"]
                    if not row["matches_uninterrupted_reference"] or row["timed_jit_warnings"]:
                        raise ValueError("remaining service correctness/JIT timing gate failed")
                    rows.append(row)
                result["cases"][key] = rows
                progress("candidate_case_measured", candidate=cid, round=round_id, workload_id=key)
            result["ready_samples"].append({"round": round_id,
                "ready_ms": 1000 * worker.cold_to_warmed_ready_s,
                "host_handoff_ms": result["cases"][keys[0]][0]["host_handoff_ms"],
                "runtime_signature": worker.runtime_signature,
                "artifact": str(out / f"{cid}-round{round_id}.json")})
            worker.close()
            result["worker"] = worker.evidence()
            if result["worker"]["worker_exit_code"] != 0:
                raise ValueError("candidate worker did not shut down normally")
            write_artifact(out / f"{cid}-round{round_id}.json", result)
            return result

        for round_id in range(3):
            with ThreadPoolExecutor(max_workers=3) as executor:
                rows = list(executor.map(lambda group: candidate(group, round_id), [[1], [2], [3]]))
            rows.append(candidate([1, 3], round_id))
            for row in rows:
                cid = row["candidate_id"]
                if cid not in report["candidates"]:
                    report["candidates"][cid] = row
                else:
                    if row["runtime_signature"] != report["candidates"][cid]["runtime_signature"]:
                        raise ValueError("independent target constructions changed runtime")
                    report["candidates"][cid]["ready_samples"].extend(row["ready_samples"])
            progress("cold_round_complete", round=round_id)
        if source is not None:
            source.close()
            report["source_runtime"] = source.evidence()
        else:
            report["source_runtime"] = prior["source_runtime"]
            report["source_cases_prior_artifact"] = args.source_calibration
            report["source_cases_prior_sha256"] = digest(prior)
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt):
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        pool.close()
        write_artifact(out / "calibration.json", report)
    progress("calibration_finished", status=report["status"], output_dir=str(out),
             error=report.get("traceback"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--source-calibration")
    parser.add_argument("--capture-target-routes", action="store_true")
    parser.add_argument("--stage", choices=["calibrate", "recalibrate-ready", "validate", "validate-shape", "pilot", "formal-ablation",
                                             "overall-pilot", "formal-overall"],
                        default="calibrate")
    parser.add_argument("--calibration")
    parser.add_argument("--pilot-report")
    parser.add_argument("--compiler-cache")
    parser.add_argument("--shape-report")
    parser.add_argument("--overall-pilot-report")
    args = parser.parse_args()
    if args.stage == "calibrate":
        result = calibrate(args)
    elif args.stage == "recalibrate-ready":
        result = recalibrate_ready(args)
    elif args.stage == "validate":
        result = validate_cost_model(args)
    elif args.stage == "validate-shape":
        result = validate_variable_shape(args)
    elif args.stage in {"overall-pilot", "formal-overall"}:
        from scripts.moe_spot_churn import run_churn_matrix
        result = run_churn_matrix(args, formal=args.stage == "formal-overall")
    else:
        result = run_recovery_matrix(args, formal=args.stage == "formal-ablation")
    return 0 if result["status"] == "passed" else 1


def recalibrate_ready(args):
    """New independent trials after *all* candidate compiler caches are populated.

    Do not select old trials by latency. Preserve the entire original calibration,
    and replace all four candidates' ready trials under one declared cache regime.
    """
    from sllm.spot.moe_route_cost import validate_calibration

    prior_path = Path(args.calibration)
    data = json.loads(prior_path.read_text())
    validate_calibration(data)
    out = Path(args.output_dir)
    plan = {**data["protocol"], "deployment_ready_regime":
            "all_candidate_shapes_and_routing_capture_warmed_before_every_ready_trial",
            "prior_calibration_artifact": str(prior_path), "prior_calibration_sha256": digest(data),
            "replace_all_candidates_all_three_ready_trials_not_latency_based_selection": True}
    write_artifact(out / "protocol.json", plan)
    data["prior_ready_samples"] = {cid: c["ready_samples"] for cid, c in data["candidates"].items()}
    for candidate in data["candidates"].values():
        candidate["ready_samples"] = []
    data["protocol"], data["protocol_sha256"] = plan, digest(plan)
    data["status"] = "running"
    pool = GPUPool(out / "worker_logs", args.model, args.model_revision, GPU_UUIDS, args.timeout_s,
                   compiler_cache=args.compiler_cache)

    def trial(group, repeat):
        cid = "gpu" + "_".join(map(str, group)) + "-tp" + str(len(group))
        worker = pool.launch(f"{cid}-matched-ready{repeat}", group,
                             capture=data["candidates"][cid]["runtime"]["routing_capture"])
        try:
            case = data["source_cases"]["train-code"]
            worker.warm(case["frozen"]["tokens"])
            if worker.runtime_signature != data["candidates"][cid]["runtime_signature"]:
                raise ValueError("matched-cache ready trial changed actual runtime")
            rid = f"{cid}-ready{repeat}-correctness"
            began = time.monotonic()
            ack = worker.command_event({"op": "install_replay_tokens", "request_id": rid,
                                       "token_ids": case["frozen"]["tokens"]}, "replay_tokens_installed")
            handoff_ms = 1000 * (time.monotonic() - began)
            row = worker.measure(rid, output_tokens=256, installed=True)
            if (case["client_prefix"] + row["output_token_ids"] != case["reference"]["output_token_ids"]
                    or row["timed_jit_warnings"] or ack["gpu_kv_restored"]
                    or ack["frozen_prefix_sha256"] != case["frozen"]["full_prefix_sha256"]):
                raise ValueError("matched-cache ready trial output/JIT/handoff gate failed")
            artifact = out / f"{cid}-ready{repeat}.json"
            evidence = {"round": repeat, "ready_ms": 1000 * worker.cold_to_warmed_ready_s,
                        "host_handoff_ms": handoff_ms, "runtime_signature": worker.runtime_signature,
                        "artifact": str(artifact)}
            worker.close()
            write_artifact(artifact, {"status": "passed", "ready_sample": evidence,
                           "correctness_output": row, "worker": worker.evidence(),
                           "warmup_jit_warnings_included_in_ready_cost": worker.warnings})
            progress("matched_cache_ready_trial", candidate=cid, repeat=repeat,
                     ready_s=evidence["ready_ms"] / 1000)
            return cid, evidence
        finally:
            worker.close()

    try:
        for repeat in range(3):
            with ThreadPoolExecutor(max_workers=3) as executor:
                measured = list(executor.map(lambda group: trial(group, repeat), [[1], [2], [3]]))
            measured.append(trial([1, 3], repeat))
            for cid, row in measured:
                data["candidates"][cid]["ready_samples"].append(row)
        data["status"] = "passed"
        validate_calibration(data)
    except (Exception, KeyboardInterrupt):
        data["status"] = "failed"
        data["traceback"] = traceback.format_exc()
    finally:
        pool.close()
        write_artifact(out / "calibration.json", data)
    progress("matched_cache_calibration_finished", status=data["status"], error=data.get("traceback"))
    return data


def validate_cost_model(args):
    data = json.loads(Path(args.calibration).read_text())
    diagnostic = validate_held_out_predictions(data)
    result = {"status": "passed", "calibration_artifact": args.calibration,
              "calibration_sha256": digest(data), "diagnostic": diagnostic,
              "routing_predictor_implemented": True, "optimization_benefit_established": False}
    write_artifact(Path(args.output_dir) / "cost_validation.json", result)
    progress("held_out_cost_validation", **result)
    return result


def validate_variable_shape(args):
    from sllm.spot.moe_route_cost import predict_candidates, validate_calibration

    data = json.loads(Path(args.calibration).read_text())
    validate_calibration(data)
    out = Path(args.output_dir)
    plan = {"scope": "held_out_variable_remaining_work_validation",
        "calibration_sha256": digest(data), "native_boundaries": [128, 384],
        "workloads": data["protocol"]["validation_ids"], "repeats": 3,
        "candidates": [[1], [2], [3], [1, 3]], "relative_mean_error_gate": 0.20,
        "prediction": "scaled_prefill_TTFT_plus_remaining_decode_step",
        "no_training_on_this_validation": True}
    write_artifact(out / "protocol.json", plan)
    result = {"status": "running", "protocol": plan, "rows": [], "source_cases": {},
              "calibration_sha256": digest(data), "formal_run": False}
    pool = GPUPool(out / "worker_logs", args.model, args.model_revision, GPU_UUIDS, args.timeout_s,
                   compiler_cache=args.compiler_cache)
    try:
        source = pool.launch("shape-validation-source", [0], capture=True)
        first_key = plan["workloads"][0]
        source.warm(data["source_cases"][first_key]["prompt_token_ids"])
        for key in plan["workloads"]:
            original = data["source_cases"][key]
            for boundary in plan["native_boundaries"]:
                rid = f"{key}-boundary{boundary}"
                freeze = source.freeze(rid, original["prompt_token_ids"], threshold=boundary)
                if freeze["client_prefix"] != original["reference"]["output_token_ids"][:boundary]:
                    raise ValueError("shape validation native prefix differs from reference")
                result["source_cases"][rid] = {"workload_id": key, "reference": original["reference"], **freeze}
                source.abort_resume(rid)
        source.close()
        def measure_candidate(group):
            cid = "gpu" + "_".join(map(str, group)) + "-tp" + str(len(group))
            worker = pool.launch("shape-validation-" + cid, group,
                                 capture=data["candidates"][cid]["runtime"]["routing_capture"])
            worker.warm(data["source_cases"]["train-code"]["frozen"]["tokens"])
            if worker.runtime_signature != data["candidates"][cid]["runtime_signature"]:
                raise ValueError("shape validation changed actual runtime")
            for rid, case in result["source_cases"].items():
                worker.measure(rid + "-warm-" + cid, case["frozen"]["tokens"], 64)
                predictions = {signal: predict_candidates(data, case, signal, variable_shape=True)[cid]
                               for signal in ("removed", "routes", "shuffled")}
                rows = []
                for repeat in range(3):
                    row = worker.measure(f"{rid}-{cid}-r{repeat}", case["frozen"]["tokens"],
                                         case["frozen"]["remaining_tokens"])
                    if (case["client_prefix"] + row["output_token_ids"] != case["reference"]["output_token_ids"]
                            or row["timed_jit_warnings"]):
                        raise ValueError("shape validation output/JIT gate failed")
                    rows.append(row)
                actual = 1000 * sum(row["latency_s"] for row in rows) / len(rows)
                result["rows"].append({"case_id": rid, "candidate": cid, "samples": rows,
                    "actual_mean_latency_ms": actual, "predictions": predictions,
                    "relative_errors": {signal: abs(predictions[signal]["route_service_ms"] - actual) / actual
                                        for signal in predictions}})
                progress("shape_validation_case", case=rid, candidate=cid)
            worker.close()
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(measure_candidate, [[1], [2], [3]]))
        measure_candidate([1, 3])
        result["mean_relative_error_by_signal"] = {
            signal: sum(row["relative_errors"][signal] for row in result["rows"]) / len(result["rows"])
            for signal in ("removed", "routes", "shuffled")}
        result["mean_relative_error"] = sum(result["mean_relative_error_by_signal"][signal]
                                            for signal in ("removed", "routes")) / 2
        result["shape_gate_passed"] = all(result["mean_relative_error_by_signal"][signal]
            <= plan["relative_mean_error_gate"] for signal in ("removed", "routes"))
        result["status"] = "passed" if result["shape_gate_passed"] else "failed"
    except (Exception, KeyboardInterrupt):
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
    finally:
        pool.close()
        write_artifact(out / "shape_validation.json", result)
    progress("shape_validation_finished", status=result["status"],
             mean_relative_error=result.get("mean_relative_error"), error=result.get("traceback"))
    return result


def run_recovery_matrix(args, formal=False):
    from scripts.moe_recovery_actor import check_tp2_replay_canary, execute_recovery
    from sllm.spot.moe_route_cost import validate_calibration

    data = json.loads(Path(args.calibration).read_text())
    validate_calibration(data)
    if formal:
        gate = json.loads(Path(args.pilot_report).read_text())
        if gate["status"] != "passed" or gate["calibration_sha256"] != digest(data):
            raise ValueError("matching real GPU pilot must pass before formal repetitions")
    out = Path(args.output_dir)
    plan = {"scope": "formal_single_event_migration_reparallelization" if formal else "GPU_recovery_pilots",
        "calibration_artifact": args.calibration, "calibration_sha256": digest(data),
        "repetitions": 3 if formal else 1, "comparisons": ["migration", "reparallelization"],
        "order_by_repeat": [["original", "moe_aware"], ["moe_aware", "original"], ["original", "moe_aware"]],
        "native_freeze_generated_tokens": 256, "workload": "formal-serving",
        "migration": {"TP": 1, "source": [0], "resident_targets": [[1], [2], [3]]},
        "reparallelization": {"source": [0], "candidates": [[1], [1, 3]],
                              "only_selected_target_created": True},
        "recovery_method": "token_replay", "initialization_and_reference_excluded_from_outage": True,
        "no_forced_different_selection": True, "shared_compiler_cache_within_stage": True,
        "overall_multiple_wall_clock_churn_not_claimed_by_this_matrix": True}
    write_artifact(out / "protocol.json", plan)
    pool = GPUPool(out / "worker_logs", args.model, args.model_revision, GPU_UUIDS, args.timeout_s,
                   compiler_cache=args.compiler_cache)
    result = {"status": "running", "scope": plan["scope"], "protocol": plan,
              "calibration_sha256": digest(data), "runs": [], "overall_runs_completed": 0,
              "formal_experiment": formal}
    try:
        if not formal:
            result["manual_tp2_correctness_canary"] = check_tp2_replay_canary(pool, data, "manual-tp2-canary")
            write_artifact(out / "manual_tp2_canary.json", result["manual_tp2_correctness_canary"])
            progress("manual_tp2_canary", status="passed")
        for kind in ("migration", "reparallelization"):
            for repeat in range(3 if formal else 1):
                order = plan["order_by_repeat"][repeat]
                for version in order:
                    label = f"{kind}-r{repeat}-{version}"
                    row = {"kind": kind, "repeat": repeat, "version": version,
                           "status": "running", "formal_run": formal}
                    result["runs"].append(row)
                    prompt = data["source_cases"]["formal-serving"]["prompt_token_ids"]
                    resident = []
                    if kind == "migration":
                        def create_resident(group):
                            worker = pool.launch(label + "-resident" + str(group[0]), group,
                                capture=data["candidates"][f"gpu{group[0]}-tp1"]["runtime"]["routing_capture"])
                            worker.warm(data["source_cases"]["train-code"]["frozen"]["tokens"])
                            return worker
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            pending = [executor.submit(create_resident, group) for group in [[1], [2], [3]]]
                            source_future = executor.submit(pool.launch, label + "-source", [0], True)
                            source = source_future.result()
                            source.warm(prompt)
                            resident = [f.result() for f in pending]
                    else:
                        source = pool.launch(label + "-source", [0], capture=True)
                        source.warm(prompt)
                    reference = source.measure(label + "-reference", prompt, 512)
                    freeze = source.freeze(label + "-freeze", prompt)
                    if freeze["client_prefix"] != reference["output_token_ids"][:256]:
                        raise ValueError("formal fresh source prefix differs from its own reference")
                    case = {"workload_id": "formal-serving", "reference": reference, **freeze}
                    execution, target = execute_recovery(pool, data, source, case, kind,
                                                         version == "moe_aware", label, resident)
                    row.update(execution)
                    row["formal_run"] = formal
                    row["repeat"], row["version"] = repeat, version
                    target.close()
                    for worker in resident:
                        worker.close()
                    row["target_worker"] = target.evidence()
                    write_artifact(out / (label + ".json"), row)
                    progress("recovery_run_complete", kind=kind, repeat=repeat, version=version,
                        selected_candidate=row["decision"]["selected_candidate"],
                        actual_TP=row["runtime_evidence"]["tensor_parallel_size"],
                        notice_to_complete_s=row["notice_to_complete_s"], formal_run=formal)
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt):
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        if result["runs"]:
            result["runs"][-1]["status"] = "failed"
    finally:
        pool.close()
        write_artifact(out / ("formal_ablation.json" if formal else "pilot.json"), result)
    progress("recovery_matrix_finished", status=result["status"], formal=formal,
             completed=sum(row["status"] == "passed" for row in result["runs"]),
             error=result.get("traceback"))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
