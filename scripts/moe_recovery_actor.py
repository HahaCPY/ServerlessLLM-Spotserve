"""Real planned local-GPU execution through SpotServe's shared executor."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from scripts.run_granite_moe_replay_pilot import digest
from sllm.spot.moe_ablation import verify_measured_runtime
from sllm.spot.moe_route_cost import plan_calibrated_recovery
from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.reparallelization_executor import ReparallelizationExecutor


def actual_runtime_evidence(worker, ack, artifact):
    return {"artifact": artifact, "observation_source": "actual_worker_runtime",
            "tensor_parallel_size": worker.runtime["tensor_parallel_size"],
            "pipeline_parallel_size": worker.runtime["pipeline_parallel_size"],
            "data_parallel_size": worker.runtime["data_parallel_size"], "replica_count": 1,
            "enable_expert_parallel": worker.runtime["enable_expert_parallel"],
            "physical_gpu_indices": sorted(row["physical_gpu_index"] for row in worker.ranks),
            "profile_runtime_signature": worker.runtime_signature,
            "ranks": worker.ranks, "frozen_prefix_sha256": ack["frozen_prefix_sha256"],
            "state_install_verified": ack["install_mode"] == "host_token_handoff_excluding_gpu_prefill",
            "state_install_definition": ack["install_mode"], "direct_gpu_kv_restore": False}


def execute_recovery(pool, data, source, source_case, kind, moe_aware, label,
                     resident_workers=(), available=(1, 2, 3), live=False):
    planning_start = time.monotonic()
    resident = {tuple(worker.group): worker for worker in resident_workers if not worker.closed}
    decision = plan_calibrated_recovery(data, source_case, kind, moe_aware, available,
                                        resident_groups=resident.keys())
    planner_s = time.monotonic() - planning_start
    selected = decision["selected_cost"]
    request_id = label + "-remaining"
    state = {"decision": decision, "planner_s": planner_s}
    timeline = []
    delivery = ThreadPoolExecutor(max_workers=1)

    def create(plan):
        timeline.append({"phase": "create_start", "monotonic_s": time.monotonic()})
        group = selected["physical_gpu_indices"]
        if tuple(group) in resident:
            worker = resident[tuple(group)]
        else:
            worker = pool.launch(label + "-target", group,
                                 capture=data["candidates"][selected["candidate_id"]]["runtime"]["routing_capture"])
            warm_tokens = data["source_cases"]["train-code"]["frozen"]["tokens"]
            worker.warm(warm_tokens)
        timeline.append({"phase": "warmed_ready", "monotonic_s": time.monotonic()})
        return worker

    def ready(worker, plan):
        began = time.monotonic()
        ack = worker.command_event({"op": "install_replay_tokens", "request_id": request_id,
                                    "token_ids": source_case["frozen"]["tokens"]}, "replay_tokens_installed")
        state["host_handoff_s"] = time.monotonic() - began
        state["handoff_ack"] = ack
        timeline.append({"phase": "host_tokens_installed", "monotonic_s": time.monotonic()})
        return ack["token_count"] == len(source_case["frozen"]["tokens"]) and not ack["gpu_kv_restored"]

    def verify(worker, plan):
        evidence = actual_runtime_evidence(worker, state["handoff_ack"], label + "-runtime")
        state["runtime_evidence"] = evidence
        state["runtime_verification"] = verify_measured_runtime(decision, evidence)
        return state["runtime_verification"]["configuration_applied"]

    def switch(worker, plan):
        timeline.append({"phase": "traffic_switched", "monotonic_s": time.monotonic()})
        state["future"] = delivery.submit(worker.measure, request_id, None,
                                           source_case["frozen"]["remaining_tokens"], True)

    def drain(worker):
        # The authoritative source snapshot was frozen before choosing a plan.
        timeline.append({"phase": "old_already_frozen", "monotonic_s": time.monotonic()})

    def stop(worker):
        timeline.append({"phase": "worker_stop_start", "label": worker.label,
                         "monotonic_s": time.monotonic()})
        worker.close()
        timeline.append({"phase": "worker_stopped", "label": worker.label,
                         "monotonic_s": time.monotonic()})
        if worker is source:
            state["source_termination"] = worker.evidence()

    executor = ReparallelizationExecutor(create, ready, switch, drain, stop,
                                         current=source, verify_runtime=verify,
                                         stop_current_before_create=True)
    try:
        target = asyncio.run(executor.apply(ParallelPlan.from_dict(decision["parallel_plan"])))
        if live:
            state["delivery_executor"] = delivery
            state["request_id"] = request_id
            state["timeline"] = timeline
            return state, target
        row = state.pop("future").result(timeout=pool.timeout)
        combined = source_case["client_prefix"] + row["output_token_ids"]
        if (combined != source_case["reference"]["output_token_ids"]
                or len(combined) != 512 or row["timed_jit_warnings"]):
            raise ValueError("planned replay/client sequence/JIT timing gate failed")
        notice = source_case["notice_monotonic_s"]
        state.update({"status": "passed", "kind": kind, "moe_aware": moe_aware,
                      "source_freeze": source_case, "target_output": row,
                      "output_sha256": digest(combined), "combined_output_tokens": len(combined),
                      "matches_uninterrupted_reference": True,
                      "notice_to_first_token_s": row["first_token_monotonic_s"] - notice,
                      "notice_to_complete_s": row["completed_monotonic_s"] - notice,
                      "maximum_delivery_gap_including_handoff_s": max(
                          row["first_token_monotonic_s"]
                              - source_case.get("source_last_token_monotonic_s", notice),
                          row["max_delivery_gap_s"]),
                      "notice_to_first_four_tokens_s": row["started_monotonic_s"]
                          + row["first_four_tokens_s"] - notice,
                      "timeline": timeline, "target_worker": target.evidence(),
                      "configuration_applied": True,
                      "recovery_method": "token_replay_not_direct_KV_restore",
                      "actor_kind": "local_physical_GPU_adapter_not_Ray_cluster"})
        return state, target
    finally:
        if not live or "delivery_executor" not in state:
            delivery.shutdown(wait=True)


def check_tp2_replay_canary(pool, data, label):
    """A manually requested TP2 correctness check, never an ablation outcome."""
    case = data["source_cases"]["formal-serving"]
    worker = pool.launch(label, [1, 3],
                         capture=data["candidates"]["gpu1_3-tp2"]["runtime"]["routing_capture"])
    try:
        worker.warm(data["source_cases"]["train-code"]["frozen"]["tokens"])
        rid = label + "-probe"
        ack = worker.command_event({"op": "install_replay_tokens", "request_id": rid,
                                    "token_ids": case["frozen"]["tokens"]}, "replay_tokens_installed")
        worker.conn.send({"op": "generate", "request_id": rid, "use_installed_replay": True,
                          "max_new_tokens": 256, "pause_after_new_tokens": 1})
        worker.wait(lambda e: e.get("event") == "paused" and e.get("request_id") == rid)
        worker.command_event({"op": "pause_generation"}, "generation_paused")
        metadata = worker.command_event({"op": "metadata", "request_id": rid}, "metadata")["result"]
        if (metadata.get("prompt_tokens") != case["frozen"]["tokens"]
                or metadata.get("output_tokens") != case["reference"]["output_token_ids"][256:257]
                or not metadata.get("allocated_kv_block_count")):
            raise ValueError("actual TP2 EngineCore replay prefix/first token/KV differs")
        worker.command_event({"op": "resume_generation"}, "generation_resumed")
        worker.command_event({"op": "resume", "request_id": rid}, "resumed")
        final = worker.wait(lambda e: e.get("request_id") == rid and e.get("finished"))
        combined = case["client_prefix"] + final["cumulative_token_ids"]
        if combined != case["reference"]["output_token_ids"] or len(combined) != 512:
            raise ValueError("TP2 resumed client output differs from uninterrupted reference")
        return {"status": "passed", "scope": "manual_TP2_runtime_and_EngineCore_correctness_canary",
                "formal_ablation_run": False, "selection_forced_for_correctness_only": True,
                "actual_runtime": worker.evidence(), "handoff_ack": ack,
                "engine_core_prompt_sha256": digest(metadata["prompt_tokens"]),
                "allocated_kv_block_count": metadata["allocated_kv_block_count"],
                "combined_output_sha256": digest(combined), "combined_output_tokens": len(combined)}
    finally:
        worker.close()
