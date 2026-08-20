# 系統執行時所收集的統計資料

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class JsonlMetricsWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: Dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **event}
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")


def make_instance_state_event(
    model: str,
    instance_id: str,
    from_state: Optional[str],
    to_state: str,
    node_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "type": "instance_state",
        "model": model,
        "instance_id": instance_id,
        "node_id": node_id,
        "from": from_state,
        "to": to_state,
        "reason": reason,
    }


def make_replanning_event(
    model: str,
    event: str,
    decision: Dict[str, Any],
    node_id: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> Dict[str, Any]:
    parallel_plan = decision.get("parallel_plan") or {}
    selected_config = decision.get("selected_config") or {}
    availability = decision.get("availability") or {}
    workload_cost_model = decision.get("workload_cost_model") or {}
    execution = decision.get("execution") or {}
    target_nodes = [str(node) for node in parallel_plan.get("target_nodes", [])]
    ready_nodes = [str(node) for node in availability.get("ready_nodes", [])]
    source_node = str(node_id) if node_id is not None else ""
    cross_node_target = bool(
        source_node
        and target_nodes
        and any(target_node != source_node for target_node in target_nodes)
    )
    return {
        "type": "reparallelization",
        "model": model,
        "event": event,
        "node_id": node_id,
        "instance_id": instance_id,
        "action": decision.get("action"),
        "available_gpus": availability.get("available_gpus", 0),
        "unavailable_gpus": availability.get("unavailable_gpus", 0),
        "candidate_count": decision.get("candidate_count", 0),
        "worker_node_count": decision.get("worker_node_count", 0),
        "ready_worker_node_count": decision.get(
            "ready_worker_node_count", len(ready_nodes)
        ),
        "synthetic_worker_node_count": decision.get(
            "synthetic_worker_node_count", 0
        ),
        "runtime_worker_node_count": decision.get(
            "runtime_worker_node_count",
            decision.get("physical_worker_node_count", 0),
        ),
        "physical_worker_node_count": decision.get(
            "physical_worker_node_count", 0
        ),
        "selected_total_gpus": decision.get("selected_total_gpus", 0),
        "selected_tensor_parallel_size": decision.get(
            "selected_tensor_parallel_size", 0
        ),
        "selected_pipeline_parallel_size": decision.get(
            "selected_pipeline_parallel_size", 0
        ),
        "selected_data_parallel_size": decision.get(
            "selected_data_parallel_size", 0
        ),
        "selected_replica_count": decision.get(
            "selected_replica_count", 0
        ),
        "selected_enable_expert_parallel": decision.get(
            "selected_enable_expert_parallel", False
        ),
        "selected_effective_expert_parallel_size": decision.get(
            "selected_effective_expert_parallel_size", 0
        ),
        "selected_expert_parallel_size": decision.get(
            "selected_expert_parallel_size", 0
        ),
        "selected_score": decision.get(
            "selected_score", selected_config.get("score", 0.0)
        ),
        "selected_base_score": decision.get(
            "selected_base_score", selected_config.get("base_score", 0.0)
        ),
        "selected_workload_score_delta": decision.get(
            "selected_workload_score_delta",
            selected_config.get("workload_score_delta", 0.0),
        ),
        "selected_arrival_rate_req_s": decision.get(
            "selected_arrival_rate_req_s",
            selected_config.get("arrival_rate_req_s", 0.0),
        ),
        "selected_batch_size": decision.get(
            "selected_batch_size", selected_config.get("batch_size", 0)
        ),
        "selected_latency_estimate_ms": decision.get(
            "selected_latency_estimate_ms",
            selected_config.get("latency_estimate_ms", 0.0),
        ),
        "selected_throughput_estimate_req_s": decision.get(
            "selected_throughput_estimate_req_s",
            selected_config.get("throughput_estimate_req_s", 0.0),
        ),
        "selected_load_time_estimate_ms": decision.get(
            "selected_load_time_estimate_ms",
            selected_config.get("load_time_estimate_ms", 0.0),
        ),
        "selected_migration_cost_estimate_ms": decision.get(
            "selected_migration_cost_estimate_ms",
            selected_config.get("migration_cost_estimate_ms", 0.0),
        ),
        "selected_queue_penalty_ms": decision.get(
            "selected_queue_penalty_ms",
            selected_config.get("queue_penalty_ms", 0.0),
        ),
        "selected_replan_window_cost_ms": decision.get(
            "selected_replan_window_cost_ms",
            selected_config.get("replan_window_cost_ms", 0.0),
        ),
        "workload_cost_model_enabled": bool(
            workload_cost_model.get("enabled", False)
        ),
        "workload_arrival_rate_req_s": workload_cost_model.get(
            "arrival_rate_req_s", 0.0
        ),
        "workload_batch_size": workload_cost_model.get("batch_size", 0),
        "workload_latency_estimate_ms": workload_cost_model.get(
            "latency_estimate_ms", 0.0
        ),
        "target_nodes": target_nodes,
        "target_worker_node_count": len(set(target_nodes)),
        "cross_node_target": cross_node_target,
        "multi_worker_target": len(set(target_nodes)) > 1,
        "parallel_plan": parallel_plan or None,
        "execution": execution or None,
        "execution_status": execution.get("status", ""),
        "execution_duration_ms": execution.get("duration_ms", 0.0),
    }


def make_context_migration_event(
    model: str,
    decision: Dict[str, Any],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    plans = decision.get("plans", [])
    prefix_warmup = (
        decision.get("prefix_warmup")
        or decision.get("kv_cache_migration")
        or {}
    )
    kv_restore = decision.get("kv_restore") or {}
    prefix_warmup_attempts = int(prefix_warmup.get("attempted", 0) or 0)
    prefix_warmup_successes = int(prefix_warmup.get("succeeded", 0) or 0)
    kv_restore_attempts = int(kv_restore.get("attempted", 0) or 0)
    kv_restore_successes = int(kv_restore.get("succeeded", 0) or 0)
    kv_restore_restored_blocks = int(
        kv_restore.get("restored_blocks", 0) or 0
    )
    true_kv_block_transfer = bool(
        kv_restore_successes > 0 and kv_restore_restored_blocks > 0
    )
    return {
        "type": "context_migration",
        "model": model,
        "action": decision.get("action"),
        "reason": reason,
        "context_migration_plan_count": len(plans),
        "migration_plan_count": len(plans),
        "unassigned_context_count": len(
            decision.get("unassigned_contexts", [])
        ),
        "total_estimated_cost": decision.get("total_estimated_cost", 0.0),
        "total_reusable_tokens": decision.get("total_reusable_tokens", 0),
        "total_context_tokens": decision.get("total_context_tokens", 0),
        "total_reusable_context_blocks": decision.get(
            "total_reusable_context_blocks", 0
        ),
        "total_context_blocks": decision.get("total_context_blocks", 0),
        "reuse_ratio": decision.get("reuse_ratio", 0.0),
        "plans": plans,
        "prefix_warmup": prefix_warmup or None,
        "prefix_warmup_attempts": prefix_warmup_attempts,
        "prefix_warmup_successes": prefix_warmup_successes,
        "prefix_warmup_tokens": int(
            prefix_warmup.get(
                "warmed_tokens", prefix_warmup.get("total_tokens", 0)
            )
            or 0
        ),
        "kv_restore": kv_restore or None,
        "kv_restore_attempts": kv_restore_attempts,
        "kv_restore_successes": kv_restore_successes,
        "kv_restore_restored_blocks": kv_restore_restored_blocks,
        "true_kv_block_transfer": true_kv_block_transfer,
        "kv_cache_migration": decision.get("kv_cache_migration"),
    }


def make_state_recovery_event(
    model: str,
    request_id: str,
    decision: Dict[str, Any],
    source_instance_id: Optional[str] = None,
    target_instance_id: Optional[str] = None,
) -> Dict[str, Any]:
    plan = decision.get("plan") or {}
    return {
        "type": "state_recovery",
        "model": model,
        "request_id": request_id,
        "action": decision.get("action"),
        "source_instance_id": (
            source_instance_id or plan.get("source_instance_id")
        ),
        "target_instance_id": (
            target_instance_id or plan.get("target_instance_id")
        ),
        "state_available": decision.get("state_available", False),
        "restore_supported": decision.get("restore_supported", False),
        "fallback_used": decision.get("fallback_used", False),
        "recovered_tokens": decision.get("recovered_tokens", 0),
        "reason": decision.get("reason"),
        "plan": plan or None,
    }


def make_risk_aware_scheduling_event(
    model: str,
    policy: str,
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = decision.get("candidates", [])
    selected_node_id = decision.get("selected_node_id")
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("node_id") == selected_node_id
        ),
        candidates[0] if candidates else {},
    )
    return {
        "type": "risk_aware_scheduling",
        "model": model,
        "policy": policy,
        "action": decision.get("action"),
        "requested_gpus": decision.get("requested_gpus", 0),
        "selected_node_id": decision.get("selected_node_id"),
        "candidate_count": len(candidates),
        "selected_score": selected.get("score", 0.0),
        "selected_spot_risk": selected.get("spot_risk", 0.0),
        "selected_remaining_lifetime_s": selected.get(
            "remaining_lifetime_s", 0.0
        ),
        "selected_loading_cost": selected.get("loading_cost", 0.0),
        "selected_metadata_source": selected.get("metadata_source", ""),
        "selected_provider": selected.get("provider", ""),
        "selected_confidence": selected.get("confidence", 0.0),
        "decision": decision,
    }


def make_request_event(
    request_id: str,
    model: str,
    policy: str,
    success: bool,
    latency_ms: float,
    retry_count: int = 0,
    failed_attempts: int = 0,
    recovered_tokens: int = 0,
    recovery_fallback: bool = False,
    state_restore_attempts: int = 0,
    state_restore_successes: int = 0,
    state_restore_fallback: bool = False,
    state_restored_tokens: int = 0,
    supports_state_restore: bool = False,
    state_kind: str = "",
    state_restore_reason: str = "",
    state_restored_blocks: int = 0,
    state_restore_staged: bool = False,
) -> Dict[str, Any]:
    return {
        "type": "request",
        "request_id": request_id,
        "model": model,
        "policy": policy,
        "success": success,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "failed_attempts": failed_attempts,
        "recovered_tokens": recovered_tokens,
        "recovery_fallback": recovery_fallback,
        "state_restore_attempts": state_restore_attempts,
        "state_restore_successes": state_restore_successes,
        "state_restore_fallback": state_restore_fallback,
        "state_restored_tokens": state_restored_tokens,
        "supports_state_restore": supports_state_restore,
        "state_kind": state_kind,
        "state_restore_reason": state_restore_reason,
        "state_restored_blocks": state_restored_blocks,
        "state_restore_staged": state_restore_staged,
    }
