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
    return {
        "type": "reparallelization",
        "model": model,
        "event": event,
        "node_id": node_id,
        "instance_id": instance_id,
        "action": decision.get("action"),
        "available_gpus": decision.get("availability", {}).get(
            "available_gpus", 0
        ),
        "unavailable_gpus": decision.get("availability", {}).get(
            "unavailable_gpus", 0
        ),
        "candidate_count": decision.get("candidate_count", 0),
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
        "target_nodes": parallel_plan.get("target_nodes", []),
        "parallel_plan": parallel_plan or None,
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
    }
