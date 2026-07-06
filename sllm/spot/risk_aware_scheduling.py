# 排序後選「風險低、剩餘時間長、載入成本低、GPU 餘量足」的 node
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

READY_STATE = "ready"


@dataclass(frozen=True)
class NodeRiskScore:
    node_id: str
    state: str
    free_gpu: int
    total_gpu: int
    spot_risk: float
    remaining_lifetime_s: float
    loading_cost: float
    score: float
    reason: str = "risk_aware_ranking"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "free_gpu": self.free_gpu,
            "total_gpu": self.total_gpu,
            "spot_risk": self.spot_risk,
            "remaining_lifetime_s": self.remaining_lifetime_s,
            "loading_cost": self.loading_cost,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SchedulingDecision:
    action: str
    model_name: str
    requested_gpus: int
    selected_node_id: Optional[str]
    candidates: List[NodeRiskScore]
    reason: str = "risk_aware_scheduling"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "model_name": self.model_name,
            "requested_gpus": self.requested_gpus,
            "selected_node_id": self.selected_node_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


def _float_value(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: float,
) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return default


def _bounded(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def node_risk_score(
    node_id: str,
    node_info: Mapping[str, Any],
    requested_gpus: int,
    scheduler_config: Optional[Mapping[str, Any]] = None,
) -> NodeRiskScore:
    scheduler_config = scheduler_config or {}
    state = str(node_info.get("state", READY_STATE))
    free_gpu = int(node_info.get("free_gpu", 0) or 0)
    total_gpu = int(node_info.get("total_gpu", free_gpu) or 0)

    spot_risk = _bounded(
        _float_value(
            node_info,
            ("spot_risk", "risk_score", "preemption_risk"),
            float(scheduler_config.get("default_spot_risk", 0.0) or 0.0),
        ),
        0.0,
        1.0,
    )
    remaining_lifetime_s = max(
        0.0,
        _float_value(
            node_info,
            ("remaining_lifetime_s", "expected_remaining_lifetime_s"),
            float(
                scheduler_config.get(
                    "default_remaining_lifetime_s", 3600.0
                )
                or 0.0
            ),
        ),
    )
    loading_cost = max(
        0.0,
        _float_value(
            node_info,
            ("loading_cost", "model_loading_cost", "load_cost"),
            float(scheduler_config.get("default_loading_cost", 0.0) or 0.0),
        ),
    )

    max_remaining_lifetime_s = max(
        1.0,
        float(scheduler_config.get("max_remaining_lifetime_s", 3600.0) or 1.0),
    )
    max_loading_cost = max(
        1.0,
        float(scheduler_config.get("max_loading_cost", 60.0) or 1.0),
    )
    risk_weight = float(scheduler_config.get("risk_weight", 1.0) or 0.0)
    lifetime_weight = float(
        scheduler_config.get("lifetime_weight", 1.0) or 0.0
    )
    loading_cost_weight = float(
        scheduler_config.get("loading_cost_weight", 1.0) or 0.0
    )
    free_gpu_weight = float(
        scheduler_config.get("free_gpu_weight", 0.05) or 0.0
    )

    lifetime_penalty = 1.0 - _bounded(
        remaining_lifetime_s / max_remaining_lifetime_s,
        0.0,
        1.0,
    )
    loading_penalty = _bounded(loading_cost / max_loading_cost, 0.0, 1.0)
    free_gpu_headroom = max(free_gpu - requested_gpus, 0)
    free_gpu_bonus = free_gpu_headroom / max(total_gpu, requested_gpus, 1)

    score = (
        risk_weight * spot_risk
        + lifetime_weight * lifetime_penalty
        + loading_cost_weight * loading_penalty
        - free_gpu_weight * free_gpu_bonus
    )

    return NodeRiskScore(
        node_id=str(node_id),
        state=state,
        free_gpu=free_gpu,
        total_gpu=total_gpu,
        spot_risk=spot_risk,
        remaining_lifetime_s=remaining_lifetime_s,
        loading_cost=loading_cost,
        score=float(score),
    )


def rank_nodes_by_spot_risk(
    worker_nodes: Mapping[str, Mapping[str, Any]],
    requested_gpus: int,
    scheduler_config: Optional[Mapping[str, Any]] = None,
) -> List[NodeRiskScore]:
    candidates = []
    for node_id, node_info in worker_nodes.items():
        state = node_info.get("state", READY_STATE)
        if state != READY_STATE:
            continue
        if int(node_info.get("free_gpu", 0) or 0) < requested_gpus:
            continue
        candidates.append(
            node_risk_score(
                node_id=node_id,
                node_info=node_info,
                requested_gpus=requested_gpus,
                scheduler_config=scheduler_config,
            )
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.spot_risk,
            -candidate.remaining_lifetime_s,
            candidate.loading_cost,
            candidate.node_id,
        ),
    )


def plan_risk_aware_scheduling(
    model_name: str,
    worker_nodes: Mapping[str, Mapping[str, Any]],
    requested_gpus: int,
    scheduler_config: Optional[Mapping[str, Any]] = None,
) -> SchedulingDecision:
    candidates = rank_nodes_by_spot_risk(
        worker_nodes=worker_nodes,
        requested_gpus=requested_gpus,
        scheduler_config=scheduler_config,
    )
    selected_node_id = candidates[0].node_id if candidates else None
    return SchedulingDecision(
        action="allocate" if selected_node_id is not None else "no_capacity",
        model_name=model_name,
        requested_gpus=requested_gpus,
        selected_node_id=selected_node_id,
        candidates=candidates,
    )
