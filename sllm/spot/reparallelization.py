from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


READY = "ready"
PREEMPTING = "preempting"
DEAD = "dead"


@dataclass(frozen=True)
class GpuAvailability:
    total_gpus: int
    available_gpus: int
    unavailable_gpus: int
    ready_nodes: List[str]
    preempting_nodes: List[str]
    dead_nodes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_gpus": self.total_gpus,
            "available_gpus": self.available_gpus,
            "unavailable_gpus": self.unavailable_gpus,
            "ready_nodes": list(self.ready_nodes),
            "preempting_nodes": list(self.preempting_nodes),
            "dead_nodes": list(self.dead_nodes),
        }


@dataclass(frozen=True)
class ParallelPlan:
    model_name: str
    backend: str
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    num_replicas: int = 1
    num_gpus: int = 1
    target_nodes: List[str] = field(default_factory=list)
    reason: str = "replan"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParallelPlan":
        """Build a validated plan from a planner/controller response.

        The planner serialises plans before they cross the Ray actor boundary.
        Keeping deserialisation here prevents the deployment adapter from
        depending on the planner's internal dictionary shape.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("parallel plan must be a mapping")
        required = {
            "model_name",
            "backend",
            "tensor_parallel_size",
            "data_parallel_size",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                "parallel plan is missing required fields: "
                + ", ".join(missing)
            )
        return cls(
            model_name=str(payload["model_name"]),
            backend=str(payload["backend"]),
            tensor_parallel_size=max(1, int(payload["tensor_parallel_size"])),
            data_parallel_size=max(1, int(payload["data_parallel_size"])),
            pipeline_parallel_size=max(
                1, int(payload.get("pipeline_parallel_size", 1) or 1)
            ),
            expert_parallel_size=max(
                1, int(payload.get("expert_parallel_size", 1) or 1)
            ),
            num_replicas=max(
                1,
                int(
                    payload.get(
                        "num_replicas",
                        payload.get("data_parallel_size", 1),
                    )
                    or 1
                ),
            ),
            num_gpus=max(1, int(payload.get("num_gpus", 1) or 1)),
            target_nodes=[str(node) for node in payload.get("target_nodes", [])],
            reason=str(payload.get("reason", "replan")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "num_replicas": self.num_replicas,
            "num_gpus": self.num_gpus,
            "target_nodes": list(self.target_nodes),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    total_gpus: int
    unused_gpus: int
    score: float
    reason: str
    expert_parallel_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "total_gpus": self.total_gpus,
            "unused_gpus": self.unused_gpus,
            "score": self.score,
            "reason": self.reason,
        }


def _gpu_count(node_info: Mapping[str, Any]) -> int:
    return int(node_info.get("total_gpu", node_info.get("GPU", 0)) or 0)


def apply_spot_event_to_worker_nodes(
    worker_nodes: Mapping[str, Mapping[str, Any]],
    event: str,
    node_id: Optional[str] = None,
    node_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    nodes = {str(key): dict(value) for key, value in worker_nodes.items()}
    if node_id is None:
        return nodes
    node = nodes.setdefault(
        str(node_id),
        {
            "ray_node_id": None,
            "address": None,
            "free_gpu": 0,
            "total_gpu": 0,
        },
    )
    if node_info:
        # A capacity-add trace may carry the newly discovered GPU count and
        # address.  Preserve those fields while changing only the lifecycle
        # state below.
        node.update(dict(node_info))
    if event == "add":
        node.setdefault("free_gpu", node.get("total_gpu", 0))
        node.setdefault("total_gpu", node.get("free_gpu", 0))
        node["state"] = READY
    elif event == "remove":
        node["state"] = DEAD
        node["free_gpu"] = 0
    elif event == "preempt":
        node["state"] = PREEMPTING
    elif event == "recover":
        node["state"] = READY
    elif event == "dead":
        node["state"] = DEAD
    return nodes


def summarize_gpu_availability(
    worker_nodes: Mapping[str, Mapping[str, Any]]
) -> GpuAvailability:
    total_gpus = 0
    available_gpus = 0
    ready_nodes: List[str] = []
    preempting_nodes: List[str] = []
    dead_nodes: List[str] = []

    for node_id, node_info in sorted(worker_nodes.items()):
        state = node_info.get("state", READY)
        node_gpus = _gpu_count(node_info)
        total_gpus += node_gpus
        if state == READY:
            available_gpus += node_gpus
            ready_nodes.append(str(node_id))
        elif state == PREEMPTING:
            preempting_nodes.append(str(node_id))
        elif state == DEAD:
            dead_nodes.append(str(node_id))

    return GpuAvailability(
        total_gpus=total_gpus,
        available_gpus=available_gpus,
        unavailable_gpus=max(total_gpus - available_gpus, 0),
        ready_nodes=ready_nodes,
        preempting_nodes=preempting_nodes,
        dead_nodes=dead_nodes,
    )


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default) or default)
    return max(value, 1)


def generate_parallel_candidates(
    available_gpus: int,
    planner_config: Optional[Mapping[str, Any]] = None,
) -> List[ParallelConfig]:
    if available_gpus <= 0:
        return []
    planner_config = planner_config or {}
    max_tensor_parallel_size = min(
        _positive_int(
            planner_config, "max_tensor_parallel_size", available_gpus
        ),
        available_gpus,
    )
    max_pipeline_parallel_size = min(
        _positive_int(
            planner_config, "max_pipeline_parallel_size", available_gpus
        ),
        available_gpus,
    )
    min_data_parallel_size = _positive_int(
        planner_config, "min_data_parallel_size", 1
    )
    target_replica_gpus = _positive_int(
        planner_config, "target_replica_gpus", 1
    )

    candidates: List[ParallelConfig] = []
    for tensor_parallel_size in range(1, max_tensor_parallel_size + 1):
        for pipeline_parallel_size in range(1, max_pipeline_parallel_size + 1):
            replica_gpus = tensor_parallel_size * pipeline_parallel_size
            if replica_gpus > available_gpus:
                continue
            data_parallel_size = available_gpus // replica_gpus
            if data_parallel_size < min_data_parallel_size:
                continue
            total_gpus = replica_gpus * data_parallel_size
            unused_gpus = available_gpus - total_gpus
            replica_distance = abs(replica_gpus - target_replica_gpus)
            score = (
                total_gpus * 10000
                + data_parallel_size * 100
                - replica_distance * 1000
                - unused_gpus
            )
            candidates.append(
                ParallelConfig(
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    data_parallel_size=data_parallel_size,
                    expert_parallel_size=1,
                    total_gpus=total_gpus,
                    unused_gpus=unused_gpus,
                    score=float(score),
                    reason=(
                        "max_gpu_utilization_then_replica_count_"
                        "then_current_replica_shape"
                    ),
                )
            )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.data_parallel_size,
            candidate.tensor_parallel_size,
            -candidate.pipeline_parallel_size,
            -candidate.unused_gpus,
        ),
        reverse=True,
    )


def select_target_nodes(
    worker_nodes: Mapping[str, Mapping[str, Any]],
    required_gpus: int,
) -> List[str]:
    if required_gpus <= 0:
        return []

    selected_nodes: List[str] = []
    selected_gpus = 0
    ready_nodes = []
    for node_id, node_info in worker_nodes.items():
        if node_info.get("state", READY) != READY:
            continue
        ready_nodes.append((str(node_id), _gpu_count(node_info)))

    ready_nodes.sort(key=lambda item: (-item[1], item[0]))
    for node_id, node_gpus in ready_nodes:
        if node_gpus <= 0:
            continue
        selected_nodes.append(node_id)
        selected_gpus += node_gpus
        if selected_gpus >= required_gpus:
            break

    if selected_gpus < required_gpus:
        return []
    return selected_nodes


def _candidate_score(
    total_gpus: int,
    data_parallel_size: int,
    replica_gpus: int,
    target_replica_gpus: int,
    unused_gpus: int,
) -> float:
    replica_distance = abs(replica_gpus - target_replica_gpus)
    return float(
        total_gpus * 10000
        + data_parallel_size * 100
        - replica_distance * 1000
        - unused_gpus
    )


def _supported_config_candidates(
    backend_capability: Any,
    available_gpus: int,
    worker_nodes: Mapping[str, Mapping[str, Any]],
    planner_config: Mapping[str, Any],
) -> List[ParallelConfig]:
    supported_configs = getattr(
        backend_capability,
        "supported_configs",
        None,
    )
    if supported_configs is None and isinstance(backend_capability, Mapping):
        supported_configs = backend_capability.get("supported_configs")
    if not supported_configs:
        return []

    target_replica_gpus = _positive_int(
        planner_config, "target_replica_gpus", 1
    )
    candidates: List[ParallelConfig] = []
    for plan in supported_configs:
        if isinstance(plan, Mapping):
            tensor_parallel_size = int(
                plan.get("tensor_parallel_size", 1) or 1
            )
            pipeline_parallel_size = int(
                plan.get("pipeline_parallel_size", 1) or 1
            )
            data_parallel_size = int(plan.get("data_parallel_size", 1) or 1)
            expert_parallel_size = int(
                plan.get("expert_parallel_size", 1) or 1
            )
            total_gpus = int(plan.get("num_gpus", 1) or 1)
            reason = str(plan.get("reason", "backend_capability"))
        else:
            tensor_parallel_size = int(plan.tensor_parallel_size)
            pipeline_parallel_size = int(plan.pipeline_parallel_size)
            data_parallel_size = int(plan.data_parallel_size)
            expert_parallel_size = int(plan.expert_parallel_size)
            total_gpus = int(plan.num_gpus)
            reason = str(plan.reason)

        if total_gpus > available_gpus:
            continue
        if not select_target_nodes(worker_nodes, total_gpus):
            continue
        replica_gpus = tensor_parallel_size * pipeline_parallel_size
        unused_gpus = available_gpus - total_gpus
        candidates.append(
            ParallelConfig(
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                data_parallel_size=data_parallel_size,
                expert_parallel_size=expert_parallel_size,
                total_gpus=total_gpus,
                unused_gpus=unused_gpus,
                score=_candidate_score( #!! score
                    total_gpus=total_gpus,
                    data_parallel_size=data_parallel_size,
                    replica_gpus=replica_gpus,
                    target_replica_gpus=target_replica_gpus,
                    unused_gpus=unused_gpus,
                ),
                reason=reason,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.data_parallel_size,
            candidate.tensor_parallel_size,
            candidate.expert_parallel_size,
            -candidate.pipeline_parallel_size,
            -candidate.unused_gpus,
        ),
        reverse=True,
    )


def _has_supported_configs(backend_capability: Any) -> bool:
    supported_configs = getattr(
        backend_capability,
        "supported_configs",
        None,
    )
    if supported_configs is None and isinstance(backend_capability, Mapping):
        supported_configs = backend_capability.get("supported_configs")
    return bool(supported_configs)


def _get_backend_capability(model_config: Mapping[str, Any]):
    configured_capability = model_config.get("backend_capability")
    if configured_capability is not None:
        return configured_capability
    try:
        from sllm.backends.capability import get_backend_capability
    except Exception:
        return None
    try:
        return get_backend_capability(model_config)
    except Exception:
        return None


def build_parallel_plan(
    model_name: str,
    backend: str,
    parallel_config: ParallelConfig,
    worker_nodes: Mapping[str, Mapping[str, Any]],
    reason: str = "replan",
) -> ParallelPlan:
    return ParallelPlan(
        model_name=model_name,
        backend=backend,
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        data_parallel_size=parallel_config.data_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        expert_parallel_size=parallel_config.expert_parallel_size,
        num_replicas=parallel_config.data_parallel_size,
        num_gpus=parallel_config.total_gpus,
        target_nodes=select_target_nodes(
            worker_nodes, parallel_config.total_gpus
        ),
        reason=reason,
    )


def plan_dynamic_reparallelization(
    model_name: str,
    worker_nodes: Mapping[str, Mapping[str, Any]],
    model_config: Optional[Mapping[str, Any]] = None,
    planner_config: Optional[Mapping[str, Any]] = None,
    event: Optional[str] = None,
    node_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    model_config = model_config or {}
    backend_name = str(backend or model_config.get("backend", "unknown"))
    planner_config = dict(planner_config or {})
    if "target_replica_gpus" not in planner_config:
        planner_config["target_replica_gpus"] = max(
            int(
                planner_config.get("model_gpu_requirement")
                or model_config.get("num_gpus", 1)
                or 1
            ),
            1,
        )

    availability = summarize_gpu_availability(worker_nodes)
    backend_capability = _get_backend_capability(model_config)
    has_backend_capability_configs = _has_supported_configs(
        backend_capability
    )
    candidates = _supported_config_candidates(
        backend_capability=backend_capability,
        available_gpus=availability.available_gpus,
        worker_nodes=worker_nodes,
        planner_config=planner_config,
    )
    if not candidates and not has_backend_capability_configs:
        candidates = generate_parallel_candidates(
            availability.available_gpus, planner_config
        )
    selected = candidates[0] if candidates else None
    parallel_plan = (
        build_parallel_plan(
            model_name=model_name,
            backend=backend_name,
            parallel_config=selected,
            worker_nodes=worker_nodes,
            reason=f"{event or 'manual'}_replan",
        )
        if selected
        else None
    )
    action = "reparallelize" if selected is not None else "no_capacity"

    decision = {
        "model": model_name,
        "backend": backend_name,
        "event": event,
        "node_id": node_id,
        "instance_id": instance_id,
        "action": action,
        "candidate_count": len(candidates),
        "availability": availability.to_dict(),
        "parallel_plan": parallel_plan.to_dict() if parallel_plan else None,
        "selected_config": selected.to_dict() if selected else None,
        "top_candidates": [
            candidate.to_dict() for candidate in candidates[:5]
        ],
    }
    if selected is not None:
        decision.update(
            {
                "selected_total_gpus": selected.total_gpus,
                "selected_tensor_parallel_size": (
                    selected.tensor_parallel_size
                ),
                "selected_pipeline_parallel_size": (
                    selected.pipeline_parallel_size
                ),
                "selected_data_parallel_size": selected.data_parallel_size,
            }
        )
    else:
        decision.update(
            {
                "selected_total_gpus": 0,
                "selected_tensor_parallel_size": 0,
                "selected_pipeline_parallel_size": 0,
                "selected_data_parallel_size": 0,
            }
        )
    return decision
