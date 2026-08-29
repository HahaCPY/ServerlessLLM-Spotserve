from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ContextMetadata:
    request_id: Optional[str]
    instance_id: str
    node_id: str
    num_tokens: int = 0
    context_blocks: int = 0
    tokens: Tuple[int, ...] = ()
    cache_block_size: int = 0
    cache_dtype: str = ""
    cache_layout: str = ""
    reusable_tokens_by_target: Mapping[str, int] = field(
        default_factory=dict
    )
    reusable_blocks_by_target: Mapping[str, int] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextMetadata":
        return cls(
            request_id=payload.get("request_id"),
            instance_id=str(payload["instance_id"]),
            node_id=str(payload["node_id"]),
            num_tokens=max(0, int(payload.get("num_tokens", 0) or 0)),
            context_blocks=max(0, int(payload.get("context_blocks", 0) or 0)),
            tokens=tuple(
                int(token) for token in (payload.get("tokens", []) or [])
            ),
            cache_block_size=max(
                0, int(payload.get("cache_block_size", 0) or 0)
            ),
            cache_dtype=str(payload.get("cache_dtype", "") or ""),
            cache_layout=str(payload.get("cache_layout", "") or ""),
            reusable_tokens_by_target={
                str(key): int(value)
                for key, value in (
                    payload.get("reusable_tokens_by_target", {}) or {}
                ).items()
            },
            reusable_blocks_by_target={
                str(key): int(value)
                for key, value in (
                    payload.get("reusable_blocks_by_target", {}) or {}
                ).items()
            },
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "num_tokens": self.num_tokens,
            "context_blocks": self.context_blocks,
            "tokens": list(self.tokens),
            "cache_block_size": self.cache_block_size,
            "cache_dtype": self.cache_dtype,
            "cache_layout": self.cache_layout,
            "reusable_tokens_by_target": dict(self.reusable_tokens_by_target),
            "reusable_blocks_by_target": dict(
                self.reusable_blocks_by_target
            ),
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class MigrationTarget:
    instance_id: str
    node_id: str
    capacity: int = 1
    warmup_cost: float = 0.0
    concurrency: int = 0
    max_queue_length: int = 0
    queue_depth: Optional[int] = None
    queue_penalty: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MigrationTarget":
        return cls(
            instance_id=str(payload["instance_id"]),
            node_id=str(payload["node_id"]),
            capacity=max(0, int(payload.get("capacity", 1) or 0)),
            warmup_cost=float(payload.get("warmup_cost", 0.0) or 0.0),
            concurrency=max(0, int(payload.get("concurrency", 0) or 0)),
            max_queue_length=max(
                0, int(payload.get("max_queue_length", 0) or 0)
            ),
            queue_depth=_as_non_negative_int(payload.get("queue_depth")),
            queue_penalty=float(payload.get("queue_penalty", 0.0) or 0.0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "capacity": self.capacity,
            "warmup_cost": self.warmup_cost,
            "concurrency": self.concurrency,
            "max_queue_length": self.max_queue_length,
            "queue_depth": self.queue_depth,
            "queue_penalty": self.queue_penalty,
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class MigrationPlan:
    request_id: Optional[str]
    old_instance_id: str
    new_instance_id: str
    old_node_id: str
    new_node_id: str
    estimated_cost: float
    kv_migration_cost: float = 0.0
    reusable_tokens: int = 0
    reusable_context_blocks: int = 0
    reason: str = "low_cost_mapping"
    expert_locality_available: bool = False
    hot_expert_locality_ratio: float = 0.0
    estimated_remote_routing_ratio: float = 0.0
    estimated_remote_routed_tokens: int = 0
    expert_dispatch_cost: float = 0.0
    queue_depth: int = 0
    queue_pressure: float = 0.0
    queue_penalty_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "old_instance_id": self.old_instance_id,
            "new_instance_id": self.new_instance_id,
            "old_node_id": self.old_node_id,
            "new_node_id": self.new_node_id,
            "estimated_cost": self.estimated_cost,
            "kv_migration_cost": self.kv_migration_cost,
            "reusable_tokens": self.reusable_tokens,
            "reusable_context_blocks": self.reusable_context_blocks,
            "reason": self.reason,
            "expert_locality_available": self.expert_locality_available,
            "hot_expert_locality_ratio": self.hot_expert_locality_ratio,
            "estimated_remote_routing_ratio": (
                self.estimated_remote_routing_ratio
            ),
            "estimated_remote_routed_tokens": (
                self.estimated_remote_routed_tokens
            ),
            "expert_dispatch_cost": self.expert_dispatch_cost,
            "queue_depth": self.queue_depth,
            "queue_pressure": self.queue_pressure,
            "queue_penalty_cost": self.queue_penalty_cost,
        }

@dataclass(frozen=True)
class MigrationDecision:
    action: str
    plans: List[MigrationPlan]
    unassigned_contexts: List[Dict[str, Any]]
    total_estimated_cost: float
    total_reusable_tokens: int
    total_context_tokens: int
    total_reusable_context_blocks: int
    total_context_blocks: int
    reuse_ratio: float
    cost_matrix: List[List[float]]
    total_kv_migration_cost: float = 0.0
    moe_route_histogram_available_count: int = 0
    moe_target_placement_available_count: int = 0
    total_estimated_remote_routed_tokens: int = 0
    total_expert_dispatch_cost: float = 0.0
    total_queue_penalty_cost: float = 0.0
    avg_queue_pressure: float = 0.0
    max_queue_depth: int = 0
    avg_hot_expert_locality_ratio: float = 0.0
    avg_estimated_remote_routing_ratio: float = 0.0
    moe_route_histogram_source: str = "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "plans": [plan.to_dict() for plan in self.plans],
            "unassigned_contexts": list(self.unassigned_contexts),
            "total_estimated_cost": self.total_estimated_cost,
            "total_reusable_tokens": self.total_reusable_tokens,
            "total_context_tokens": self.total_context_tokens,
            "total_reusable_context_blocks": (
                self.total_reusable_context_blocks
            ),
            "total_context_blocks": self.total_context_blocks,
            "reuse_ratio": self.reuse_ratio,
            "cost_matrix": [list(row) for row in self.cost_matrix],
            "total_kv_migration_cost": self.total_kv_migration_cost,
            "moe_route_histogram_available_count": (
                self.moe_route_histogram_available_count
            ),
            "moe_target_placement_available_count": (
                self.moe_target_placement_available_count
            ),
            "total_estimated_remote_routed_tokens": (
                self.total_estimated_remote_routed_tokens
            ),
            "total_expert_dispatch_cost": self.total_expert_dispatch_cost,
            "total_queue_penalty_cost": self.total_queue_penalty_cost,
            "avg_queue_pressure": self.avg_queue_pressure,
            "max_queue_depth": self.max_queue_depth,
            "avg_hot_expert_locality_ratio": (
                self.avg_hot_expert_locality_ratio
            ),
            "avg_estimated_remote_routing_ratio": (
                self.avg_estimated_remote_routing_ratio
            ),
            "moe_route_histogram_source": self.moe_route_histogram_source,
        }


def _positive_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = config.get(key, default)
    if value is None:
        value = default
    return max(0.0, float(value))


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _expert_key_from_parts(parts: Sequence[Any]) -> str:
    cleaned = [str(part) for part in parts if str(part) not in {"", "None"}]
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and all(part.isdigit() for part in cleaned[-2:]):
        return f"layer:{cleaned[-2]}/expert:{cleaned[-1]}"
    if len(cleaned) == 1:
        return cleaned[0]
    return "/".join(cleaned)


def _flatten_expert_histogram(
    value: Any,
    request_id: Optional[str] = None,
) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    if request_id and isinstance(value.get(request_id), Mapping):
        value = value[request_id]

    histogram: Dict[str, int] = {}

    def visit(node: Any, parts: Tuple[Any, ...]) -> None:
        parsed = _as_non_negative_int(node)
        if parsed is not None:
            key = _expert_key_from_parts(parts)
            if key and parsed > 0:
                histogram[key] = histogram.get(key, 0) + parsed
            return
        if isinstance(node, Mapping):
            if "layer_id" in node and "expert_id" in node:
                routed_tokens = _as_non_negative_int(
                    node.get("routed_tokens", node.get("tokens", 0))
                )
                if routed_tokens:
                    key = _expert_key_from_parts(
                        (node["layer_id"], node["expert_id"])
                    )
                    histogram[key] = histogram.get(key, 0) + routed_tokens
                return
            for key, child in node.items():
                visit(child, parts + (key,))
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, parts + (index,))

    visit(value, ())
    return histogram


def _normalize_expert_key(value: Any) -> str:
    text = str(value)
    if "layer:" in text and "expert:" in text:
        return text
    parts = [part for part in text.replace("/", ":").split(":") if part]
    if len(parts) >= 2 and all(part.isdigit() for part in parts[-2:]):
        return _expert_key_from_parts(parts[-2:])
    return text


def _placement_expert_keys(value: Any) -> set[str]:
    keys: set[str] = set()

    def visit(node: Any, parent_key: Optional[str] = None) -> None:
        if isinstance(node, Mapping):
            if "layer_id" in node and "expert_id" in node:
                keys.add(_expert_key_from_parts((node["layer_id"], node["expert_id"])))
            elif parent_key is not None:
                keys.add(_normalize_expert_key(parent_key))
            for key, child in node.items():
                if "expert" in str(key) or "layer" in str(key):
                    keys.add(_normalize_expert_key(key))
                if isinstance(child, (Mapping, list, tuple)):
                    visit(child, str(key))
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return {key for key in keys if key}


def _metadata_value(
    item: ContextMetadata | MigrationTarget,
    *keys: str,
) -> Any:
    metadata = dict(item.metadata or {})
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def source_expert_route_histogram(
    source: ContextMetadata,
) -> Dict[str, int]:
    raw = _metadata_value(
        source,
        "per_request_expert_route_histogram",
    )
    return _flatten_expert_histogram(raw, request_id=source.request_id)


def source_has_available_expert_route_histogram(
    source: ContextMetadata,
) -> bool:
    histogram = source_expert_route_histogram(source)
    if not histogram:
        return False
    return _to_bool(
        _metadata_value(source, "moe_route_histogram_available"),
        default=True,
    )


def target_expert_placement_keys(target: MigrationTarget) -> set[str]:
    raw = _metadata_value(
        target,
        "expert_placement_snapshot",
        "expert_placement",
    )
    return _placement_expert_keys(raw)


def estimate_expert_dispatch_cost(
    source: ContextMetadata,
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    planner_config = planner_config or {}
    enabled = _to_bool(
        planner_config.get("enable_moe_expert_locality"),
        default=("expert_dispatch_weight" in planner_config),
    )
    histogram = source_expert_route_histogram(source)
    placement_keys = target_expert_placement_keys(target)
    histogram_available = bool(histogram)
    placement_available = bool(placement_keys)
    source_declares_histogram = _to_bool(
        _metadata_value(source, "moe_route_histogram_available"),
        default=histogram_available,
    )
    target_declares_placement = _to_bool(
        _metadata_value(target, "expert_placement_available"),
        default=placement_available,
    )
    if (
        not enabled
        or not histogram_available
        or not placement_available
        or not source_declares_histogram
        or not target_declares_placement
    ):
        return {
            "available": False,
            "histogram_available": histogram_available,
            "placement_available": placement_available,
            "locality_ratio": 0.0,
            "estimated_remote_routing_ratio": 0.0,
            "estimated_remote_routed_tokens": 0,
            "cost": 0.0,
            "route_histogram_source": str(
                _metadata_value(source, "moe_route_histogram_source")
                or "unavailable"
            ),
        }

    total_tokens = sum(histogram.values())
    local_tokens = sum(
        tokens
        for expert_key, tokens in histogram.items()
        if _normalize_expert_key(expert_key) in placement_keys
    )
    locality_ratio = local_tokens / total_tokens if total_tokens else 0.0
    remote_tokens = max(total_tokens - local_tokens, 0)
    remote_ratio = 1.0 - locality_ratio if total_tokens else 0.0
    weight = _positive_float(planner_config, "expert_dispatch_weight", 1.0)
    cost = remote_ratio * weight
    return {
        "available": True,
        "histogram_available": True,
        "placement_available": True,
        "locality_ratio": float(locality_ratio),
        "estimated_remote_routing_ratio": float(remote_ratio),
        "estimated_remote_routed_tokens": int(remote_tokens),
        "cost": float(cost),
        "route_histogram_source": str(
            _metadata_value(source, "moe_route_histogram_source")
            or "runtime_or_instrumentation"
        ),
    }


def estimate_queue_penalty_cost(
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
    planned_requests_ahead: int = 0,
) -> Dict[str, Any]:
    """Estimate target-side queue pressure for a migration assignment.

    The planner still uses target capacity as a hard filter.  This adds a soft
    penalty among targets that can accept the request, so a nearly full target
    is less attractive than an otherwise equivalent idle target.
    """
    planner_config = planner_config or {}
    current_depth = (
        target.queue_depth
        if target.queue_depth is not None
        else target.concurrency
    )
    queue_depth = max(0, int(current_depth or 0)) + max(
        0, int(planned_requests_ahead or 0)
    )
    queue_capacity = target.max_queue_length
    if queue_capacity <= 0 and target.capacity > 0:
        queue_capacity = queue_depth + target.capacity
    queue_pressure = (
        min(1.0, queue_depth / queue_capacity)
        if queue_capacity > 0
        else 0.0
    )
    queue_depth_weight = _positive_float(
        planner_config, "queue_penalty_weight", 0.0
    )
    queue_pressure_weight = _positive_float(
        planner_config, "queue_pressure_weight", 0.0
    )
    explicit_penalty = max(0.0, float(target.queue_penalty or 0.0))
    cost = (
        explicit_penalty
        + queue_depth * queue_depth_weight
        + queue_pressure * queue_pressure_weight
    )
    return {
        "queue_depth": queue_depth,
        "queue_capacity": max(0, int(queue_capacity or 0)),
        "queue_pressure": float(queue_pressure),
        "cost": float(cost),
    }


def _reuse_value(
    source: ContextMetadata,
    target: MigrationTarget,
    values_by_target: Mapping[str, int],
    total_value: int,
    same_node_ratio: float,
    cross_node_ratio: float,
) -> int:
    if target.instance_id in values_by_target:
        return max(0, min(total_value, int(values_by_target[target.instance_id])))
    if target.node_id in values_by_target:
        return max(0, min(total_value, int(values_by_target[target.node_id])))

    ratio = (
        same_node_ratio
        if source.node_id == target.node_id
        else cross_node_ratio
    )
    return max(0, min(total_value, int(round(total_value * ratio))))


def reusable_context(
    source: ContextMetadata,
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, int]:
    planner_config = planner_config or {}
    same_node_token_reuse_ratio = _positive_float(
        planner_config, "same_node_token_reuse_ratio", 1.0
    )
    cross_node_token_reuse_ratio = _positive_float(
        planner_config, "cross_node_token_reuse_ratio", 0.0
    )
    same_node_block_reuse_ratio = _positive_float(
        planner_config, "same_node_block_reuse_ratio", 1.0
    )
    cross_node_block_reuse_ratio = _positive_float(
        planner_config, "cross_node_block_reuse_ratio", 0.0
    )

    reusable_tokens = _reuse_value(
        source,
        target,
        source.reusable_tokens_by_target,
        source.num_tokens,
        same_node_token_reuse_ratio,
        cross_node_token_reuse_ratio,
    )
    reusable_blocks = _reuse_value(
        source,
        target,
        source.reusable_blocks_by_target,
        source.context_blocks,
        same_node_block_reuse_ratio,
        cross_node_block_reuse_ratio,
    )
    return reusable_tokens, reusable_blocks


def estimate_kv_migration_cost(
    source: ContextMetadata,
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
    include_warmup: bool = True,
) -> Dict[str, Any]:
    planner_config = planner_config or {}
    token_transfer_cost = _positive_float(
        planner_config, "token_transfer_cost", 1.0
    )
    context_block_transfer_cost = _positive_float(
        planner_config, "context_block_transfer_cost", 4.0
    )
    base_migration_cost = _positive_float(
        planner_config, "base_migration_cost", 0.0
    )
    cross_node_penalty = _positive_float(
        planner_config, "cross_node_penalty", 10.0
    )

    reusable_tokens, reusable_blocks = reusable_context(
        source, target, planner_config
    )
    non_reusable_tokens = max(source.num_tokens - reusable_tokens, 0)
    non_reusable_blocks = max(source.context_blocks - reusable_blocks, 0)
    token_cost = non_reusable_tokens * token_transfer_cost
    block_cost = non_reusable_blocks * context_block_transfer_cost
    warmup_cost = target.warmup_cost if include_warmup else 0.0
    cross_node_cost = (
        cross_node_penalty if source.node_id != target.node_id else 0.0
    )
    cost = (
        base_migration_cost
        + token_cost
        + block_cost
        + warmup_cost
        + cross_node_cost
    )
    return {
        "cost": float(cost),
        "base_migration_cost": float(base_migration_cost),
        "token_migration_cost": float(token_cost),
        "context_block_migration_cost": float(block_cost),
        "warmup_cost": float(warmup_cost),
        "cross_node_penalty_cost": float(cross_node_cost),
        "reusable_tokens": int(reusable_tokens),
        "reusable_context_blocks": int(reusable_blocks),
        "non_reusable_tokens": int(non_reusable_tokens),
        "non_reusable_context_blocks": int(non_reusable_blocks),
    }


def estimate_migration_cost(
    source: ContextMetadata,
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
    include_warmup: bool = True,
    planned_requests_ahead: int = 0,
) -> Tuple[float, int, int]:
    planner_config = planner_config or {}
    kv_migration = estimate_kv_migration_cost(
        source,
        target,
        planner_config,
        include_warmup=include_warmup,
    )
    estimated_cost = float(kv_migration["cost"])
    expert_dispatch = estimate_expert_dispatch_cost(
        source, target, planner_config
    )
    estimated_cost += float(expert_dispatch.get("cost", 0.0) or 0.0)
    queue_penalty = estimate_queue_penalty_cost(
        target,
        planner_config,
        planned_requests_ahead=planned_requests_ahead,
    )
    estimated_cost += float(queue_penalty.get("cost", 0.0) or 0.0)
    return (
        float(estimated_cost),
        int(kv_migration["reusable_tokens"]),
        int(kv_migration["reusable_context_blocks"]),
    )


def build_cost_matrix(
    sources: Sequence[ContextMetadata],
    targets: Sequence[MigrationTarget],
    planner_config: Optional[Mapping[str, Any]] = None,
    include_warmup: bool = True,
) -> Tuple[List[List[float]], List[MigrationTarget]]:
    target_slots: List[MigrationTarget] = []
    for target in targets:
        for _ in range(target.capacity):
            target_slots.append(target)

    cost_matrix: List[List[float]] = []
    for source in sources:
        row = []
        for target in target_slots:
            estimated_cost, _, _ = estimate_migration_cost(
                source, target, planner_config, include_warmup=include_warmup
            )
            row.append(estimated_cost)
        cost_matrix.append(row)
    return cost_matrix, target_slots


def build_candidate_component_costs(
    sources: Sequence[ContextMetadata],
    targets: Sequence[MigrationTarget],
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    planner_config = planner_config or {}
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source in sources:
        source_key = str(source.request_id or source.instance_id)
        rows[source_key] = {}
        for target in targets:
            kv_migration = estimate_kv_migration_cost(
                source,
                target,
                planner_config,
                include_warmup=True,
            )
            expert_dispatch = estimate_expert_dispatch_cost(
                source,
                target,
                planner_config,
            )
            queue_penalty = estimate_queue_penalty_cost(
                target,
                planner_config,
                planned_requests_ahead=0,
            )
            rows[source_key][target.instance_id] = {
                "total_estimated_cost": float(
                    kv_migration["cost"]
                    + float(expert_dispatch.get("cost", 0.0) or 0.0)
                    + float(queue_penalty.get("cost", 0.0) or 0.0)
                ),
                "kv_migration_cost": float(kv_migration["cost"]),
                "expert_dispatch_cost": float(
                    expert_dispatch.get("cost", 0.0) or 0.0
                ),
                "queue_penalty_cost": float(
                    queue_penalty.get("cost", 0.0) or 0.0
                ),
                "reusable_tokens": int(kv_migration["reusable_tokens"]),
                "reusable_context_blocks": int(
                    kv_migration["reusable_context_blocks"]
                ),
                "hot_expert_locality_ratio": float(
                    expert_dispatch.get("locality_ratio", 0.0) or 0.0
                ),
                "estimated_remote_routing_ratio": float(
                    expert_dispatch.get(
                        "estimated_remote_routing_ratio", 0.0
                    )
                    or 0.0
                ),
                "estimated_remote_routed_tokens": int(
                    expert_dispatch.get(
                        "estimated_remote_routed_tokens", 0
                    )
                    or 0
                ),
                "queue_depth": int(queue_penalty.get("queue_depth", 0) or 0),
                "queue_pressure": float(
                    queue_penalty.get("queue_pressure", 0.0) or 0.0
                ),
                "expert_locality_available": bool(
                    expert_dispatch.get("available", False)
                ),
            }
    return rows


def _fixed_warmup_assignment(
    sources: Sequence[ContextMetadata],
    targets: Sequence[MigrationTarget],
    planner_config: Mapping[str, Any],
    unmatched_penalty: float,
) -> List[int]:
    capacities = tuple(target.capacity for target in targets)

    @lru_cache(maxsize=None)
    def solve(
        source_index: int,
        remaining_capacities: Tuple[int, ...],
        opened_targets: int,
    ) -> Tuple[float, Tuple[int, ...]]:
        if source_index >= len(sources):
            return 0.0, ()

        best_cost, best_choices = solve(
            source_index + 1,
            remaining_capacities,
            opened_targets,
        )
        best_cost += unmatched_penalty
        best_choices = (-1,) + best_choices

        source = sources[source_index]
        for target_index, target in enumerate(targets):
            if remaining_capacities[target_index] <= 0:
                continue
            planned_requests_ahead = (
                capacities[target_index] - remaining_capacities[target_index]
            )
            cost, _, _ = estimate_migration_cost(
                source,
                target,
                planner_config,
                include_warmup=False,
                planned_requests_ahead=planned_requests_ahead,
            )
            target_mask = 1 << target_index
            if not opened_targets & target_mask:
                cost += target.warmup_cost

            next_capacities = list(remaining_capacities)
            next_capacities[target_index] -= 1
            rest_cost, rest_choices = solve(
                source_index + 1,
                tuple(next_capacities),
                opened_targets | target_mask,
            )
            total_cost = cost + rest_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_choices = (target_index,) + rest_choices

        return best_cost, best_choices

    _, choices = solve(0, capacities, 0)
    return list(choices)


def plan_low_cost_migration(
    sources: Sequence[ContextMetadata],
    targets: Sequence[MigrationTarget],
    planner_config: Optional[Mapping[str, Any]] = None,
) -> MigrationDecision:
    planner_config = planner_config or {}
    cost_matrix, _ = build_cost_matrix(
        sources,
        targets,
        planner_config,
        include_warmup=False,
    )
    unmatched_penalty = _positive_float(
        planner_config, "unmatched_penalty", 1_000_000.0
    )
    assignments = _fixed_warmup_assignment(
        sources=sources,
        targets=targets,
        planner_config=planner_config,
        unmatched_penalty=unmatched_penalty,
    )
    plans: List[MigrationPlan] = []
    unassigned_contexts: List[Dict[str, Any]] = []
    opened_targets = set()
    assigned_counts_by_target: Dict[int, int] = {}

    for source_index, assigned_slot in enumerate(assignments):
        source = sources[source_index]
        if assigned_slot < 0:
            unassigned_contexts.append(source.to_dict())
            continue

        target = targets[assigned_slot]
        include_warmup = assigned_slot not in opened_targets
        planned_requests_ahead = assigned_counts_by_target.get(
            assigned_slot, 0
        )
        estimated_cost, reusable_tokens, reusable_blocks = (
            estimate_migration_cost(
                source,
                target,
                planner_config,
                include_warmup=include_warmup,
                planned_requests_ahead=planned_requests_ahead,
            )
        )
        opened_targets.add(assigned_slot)
        assigned_counts_by_target[assigned_slot] = planned_requests_ahead + 1
        expert_dispatch = estimate_expert_dispatch_cost(
            source, target, planner_config
        )
        kv_migration = estimate_kv_migration_cost(
            source,
            target,
            planner_config,
            include_warmup=include_warmup,
        )
        queue_penalty = estimate_queue_penalty_cost(
            target,
            planner_config,
            planned_requests_ahead=planned_requests_ahead,
        )
        plans.append(
            MigrationPlan(
                request_id=source.request_id,
                old_instance_id=source.instance_id,
                new_instance_id=target.instance_id,
                old_node_id=source.node_id,
                new_node_id=target.node_id,
                estimated_cost=estimated_cost,
                kv_migration_cost=float(kv_migration.get("cost", 0.0) or 0.0),
                reusable_tokens=reusable_tokens,
                reusable_context_blocks=reusable_blocks,
                reason=(
                    "kv_and_expert_locality"
                    if expert_dispatch.get("available")
                    else "low_cost_mapping"
                ),
                expert_locality_available=bool(
                    expert_dispatch.get("available", False)
                ),
                hot_expert_locality_ratio=float(
                    expert_dispatch.get("locality_ratio", 0.0) or 0.0
                ),
                estimated_remote_routing_ratio=float(
                    expert_dispatch.get(
                        "estimated_remote_routing_ratio", 0.0
                    )
                    or 0.0
                ),
                estimated_remote_routed_tokens=int(
                    expert_dispatch.get(
                        "estimated_remote_routed_tokens", 0
                    )
                    or 0
                ),
                expert_dispatch_cost=float(
                    expert_dispatch.get("cost", 0.0) or 0.0
                ),
                queue_depth=int(queue_penalty.get("queue_depth", 0) or 0),
                queue_pressure=float(
                    queue_penalty.get("queue_pressure", 0.0) or 0.0
                ),
                queue_penalty_cost=float(
                    queue_penalty.get("cost", 0.0) or 0.0
                ),
            )
        )

    total_context_tokens = sum(source.num_tokens for source in sources)
    total_context_blocks = sum(source.context_blocks for source in sources)
    total_reusable_tokens = sum(plan.reusable_tokens for plan in plans)
    total_reusable_blocks = sum(
        plan.reusable_context_blocks for plan in plans
    )
    total_estimated_cost = sum(plan.estimated_cost for plan in plans)
    total_kv_migration_cost = sum(
        plan.kv_migration_cost for plan in plans
    )
    total_expert_dispatch_cost = sum(
        plan.expert_dispatch_cost for plan in plans
    )
    total_queue_penalty_cost = sum(
        plan.queue_penalty_cost for plan in plans
    )
    queue_pressures = [plan.queue_pressure for plan in plans]
    max_queue_depth = max(
        (plan.queue_depth for plan in plans),
        default=0,
    )
    total_estimated_remote_routed_tokens = sum(
        plan.estimated_remote_routed_tokens for plan in plans
    )
    locality_ratios = [
        plan.hot_expert_locality_ratio
        for plan in plans
        if plan.expert_locality_available
    ]
    remote_ratios = [
        plan.estimated_remote_routing_ratio
        for plan in plans
        if plan.expert_locality_available
    ]
    route_histogram_available_count = sum(
        1 for source in sources if source_has_available_expert_route_histogram(source)
    )
    target_placement_available_count = sum(
        1 for target in targets if target_expert_placement_keys(target)
    )
    route_sources = sorted(
        {
            str(_metadata_value(source, "moe_route_histogram_source") or "")
            for source in sources
            if source_has_available_expert_route_histogram(source)
        }
    )
    reuse_denominator = total_context_blocks or total_context_tokens
    reuse_numerator = (
        total_reusable_blocks if total_context_blocks else total_reusable_tokens
    )
    reuse_ratio = (
        reuse_numerator / reuse_denominator if reuse_denominator else 0.0
    )

    if not sources:
        action = "no_context"
    elif plans:
        action = "migrate"
    else:
        action = "no_target_capacity"

    return MigrationDecision(
        action=action,
        plans=plans,
        unassigned_contexts=unassigned_contexts,
        total_estimated_cost=float(total_estimated_cost),
        total_reusable_tokens=total_reusable_tokens,
        total_context_tokens=total_context_tokens,
        total_reusable_context_blocks=total_reusable_blocks,
        total_context_blocks=total_context_blocks,
        reuse_ratio=float(reuse_ratio),
        cost_matrix=cost_matrix,
        total_kv_migration_cost=float(total_kv_migration_cost),
        moe_route_histogram_available_count=route_histogram_available_count,
        moe_target_placement_available_count=target_placement_available_count,
        total_estimated_remote_routed_tokens=(
            total_estimated_remote_routed_tokens
        ),
        total_expert_dispatch_cost=float(total_expert_dispatch_cost),
        total_queue_penalty_cost=float(total_queue_penalty_cost),
        avg_queue_pressure=(
            sum(queue_pressures) / len(queue_pressures)
            if queue_pressures
            else 0.0
        ),
        max_queue_depth=max_queue_depth,
        avg_hot_expert_locality_ratio=(
            sum(locality_ratios) / len(locality_ratios)
            if locality_ratios
            else 0.0
        ),
        avg_estimated_remote_routing_ratio=(
            sum(remote_ratios) / len(remote_ratios)
            if remote_ratios
            else 0.0
        ),
        moe_route_histogram_source=(
            ",".join(source for source in route_sources if source)
            or "unavailable"
        ),
    )


def plan_low_cost_migration_from_dict(
    payload: Mapping[str, Any],
) -> MigrationDecision:
    sources = [
        ContextMetadata.from_dict(row) for row in payload.get("sources", [])
    ]
    targets = [
        MigrationTarget.from_dict(row) for row in payload.get("targets", [])
    ]
    return plan_low_cost_migration(
        sources=sources,
        targets=targets,
        planner_config=payload.get("planner_config", {}),
    )
