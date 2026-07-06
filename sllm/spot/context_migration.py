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
    reusable_tokens_by_target: Mapping[str, int] = field(
        default_factory=dict
    )
    reusable_blocks_by_target: Mapping[str, int] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextMetadata":
        return cls(
            request_id=payload.get("request_id"),
            instance_id=str(payload["instance_id"]),
            node_id=str(payload["node_id"]),
            num_tokens=max(0, int(payload.get("num_tokens", 0) or 0)),
            context_blocks=max(0, int(payload.get("context_blocks", 0) or 0)),
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
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "num_tokens": self.num_tokens,
            "context_blocks": self.context_blocks,
            "reusable_tokens_by_target": dict(self.reusable_tokens_by_target),
            "reusable_blocks_by_target": dict(
                self.reusable_blocks_by_target
            ),
        }

@dataclass(frozen=True)
class MigrationTarget:
    instance_id: str
    node_id: str
    capacity: int = 1
    warmup_cost: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MigrationTarget":
        return cls(
            instance_id=str(payload["instance_id"]),
            node_id=str(payload["node_id"]),
            capacity=max(0, int(payload.get("capacity", 1) or 0)),
            warmup_cost=float(payload.get("warmup_cost", 0.0) or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "capacity": self.capacity,
            "warmup_cost": self.warmup_cost,
        }

@dataclass(frozen=True)
class MigrationPlan:
    request_id: Optional[str]
    old_instance_id: str
    new_instance_id: str
    old_node_id: str
    new_node_id: str
    estimated_cost: float
    reusable_tokens: int = 0
    reusable_context_blocks: int = 0
    reason: str = "low_cost_mapping"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "old_instance_id": self.old_instance_id,
            "new_instance_id": self.new_instance_id,
            "old_node_id": self.old_node_id,
            "new_node_id": self.new_node_id,
            "estimated_cost": self.estimated_cost,
            "reusable_tokens": self.reusable_tokens,
            "reusable_context_blocks": self.reusable_context_blocks,
            "reason": self.reason,
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


def estimate_migration_cost(
    source: ContextMetadata,
    target: MigrationTarget,
    planner_config: Optional[Mapping[str, Any]] = None,
    include_warmup: bool = True,
) -> Tuple[float, int, int]:
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
    token_cost = max(source.num_tokens - reusable_tokens, 0)
    block_cost = max(source.context_blocks - reusable_blocks, 0)
    estimated_cost = (
        base_migration_cost
        + token_cost * token_transfer_cost
        + block_cost * context_block_transfer_cost
    )
    if include_warmup:
        estimated_cost += target.warmup_cost
    if source.node_id != target.node_id:
        estimated_cost += cross_node_penalty
    return float(estimated_cost), reusable_tokens, reusable_blocks


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
            cost, _, _ = estimate_migration_cost(
                source,
                target,
                planner_config,
                include_warmup=False,
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

    for source_index, assigned_slot in enumerate(assignments):
        source = sources[source_index]
        if assigned_slot < 0:
            unassigned_contexts.append(source.to_dict())
            continue

        target = targets[assigned_slot]
        include_warmup = assigned_slot not in opened_targets
        estimated_cost, reusable_tokens, reusable_blocks = (
            estimate_migration_cost(
                source,
                target,
                planner_config,
                include_warmup=include_warmup,
            )
        )
        opened_targets.add(assigned_slot)
        plans.append(
            MigrationPlan(
                request_id=source.request_id,
                old_instance_id=source.instance_id,
                new_instance_id=target.instance_id,
                old_node_id=source.node_id,
                new_node_id=target.node_id,
                estimated_cost=estimated_cost,
                reusable_tokens=reusable_tokens,
                reusable_context_blocks=reusable_blocks,
            )
        )

    total_context_tokens = sum(source.num_tokens for source in sources)
    total_context_blocks = sum(source.context_blocks for source in sources)
    total_reusable_tokens = sum(plan.reusable_tokens for plan in plans)
    total_reusable_blocks = sum(
        plan.reusable_context_blocks for plan in plans
    )
    total_estimated_cost = sum(plan.estimated_cost for plan in plans)
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
