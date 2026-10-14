import json
import math
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ExpertShard:
    layer_id: int
    expert_id: int
    rank_id: str
    node_id: str
    gpu_id: str
    physical_expert_id: Optional[int] = None
    weight_size_bytes: Optional[int] = None
    weight_resident: bool = True
    routed_tokens: int = 0
    recent_execution_count: int = 0
    load_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "physical_expert_id": self.physical_expert_id,
            "rank_id": self.rank_id,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "weight_size_bytes": self.weight_size_bytes,
            "weight_resident": self.weight_resident,
            "routed_tokens": self.routed_tokens,
            "recent_execution_count": self.recent_execution_count,
            "load_score": self.load_score,
        }


@dataclass(frozen=True)
class ExpertPlacementState:
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    vllm_data_parallel_size: int
    sllm_replica_count: int
    expert_parallel_enabled: bool
    effective_expert_parallel_size: int
    expert_parallel_size_source: str
    expert_physical_replication_factor: int = 1
    placement_epoch: int = 0
    placement_source: str = "unavailable"
    shards: tuple[ExpertShard, ...] = field(default_factory=tuple)

    @property
    def placement_available(self) -> bool:
        return bool(self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "vllm_data_parallel_size": self.vllm_data_parallel_size,
            "sllm_replica_count": self.sllm_replica_count,
            "expert_parallel_enabled": self.expert_parallel_enabled,
            "effective_expert_parallel_size": (
                self.effective_expert_parallel_size
            ),
            "expert_parallel_size_source": self.expert_parallel_size_source,
            "expert_physical_replication_factor": (
                self.expert_physical_replication_factor
            ),
            "expert_placement_available": self.placement_available,
            "placement_epoch": self.placement_epoch,
            "placement_source": self.placement_source,
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class ExpertPlacementPlan:
    model_name: str
    target_parallel_plan: Mapping[str, Any]
    expert_to_target_rank: Mapping[str, str] = field(default_factory=dict)
    expert_to_target_ranks: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    placement_epoch: int = 0
    placement_source: str = "unavailable"
    placement_fingerprint: str = ""
    required_expert_count: int = 0
    covered_expert_count: int = 0
    planned_shard_count: int = 0
    target_rank_count: int = 0
    expert_physical_replication_factor: int = 1
    sllm_replica_count: int = 1
    physical_weight_migration: bool = False
    live_expert_remap: bool = False
    allow_active_requests: bool = False
    require_cross_node: bool = False
    physical_migration_required: bool = False
    expert_placement_snapshot: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    shards: tuple[ExpertShard, ...] = field(default_factory=tuple)
    movement_observation_available: bool = False
    movement_source: str = "unavailable"
    moved_expert_count: int = 0
    stationary_expert_count: int = 0
    unknown_movement_expert_count: int = 0
    moved_weight_bytes: Optional[int] = None
    estimated_expert_weight_movement_cost_ms: Optional[float] = None
    expert_movement_diff: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    estimated_dispatch_cost: float = 0.0
    estimated_load_balance_penalty: float = 0.0
    placement_strategy: str = "linear"
    selection_reason: str = "configured_strategy"
    hotness_snapshot_id: str = ""
    hotness_snapshot_available: bool = False
    hotness_snapshot_reason: str = "not_requested"
    hotness_snapshot: Mapping[str, Any] = field(default_factory=dict)
    per_layer_rank_load: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    per_layer_load_imbalance: Mapping[str, float] = field(
        default_factory=dict
    )
    per_layer_load_cv: Mapping[str, float] = field(default_factory=dict)
    hotness_layer_coverage: float = 0.0
    predicted_improvement_ratio: float = 0.0
    placement_objective: Optional[float] = None
    source_rank_locality_available: bool = False
    source_rank_locality_reason: str = "not_requested"
    estimated_remote_assignments: float = 0.0
    estimated_total_assignments: float = 0.0
    estimated_remote_ratio: Optional[float] = None
    per_source_remote_assignments: Mapping[str, float] = field(
        default_factory=dict
    )
    reason: str = "metadata_only"

    @property
    def placement_available(self) -> bool:
        return bool(self.expert_to_target_rank or self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_parallel_plan": dict(self.target_parallel_plan),
            "expert_to_target_rank": dict(self.expert_to_target_rank),
            "expert_to_target_ranks": {
                key: list(value)
                for key, value in self.expert_to_target_ranks.items()
            },
            "placement_epoch": self.placement_epoch,
            "placement_source": self.placement_source,
            "placement_fingerprint": self.placement_fingerprint,
            "expert_placement_available": self.placement_available,
            "required_expert_count": self.required_expert_count,
            "covered_expert_count": self.covered_expert_count,
            "planned_shard_count": (
                self.planned_shard_count or len(self.shards)
            ),
            "target_rank_count": self.target_rank_count,
            "expert_physical_replication_factor": (
                self.expert_physical_replication_factor
            ),
            "sllm_replica_count": self.sllm_replica_count,
            "physical_weight_migration": self.physical_weight_migration,
            "live_expert_remap": self.live_expert_remap,
            "allow_active_requests": self.allow_active_requests,
            "require_cross_node": self.require_cross_node,
            "expert_placement_physical_migration_required": (
                self.physical_migration_required
            ),
            "expert_placement_snapshot": {
                str(key): dict(value)
                for key, value in self.expert_placement_snapshot.items()
            },
            "shards": [shard.to_dict() for shard in self.shards],
            "movement_observation_available": (
                self.movement_observation_available
            ),
            "movement_source": self.movement_source,
            "moved_expert_count": self.moved_expert_count,
            "stationary_expert_count": self.stationary_expert_count,
            "unknown_movement_expert_count": (
                self.unknown_movement_expert_count
            ),
            "moved_weight_bytes": self.moved_weight_bytes,
            "estimated_expert_weight_movement_cost_ms": (
                self.estimated_expert_weight_movement_cost_ms
            ),
            "expert_movement_diff": {
                str(key): dict(value)
                for key, value in self.expert_movement_diff.items()
            },
            "estimated_dispatch_cost": self.estimated_dispatch_cost,
            "estimated_load_balance_penalty": (
                self.estimated_load_balance_penalty
            ),
            "placement_strategy": self.placement_strategy,
            "selection_reason": self.selection_reason,
            "hotness_snapshot_id": self.hotness_snapshot_id,
            "hotness_snapshot_available": self.hotness_snapshot_available,
            "hotness_snapshot_reason": self.hotness_snapshot_reason,
            "hotness_snapshot": dict(self.hotness_snapshot),
            "per_layer_rank_load": {
                str(layer): {
                    str(rank): float(load)
                    for rank, load in rank_load.items()
                }
                for layer, rank_load in self.per_layer_rank_load.items()
            },
            "per_layer_load_imbalance": {
                str(layer): float(value)
                for layer, value in self.per_layer_load_imbalance.items()
            },
            "per_layer_load_cv": {
                str(layer): float(value)
                for layer, value in self.per_layer_load_cv.items()
            },
            "hotness_layer_coverage": self.hotness_layer_coverage,
            "predicted_improvement_ratio": self.predicted_improvement_ratio,
            "placement_objective": self.placement_objective,
            "source_rank_locality_available": (
                self.source_rank_locality_available
            ),
            "source_rank_locality_reason": self.source_rank_locality_reason,
            "estimated_remote_assignments": self.estimated_remote_assignments,
            "estimated_total_assignments": self.estimated_total_assignments,
            "estimated_remote_ratio": self.estimated_remote_ratio,
            "per_source_remote_assignments": {
                str(rank): float(value)
                for rank, value in self.per_source_remote_assignments.items()
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HotnessSnapshot:
    snapshot_id: str
    available: bool
    reason: str
    alpha: float
    hotness: Mapping[str, float] = field(default_factory=dict)
    global_counts: Mapping[str, float] = field(default_factory=dict)
    recent_counts: Mapping[str, float] = field(default_factory=dict)
    layer_sample_counts: Mapping[str, int] = field(default_factory=dict)
    sample_count: int = 0
    recent_event_count: int = 0
    observation_time_s: Optional[float] = None
    age_s: Optional[float] = None
    source: str = "unavailable"
    counter_unit: str = "topk_routed_assignments"
    aggregation_scope: str = "deployment"
    key_granularity: str = "layer_expert"
    complete_sparse: bool = False
    global_reset_marker: str = ""
    layer_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "available": self.available,
            "reason": self.reason,
            "alpha": self.alpha,
            "hotness": {
                str(key): float(value) for key, value in self.hotness.items()
            },
            "global_counts": {
                str(key): float(value)
                for key, value in self.global_counts.items()
            },
            "recent_counts": {
                str(key): float(value)
                for key, value in self.recent_counts.items()
            },
            "layer_sample_counts": {
                str(key): int(value)
                for key, value in self.layer_sample_counts.items()
            },
            "sample_count": self.sample_count,
            "recent_event_count": self.recent_event_count,
            "observation_time_s": self.observation_time_s,
            "age_s": self.age_s,
            "source": self.source,
            "counter_unit": self.counter_unit,
            "aggregation_scope": self.aggregation_scope,
            "key_granularity": self.key_granularity,
            "complete_sparse": self.complete_sparse,
            "global_reset_marker": self.global_reset_marker,
            "layer_coverage": self.layer_coverage,
        }


def _as_positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed <= 0:
        return int(default)
    return parsed


def _as_non_negative_int(value: Any, default: int = -1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    if parsed < 0:
        return int(default)
    return parsed


def _first_positive_config_int(
    config: Mapping[str, Any],
    *keys: str,
) -> int:
    for key in keys:
        value = _as_positive_int(config.get(key), 0)
        if value > 0:
            return value
    return 0


def _model_config_sources(
    model_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [model_config]
    for key in ("runtime_metadata", "model_config", "hf_config", "moe_config"):
        nested = model_config.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
            nested_profile = nested.get("model_resource_profile")
            if isinstance(nested_profile, Mapping):
                sources.append(nested_profile)
    backend_config = model_config.get("backend_config", {})
    if isinstance(backend_config, Mapping):
        sources.append(backend_config)
        for key in (
            "runtime_metadata",
            "model_config",
            "hf_config",
            "moe_config",
        ):
            nested = backend_config.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
                nested_profile = nested.get("model_resource_profile")
                if isinstance(nested_profile, Mapping):
                    sources.append(nested_profile)

        model_path = backend_config.get("pretrained_model_name_or_path")
        if model_path is None:
            model_path = backend_config.get("model")
        try:
            config_path = Path(str(model_path)) / "config.json"
            if config_path.is_file():
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(parsed, Mapping):
                    sources.append(parsed)
        except Exception:
            pass

    return tuple(sources)


def _infer_topology_from_placement_snapshot(
    config: Mapping[str, Any],
) -> tuple[int, int]:
    snapshot = config.get("expert_placement_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        snapshot = config.get("expert_placement")
    runtime_shards = config.get("runtime_expert_placement_shards")
    if (
        (not isinstance(snapshot, Mapping) or not snapshot)
        and not isinstance(runtime_shards, Mapping)
    ):
        return 0, 0

    max_layer_id = -1
    max_expert_id = -1

    def observe_entry(key: Any, value: Any) -> None:
        nonlocal max_layer_id, max_expert_id
        layer_id = -1
        expert_id = -1
        if isinstance(value, Mapping):
            layer_id = _as_non_negative_int(value.get("layer_id"), -1)
            expert_id = _as_non_negative_int(value.get("expert_id"), -1)
        if layer_id < 0 or expert_id < 0:
            key_text = str(key)
            for part in key_text.split("/"):
                name, _, raw_value = part.partition(":")
                if name == "layer":
                    layer_id = _as_non_negative_int(raw_value, -1)
                elif name == "expert":
                    expert_id = _as_non_negative_int(raw_value, -1)
        if layer_id >= 0:
            max_layer_id = max(max_layer_id, layer_id)
        if expert_id >= 0:
            max_expert_id = max(max_expert_id, expert_id)

    if isinstance(runtime_shards, Mapping):
        for key, rows in runtime_shards.items():
            observe_entry(key, {})
            if isinstance(rows, Mapping):
                observe_entry(key, rows)
            elif isinstance(rows, (list, tuple)):
                for row in rows:
                    observe_entry(key, row)

    if not isinstance(snapshot, Mapping) or not snapshot:
        if max_layer_id < 0 or max_expert_id < 0:
            return 0, 0
        return max_layer_id + 1, max_expert_id + 1

    for key, value in snapshot.items():
        observe_entry(key, value)

    if max_layer_id < 0 or max_expert_id < 0:
        return 0, 0
    return max_layer_id + 1, max_expert_id + 1


def infer_moe_topology(
    model_config: Mapping[str, Any],
) -> tuple[int, int]:
    """Return logical MoE topology as (num_layers, experts_per_layer)."""
    sources = _model_config_sources(model_config)
    for config in sources:
        num_layers = _first_positive_config_int(
            config,
            "num_hidden_layers",
            "n_layer",
            "num_layers",
        )
        num_experts = _first_positive_config_int(
            config,
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
        )
        if num_layers > 0 and num_experts > 0:
            return num_layers, num_experts
    for config in sources:
        num_layers, num_experts = _infer_topology_from_placement_snapshot(
            config
        )
        if num_layers > 0 and num_experts > 0:
            return num_layers, num_experts
    return 0, 0


def looks_like_moe_model(
    model_name: str,
    model_config: Mapping[str, Any],
) -> bool:
    if infer_moe_topology(model_config) != (0, 0):
        return True
    backend_config = model_config.get("backend_config", {})
    if not isinstance(backend_config, Mapping):
        backend_config = {}
    if bool(backend_config.get("enable_expert_parallel", False)):
        return True
    model_id = str(
        backend_config.get("pretrained_model_name_or_path")
        or backend_config.get("model")
        or model_config.get("model")
        or model_name
    )
    return "moe" in model_id.lower() or "moe" in str(model_name).lower()


def _plan_fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return format(zlib.crc32(blob) & 0xFFFFFFFF, "08x")


def _config_sources(
    planner_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    for source in (
        planner_config,
        planner_config.get("moe_planner_config"),
        planner_config.get("expert_placement_config"),
    ):
        if isinstance(source, Mapping):
            sources.append(source)
    sources.extend(_model_config_sources(model_config))
    return tuple(sources)


def _first_config_value(
    sources: tuple[Mapping[str, Any], ...],
    *keys: str,
) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                return source.get(key)
    return None


def _first_non_negative_float(
    sources: tuple[Mapping[str, Any], ...],
    *keys: str,
    default: float = 0.0,
) -> float:
    value = _first_config_value(sources, *keys)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, parsed)


def _expert_key_from_entry(key: Any, value: Any) -> str:
    layer_id = -1
    expert_id = -1
    if isinstance(value, Mapping):
        layer_id = _as_non_negative_int(value.get("layer_id"), -1)
        expert_id = _as_non_negative_int(value.get("expert_id"), -1)
    if layer_id >= 0 and expert_id >= 0:
        return f"layer:{layer_id}/expert:{expert_id}"

    key_text = str(key)
    parsed_layer_id = -1
    parsed_expert_id = -1
    for part in key_text.split("/"):
        name, _, raw_value = part.partition(":")
        if name == "layer":
            parsed_layer_id = _as_non_negative_int(raw_value, -1)
        elif name == "expert":
            parsed_expert_id = _as_non_negative_int(raw_value, -1)
    if parsed_layer_id >= 0 and parsed_expert_id >= 0:
        return f"layer:{parsed_layer_id}/expert:{parsed_expert_id}"
    return key_text


def _normalize_expert_placement_snapshot(
    snapshot: Any,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(snapshot, Mapping):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    shards = snapshot.get("shards")
    if isinstance(shards, (list, tuple)):
        for shard in shards:
            if not isinstance(shard, Mapping):
                continue
            expert_key = _expert_key_from_entry("", shard)
            if expert_key:
                normalized[expert_key] = dict(shard)

    for key, value in snapshot.items():
        if key == "shards":
            continue
        if isinstance(value, Mapping):
            expert_key = _expert_key_from_entry(key, value)
            payload = dict(value)
        else:
            expert_key = str(key)
            payload = {"rank_id": str(value)}
        if not expert_key:
            continue
        if expert_key.startswith("layer:"):
            for part in expert_key.split("/"):
                name, _, raw_value = part.partition(":")
                if name == "layer" and "layer_id" not in payload:
                    payload["layer_id"] = _as_non_negative_int(raw_value, -1)
                elif name == "expert" and "expert_id" not in payload:
                    payload["expert_id"] = _as_non_negative_int(raw_value, -1)
        normalized[expert_key] = payload

    return normalized


def _observed_runtime_expert_placement(
    metadata: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    profile = metadata.get("model_resource_profile")
    sources = (metadata, profile) if isinstance(profile, Mapping) else (metadata,)
    for source in sources:
        if not source.get("runtime_expert_placement_available"):
            continue
        shards = source.get("runtime_expert_placement_shards")
        if not isinstance(shards, Mapping):
            continue
        observed: Dict[str, Dict[str, Any]] = {}
        for key, rows in shards.items():
            # A duplicated expert needs replica-aware comparison; do not guess
            # which copy corresponds to the single-rank logical plan.
            if not isinstance(rows, (list, tuple)) or len(rows) != 1:
                continue
            row = rows[0]
            if not isinstance(row, Mapping) or row.get("weight_resident") is not True:
                continue
            ep_rank = _as_non_negative_int(row.get("ep_rank"), -1)
            if ep_rank < 0:
                continue
            expert_key = _expert_key_from_entry(key, row)
            observed[expert_key] = {
                "rank_id": f"replica:0/ep-rank:{ep_rank}",
                "weight_size_bytes": _entry_weight_size_bytes(row),
                "weight_resident": True,
            }
        if observed:
            return observed
    return {}


def _current_expert_placement_snapshot(
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], str]:
    runtime_metadata = model_config.get("runtime_metadata")
    if isinstance(runtime_metadata, Mapping):
        observed = _observed_runtime_expert_placement(runtime_metadata)
        if observed:
            return observed, "runtime_expert_placement_shards"

    planner_snapshot = planner_config.get("current_expert_placement_snapshot")
    normalized = _normalize_expert_placement_snapshot(planner_snapshot)
    if normalized:
        return normalized, "planner_current_expert_placement_snapshot"

    if isinstance(runtime_metadata, Mapping):
        profile = runtime_metadata.get("model_resource_profile")
        if (
            "runtime_expert_placement_available" in runtime_metadata
            or isinstance(profile, Mapping)
            and "runtime_expert_placement_available" in profile
        ):
            return {}, "runtime_placement_unavailable"
        if runtime_metadata.get("placement_source") == "derived_from_model_config":
            return {}, "runtime_placement_unavailable"
        normalized = _normalize_expert_placement_snapshot(
            runtime_metadata.get("expert_placement_snapshot")
            or runtime_metadata.get("expert_placement")
        )
        if normalized:
            return normalized, "runtime_metadata"
        if isinstance(profile, Mapping):
            if profile.get("placement_source") == "derived_from_model_config":
                return {}, "runtime_placement_unavailable"
            normalized = _normalize_expert_placement_snapshot(
                profile.get("expert_placement_snapshot")
                or profile.get("expert_placement")
            )
            if normalized:
                return normalized, "runtime_metadata.model_resource_profile"

    backend_config = model_config.get("backend_config")
    if isinstance(backend_config, Mapping):
        backend_runtime = backend_config.get("runtime_metadata")
        if isinstance(backend_runtime, Mapping):
            observed = _observed_runtime_expert_placement(backend_runtime)
            if observed:
                return observed, "runtime_expert_placement_shards"
            profile = backend_runtime.get("model_resource_profile")
            if (
                "runtime_expert_placement_available" in backend_runtime
                or isinstance(profile, Mapping)
                and "runtime_expert_placement_available" in profile
            ):
                return {}, "runtime_placement_unavailable"
            if backend_runtime.get("placement_source") == "derived_from_model_config":
                return {}, "runtime_placement_unavailable"
            normalized = _normalize_expert_placement_snapshot(
                backend_runtime.get("expert_placement_snapshot")
                or backend_runtime.get("expert_placement")
            )
            if normalized:
                return normalized, "backend_config.runtime_metadata"
            if isinstance(profile, Mapping):
                if profile.get("placement_source") == "derived_from_model_config":
                    return {}, "runtime_placement_unavailable"
                normalized = _normalize_expert_placement_snapshot(
                    profile.get("expert_placement_snapshot")
                    or profile.get("expert_placement")
                )
                if normalized:
                    return (
                        normalized,
                        "backend_config.runtime_metadata.model_resource_profile",
                    )

        if backend_config.get("placement_source") == "derived_from_model_config":
            return {}, "runtime_placement_unavailable"
        normalized = _normalize_expert_placement_snapshot(
            backend_config.get("expert_placement_snapshot")
            or backend_config.get("expert_placement")
        )
        if normalized:
            return normalized, "backend_config"

    if model_config.get("placement_source") == "derived_from_model_config":
        return {}, "runtime_placement_unavailable"
    normalized = _normalize_expert_placement_snapshot(
        model_config.get("expert_placement_snapshot")
        or model_config.get("expert_placement")
    )
    if normalized:
        return normalized, "model_config"

    return {}, "unavailable"


def _rank_signature(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    match = re.search(r"ep-rank[:\-]?(\d+)", text)
    if match:
        return f"ep-rank:{int(match.group(1))}"
    match = re.search(r"rank[:\-]?(\d+)", text)
    if match:
        return f"rank:{int(match.group(1))}"
    return text


def _location_value(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    if value is None:
        return ""
    return str(value)


def _entry_weight_size_bytes(
    *entries: Mapping[str, Any],
    default: Optional[int] = None,
) -> Optional[int]:
    for entry in entries:
        for key in ("weight_size_bytes", "expert_weight_size_bytes"):
            parsed = _as_positive_int(entry.get(key), 0)
            if parsed > 0:
                return parsed
    return default if default and default > 0 else None


def _default_expert_weight_size_bytes(
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> Optional[int]:
    sources = _config_sources(planner_config, model_config)
    value = _first_config_value(
        sources,
        "expert_weight_size_bytes",
        "default_expert_weight_size_bytes",
        "moe_expert_weight_size_bytes",
    )
    return _as_positive_int(value, 0) or None


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _canonical_expert_key(key: Any) -> Optional[tuple[int, int]]:
    match = re.fullmatch(r"layer:(\d+)/expert:(\d+)", str(key))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _hotness_contract_source(
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    backend_config = model_config.get("backend_config")
    if not isinstance(backend_config, Mapping):
        backend_config = {}
    candidates = (
        (planner_config.get("hotness_snapshot"), "planner.hotness_snapshot"),
        (planner_config, "planner_config"),
        (model_config.get("runtime_metadata"), "runtime_metadata"),
        (backend_config.get("runtime_metadata"), "backend.runtime_metadata"),
        (backend_config, "backend_config"),
        (model_config, "model_config"),
    )
    for candidate, label in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if isinstance(candidate.get("global_expert_hotness"), Mapping) or isinstance(
            candidate.get("recent_window_expert_hotness"), Mapping
        ):
            return candidate, label
    return {}, "unavailable"


def _canonical_histogram(
    raw: Any,
    *,
    num_layers: int,
    num_experts: int,
) -> tuple[Dict[str, float], str]:
    if raw is None:
        return {}, "missing"
    if not isinstance(raw, Mapping):
        return {}, "not_mapping"
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        parsed_key = _canonical_expert_key(key)
        parsed_value = _finite_float(value)
        if parsed_key is None:
            if parsed_value is not None and parsed_value > 0:
                return {}, "noncanonical_expert_key"
            continue
        layer_id, expert_id = parsed_key
        if layer_id >= num_layers or expert_id >= num_experts:
            return {}, "expert_key_out_of_range"
        if parsed_value is None or parsed_value < 0:
            return {}, "invalid_histogram_value"
        normalized[f"layer:{layer_id}/expert:{expert_id}"] = parsed_value
    return normalized, "ok"


def build_hotness_snapshot(
    *,
    num_layers: int,
    num_experts: int,
    model_config: Optional[Mapping[str, Any]] = None,
    planner_config: Optional[Mapping[str, Any]] = None,
) -> HotnessSnapshot:
    """Freeze and normalize one deployment-level routing snapshot.

    Counts are normalized independently for every MoE layer.  Generic
    ``expert:{id}`` keys are deliberately rejected because they cannot be
    safely applied to every layer.
    """
    model_config = model_config or {}
    planner_config = planner_config or {}
    sources = _config_sources(planner_config, model_config)
    alpha_value = _first_config_value(sources, "hotness_alpha")
    alpha = _finite_float(0.8 if alpha_value is None else alpha_value)
    if alpha is None or not 0.0 <= alpha <= 1.0:
        alpha = 0.8

    source, source_label = _hotness_contract_source(
        model_config, planner_config
    )
    global_counts, global_reason = _canonical_histogram(
        source.get("global_expert_hotness"),
        num_layers=num_layers,
        num_experts=num_experts,
    )
    recent_counts, recent_reason = _canonical_histogram(
        source.get("recent_window_expert_hotness"),
        num_layers=num_layers,
        num_experts=num_experts,
    )
    complete_sparse = _to_bool(
        source.get(
            "moe_route_histogram_complete_sparse",
            source.get("hotness_histogram_complete_sparse"),
        ),
        False,
    )
    counter_unit = str(
        source.get("moe_route_counter_unit")
        or source.get("counter_unit")
        or "topk_routed_assignments"
    )
    aggregation_scope = str(
        source.get("moe_route_aggregation_scope")
        or source.get("aggregation_scope")
        or "deployment"
    )
    key_granularity = str(
        source.get("moe_route_histogram_key_granularity")
        or source.get("key_granularity")
        or "layer_expert"
    )
    global_reset_marker = str(
        source.get("moe_route_global_reset_marker")
        or source.get("global_reset_marker")
        or ""
    )
    observation_time_s = _finite_float(
        source.get(
            "moe_route_last_observation_time_s",
            source.get("routing_last_observation_time_s"),
        )
    )
    now_s = _finite_float(
        _first_config_value(sources, "hotness_snapshot_time_s")
    )
    if now_s is None:
        now_s = time.time()
    explicit_age_s = _finite_float(source.get("moe_route_histogram_age_s"))
    age_s = explicit_age_s
    if age_s is None and observation_time_s is not None:
        age_s = max(0.0, now_s - observation_time_s)
    recent_event_count = _as_non_negative_int(
        source.get("moe_route_recent_event_count"), 0
    )

    hotness: Dict[str, float] = {}
    layer_sample_counts: Dict[str, int] = {}
    covered_layers = 0
    invalid_reason = ""
    if source_label == "unavailable":
        invalid_reason = "hotness_histogram_unavailable"
    elif global_reason not in {"ok", "missing"}:
        invalid_reason = f"invalid_global_histogram:{global_reason}"
    elif recent_reason not in {"ok", "missing"}:
        invalid_reason = f"invalid_recent_histogram:{recent_reason}"
    elif counter_unit != "topk_routed_assignments":
        invalid_reason = "unsupported_counter_unit"
    elif key_granularity != "layer_expert":
        invalid_reason = "unsupported_key_granularity"
    elif aggregation_scope not in {"deployment", "engine", "ep_group"}:
        invalid_reason = "unsupported_aggregation_scope"
    elif source.get("moe_route_metadata_consistent") is False:
        invalid_reason = "inconsistent_routing_metadata"
    elif str(source.get("moe_route_tp_aggregation") or "") in {
        "unverified",
        "sum_unverified_tp_workers",
    }:
        invalid_reason = "unverified_tp_aggregation"

    if not invalid_reason:
        for layer_id in range(num_layers):
            keys = [
                f"layer:{layer_id}/expert:{expert_id}"
                for expert_id in range(num_experts)
            ]
            global_complete = complete_sparse or all(
                key in global_counts for key in keys
            )
            recent_complete = complete_sparse or all(
                key in recent_counts for key in keys
            )
            global_total = sum(global_counts.get(key, 0.0) for key in keys)
            recent_total = sum(recent_counts.get(key, 0.0) for key in keys)
            global_valid = global_complete and global_total > 0.0
            recent_valid = recent_complete and recent_total > 0.0
            if not global_valid and not recent_valid:
                continue
            covered_layers += 1
            layer_sample_counts[str(layer_id)] = int(
                global_total if global_valid else recent_total
            )
            for key in keys:
                global_share = (
                    global_counts.get(key, 0.0) / global_total
                    if global_valid
                    else 0.0
                )
                recent_share = (
                    recent_counts.get(key, 0.0) / recent_total
                    if recent_valid
                    else 0.0
                )
                if global_valid and recent_valid:
                    value = alpha * recent_share + (1.0 - alpha) * global_share
                elif recent_valid:
                    value = recent_share
                else:
                    value = global_share
                hotness[key] = float(value)

    sample_count = sum(layer_sample_counts.values())
    minimum_samples = _as_non_negative_int(
        _first_config_value(sources, "minimum_hotness_samples"), 1
    )
    max_age_s = _first_non_negative_float(
        sources, "hotness_max_age_s", default=0.0
    )
    require_reset_marker = _to_bool(
        _first_config_value(sources, "require_hotness_reset_marker"), False
    )
    if not invalid_reason and not hotness:
        invalid_reason = "no_complete_sampled_layers"
    elif not invalid_reason and sample_count < minimum_samples:
        invalid_reason = "insufficient_hotness_samples"
    elif not invalid_reason and max_age_s > 0.0 and age_s is None:
        invalid_reason = "hotness_freshness_unknown"
    elif not invalid_reason and max_age_s > 0.0 and age_s > max_age_s:
        invalid_reason = "stale_hotness_snapshot"
    elif not invalid_reason and require_reset_marker and not global_reset_marker:
        invalid_reason = "global_counter_reset_marker_unknown"

    fingerprint_payload = {
        "alpha": alpha,
        "global": global_counts,
        "recent": recent_counts,
        "source": source_label,
        "counter_unit": counter_unit,
        "aggregation_scope": aggregation_scope,
        "key_granularity": key_granularity,
        "global_reset_marker": global_reset_marker,
    }
    snapshot_id = _plan_fingerprint(fingerprint_payload)
    return HotnessSnapshot(
        snapshot_id=snapshot_id,
        available=not invalid_reason,
        reason=invalid_reason or "valid_hotness_snapshot",
        alpha=alpha,
        hotness=hotness,
        global_counts=global_counts,
        recent_counts=recent_counts,
        layer_sample_counts=layer_sample_counts,
        sample_count=sample_count,
        recent_event_count=recent_event_count,
        observation_time_s=observation_time_s,
        age_s=age_s,
        source=source_label,
        counter_unit=counter_unit,
        aggregation_scope=aggregation_scope,
        key_granularity=key_granularity,
        complete_sparse=complete_sparse,
        global_reset_marker=global_reset_marker,
        layer_coverage=(
            covered_layers / float(num_layers) if num_layers > 0 else 0.0
        ),
    )


def _expert_hotness_lookup(
    expert_key: str,
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> float:
    sources = _config_sources(planner_config, model_config)
    candidates = {str(expert_key)}
    normalized = str(expert_key).replace("/", "/")
    for part in str(expert_key).split("/"):
        name, _, raw_value = part.partition(":")
        if name == "expert":
            candidates.add(f"expert:{raw_value}")
            candidates.add(raw_value)
        elif name == "layer":
            candidates.add(f"layer:{raw_value}")
    score = 0.0
    for source in sources:
        for field in ("global_expert_hotness", "recent_window_expert_hotness"):
            histogram = source.get(field)
            if not isinstance(histogram, Mapping):
                continue
            for key, value in histogram.items():
                if str(key) not in candidates:
                    continue
                try:
                    score += float(value or 0)
                except (TypeError, ValueError):
                    continue
    return score


def _expert_hotness_order(
    layer_id: int,
    num_experts: int,
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> list[int]:
    ordered = list(range(num_experts))
    ordered.sort(
        key=lambda expert_id: (
            -_expert_hotness_lookup(
                f"layer:{layer_id}/expert:{expert_id}",
                model_config,
                planner_config,
            ),
            expert_id,
        )
    )
    return ordered


def _rank_index(value: Any, rank_count: int) -> Optional[int]:
    match = re.search(r"(?:ep-)?rank[:\-]?(\d+)", str(value or ""))
    if match is None:
        return None
    rank_index = int(match.group(1))
    return rank_index if 0 <= rank_index < rank_count else None


def _current_layer_assignment(
    *,
    layer_id: int,
    num_experts: int,
    rank_count: int,
    current_snapshot: Mapping[str, Mapping[str, Any]],
) -> Optional[list[int]]:
    assignment: list[int] = []
    for expert_id in range(num_experts):
        entry = current_snapshot.get(f"layer:{layer_id}/expert:{expert_id}")
        if not isinstance(entry, Mapping):
            return None
        rank_index = _rank_index(entry.get("rank_id"), rank_count)
        if rank_index is None:
            return None
        assignment.append(rank_index)
    return assignment


def _default_slot_capacities(
    num_experts: int,
    rank_count: int,
    current_assignment: Optional[list[int]],
) -> list[int]:
    if current_assignment is not None:
        counts = [current_assignment.count(rank) for rank in range(rank_count)]
        if sum(counts) == num_experts and all(count >= 0 for count in counts):
            return counts
    base, remainder = divmod(num_experts, rank_count)
    return [base + int(rank < remainder) for rank in range(rank_count)]


def _hotness_balanced_assignment(
    *,
    layer_id: int,
    num_experts: int,
    rank_count: int,
    hotness: Mapping[str, float],
    current_assignment: Optional[list[int]],
) -> list[int]:
    capacities = _default_slot_capacities(
        num_experts, rank_count, current_assignment
    )
    assignment = [-1] * num_experts
    rank_load = [0.0] * rank_count
    rank_counts = [0] * rank_count
    ordered_experts = sorted(
        range(num_experts),
        key=lambda expert_id: (
            -float(
                hotness.get(
                    f"layer:{layer_id}/expert:{expert_id}", 0.0
                )
            ),
            expert_id,
        ),
    )
    for expert_id in ordered_experts:
        feasible = [
            rank
            for rank in range(rank_count)
            if rank_counts[rank] < capacities[rank]
        ]
        if not feasible:
            raise ValueError("expert_slot_capacity_exhausted")
        owner = (
            current_assignment[expert_id]
            if current_assignment is not None
            else None
        )
        rank = min(
            feasible,
            key=lambda candidate: (
                rank_load[candidate],
                0 if candidate == owner else 1,
                candidate,
            ),
        )
        assignment[expert_id] = rank
        rank_counts[rank] += 1
        rank_load[rank] += float(
            hotness.get(f"layer:{layer_id}/expert:{expert_id}", 0.0)
        )
    return assignment


_SOURCE_EXPERT_KEY = re.compile(
    r"^source-rank:(\d+)/layer:(\d+)/expert:(\d+)$"
)


def _source_rank_expert_snapshot(
    *,
    rank_count: int,
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> tuple[Dict[tuple[int, int, int], float], str]:
    source, _ = _hotness_contract_source(model_config, planner_config)
    if not _to_bool(
        source.get("moe_source_rank_expert_histogram_available"), False
    ):
        return {}, "source_rank_expert_histogram_unavailable"
    if str(source.get("moe_source_rank_expert_observation_point") or "") != (
        "pre_dispatch"
    ):
        return {}, "source_rank_observation_point_not_pre_dispatch"
    if str(source.get("moe_source_rank_expert_key_granularity") or "") != (
        "source_rank_layer_expert"
    ):
        return {}, "source_rank_key_granularity_mismatch"
    if str(source.get("moe_source_rank_semantics") or "") != (
        "vllm_dp_rank_equals_effective_ep_rank"
    ):
        return {}, "source_rank_semantics_unsupported"
    histogram = source.get("recent_window_source_rank_expert_hotness")
    histogram_source = "recent_window_source_rank_expert_hotness"
    if not isinstance(histogram, Mapping) or not histogram:
        histogram = source.get("global_source_rank_expert_hotness")
        histogram_source = "global_source_rank_expert_hotness"
    if not isinstance(histogram, Mapping) or not histogram:
        return {}, "source_rank_expert_histogram_empty"
    parsed: Dict[tuple[int, int, int], float] = {}
    for raw_key, raw_value in histogram.items():
        match = _SOURCE_EXPERT_KEY.fullmatch(str(raw_key))
        value = _finite_float(raw_value)
        if match is None or value is None or value < 0.0:
            return {}, "source_rank_expert_histogram_invalid"
        source_rank, layer_id, expert_id = map(int, match.groups())
        if source_rank >= rank_count:
            return {}, "source_rank_out_of_range"
        if value > 0.0:
            parsed[(source_rank, layer_id, expert_id)] = value
    if not parsed:
        return {}, "source_rank_expert_histogram_empty"
    return parsed, histogram_source


def _hotness_locality_assignment(
    *,
    layer_id: int,
    num_experts: int,
    rank_count: int,
    hotness: Mapping[str, float],
    source_expert_counts: Mapping[tuple[int, int, int], float],
    locality_weight: float,
    current_assignment: Optional[list[int]],
) -> list[int]:
    capacities = _default_slot_capacities(
        num_experts, rank_count, current_assignment
    )
    assignment = [-1] * num_experts
    rank_load = [0.0] * rank_count
    rank_counts = [0] * rank_count
    layer_total = sum(
        value
        for (source_rank, source_layer, expert_id), value
        in source_expert_counts.items()
        if source_layer == layer_id
    )
    ordered_experts = sorted(
        range(num_experts),
        key=lambda expert_id: (
            -float(hotness.get(f"layer:{layer_id}/expert:{expert_id}", 0.0)),
            expert_id,
        ),
    )
    for expert_id in ordered_experts:
        feasible = [
            rank
            for rank in range(rank_count)
            if rank_counts[rank] < capacities[rank]
        ]
        if not feasible:
            raise ValueError("expert_slot_capacity_exhausted")
        owner = (
            current_assignment[expert_id]
            if current_assignment is not None
            else None
        )

        def score(candidate: int) -> tuple[float, int, int]:
            remote = sum(
                source_expert_counts.get(
                    (source_rank, layer_id, expert_id), 0.0
                )
                for source_rank in range(rank_count)
                if source_rank != candidate
            )
            remote_share = remote / layer_total if layer_total > 0.0 else 0.0
            projected_load = rank_load[candidate] + float(
                hotness.get(f"layer:{layer_id}/expert:{expert_id}", 0.0)
            )
            return (
                projected_load + locality_weight * remote_share,
                0 if candidate == owner else 1,
                candidate,
            )

        rank = min(feasible, key=score)
        assignment[expert_id] = rank
        rank_counts[rank] += 1
        rank_load[rank] += float(
            hotness.get(f"layer:{layer_id}/expert:{expert_id}", 0.0)
        )
    return assignment


def _remote_dispatch_diagnostics(
    *,
    assignments: Mapping[int, list[int]],
    source_expert_counts: Mapping[tuple[int, int, int], float],
) -> tuple[float, float, Optional[float], Dict[str, float]]:
    total = 0.0
    remote = 0.0
    per_source: Dict[str, float] = {}
    for (source_rank, layer_id, expert_id), count in source_expert_counts.items():
        assignment = assignments.get(layer_id)
        if assignment is None or expert_id >= len(assignment):
            continue
        total += float(count)
        if assignment[expert_id] != source_rank:
            remote += float(count)
            source_key = str(source_rank)
            per_source[source_key] = per_source.get(source_key, 0.0) + float(
                count
            )
    ratio = remote / total if total > 0.0 else None
    return remote, total, ratio, per_source


def _load_diagnostics(
    *,
    assignments: Mapping[int, list[int]],
    hotness: Mapping[str, float],
    rank_count: int,
) -> tuple[
    Dict[str, Dict[str, float]],
    Dict[str, float],
    Dict[str, float],
    float,
]:
    per_layer_rank_load: Dict[str, Dict[str, float]] = {}
    per_layer_imbalance: Dict[str, float] = {}
    per_layer_cv: Dict[str, float] = {}
    for layer_id, assignment in assignments.items():
        rank_load = [0.0] * rank_count
        for expert_id, rank in enumerate(assignment):
            rank_load[rank] += float(
                hotness.get(f"layer:{layer_id}/expert:{expert_id}", 0.0)
            )
        mean = sum(rank_load) / float(rank_count)
        if mean <= 0.0:
            continue
        imbalance = max(rank_load) / mean - 1.0
        variance = sum((value - mean) ** 2 for value in rank_load) / rank_count
        per_layer_rank_load[str(layer_id)] = {
            str(rank): float(value) for rank, value in enumerate(rank_load)
        }
        per_layer_imbalance[str(layer_id)] = float(imbalance)
        per_layer_cv[str(layer_id)] = float(math.sqrt(variance) / mean)
    penalty = (
        sum(per_layer_imbalance.values()) / len(per_layer_imbalance)
        if per_layer_imbalance
        else 0.0
    )
    return per_layer_rank_load, per_layer_imbalance, per_layer_cv, penalty


def _estimate_expert_weight_movement_cost_ms(
    moved_weight_bytes: Optional[int],
    moved_expert_count: int,
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> Optional[float]:
    if moved_expert_count <= 0:
        return 0.0

    sources = _config_sources(planner_config, model_config)
    cost_ms = _first_non_negative_float(
        sources,
        "expert_weight_movement_cost_ms",
        default=0.0,
    )
    cost_ms_per_gib = _first_non_negative_float(
        sources,
        "expert_weight_movement_cost_ms_per_gib",
        "expert_weight_movement_cost_ms_per_gb",
        default=0.0,
    )
    cost_ms_per_expert = _first_non_negative_float(
        sources,
        "expert_weight_movement_cost_ms_per_expert",
        default=0.0,
    )
    bandwidth_bytes_per_s = _first_non_negative_float(
        sources,
        "expert_weight_movement_bandwidth_bytes_per_s",
        default=0.0,
    )
    if not any(
        value > 0.0
        for value in (
            cost_ms,
            cost_ms_per_gib,
            cost_ms_per_expert,
            bandwidth_bytes_per_s,
        )
    ):
        return None

    if moved_weight_bytes is None:
        if cost_ms_per_gib > 0.0 or bandwidth_bytes_per_s > 0.0:
            return None
        if cost_ms <= 0.0 and cost_ms_per_expert <= 0.0:
            return None
        return float(cost_ms + moved_expert_count * cost_ms_per_expert)

    gib = moved_weight_bytes / float(1024**3)
    if cost_ms_per_gib > 0.0:
        cost_ms += gib * cost_ms_per_gib
    elif bandwidth_bytes_per_s > 0.0 and moved_weight_bytes > 0:
        cost_ms += (moved_weight_bytes / bandwidth_bytes_per_s) * 1000.0
    cost_ms += moved_expert_count * cost_ms_per_expert
    return float(cost_ms)


def _placement_changed(
    current: Mapping[str, Any],
    planned: Mapping[str, Any],
) -> tuple[Optional[bool], str]:
    compared = False

    current_rank = _rank_signature(current.get("rank_id"))
    planned_rank = _rank_signature(planned.get("rank_id"))
    if current_rank:
        compared = True
        if current_rank != planned_rank:
            return True, "rank_changed"

    for key, reason in (
        ("node_id", "node_changed"),
        ("gpu_id", "gpu_changed"),
        ("physical_expert_id", "physical_expert_changed"),
    ):
        current_value = _location_value(current, key)
        if not current_value:
            continue
        compared = True
        if current_value != _location_value(planned, key):
            return True, reason

    if not compared:
        return None, "current_location_unavailable"
    return False, "placement_unchanged"


def _build_expert_movement_diff(
    planned_snapshot: Mapping[str, Mapping[str, Any]],
    current_snapshot: Mapping[str, Mapping[str, Any]],
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not current_snapshot:
        return {
            "movement_observation_available": False,
            "moved_expert_count": 0,
            "stationary_expert_count": 0,
            "unknown_movement_expert_count": len(planned_snapshot),
            "moved_weight_bytes": None,
            "estimated_expert_weight_movement_cost_ms": None,
            "expert_movement_diff": {},
        }

    default_weight_size = _default_expert_weight_size_bytes(
        model_config, planner_config
    )
    moved_experts = 0
    stationary_experts = 0
    unknown_experts = 0
    moved_weight_bytes = 0
    moved_weight_size_known = True
    movement_diff: Dict[str, Dict[str, Any]] = {}

    for expert_key, planned_entry in planned_snapshot.items():
        current_entry = current_snapshot.get(expert_key)
        if current_entry is None:
            unknown_experts += 1
            movement_diff[expert_key] = {
                "status": "unknown",
                "reason": "missing_current_expert",
                "to_rank_id": planned_entry.get("rank_id", ""),
                "to_node_id": planned_entry.get("node_id", ""),
                "to_gpu_id": planned_entry.get("gpu_id", ""),
            }
            continue

        changed, reason = _placement_changed(current_entry, planned_entry)
        if changed is None:
            unknown_experts += 1
            movement_diff[expert_key] = {
                "status": "unknown",
                "reason": reason,
                "to_rank_id": planned_entry.get("rank_id", ""),
                "to_node_id": planned_entry.get("node_id", ""),
                "to_gpu_id": planned_entry.get("gpu_id", ""),
            }
            continue

        weight_size = _entry_weight_size_bytes(
            planned_entry,
            current_entry,
            default=default_weight_size,
        )
        if changed:
            moved_experts += 1
            if weight_size is None:
                moved_weight_size_known = False
            else:
                moved_weight_bytes += weight_size
            movement_diff[expert_key] = {
                "status": "moved",
                "reason": reason,
                "weight_size_bytes": weight_size,
                "from_rank_id": current_entry.get("rank_id", ""),
                "from_node_id": current_entry.get("node_id", ""),
                "from_gpu_id": current_entry.get("gpu_id", ""),
                "to_rank_id": planned_entry.get("rank_id", ""),
                "to_node_id": planned_entry.get("node_id", ""),
                "to_gpu_id": planned_entry.get("gpu_id", ""),
            }
        else:
            stationary_experts += 1

    complete_moved_weight_bytes = (
        moved_weight_bytes if moved_weight_size_known and not unknown_experts else None
    )
    movement_cost_ms = (
        _estimate_expert_weight_movement_cost_ms(
            complete_moved_weight_bytes,
            moved_experts,
            model_config,
            planner_config,
        )
        if not unknown_experts
        else None
    )
    return {
        "movement_observation_available": True,
        "moved_expert_count": moved_experts,
        "stationary_expert_count": stationary_experts,
        "unknown_movement_expert_count": unknown_experts,
        "moved_weight_bytes": complete_moved_weight_bytes,
        "estimated_expert_weight_movement_cost_ms": movement_cost_ms,
        "expert_movement_diff": movement_diff,
    }


def build_logical_expert_placement_plan(
    model_name: str,
    target_parallel_plan: Mapping[str, Any],
    model_config: Optional[Mapping[str, Any]] = None,
    placement_epoch: int = 0,
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Optional[ExpertPlacementPlan]:
    """Build a deterministic logical expert placement plan.

    This describes where experts should live after re-parallelization.  It does
    not move weights or change vLLM rank mapping by itself.
    """
    model_config = model_config or {}
    planner_config = planner_config or {}
    if not looks_like_moe_model(model_name, model_config):
        return None

    num_layers, num_experts = infer_moe_topology(model_config)
    if num_layers <= 0 or num_experts <= 0:
        return ExpertPlacementPlan(
            model_name=model_name,
            target_parallel_plan=dict(target_parallel_plan),
            placement_epoch=max(0, int(placement_epoch or 0)),
            placement_source="unavailable",
            reason="moe_topology_unavailable",
        )

    target_nodes = [
        str(node) for node in target_parallel_plan.get("target_nodes", [])
    ]
    replica_count = _as_positive_int(
        target_parallel_plan.get(
            "sllm_replica_count",
            target_parallel_plan.get("replica_count", 1),
        ),
        1,
    )
    enable_ep = bool(target_parallel_plan.get("enable_expert_parallel", False))
    if enable_ep:
        target_rank_count = _as_positive_int(
            target_parallel_plan.get(
                "runtime_effective_expert_parallel_size",
                target_parallel_plan.get(
                    "effective_expert_parallel_size",
                    target_parallel_plan.get("expert_parallel_size", 1),
                ),
            ),
            1,
        )
    else:
        target_rank_count = 1
    target_rank_count = max(1, target_rank_count)

    backend_config = model_config.get("backend_config", {})
    if not isinstance(backend_config, Mapping):
        backend_config = {}
    backend = str(
        target_parallel_plan.get("backend", model_config.get("backend", ""))
    ).lower()
    placement_strategy = str(
        planner_config.get("target_expert_placement_strategy")
        or backend_config.get("expert_placement_strategy")
        or ("linear" if backend == "vllm" else "round_robin")
    ).strip().lower()
    live_expert_remap_value = planner_config.get(
        "enable_live_expert_remap", False
    )
    live_expert_remap = (
        live_expert_remap_value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(live_expert_remap_value, str)
        else bool(live_expert_remap_value)
    )
    active_remap_value = planner_config.get(
        "allow_active_expert_remap_requests", False
    )
    allow_active_requests = (
        active_remap_value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(active_remap_value, str)
        else bool(active_remap_value)
    )
    require_cross_node_value = planner_config.get(
        "require_cross_node_expert_migration", False
    )
    require_cross_node = (
        require_cross_node_value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(require_cross_node_value, str)
        else bool(require_cross_node_value)
    )
    supported_strategies = {
        "linear",
        "round_robin",
        "hotness",
        "hotspot",
        "hotspot_aware",
        "hotness_balanced",
        "hotness_locality",
    }
    if placement_strategy not in supported_strategies:
        return ExpertPlacementPlan(
            model_name=model_name,
            target_parallel_plan=dict(target_parallel_plan),
            placement_epoch=max(0, int(placement_epoch or 0)),
            placement_source="unavailable",
            reason="unsupported_expert_placement_strategy",
        )

    current_snapshot, movement_source = _current_expert_placement_snapshot(
        model_config, planner_config
    )
    hotness_planner_config = dict(planner_config)
    if placement_strategy in {"hotness_balanced", "hotness_locality"}:
        hotness_planner_config.setdefault(
            "require_hotness_reset_marker", True
        )
    hotness_snapshot = build_hotness_snapshot(
        num_layers=num_layers,
        num_experts=num_experts,
        model_config=model_config,
        planner_config=hotness_planner_config,
    )
    source_expert_counts: Dict[tuple[int, int, int], float] = {}
    source_rank_locality_reason = "not_requested"
    locality_weight = _first_non_negative_float(
        _config_sources(planner_config, model_config),
        "source_rank_locality_weight",
        "locality_weight",
        default=1.0,
    )
    if placement_strategy == "hotness_locality":
        source_expert_counts, source_rank_locality_reason = (
            _source_rank_expert_snapshot(
                rank_count=target_rank_count,
                model_config=model_config,
                planner_config=planner_config,
            )
        )
    layer_assignments: Dict[int, list[int]] = {}
    current_assignments: Dict[int, list[int]] = {}
    for layer_id in range(num_layers):
        current_assignment = _current_layer_assignment(
            layer_id=layer_id,
            num_experts=num_experts,
            rank_count=target_rank_count,
            current_snapshot=current_snapshot,
        )
        if current_assignment is not None:
            current_assignments[layer_id] = current_assignment

        if placement_strategy == "linear":
            base_experts, remainder = divmod(num_experts, target_rank_count)
            assignment = [
                rank_index
                for rank_index in range(target_rank_count)
                for _ in range(base_experts + int(rank_index < remainder))
            ]
        elif placement_strategy == "round_robin":
            assignment = [
                expert_id % target_rank_count
                for expert_id in range(num_experts)
            ]
        elif placement_strategy in {"hotness", "hotspot", "hotspot_aware"}:
            # Retained only as the legacy/synthetic contiguous grouping case.
            assignment = [0] * num_experts
            ordered_experts = _expert_hotness_order(
                layer_id, num_experts, model_config, planner_config
            )
            for position, expert_id in enumerate(ordered_experts):
                assignment[expert_id] = min(
                    (position * target_rank_count)
                    // max(1, len(ordered_experts)),
                    target_rank_count - 1,
                )
        elif (
            placement_strategy == "hotness_locality"
            and hotness_snapshot.available
            and source_expert_counts
        ):
            assignment = _hotness_locality_assignment(
                layer_id=layer_id,
                num_experts=num_experts,
                rank_count=target_rank_count,
                hotness=hotness_snapshot.hotness,
                source_expert_counts=source_expert_counts,
                locality_weight=locality_weight,
                current_assignment=current_assignment,
            )
        elif hotness_snapshot.available and placement_strategy == (
            "hotness_balanced"
        ):
            assignment = _hotness_balanced_assignment(
                layer_id=layer_id,
                num_experts=num_experts,
                rank_count=target_rank_count,
                hotness=hotness_snapshot.hotness,
                current_assignment=current_assignment,
            )
        elif current_assignment is not None:
            assignment = list(current_assignment)
        else:
            base_experts, remainder = divmod(num_experts, target_rank_count)
            assignment = [
                rank_index
                for rank_index in range(target_rank_count)
                for _ in range(base_experts + int(rank_index < remainder))
            ]
        layer_assignments[layer_id] = assignment

    selection_reason = "configured_strategy"
    predicted_improvement_ratio = 0.0
    if placement_strategy in {"hotness_balanced", "hotness_locality"}:
        if not hotness_snapshot.available:
            selection_reason = (
                "retain_current:" if len(current_assignments) == num_layers
                else "deterministic_linear_fallback:"
            ) + hotness_snapshot.reason
        elif placement_strategy == "hotness_locality" and not source_expert_counts:
            selection_reason = (
                "retain_current:" if len(current_assignments) == num_layers
                else "deterministic_linear_fallback:"
            ) + source_rank_locality_reason
        elif len(current_assignments) == num_layers:
            _, _, _, current_penalty = _load_diagnostics(
                assignments=current_assignments,
                hotness=hotness_snapshot.hotness,
                rank_count=target_rank_count,
            )
            _, _, _, candidate_penalty = _load_diagnostics(
                assignments=layer_assignments,
                hotness=hotness_snapshot.hotness,
                rank_count=target_rank_count,
            )
            current_selection_objective = current_penalty
            candidate_selection_objective = candidate_penalty
            if placement_strategy == "hotness_locality":
                _, _, current_remote_ratio, _ = (
                    _remote_dispatch_diagnostics(
                        assignments=current_assignments,
                        source_expert_counts=source_expert_counts,
                    )
                )
                _, _, candidate_remote_ratio, _ = (
                    _remote_dispatch_diagnostics(
                        assignments=layer_assignments,
                        source_expert_counts=source_expert_counts,
                    )
                )
                current_selection_objective += locality_weight * float(
                    current_remote_ratio or 0.0
                )
                candidate_selection_objective += locality_weight * float(
                    candidate_remote_ratio or 0.0
                )
            if current_selection_objective > 0.0:
                predicted_improvement_ratio = max(
                    0.0,
                    (
                        current_selection_objective
                        - candidate_selection_objective
                    )
                    / current_selection_objective,
                )
            moved_count = sum(
                int(candidate_rank != current_assignments[layer_id][expert_id])
                for layer_id, assignment in layer_assignments.items()
                for expert_id, candidate_rank in enumerate(assignment)
            )
            sources = _config_sources(planner_config, model_config)
            minimum_improvement = _first_non_negative_float(
                sources, "minimum_improvement_ratio", default=0.0
            )
            max_moved_experts = _as_non_negative_int(
                _first_config_value(
                    sources,
                    "maximum_moved_experts",
                    "max_moved_experts",
                    "movement_budget_experts",
                ),
                0,
            )
            default_weight_size = _default_expert_weight_size_bytes(
                model_config, planner_config
            )
            moved_weight_sizes: list[int] = []
            moved_weight_sizes_known = True
            for layer_id, assignment in layer_assignments.items():
                for expert_id, candidate_rank in enumerate(assignment):
                    if candidate_rank == current_assignments[layer_id][expert_id]:
                        continue
                    expert_key = f"layer:{layer_id}/expert:{expert_id}"
                    weight_size = _entry_weight_size_bytes(
                        current_snapshot.get(expert_key, {}),
                        default=default_weight_size,
                    )
                    if weight_size is None:
                        moved_weight_sizes_known = False
                    else:
                        moved_weight_sizes.append(weight_size)
            estimated_moved_bytes = (
                sum(moved_weight_sizes)
                if moved_weight_sizes_known
                else None
            )
            max_moved_bytes = _as_non_negative_int(
                _first_config_value(
                    sources,
                    "maximum_moved_weight_bytes",
                    "max_moved_weight_bytes",
                    "movement_budget_bytes",
                ),
                0,
            )
            estimated_movement_cost = _estimate_expert_weight_movement_cost_ms(
                estimated_moved_bytes,
                moved_count,
                model_config,
                planner_config,
            )
            max_movement_cost_ms = _first_non_negative_float(
                sources,
                "maximum_movement_cost_ms",
                "movement_budget_ms",
                default=0.0,
            )
            movement_weight = _first_non_negative_float(
                sources,
                "hotness_movement_weight",
                "movement_weight",
                default=0.0,
            )
            movement_reference_ms = _first_non_negative_float(
                sources,
                "movement_normalization_reference_ms",
                default=0.0,
            )
            source, _ = _hotness_contract_source(
                model_config, planner_config
            )
            last_change_s = _finite_float(
                source.get(
                    "placement_last_changed_time_s",
                    source.get("expert_placement_last_changed_time_s"),
                )
            )
            snapshot_time_s = _finite_float(
                _first_config_value(sources, "hotness_snapshot_time_s")
            )
            if snapshot_time_s is None:
                snapshot_time_s = time.time()
            cooldown_s = _first_non_negative_float(
                sources, "placement_cooldown_s", default=0.0
            )
            require_known_movement = _to_bool(
                _first_config_value(
                    sources, "require_known_hotness_movement_cost"
                ),
                True,
            )
            retain_reason = ""
            if moved_count == 0:
                retain_reason = "current_already_selected"
            elif current_selection_objective <= 0.0:
                retain_reason = "current_already_balanced"
            elif predicted_improvement_ratio < minimum_improvement:
                retain_reason = "minimum_improvement_not_met"
            elif max_moved_experts > 0 and moved_count > max_moved_experts:
                retain_reason = "moved_expert_budget_exceeded"
            elif (
                max_moved_bytes > 0
                and estimated_moved_bytes is not None
                and estimated_moved_bytes > max_moved_bytes
            ):
                retain_reason = "moved_byte_budget_exceeded"
            elif moved_count and require_known_movement and estimated_moved_bytes is None:
                retain_reason = "movement_cost_unknown"
            elif (
                max_movement_cost_ms > 0.0
                and estimated_movement_cost is not None
                and estimated_movement_cost > max_movement_cost_ms
            ):
                retain_reason = "movement_time_budget_exceeded"
            elif movement_weight > 0.0 and movement_reference_ms <= 0.0:
                retain_reason = "movement_normalization_reference_missing"
            elif movement_weight > 0.0 and estimated_movement_cost is None:
                retain_reason = "movement_cost_unknown"
            elif (
                movement_weight > 0.0
                and candidate_selection_objective
                + movement_weight
                * estimated_movement_cost
                / movement_reference_ms
                >= current_selection_objective
            ):
                retain_reason = "movement_adjusted_objective_not_improved"
            elif (
                cooldown_s > 0.0
                and last_change_s is not None
                and snapshot_time_s - last_change_s < cooldown_s
            ):
                retain_reason = "placement_cooldown_active"
            if retain_reason:
                layer_assignments = {
                    layer_id: list(assignment)
                    for layer_id, assignment in current_assignments.items()
                }
                selection_reason = f"retain_current:{retain_reason}"
            else:
                selection_reason = (
                    "source_rank_locality_improvement_accepted"
                    if placement_strategy == "hotness_locality"
                    else "hotness_balance_improvement_accepted"
                )
        else:
            selection_reason = "hotness_balance_without_current_reference"

    expert_to_target_rank: Dict[str, str] = {}
    expert_to_target_ranks: Dict[str, tuple[str, ...]] = {}
    expert_placement_snapshot: Dict[str, Dict[str, Any]] = {}
    shards: list[ExpertShard] = []
    default_weight_size = _default_expert_weight_size_bytes(
        model_config, planner_config
    )
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            expert_key = f"layer:{layer_id}/expert:{expert_id}"
            rank_index = layer_assignments[layer_id][expert_id]
            ranks: list[str] = []
            for replica_id in range(replica_count):
                rank_id = f"replica:{replica_id}/ep-rank:{rank_index}"
                node_index = (
                    replica_id * target_rank_count + rank_index
                )
                node_id = (
                    target_nodes[node_index % len(target_nodes)]
                    if target_nodes
                    else ""
                )
                gpu_id = str(rank_index)
                shard = ExpertShard(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    rank_id=rank_id,
                    node_id=node_id,
                    gpu_id=gpu_id,
                    physical_expert_id=expert_id,
                    weight_size_bytes=default_weight_size,
                    weight_resident=False,
                    load_score=float(
                        hotness_snapshot.hotness.get(expert_key, 0.0)
                    ),
                )
                shards.append(shard)
                ranks.append(rank_id)
                if replica_id == 0:
                    expert_to_target_rank[expert_key] = rank_id
                    expert_placement_snapshot[expert_key] = shard.to_dict()
            expert_to_target_ranks[expert_key] = tuple(ranks)

    fingerprint_payload = {
        "model_name": model_name,
        "target_parallel_shape": {
            key: target_parallel_plan.get(key)
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "effective_expert_parallel_size",
                "replica_count",
            )
        },
        "expert_to_target_ranks": {
            key: list(value)
            for key, value in expert_to_target_ranks.items()
        },
    }
    fingerprint = _plan_fingerprint(fingerprint_payload)
    logical_expert_count = num_layers * num_experts
    movement = _build_expert_movement_diff(
        expert_placement_snapshot,
        current_snapshot,
        model_config,
        planner_config,
    )
    load_source = (
        hotness_snapshot.hotness
        if hotness_snapshot.hotness
        else {
            f"layer:{layer_id}/expert:{expert_id}": 1.0 / num_experts
            for layer_id in range(num_layers)
            for expert_id in range(num_experts)
        }
    )
    (
        per_layer_rank_load,
        per_layer_load_imbalance,
        per_layer_load_cv,
        load_balance_penalty,
    ) = _load_diagnostics(
        assignments=layer_assignments,
        hotness=load_source,
        rank_count=target_rank_count,
    )
    (
        estimated_remote_assignments,
        estimated_total_assignments,
        estimated_remote_ratio,
        per_source_remote_assignments,
    ) = _remote_dispatch_diagnostics(
        assignments=layer_assignments,
        source_expert_counts=source_expert_counts,
    )
    objective: Optional[float] = float(load_balance_penalty)
    if placement_strategy == "hotness_locality" and estimated_remote_ratio is not None:
        objective += locality_weight * estimated_remote_ratio
    movement_weight = _first_non_negative_float(
        _config_sources(planner_config, model_config),
        "hotness_movement_weight",
        "movement_weight",
        default=0.0,
    )
    if movement_weight > 0.0:
        movement_cost = movement["estimated_expert_weight_movement_cost_ms"]
        reference_cost = _first_non_negative_float(
            _config_sources(planner_config, model_config),
            "movement_normalization_reference_ms",
            default=0.0,
        )
        if movement_cost is None or reference_cost <= 0.0:
            objective = None
        else:
            objective += movement_weight * float(movement_cost) / reference_cost
    return ExpertPlacementPlan(
        model_name=model_name,
        target_parallel_plan=dict(target_parallel_plan),
        expert_to_target_rank=expert_to_target_rank,
        expert_to_target_ranks=expert_to_target_ranks,
        placement_epoch=max(0, int(placement_epoch or 0)),
        placement_source="logical_reparallelization_planner",
        placement_fingerprint=fingerprint,
        required_expert_count=logical_expert_count,
        covered_expert_count=len(expert_to_target_rank),
        planned_shard_count=len(shards),
        target_rank_count=target_rank_count,
        expert_physical_replication_factor=1,
        sllm_replica_count=replica_count,
        physical_weight_migration=False,
        live_expert_remap=live_expert_remap,
        allow_active_requests=live_expert_remap and allow_active_requests,
        require_cross_node=live_expert_remap and require_cross_node,
        physical_migration_required=live_expert_remap,
        expert_placement_snapshot=expert_placement_snapshot,
        shards=tuple(shards),
        movement_observation_available=bool(
            movement["movement_observation_available"]
        ),
        movement_source=movement_source,
        moved_expert_count=int(movement["moved_expert_count"]),
        stationary_expert_count=int(movement["stationary_expert_count"]),
        unknown_movement_expert_count=int(
            movement["unknown_movement_expert_count"]
        ),
        moved_weight_bytes=movement["moved_weight_bytes"],
        estimated_expert_weight_movement_cost_ms=(
            movement["estimated_expert_weight_movement_cost_ms"]
        ),
        expert_movement_diff=movement["expert_movement_diff"],
        estimated_load_balance_penalty=float(load_balance_penalty),
        placement_strategy=placement_strategy,
        selection_reason=selection_reason,
        hotness_snapshot_id=hotness_snapshot.snapshot_id,
        hotness_snapshot_available=hotness_snapshot.available,
        hotness_snapshot_reason=hotness_snapshot.reason,
        hotness_snapshot=hotness_snapshot.to_dict(),
        per_layer_rank_load=per_layer_rank_load,
        per_layer_load_imbalance=per_layer_load_imbalance,
        per_layer_load_cv=per_layer_load_cv,
        hotness_layer_coverage=hotness_snapshot.layer_coverage,
        predicted_improvement_ratio=predicted_improvement_ratio,
        placement_objective=objective,
        source_rank_locality_available=bool(source_expert_counts),
        source_rank_locality_reason=(
            source_rank_locality_reason
            if placement_strategy == "hotness_locality"
            else "not_requested"
        ),
        estimated_remote_assignments=estimated_remote_assignments,
        estimated_total_assignments=estimated_total_assignments,
        estimated_remote_ratio=estimated_remote_ratio,
        per_source_remote_assignments=per_source_remote_assignments,
        reason="logical_expert_placement_plan",
    )
