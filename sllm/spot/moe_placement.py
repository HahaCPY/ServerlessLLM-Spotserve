import json
import re
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
    weight_size_bytes: int = 0
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
    expert_placement_snapshot: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    shards: tuple[ExpertShard, ...] = field(default_factory=tuple)
    movement_observation_available: bool = False
    movement_source: str = "unavailable"
    moved_expert_count: int = 0
    stationary_expert_count: int = 0
    unknown_movement_expert_count: int = 0
    moved_weight_bytes: int = 0
    estimated_expert_weight_movement_cost_ms: float = 0.0
    expert_movement_diff: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    estimated_dispatch_cost: float = 0.0
    estimated_load_balance_penalty: float = 0.0
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
            "reason": self.reason,
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
    if not isinstance(snapshot, Mapping) or not snapshot:
        return 0, 0

    max_layer_id = -1
    max_expert_id = -1
    for key, value in snapshot.items():
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

    if max_layer_id < 0 or max_expert_id < 0:
        return 0, 0
    return max_layer_id + 1, max_expert_id + 1


def infer_moe_topology(
    model_config: Mapping[str, Any],
) -> tuple[int, int]:
    """Return logical MoE topology as (num_layers, experts_per_layer)."""
    for config in _model_config_sources(model_config):
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


def _current_expert_placement_snapshot(
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], str]:
    planner_snapshot = planner_config.get("current_expert_placement_snapshot")
    normalized = _normalize_expert_placement_snapshot(planner_snapshot)
    if normalized:
        return normalized, "planner_current_expert_placement_snapshot"

    runtime_metadata = model_config.get("runtime_metadata")
    if isinstance(runtime_metadata, Mapping):
        normalized = _normalize_expert_placement_snapshot(
            runtime_metadata.get("expert_placement_snapshot")
            or runtime_metadata.get("expert_placement")
        )
        if normalized:
            return normalized, "runtime_metadata"
        profile = runtime_metadata.get("model_resource_profile")
        if isinstance(profile, Mapping):
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
            normalized = _normalize_expert_placement_snapshot(
                backend_runtime.get("expert_placement_snapshot")
                or backend_runtime.get("expert_placement")
            )
            if normalized:
                return normalized, "backend_config.runtime_metadata"
            profile = backend_runtime.get("model_resource_profile")
            if isinstance(profile, Mapping):
                normalized = _normalize_expert_placement_snapshot(
                    profile.get("expert_placement_snapshot")
                    or profile.get("expert_placement")
                )
                if normalized:
                    return (
                        normalized,
                        "backend_config.runtime_metadata.model_resource_profile",
                    )

        normalized = _normalize_expert_placement_snapshot(
            backend_config.get("expert_placement_snapshot")
            or backend_config.get("expert_placement")
        )
        if normalized:
            return normalized, "backend_config"

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
    default: int = 0,
) -> int:
    for entry in entries:
        for key in ("weight_size_bytes", "expert_weight_size_bytes"):
            parsed = _as_positive_int(entry.get(key), 0)
            if parsed > 0:
                return parsed
    return max(0, int(default or 0))


def _default_expert_weight_size_bytes(
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> int:
    sources = _config_sources(planner_config, model_config)
    value = _first_config_value(
        sources,
        "expert_weight_size_bytes",
        "default_expert_weight_size_bytes",
        "moe_expert_weight_size_bytes",
    )
    return _as_positive_int(value, 0)


def _estimate_expert_weight_movement_cost_ms(
    moved_weight_bytes: int,
    moved_expert_count: int,
    model_config: Mapping[str, Any],
    planner_config: Mapping[str, Any],
) -> float:
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
            "moved_weight_bytes": 0,
            "estimated_expert_weight_movement_cost_ms": 0.0,
            "expert_movement_diff": {},
        }

    default_weight_size = _default_expert_weight_size_bytes(
        model_config, planner_config
    )
    moved_experts = 0
    stationary_experts = 0
    unknown_experts = 0
    moved_weight_bytes = 0
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

    movement_cost_ms = _estimate_expert_weight_movement_cost_ms(
        moved_weight_bytes,
        moved_experts,
        model_config,
        planner_config,
    )
    return {
        "movement_observation_available": True,
        "moved_expert_count": moved_experts,
        "stationary_expert_count": stationary_experts,
        "unknown_movement_expert_count": unknown_experts,
        "moved_weight_bytes": moved_weight_bytes,
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
            rank_index = expert_id % target_rank_count
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
                )
                shards.append(shard)
                ranks.append(rank_id)
                if replica_id == 0:
                    expert_to_target_rank[expert_key] = rank_id
                    expert_placement_snapshot[expert_key] = shard.to_dict()
            expert_to_target_ranks[expert_key] = tuple(ranks)

    fingerprint_payload = {
        "model_name": model_name,
        "placement_epoch": max(0, int(placement_epoch or 0)),
        "target_parallel_plan": dict(target_parallel_plan),
        "expert_to_target_ranks": {
            key: list(value)
            for key, value in expert_to_target_ranks.items()
        },
    }
    fingerprint = _plan_fingerprint(fingerprint_payload)
    logical_expert_count = num_layers * num_experts
    current_snapshot, movement_source = _current_expert_placement_snapshot(
        model_config, planner_config
    )
    movement = _build_expert_movement_diff(
        expert_placement_snapshot,
        current_snapshot,
        model_config,
        planner_config,
    )
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
        moved_weight_bytes=int(movement["moved_weight_bytes"]),
        estimated_expert_weight_movement_cost_ms=float(
            movement["estimated_expert_weight_movement_cost_ms"]
        ),
        expert_movement_diff=movement["expert_movement_diff"],
        reason="logical_expert_placement_plan",
    )
