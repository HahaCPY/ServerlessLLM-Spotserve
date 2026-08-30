from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    estimate_expert_dispatch_cost,
)


# 統一 token 格式
def _token_list(tokens: Any) -> List[int]:
    if not tokens:
        return []
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        return [int(token) for token in tokens[0]]
    if isinstance(tokens, list):
        return [int(token) for token in tokens]
    return []

# 一個 request 中斷前留下來的 inference state
@dataclass(frozen=True)
class InferenceState:
    request_id: Optional[str]
    instance_id: str = ""
    node_id: str = ""
    backend: str = ""
    model_name: str = ""
    tokens: List[int] = field(default_factory=list)
    completed_tokens: int = 0
    state_kind: str = "token_snapshot"
    supports_restore: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferenceState":
        tokens = _token_list(payload.get("tokens"))
        completed_tokens = int(
            payload.get("completed_tokens", len(tokens)) or 0
        )
        return cls(
            request_id=payload.get("request_id"),
            instance_id=str(payload.get("instance_id", "")),
            node_id=str(payload.get("node_id", "")),
            backend=str(payload.get("backend", "")),
            model_name=str(payload.get("model_name", "")),
            tokens=tokens,
            completed_tokens=max(0, completed_tokens),
            state_kind=str(payload.get("state_kind", "token_snapshot")),
            supports_restore=bool(payload.get("supports_restore", False)),
            metadata=dict(payload.get("metadata", {}) or {}),
            runtime_state=dict(payload.get("runtime_state", {}) or {}),
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: List[int],
        request_id: Optional[str] = None,
        instance_id: str = "",
        node_id: str = "",
        backend: str = "",
        model_name: str = "",
        completed_tokens: Optional[int] = None,
        state_kind: str = "token_snapshot",
        supports_restore: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        runtime_state: Optional[Mapping[str, Any]] = None,
    ) -> "InferenceState":
        return cls(
            request_id=request_id,
            instance_id=instance_id,
            node_id=node_id,
            backend=backend,
            model_name=model_name,
            tokens=[int(token) for token in tokens],
            completed_tokens=(
                len(tokens)
                if completed_tokens is None
                else max(0, int(completed_tokens))
            ),
            state_kind=state_kind,
            supports_restore=supports_restore,
            metadata=dict(metadata or {}),
            runtime_state=dict(runtime_state or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "backend": self.backend,
            "model_name": self.model_name,
            "tokens": list(self.tokens),
            "completed_tokens": self.completed_tokens,
            "state_kind": self.state_kind,
            "supports_restore": self.supports_restore,
            "metadata": dict(self.metadata),
            "runtime_state": dict(self.runtime_state),
        }


@dataclass(frozen=True)
class StateRecoveryPlan:
    request_id: Optional[str]
    action: str
    source_instance_id: str = ""
    target_instance_id: str = ""
    recovered_tokens: int = 0
    state_kind: str = ""
    fallback_policy: str = ""
    reason: str = "stateful_recovery"
    target_selection: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "action": self.action,
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
            "recovered_tokens": self.recovered_tokens,
            "state_kind": self.state_kind,
            "fallback_policy": self.fallback_policy,
            "reason": self.reason,
        }
        if self.target_selection:
            payload["target_selection"] = dict(self.target_selection)
            selected = self.target_selection.get("selected_candidate")
            if isinstance(selected, Mapping):
                for key in (
                    "model_semantic_compatible",
                    "state_serialization_compatible",
                    "kv_layout_compatible",
                    "kv_restore_compatible",
                    "ep_layout_required",
                    "ep_layout_compatible",
                    "expert_placement_mismatch",
                    "expert_locality_available",
                    "hot_expert_locality_ratio",
                    "estimated_remote_routing_ratio",
                    "estimated_remote_routed_tokens",
                    "expert_dispatch_cost",
                    "moe_route_histogram_source",
                    "moe_route_histogram_kind",
                ):
                    if key in selected:
                        payload[key] = selected[key]
        return payload


@dataclass(frozen=True)
class StateRecoveryDecision:
    action: str
    plan: Optional[StateRecoveryPlan] = None
    state_available: bool = False
    restore_supported: bool = False
    fallback_used: bool = False
    recovered_tokens: int = 0
    reason: str = "stateful_recovery"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "plan": self.plan.to_dict() if self.plan else None,
            "state_available": self.state_available,
            "restore_supported": self.restore_supported,
            "fallback_used": self.fallback_used,
            "recovered_tokens": self.recovered_tokens,
            "reason": self.reason,
        }


def _candidate_value(candidate: Mapping[str, Any], key: str) -> Any:
    """Read a runtime field from either the top-level or vLLM profile."""
    if key in candidate:
        return candidate.get(key)
    profile = candidate.get("model_resource_profile")
    if isinstance(profile, Mapping):
        value = profile.get(key)
        if value is not None:
            return value
    metadata = candidate.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _state_value(state: InferenceState, key: str) -> Any:
    if key == "model_name":
        return state.model_name
    if key == "backend":
        return state.backend
    return (state.metadata or {}).get(key)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _compatibility_check(
    compatible: bool,
    reason: str,
) -> Dict[str, Any]:
    return {"compatible": compatible, "reason": reason}


def _model_semantic_compatible(
    state: InferenceState,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(_candidate_value(candidate, "backend") or "") not in {
        "",
        str(state.backend or ""),
    }:
        return _compatibility_check(False, "backend_mismatch")
    candidate_model = _candidate_value(candidate, "model_name")
    if candidate_model and state.model_name and str(candidate_model) != str(
        state.model_name
    ):
        return _compatibility_check(False, "model_mismatch")
    for key in (
        "model_revision",
        "tokenizer_revision",
        "gate_model_revision",
        "moe_backend",
    ):
        source_value = _state_value(state, key)
        target_value = _candidate_value(candidate, key)
        if source_value is None or target_value is None:
            continue
        if str(source_value) != str(target_value):
            return _compatibility_check(False, f"{key}_mismatch")
    return _compatibility_check(True, "compatible_model_semantics")


def _state_serialization_compatible(
    state: InferenceState,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    if not state.supports_restore or state.state_kind != "vllm_kv_snapshot":
        return _compatibility_check(False, "state_restore_not_advertised")
    supported_state_kinds = _candidate_value(candidate, "supported_state_kinds")
    if isinstance(supported_state_kinds, (list, tuple, set)):
        if state.state_kind not in {str(kind) for kind in supported_state_kinds}:
            return _compatibility_check(False, "state_kind_unsupported")
    elif _candidate_value(candidate, "state_kind"):
        target_kind = str(_candidate_value(candidate, "state_kind"))
        if target_kind and target_kind != state.state_kind:
            return _compatibility_check(False, "state_kind_mismatch")
    for key in ("sampling_state_encoding", "request_metadata_encoding"):
        source_value = _state_value(state, key)
        target_value = _candidate_value(candidate, key)
        if source_value is None or target_value is None:
            continue
        if str(source_value) != str(target_value):
            return _compatibility_check(False, f"{key}_mismatch")
    return _compatibility_check(True, "compatible_state_serialization")


def _kv_layout_compatible(
    state: InferenceState,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require evidence that a restore target has the same KV/cache shape.

    Unknown optional fields are tolerated because older backends do not expose
    every runtime field.  A field exposed by both source and target must match;
    this prevents the planner from selecting a different TP/PP or cache layout
    and hoping that NIXL will repartition it.
    """
    compatibility_keys = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "cache_block_size",
        "cache_dtype",
        "cache_layout",
        "cache_config_fingerprint",
        "cache_engine",
        "kv_connector",
    )
    for key in compatibility_keys:
        source_value = _state_value(state, key)
        target_value = _candidate_value(candidate, key)
        if source_value is None or target_value is None:
            continue
        if str(source_value) != str(target_value):
            return _compatibility_check(False, f"{key}_mismatch")
    return _compatibility_check(True, "compatible_kv_layout")


def _ep_layout_compatible(
    state: InferenceState,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    ep_required = _boolish(
        _state_value(state, "state_restore_requires_ep_layout")
    ) or _boolish(_candidate_value(candidate, "state_restore_requires_ep_layout"))
    mismatch_keys: List[str] = []
    for key in (
        "effective_expert_parallel_size",
        "expert_parallel_enabled",
        "expert_placement_fingerprint",
    ):
        source_value = _state_value(state, key)
        target_value = _candidate_value(candidate, key)
        if source_value is None or target_value is None:
            continue
        if str(source_value) != str(target_value):
            mismatch_keys.append(key)
    if ep_required:
        if mismatch_keys:
            return {
                "compatible": False,
                "reason": f"{mismatch_keys[0]}_mismatch",
                "required": True,
                "mismatch_keys": mismatch_keys,
            }
        return {
            "compatible": True,
            "reason": "compatible_ep_layout",
            "required": True,
            "mismatch_keys": [],
        }
    return {
        "compatible": True,
        "reason": (
            "ep_layout_mismatch_is_locality_only"
            if mismatch_keys
            else "ep_layout_not_required"
        ),
        "required": False,
        "mismatch_keys": mismatch_keys,
    }


def _merged_candidate_metadata(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    profile = candidate.get("model_resource_profile")
    if isinstance(profile, Mapping):
        metadata.update(profile)
    nested = candidate.get("metadata")
    if isinstance(nested, Mapping):
        metadata.update(nested)
    for key, value in candidate.items():
        if key not in {"model_resource_profile", "metadata"}:
            metadata[key] = value
    return metadata


def _state_context_metadata(state: InferenceState) -> ContextMetadata:
    metadata = dict(state.metadata or {})
    try:
        context_blocks = int(metadata.get("kv_block_count", 0) or 0)
    except (TypeError, ValueError):
        context_blocks = 0
    return ContextMetadata(
        request_id=state.request_id,
        instance_id=state.instance_id,
        node_id=state.node_id,
        num_tokens=state.completed_tokens or len(state.tokens),
        context_blocks=max(0, context_blocks),
        tokens=tuple(state.tokens),
        metadata=metadata,
    )


def _candidate_migration_target(
    candidate: Mapping[str, Any],
) -> MigrationTarget:
    return MigrationTarget(
        instance_id=str(candidate.get("instance_id", "")),
        node_id=str(candidate.get("node_id", "") or ""),
        warmup_cost=float(candidate.get("warmup_cost", 0.0) or 0.0),
        concurrency=max(0, int(candidate.get("concurrency", 0) or 0)),
        metadata=_merged_candidate_metadata(candidate),
    )


def _expert_recovery_locality(
    state: InferenceState,
    candidate: Mapping[str, Any],
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    estimate = estimate_expert_dispatch_cost(
        _state_context_metadata(state),
        _candidate_migration_target(candidate),
        planner_config or {},
    )
    return {
        "expert_locality_available": bool(estimate.get("available", False)),
        "hot_expert_locality_ratio": float(
            estimate.get("locality_ratio", 0.0) or 0.0
        ),
        "estimated_remote_routing_ratio": float(
            estimate.get("estimated_remote_routing_ratio", 0.0) or 0.0
        ),
        "estimated_remote_routed_tokens": int(
            estimate.get("estimated_remote_routed_tokens", 0) or 0
        ),
        "expert_dispatch_cost": float(estimate.get("cost", 0.0) or 0.0),
        "moe_route_histogram_source": str(
            estimate.get("route_histogram_source", "unavailable")
            or "unavailable"
        ),
        "moe_route_histogram_kind": str(
            estimate.get("route_histogram_kind", "unavailable")
            or "unavailable"
        ),
    }


def _candidate_restore_score(
    state: InferenceState,
    candidate: Mapping[str, Any],
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    model_check = _model_semantic_compatible(state, candidate)
    serialization_check = _state_serialization_compatible(state, candidate)
    kv_check = _kv_layout_compatible(state, candidate)
    ep_check = _ep_layout_compatible(state, candidate)
    locality = _expert_recovery_locality(state, candidate, planner_config)
    source_node = str(state.node_id or "")
    target_node = str(candidate.get("node_id", "") or "")
    same_node = bool(source_node and target_node and source_node == target_node)
    scope_cost = 0.0 if same_node else float(
        (planner_config or {}).get("cross_node_restore_cost", 1.0) or 0.0
    )
    warmup_cost = float(candidate.get("warmup_cost", 0.0) or 0.0)
    concurrency = max(0, int(candidate.get("concurrency", 0) or 0))
    total_cost = (
        scope_cost
        + warmup_cost
        + float(locality["expert_dispatch_cost"])
        + float((planner_config or {}).get("concurrency_weight", 0.0) or 0.0)
        * concurrency
    )
    return {
        "instance_id": str(candidate.get("instance_id", "")),
        "node_id": target_node,
        "same_node": same_node,
        "model_semantic_compatible": bool(model_check["compatible"]),
        "model_semantic_reason": str(model_check["reason"]),
        "state_serialization_compatible": bool(
            serialization_check["compatible"]
        ),
        "state_serialization_reason": str(serialization_check["reason"]),
        "kv_layout_compatible": bool(kv_check["compatible"]),
        "kv_layout_reason": str(kv_check["reason"]),
        "kv_restore_compatible": bool(
            model_check["compatible"]
            and serialization_check["compatible"]
            and kv_check["compatible"]
            and ep_check["compatible"]
        ),
        "ep_layout_required": bool(ep_check.get("required", False)),
        "ep_layout_compatible": bool(ep_check["compatible"]),
        "ep_layout_reason": str(ep_check["reason"]),
        "ep_layout_mismatch_keys": list(ep_check.get("mismatch_keys", [])),
        "expert_placement_mismatch": bool(ep_check.get("mismatch_keys", [])),
        "restore_scope_cost": float(scope_cost),
        "warmup_cost": float(warmup_cost),
        "concurrency": concurrency,
        "total_estimated_cost": float(total_cost),
        **locality,
    }


def plan_compatible_state_target(
    state: InferenceState,
    candidates: Sequence[Mapping[str, Any]],
    source_instance_id: str = "",
    planner_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Choose a READY target on which the exported state may be restored.

    This is deliberately narrower than generic re-parallelization: it only
    chooses an already-ready target whose runtime proves a compatible cache
    shape.  It never changes TP/PP/EP and never creates an engine.
    """
    planner_config = planner_config or {}
    if not state.supports_restore or state.state_kind != "vllm_kv_snapshot":
        return {
            "action": "fallback_token_replay",
            "target_instance_id": None,
            "reason": "state_restore_not_advertised",
            "candidates": [],
        }

    allow_cross_node = bool(
        (state.metadata or {}).get("can_restore_cross_node", False)
    )
    ranked = []
    rejected = []
    accepted = []
    for candidate in candidates:
        instance_id = str(candidate.get("instance_id", ""))
        if not instance_id or instance_id == str(source_instance_id):
            continue
        if not bool(candidate.get("ready", False)):
            continue
        if not bool(candidate.get("supports_state_restore", False)):
            rejected.append({
                "instance_id": instance_id,
                "reason": "target_restore_unsupported",
            })
            continue
        source_node = str(state.node_id or "")
        target_node = str(candidate.get("node_id", "") or "")
        if source_node and target_node and source_node != target_node:
            if not allow_cross_node:
                rejected.append({
                    "instance_id": instance_id,
                    "reason": "cross_node_restore_unsupported",
                })
                continue
        score = _candidate_restore_score(
            state=state,
            candidate=candidate,
            planner_config=planner_config,
        )
        if not score["kv_restore_compatible"]:
            rejected.append({
                "instance_id": instance_id,
                "reason": (
                    score["ep_layout_reason"]
                    if not score["ep_layout_compatible"]
                    else score["kv_layout_reason"]
                    if not score["kv_layout_compatible"]
                    else score["state_serialization_reason"]
                    if not score["state_serialization_compatible"]
                    else score["model_semantic_reason"]
                ),
                "compatibility": score,
            })
            continue
        accepted.append(score)
        ranked.append(
            (
                float(score["total_estimated_cost"]),
                0 if source_node == target_node else 1,
                float(score["warmup_cost"]),
                int(score["concurrency"]),
                instance_id,
                score,
            )
        )

    if not ranked:
        return {
            "action": "fallback_token_replay",
            "target_instance_id": None,
            "reason": "no_compatible_ready_target",
            "candidates": rejected,
        }

    ranked.sort()
    target_instance_id = ranked[0][4]
    selected = dict(ranked[0][5])
    selected["rank"] = 0
    return {
        "action": "restore_state",
        "target_instance_id": target_instance_id,
        "reason": "planner_selected_compatible_ready_target",
        "selected_candidate": selected,
        "candidates": [
            {
                "instance_id": item[4],
                "rank": index,
                "compatibility": item[5],
            }
            for index, item in enumerate(ranked)
        ],
        "rejected_candidates": rejected,
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": len(rejected),
    }


def plan_stateful_recovery(
    request_id: Optional[str],
    source_instance_id: str,
    target_instance_id: str,
    state: Optional[InferenceState],
    restore_supported: bool,
    fallback_policy: str = "generated_token_replay",
    reason: str = "stateful_recovery",
    target_selection: Optional[Mapping[str, Any]] = None,
) -> StateRecoveryDecision:
    target_selection_dict = dict(target_selection or {})
    if state is None or not state.tokens:
        plan = StateRecoveryPlan(
            request_id=request_id,
            action="retry",
            source_instance_id=source_instance_id,
            target_instance_id=target_instance_id,
            fallback_policy="naive_retry",
            reason="no_state_available",
            target_selection=target_selection_dict,
        )
        return StateRecoveryDecision(
            action="retry",
            plan=plan,
            state_available=False,
            restore_supported=restore_supported,
            fallback_used=True,
            recovered_tokens=0,
            reason=reason if reason != "stateful_recovery" else "no_state_available",
        )

    recovered_tokens = state.completed_tokens or len(state.tokens)
    if restore_supported and state.supports_restore:
        plan = StateRecoveryPlan(
            request_id=request_id,
            action="restore_state",
            source_instance_id=source_instance_id,
            target_instance_id=target_instance_id,
            recovered_tokens=recovered_tokens,
            state_kind=state.state_kind,
            reason="backend_state_restore",
            target_selection=target_selection_dict,
        )
        return StateRecoveryDecision(
            action="restore_state",
            plan=plan,
            state_available=True,
            restore_supported=True,
            fallback_used=False,
            recovered_tokens=recovered_tokens,
            reason=reason if reason != "stateful_recovery" else "backend_state_restore",
        )

    plan = StateRecoveryPlan(
        request_id=request_id,
        action="fallback_token_replay",
        source_instance_id=source_instance_id,
        target_instance_id=target_instance_id,
        recovered_tokens=recovered_tokens,
        state_kind=state.state_kind,
        fallback_policy=fallback_policy,
        reason="state_restore_unsupported",
        target_selection=target_selection_dict,
    )
    return StateRecoveryDecision(
        action="fallback_token_replay",
        plan=plan,
        state_available=True,
        restore_supported=False,
        fallback_used=True,
        recovered_tokens=recovered_tokens,
        reason=reason if reason != "stateful_recovery" else "state_restore_unsupported",
    )
