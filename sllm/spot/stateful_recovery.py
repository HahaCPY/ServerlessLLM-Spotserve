from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
            "recovered_tokens": self.recovered_tokens,
            "state_kind": self.state_kind,
            "fallback_policy": self.fallback_policy,
            "reason": self.reason,
        }


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
        return profile.get(key)
    return None


def _state_value(state: InferenceState, key: str) -> Any:
    if key == "model_name":
        return state.model_name
    if key == "backend":
        return state.backend
    return (state.metadata or {}).get(key)


def _parallel_and_cache_compatible(
    state: InferenceState,
    candidate: Mapping[str, Any],
) -> tuple[bool, str]:
    """Require evidence that a restore target has the same cache shape.

    Unknown optional fields are tolerated because older backends do not expose
    every runtime field.  A field exposed by both source and target must match;
    this prevents the planner from selecting a different TP/PP or cache layout
    and hoping that NIXL will repartition it.
    """
    if str(_candidate_value(candidate, "backend") or "") not in {
        "",
        str(state.backend or ""),
    }:
        return False, "backend_mismatch"
    candidate_model = _candidate_value(candidate, "model_name")
    if candidate_model and state.model_name and str(candidate_model) != str(
        state.model_name
    ):
        return False, "model_mismatch"

    compatibility_keys = (
        "model_revision",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "effective_expert_parallel_size",
        "expert_parallel_enabled",
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
            return False, f"{key}_mismatch"
    return True, "compatible_parallel_and_cache_shape"


def plan_compatible_state_target(
    state: InferenceState,
    candidates: Sequence[Mapping[str, Any]],
    source_instance_id: str = "",
) -> Dict[str, Any]:
    """Choose a READY target on which the exported state may be restored.

    This is deliberately narrower than generic re-parallelization: it only
    chooses an already-ready target whose runtime proves a compatible cache
    shape.  It never changes TP/PP/EP and never creates an engine.
    """
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
        compatible, reason = _parallel_and_cache_compatible(state, candidate)
        if not compatible:
            rejected.append({"instance_id": instance_id, "reason": reason})
            continue
        ranked.append(
            (
                0 if source_node == target_node else 1,
                float(candidate.get("warmup_cost", 0.0) or 0.0),
                int(candidate.get("concurrency", 0) or 0),
                instance_id,
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
    target_instance_id = ranked[0][-1]
    return {
        "action": "restore_state",
        "target_instance_id": target_instance_id,
        "reason": "planner_selected_compatible_ready_target",
        "candidates": [
            {"instance_id": item[-1], "rank": index}
            for index, item in enumerate(ranked)
        ],
    }


def plan_stateful_recovery(
    request_id: Optional[str],
    source_instance_id: str,
    target_instance_id: str,
    state: Optional[InferenceState],
    restore_supported: bool,
    fallback_policy: str = "generated_token_replay",
    reason: str = "stateful_recovery",
) -> StateRecoveryDecision:
    if state is None or not state.tokens:
        plan = StateRecoveryPlan(
            request_id=request_id,
            action="retry",
            source_instance_id=source_instance_id,
            target_instance_id=target_instance_id,
            fallback_policy="naive_retry",
            reason="no_state_available",
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
