from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


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


def plan_stateful_recovery(
    request_id: Optional[str],
    source_instance_id: str,
    target_instance_id: str,
    state: Optional[InferenceState],
    restore_supported: bool,
    fallback_policy: str = "generated_token_replay",
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
            reason="no_state_available",
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
            reason="backend_state_restore",
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
        reason="state_restore_unsupported",
    )
