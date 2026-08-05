"""Pluggable provider/runtime risk metadata for SpotServe scheduling.

Cloud-specific risk APIs are deliberately not hard-coded into the scheduler.
Production deployments can provide ``module:callable`` through
``SLLM_RISK_PROVIDER``; local and air-gapped deployments can use a JSON file or
environment values.  Every provider is normalized to bounded values and falls
back to conservative defaults when unavailable.
"""

import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol


RISK_FIELDS = (
    "spot_risk",
    "remaining_lifetime_s",
    "loading_cost",
    "risk_score",
    "preemption_risk",
)

_SIGNAL_FIELDS = set(
    RISK_FIELDS
    + (
        "expected_remaining_lifetime_s",
        "model_loading_cost",
        "load_cost",
        "free_gpu",
        "total_gpu",
    )
)


def _has_signal(payload: Mapping[str, Any]) -> bool:
    return any(payload.get(key) is not None for key in _SIGNAL_FIELDS)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_risk(value: Any) -> float:
    return min(1.0, max(0.0, _float(value, 0.0)))


def _non_negative(value: Any) -> float:
    return max(0.0, _float(value, 0.0))


def normalize_risk_metadata(
    payload: Optional[Mapping[str, Any]],
    *,
    source: str,
    provider: str = "",
) -> Dict[str, Any]:
    """Normalize aliases, bounds, and provenance for scheduler consumption."""
    payload = payload or {}
    result: Dict[str, Any] = {}
    spot_value = next(
        (
            payload.get(key)
            for key in ("spot_risk", "risk_score", "preemption_risk")
            if payload.get(key) is not None
        ),
        None,
    )
    if spot_value is not None:
        result["spot_risk"] = _bounded_risk(spot_value)
    lifetime = next(
        (
            payload.get(key)
            for key in ("remaining_lifetime_s", "expected_remaining_lifetime_s")
            if payload.get(key) is not None
        ),
        None,
    )
    if lifetime is not None:
        result["remaining_lifetime_s"] = _non_negative(lifetime)
    loading = next(
        (
            payload.get(key)
            for key in ("loading_cost", "model_loading_cost", "load_cost")
            if payload.get(key) is not None
        ),
        None,
    )
    if loading is not None:
        result["loading_cost"] = _non_negative(loading)

    # Providers may also expose capacity/lifetime state.  Keep these fields
    # bounded and typed so scheduler ranking never receives malformed values.
    for key in ("free_gpu", "total_gpu"):
        if payload.get(key) is not None:
            try:
                result[key] = max(0, int(payload[key]))
            except (TypeError, ValueError):
                pass
    for key in ("provider", "region", "instance_type", "instance_id"):
        if payload.get(key) is not None:
            result[key] = str(payload[key])

    result["risk_metadata_source"] = source
    if provider:
        result["risk_provider"] = provider
    result["risk_observed_at"] = _non_negative(
        payload.get("observed_at", time.time())
    )
    default_confidence = 0.0 if source == "conservative" else 1.0
    result["risk_confidence"] = min(
        1.0,
        max(
            0.0,
            _float(payload.get("confidence", default_confidence), default_confidence),
        ),
    )
    return result


class RiskMetadataProvider(Protocol):
    def collect(
        self, node_id: str, node_info: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Return raw or normalized risk metadata for one worker node."""


class ConservativeRiskMetadataProvider:
    """No-provider fallback; ranking code supplies safe defaults."""

    def collect(self, node_id: str, node_info: Mapping[str, Any]):
        return normalize_risk_metadata(
            {
                "spot_risk": 0.0,
                "remaining_lifetime_s": 0.0,
                "loading_cost": 0.0,
                "confidence": 0.0,
            },
            source="conservative",
        )


class EnvironmentRiskMetadataProvider:
    """Read a deployment-provided JSON map or node-local environment values."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        config = config or {}
        self.metadata_file = str(
            config.get("risk_metadata_file")
            or os.getenv("SLLM_RISK_METADATA_FILE", "")
        )
        self.metadata_json = str(
            config.get("risk_metadata_json")
            or os.getenv("SLLM_RISK_METADATA_JSON", "")
        )
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        raw = self.metadata_json
        if self.metadata_file:
            try:
                raw = Path(self.metadata_file).read_text(encoding="utf-8")
            except OSError:
                raw = ""
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}

    def collect(self, node_id: str, node_info: Mapping[str, Any]):
        payload = self._metadata.get(str(node_id), self._metadata.get("*", {}))
        if not isinstance(payload, Mapping):
            payload = {}
        env_payload = {
            "spot_risk": os.getenv("SLLM_SPOT_RISK"),
            "remaining_lifetime_s": os.getenv("SLLM_REMAINING_LIFETIME_S"),
            "loading_cost": os.getenv("SLLM_LOADING_COST"),
            "provider": os.getenv("SLLM_RISK_PROVIDER_NAME"),
            "region": os.getenv("SLLM_NODE_REGION"),
            "instance_type": os.getenv("SLLM_INSTANCE_TYPE"),
            "confidence": os.getenv("SLLM_RISK_CONFIDENCE"),
        }
        merged = dict(env_payload)
        merged.update(dict(payload))
        merged = {key: value for key, value in merged.items() if value is not None}
        if not merged or not _has_signal(merged):
            return {}
        return normalize_risk_metadata(
            merged,
            source="environment",
            provider=str(merged.get("provider", "environment")),
        )


class CallableRiskMetadataProvider:
    """Adapter for a production provider callable loaded by import path."""

    def __init__(self, callback: Callable[..., Mapping[str, Any]], name: str):
        self.callback = callback
        self.name = name

    def collect(self, node_id: str, node_info: Mapping[str, Any]):
        try:
            payload = self.callback(node_id=node_id, node_info=dict(node_info))
        except TypeError:
            payload = self.callback(node_id, dict(node_info))
        if not isinstance(payload, Mapping) or not _has_signal(payload):
            payload = {}
        return normalize_risk_metadata(
            payload,
            source="provider",
            provider=self.name,
        )


class CompositeRiskMetadataProvider:
    def __init__(self, providers):
        self.providers = list(providers)

    def collect(self, node_id: str, node_info: Mapping[str, Any]):
        result: Dict[str, Any] = {}
        for provider in self.providers:
            try:
                payload = provider.collect(node_id, node_info)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                if payload.get("risk_metadata_source") == "conservative" or not _has_signal(payload):
                    for key, value in payload.items():
                        result.setdefault(key, value)
                else:
                    result.update(dict(payload))
        return result or normalize_risk_metadata({}, source="conservative")


def _load_callable(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("risk provider must use module:callable syntax")
    target = getattr(importlib.import_module(module_name), attribute)
    if isinstance(target, type):
        target = target()
    if not callable(target):
        raise TypeError("risk provider callable is not callable")
    return target


def build_risk_metadata_provider(
    config: Optional[Mapping[str, Any]] = None,
) -> RiskMetadataProvider:
    config = config or {}
    providers = []
    provider_path = str(
        config.get("risk_provider") or os.getenv("SLLM_RISK_PROVIDER", "")
    )
    if provider_path:
        try:
            callback = _load_callable(provider_path)
            providers.append(CallableRiskMetadataProvider(callback, provider_path))
        except Exception:
            # A bad optional provider must not prevent scheduling; use the
            # conservative fallback and expose the failure through provenance.
            pass
    providers.append(EnvironmentRiskMetadataProvider(config))
    providers.append(ConservativeRiskMetadataProvider())
    return CompositeRiskMetadataProvider(providers)
