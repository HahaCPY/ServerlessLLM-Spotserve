import json

from sllm.spot.risk_metadata_provider import (
    EnvironmentRiskMetadataProvider,
    build_risk_metadata_provider,
    normalize_risk_metadata,
)


def test_normalize_risk_metadata_bounds_values_and_records_provenance():
    result = normalize_risk_metadata(
        {
            "risk_score": 1.4,
            "expected_remaining_lifetime_s": -2,
            "model_loading_cost": 3,
            "confidence": 2,
        },
        source="provider",
        provider="fixture",
    )

    assert result["spot_risk"] == 1.0
    assert result["remaining_lifetime_s"] == 0.0
    assert result["loading_cost"] == 3.0
    assert result["risk_metadata_source"] == "provider"
    assert result["risk_provider"] == "fixture"
    assert result["risk_confidence"] == 1.0


def test_environment_provider_reads_node_specific_json(monkeypatch):
    monkeypatch.setenv(
        "SLLM_RISK_METADATA_JSON",
        json.dumps({"node-0": {"spot_risk": 0.25, "remaining_lifetime_s": 900}}),
    )
    provider = EnvironmentRiskMetadataProvider()
    metadata = provider.collect("node-0", {})
    assert metadata["spot_risk"] == 0.25
    assert metadata["remaining_lifetime_s"] == 900.0
    assert metadata["risk_metadata_source"] == "environment"


def test_provider_falls_back_conservatively(monkeypatch):
    monkeypatch.delenv("SLLM_RISK_METADATA_JSON", raising=False)
    monkeypatch.delenv("SLLM_RISK_METADATA_FILE", raising=False)
    monkeypatch.delenv("SLLM_SPOT_RISK", raising=False)
    provider = build_risk_metadata_provider()
    metadata = provider.collect("node-0", {})
    assert metadata["risk_metadata_source"] == "conservative"
    assert metadata["spot_risk"] == 0.0
    assert metadata["remaining_lifetime_s"] == 0.0
