import pytest

from scripts.check_granite_native_freeze import validate_route_histogram


def observation():
    return {
        "moe_route_histogram_available": True,
        "moe_route_histogram_source": "vllm_runtime_topk",
        "moe_route_histogram_kind": "runtime_observed_topk",
        "per_request_expert_route_histogram": {
            "layer:0/expert:0": 8, "layer:1/expert:1": 8,
        },
    }


def test_live_histogram_is_bounded_and_does_not_claim_full_prefix_coverage():
    result = validate_route_histogram(observation(), 2, 40, 8)
    assert result["observed_routed_tokens"] == 1
    assert result["covers_all_frozen_prompt_tokens"] is None
    assert result["physical_dispatch_traffic_verified"] is False


@pytest.mark.parametrize("key,value", [
    ("layer:2/expert:0", 8), ("layer:0/expert:40", 8),
    ("layer:0/expert:0", True), ("layer:0/expert:0", 0),
    ("layer:0/expert:0", 7), ("invalid", 8),
])
def test_invalid_or_incomplete_histogram_is_rejected(key, value):
    metadata = observation()
    metadata["per_request_expert_route_histogram"] = {
        key: value, "layer:1/expert:1": 8,
    }
    with pytest.raises(ValueError):
        validate_route_histogram(metadata, 2, 40, 8)


def test_unobserved_routing_is_rejected():
    metadata = observation()
    metadata["moe_route_histogram_available"] = False
    with pytest.raises(ValueError):
        validate_route_histogram(metadata, 2, 40, 8)
