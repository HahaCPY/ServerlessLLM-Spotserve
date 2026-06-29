import pytest

from sllm.spot.recovery_policy import RecoveryPolicy, normalize_policy


def test_normalize_policy_accepts_supported_values():
    assert normalize_policy(None) == RecoveryPolicy.NONE
    assert normalize_policy("none") == RecoveryPolicy.NONE
    assert normalize_policy("naive_retry") == RecoveryPolicy.NAIVE_RETRY
    assert (
        normalize_policy("generated_token_replay")
        == RecoveryPolicy.GENERATED_TOKEN_REPLAY
    )


def test_normalize_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported recovery policy"):
        normalize_policy("kv_teleport")
