from enum import Enum


class RecoveryPolicy(str, Enum):
    NONE = "none"
    NAIVE_RETRY = "naive_retry"
    GENERATED_TOKEN_REPLAY = "generated_token_replay"
    STATEFUL_RECOVERY = "stateful_recovery"


def normalize_policy(policy: str | RecoveryPolicy | None) -> RecoveryPolicy:
    if policy is None:
        return RecoveryPolicy.NONE
    if isinstance(policy, RecoveryPolicy):
        return policy
    try:
        return RecoveryPolicy(policy)
    except ValueError as exc:
        supported = ", ".join(item.value for item in RecoveryPolicy)
        raise ValueError(
            f"Unsupported recovery policy '{policy}'. Supported: {supported}"
        ) from exc
