import pytest

from sllm.backends.dummy_backend import DummyBackend
from sllm.spot.metrics import make_state_recovery_event
from sllm.spot.stateful_recovery import (
    InferenceState,
    plan_stateful_recovery,
)


@pytest.fixture(autouse=True)
def clear_forced_failures():
    DummyBackend._forced_failures_seen.clear()


def test_stateful_recovery_plan_uses_backend_restore_when_supported():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-stateful",
        instance_id="old-a",
        completed_tokens=3,
        state_kind="dummy_token_state",
        supports_restore=True,
    )

    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=True,
    )

    assert decision.action == "restore_state"
    assert decision.fallback_used is False
    assert decision.recovered_tokens == 3
    assert decision.plan.target_instance_id == "new-a"


def test_stateful_recovery_plan_falls_back_without_backend_restore():
    state = InferenceState.from_tokens(
        tokens=[1, 2, 3],
        request_id="req-stateful",
        instance_id="old-a",
        completed_tokens=3,
        supports_restore=False,
    )

    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=False,
    )

    assert decision.action == "fallback_token_replay"
    assert decision.fallback_used is True
    assert decision.recovered_tokens == 3


def test_state_recovery_metric_contains_restore_summary():
    state = InferenceState.from_tokens(
        tokens=[1, 2],
        request_id="req-stateful",
        supports_restore=True,
    )
    decision = plan_stateful_recovery(
        request_id="req-stateful",
        source_instance_id="old-a",
        target_instance_id="new-a",
        state=state,
        restore_supported=True,
    )

    event = make_state_recovery_event(
        model="dummy-stateful",
        request_id="req-stateful",
        decision=decision.to_dict(),
    )

    assert event["type"] == "state_recovery"
    assert event["action"] == "restore_state"
    assert event["recovered_tokens"] == 2
    assert event["fallback_used"] is False


@pytest.mark.asyncio
async def test_dummy_backend_exports_and_restores_inference_state():
    source = DummyBackend("dummy-stateful", {})
    target = DummyBackend("dummy-stateful", {})
    request = {
        "model": "dummy-stateful",
        "request_id": "stateful-restore-1",
        "messages": [{"role": "user", "content": "restore state"}],
        "max_tokens": 6,
        "token_latency": 0.0,
        "force_failure": "preempted",
        "force_fail_after_tokens": 2,
        "force_fail_once": True,
    }

    first_result = await source.generate(request)
    state = await source.export_inference_state(
        request_data=request,
        current_output=first_result["current_output"],
        completed_tokens=first_result["completed_tokens"],
    )
    restored = await target.restore_inference_state(
        state=state,
        request_data=request,
    )
    second_result = await target.generate(request)

    assert first_result["preempted"] is True
    assert state["supports_restore"] is True
    assert restored["restored"] is True
    assert second_result["usage"]["completion_tokens"] == 6
