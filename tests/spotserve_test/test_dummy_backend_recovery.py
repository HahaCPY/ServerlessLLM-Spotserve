import pytest

from sllm.backends.dummy_backend import DummyBackend


@pytest.fixture(autouse=True)
def clear_forced_failures():
    DummyBackend._forced_failures_seen.clear()


@pytest.mark.asyncio
async def test_dummy_backend_forced_preemption_reports_partial_tokens_once():
    backend = DummyBackend("dummy-correctness-test", {})
    request = {
        "model": "dummy-correctness-test",
        "request_id": "forced-preempt-1",
        "messages": [{"role": "user", "content": "partial token test"}],
        "max_tokens": 6,
        "token_latency": 0.0,
        "force_failure": "preempted",
        "force_fail_after_tokens": 2,
        "force_fail_once": True,
    }

    first_result = await backend.generate(request)

    assert first_result["preempted"] is True
    assert first_result["completed_tokens"] == 2
    assert first_result["current_output"] == [[1, 2]]

    retry_request = {
        **request,
        "input_tokens": first_result["current_output"][0],
        "max_tokens": 4,
    }
    second_result = await backend.generate(retry_request)

    assert "error" not in second_result
    assert second_result["usage"]["completion_tokens"] == 6


@pytest.mark.asyncio
async def test_dummy_backend_forced_failure_can_clear_current_tokens():
    backend = DummyBackend("dummy-correctness-test", {})
    request = {
        "model": "dummy-correctness-test",
        "request_id": "forced-error-1",
        "messages": [{"role": "user", "content": "fallback test"}],
        "max_tokens": 6,
        "token_latency": 0.0,
        "force_failure": "error",
        "force_fail_after_tokens": 2,
        "force_no_current_tokens": True,
        "force_fail_once": True,
    }

    with pytest.raises(RuntimeError, match="Forced dummy backend failure"):
        await backend.generate(request)

    assert await backend.get_current_tokens() == []

    second_result = await backend.generate(request)

    assert "error" not in second_result
    assert second_result["usage"]["completion_tokens"] == 6
