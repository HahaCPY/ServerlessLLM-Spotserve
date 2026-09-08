import pytest

from benchmarks.spotserve import run_benchmark
from benchmarks.spotserve.run_benchmark import (
    BenchmarkSpotEvent,
    alive_model_actor_names_from_actor_table,
    resolve_trace_instance_id,
)


def test_alive_model_actor_names_from_actor_table_matches_backend_prefix():
    actor_table = {
        "router": {
            "Name": "demo-model",
            "State": "ALIVE",
        },
        "backend": {
            "Name": "demo-model_instance-0",
            "State": "ALIVE",
        },
        "dead-backend": {
            "Name": "demo-model_instance-1",
            "State": "DEAD",
        },
        "other-prefix": {
            "Name": "demo-modelish_instance-0",
            "State": "ALIVE",
        },
        "other-model": {
            "Name": "other-model_instance-0",
            "State": "ALIVE",
        },
    }

    assert alive_model_actor_names_from_actor_table(
        actor_table,
        "demo-model",
    ) == [
        "demo-model",
        "demo-model_instance-0",
    ]


@pytest.mark.asyncio
async def test_busy_trace_selector_falls_back_to_ready_when_no_instance_is_busy(
    monkeypatch,
):
    async def fake_get_model_instance_states(
        model_name, ray_address, ray_namespace
    ):
        return {
            "instance-b": {
                "pool": "ready",
                "state": "ready",
                "concurrency": 0,
            },
            "instance-a": {
                "pool": "ready",
                "state": "ready",
                "concurrency": 0,
            },
        }

    monkeypatch.setattr(
        run_benchmark,
        "get_model_instance_states",
        fake_get_model_instance_states,
    )
    event = BenchmarkSpotEvent(
        time=1.5,
        event="preempt",
        node_id=None,
        model_name="test-model",
        instance_id=None,
        instance_index=None,
        instance_selector="busy",
    )

    selected = await resolve_trace_instance_id(event, "auto", "sllm")

    assert selected == "instance-a"
