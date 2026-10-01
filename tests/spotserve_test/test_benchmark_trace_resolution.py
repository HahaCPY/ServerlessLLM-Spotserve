import json

import pytest

from benchmarks.spotserve import run_benchmark
from benchmarks.spotserve.run_benchmark import (
    BenchmarkSpotEvent,
    alive_model_actor_names_from_actor_table,
    resolve_trace_instance_id,
    wait_for_ready_instances,
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


@pytest.mark.asyncio
async def test_http_add_trace_sends_node_info_to_controller(monkeypatch, tmp_path):
    trace_path = tmp_path / "add.jsonl"
    trace_path.write_text(
        json.dumps({
            "time": 0,
            "event": "add",
            "node_id": "0",
            "model_name": "test-model",
            "node_info": {"free_gpu": 2, "total_gpu": 4, "state": "ready"},
        }) + "\n",
        encoding="utf-8",
    )
    posted = []

    def fake_post_json(endpoint, payload, timeout_s):
        posted.append((endpoint, payload, timeout_s))
        return {"success": True, "response": {"event": "add"}}

    monkeypatch.setattr(run_benchmark, "post_json", fake_post_json)
    await run_benchmark.replay_trace_over_http(
        trace_path=str(trace_path),
        speedup=1.0,
        endpoint="http://127.0.0.1:8343/v1/chat/completions",
        log_path=tmp_path / "trace.log",
        timeout_s=30.0,
        ray_address="auto",
        ray_namespace="sllm",
    )

    assert posted == [(
        "http://127.0.0.1:8343/spot/event",
        {
            "event": "add",
            "node_id": "0",
            "instance_id": None,
            "model_name": "test-model",
            "node_info": {"free_gpu": 2, "total_gpu": 4, "state": "ready"},
        },
        30.0,
    )]


@pytest.mark.asyncio
async def test_wait_for_ready_tolerates_failed_actor_while_retry_is_starting(
    monkeypatch,
):
    snapshots = iter([
        {
            "failed-0": {"pool": "failed", "state": "dead"},
            "retry-0": {"pool": "starting", "state": "starting"},
        },
        {
            "failed-0": {"pool": "failed", "state": "dead"},
            "retry-0": {"pool": "ready", "state": "ready"},
        },
    ])

    async def fake_get_model_instance_states(*_args):
        return next(snapshots)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(
        run_benchmark,
        "get_model_instance_states",
        fake_get_model_instance_states,
    )
    monkeypatch.setattr(run_benchmark.asyncio, "sleep", no_sleep)

    await wait_for_ready_instances("test-model", 1, 30.0, "auto", "sllm")


@pytest.mark.asyncio
async def test_wait_for_ready_fails_after_three_failed_only_polls(monkeypatch):
    async def fake_get_model_instance_states(*_args):
        return {"failed-0": {"pool": "failed", "state": "dead"}}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(
        run_benchmark,
        "get_model_instance_states",
        fake_get_model_instance_states,
    )
    monkeypatch.setattr(run_benchmark.asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError, match="Instance startup failed"):
        await wait_for_ready_instances(
            "test-model", 1, 30.0, "auto", "sllm"
        )
