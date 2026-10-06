import json

import pytest

from benchmarks.spotserve import run_benchmark
from benchmarks.spotserve.run_benchmark import (
    BenchmarkSpotEvent,
    alive_model_actor_names_from_actor_table,
    resolve_trace_instance_id,
    wait_for_ready_instances,
)


@pytest.mark.asyncio
async def test_instance_membership_is_saved_before_model_cleanup(monkeypatch, tmp_path):
    states = {"a": {"pool": "ready", "state": "ready", "member_node_ids": ["0", "1"]}}
    async def snapshot(*args):
        return dict(states)
    async def noop(*args, **kwargs):
        return None
    async def requests(*args):
        return []
    monkeypatch.setattr(run_benchmark, "git_commit", lambda: "mock")
    monkeypatch.setattr(run_benchmark, "apply_scheduler_config", noop)
    monkeypatch.setattr(run_benchmark, "wait_for_ready_instances", noop)
    monkeypatch.setattr(run_benchmark, "get_model_instance_states", snapshot)
    monkeypatch.setattr(run_benchmark, "send_workload", requests)
    monkeypatch.setattr(run_benchmark, "delete_model_over_http", lambda *args: states.clear())
    monkeypatch.setattr(run_benchmark, "record_actor_cleanup_after_delete", noop)
    workload = tmp_path / "workload.jsonl"
    workload.write_text("{}\n")
    run_dir = await run_benchmark.run_one(
        {"name": "smoke", "model": "probe-model", "workload": str(workload),
         "capture_instance_states": True, "capture_instance_states_after_workload": True,
         "delete_after_run": True},
        "endpoint", tmp_path, 1, 1, skip_trace=True)
    assert not states
    assert json.loads((run_dir / "instance_states.json").read_text())["a"]["member_node_ids"] == ["0", "1"]
    assert json.loads((run_dir / "final_instance_states.json").read_text())["a"]["member_node_ids"] == ["0", "1"]


@pytest.mark.asyncio
async def test_initial_capacity_is_applied_before_deploy_and_restored_after_run(
    monkeypatch, tmp_path
):
    order = []

    async def noop(*args, **kwargs):
        return None

    async def availability(nodes, available, *args):
        order.append(("availability", list(nodes), available))
        return [
            {"node_id": node, "available": available} for node in nodes
        ]

    def deploy(*args, **kwargs):
        order.append(("deploy",))
        return {"success": True, "response": {}}

    async def requests(*args):
        return []

    monkeypatch.setattr(run_benchmark, "git_commit", lambda: "mock")
    monkeypatch.setattr(run_benchmark, "apply_scheduler_config", noop)
    monkeypatch.setattr(
        run_benchmark, "set_scheduler_worker_availability", availability
    )
    monkeypatch.setattr(run_benchmark, "deploy_config_over_http", deploy)
    monkeypatch.setattr(run_benchmark, "wait_for_ready_instances", noop)
    monkeypatch.setattr(run_benchmark, "send_workload", requests)
    workload = tmp_path / "workload.jsonl"
    workload.write_text("{}\n")

    run_dir = await run_benchmark.run_one(
        {
            "name": "capacity-trace",
            "model": "probe-model",
            "workload": str(workload),
            "deploy_config": str(tmp_path / "deploy.json"),
            "initial_unavailable_worker_nodes": ["5", "6", "7"],
            "restore_worker_nodes_after_run": ["3", "5", "6", "7"],
        },
        "endpoint",
        tmp_path,
        1,
        1,
        skip_trace=True,
    )

    assert order == [
        ("availability", ["5", "6", "7"], False),
        ("deploy",),
        ("availability", ["3", "5", "6", "7"], True),
    ]
    assert (run_dir / "initial_worker_availability.json").is_file()
    assert (run_dir / "restored_worker_availability.json").is_file()


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
    async def inline_thread_call(fn, *args):
        # The transport is already mocked; no real thread/network is needed.
        return fn(*args)
    monkeypatch.setattr(run_benchmark.asyncio, "to_thread", inline_thread_call)
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
