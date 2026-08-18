import pytest

from sllm.spot.trace_reader import SpotEvent, load_spot_trace


def test_load_spot_trace_sorts_events(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"time": 3.0, "event": "preempt", "node_id": "1"}',
                '{"time": 1.0, "event": "preempt", "instance_id": "i-0"}',
            ]
        ),
        encoding="utf-8",
    )

    events = load_spot_trace(trace_path)

    assert events == [
        SpotEvent(time=1.0, event="preempt", instance_id="i-0"),
        SpotEvent(time=3.0, event="preempt", node_id="1"),
    ]


def test_load_spot_trace_rejects_event_without_target(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"time": 1.0, "event": "preempt"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="node_id or instance_id"):
        load_spot_trace(trace_path)


def test_load_spot_trace_rejects_unknown_event(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"time": 1.0, "event": "evict", "node_id": "0"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported spot event"):
        load_spot_trace(trace_path)


def test_load_spot_trace_accepts_capacity_events(tmp_path):
    trace_path = tmp_path / "capacity-trace.jsonl"
    trace_path.write_text(
        '\n'.join([
            '{"time": 0, "event": "add", "node_id": "node-0", '
            '"node_info": {"total_gpu": 2, "free_gpu": 2}}',
            '{"time": 1, "event": "remove", "node_id": "node-0"}',
        ]),
        encoding="utf-8",
    )

    events = load_spot_trace(trace_path)

    assert [event.event for event in events] == ["add", "remove"]
    assert events[0].node_info == {"total_gpu": 2, "free_gpu": 2}
