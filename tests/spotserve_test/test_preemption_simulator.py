import asyncio

import pytest

from sllm.spot.preemption_simulator import (
    _cancel_pending_deadlines,
    _deadline_keys_for_event,
    _dispatch_auto_dead_after,
    _dispatch_event,
)
from sllm.spot.trace_reader import SpotEvent


class RemoteMethod:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    async def remote(self, **kwargs):
        self.calls.append((self.name, kwargs))
        return {"called": self.name}


class FakeController:
    def __init__(self):
        self.calls = []
        self.handle_preemption = RemoteMethod(
            self.calls, "handle_preemption"
        )
        self.handle_recover = RemoteMethod(self.calls, "handle_recover")
        self.handle_instance_dead = RemoteMethod(
            self.calls, "handle_instance_dead"
        )
        self.handle_add = RemoteMethod(self.calls, "handle_add")
        self.handle_remove = RemoteMethod(self.calls, "handle_remove")


@pytest.mark.asyncio
async def test_dispatches_recover_event_to_controller():
    controller = FakeController()
    event = SpotEvent(time=1.0, event="recover", node_id="node-0")

    result = await _dispatch_event(controller, event)

    assert result == {"called": "handle_recover"}
    assert controller.calls == [
        (
            "handle_recover",
            {
                "node_id": "node-0",
                "instance_id": None,
                "model_name": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_dispatches_dead_event_to_controller():
    controller = FakeController()
    event = SpotEvent(time=1.0, event="dead", instance_id="instance-0")

    result = await _dispatch_event(controller, event)

    assert result == {"called": "handle_instance_dead"}
    assert controller.calls[0][0] == "handle_instance_dead"


@pytest.mark.asyncio
async def test_dispatches_add_event_to_controller():
    controller = FakeController()
    event = SpotEvent(time=1.0, event="add", node_id="node-2")

    result = await _dispatch_event(controller, event)

    assert result == {"called": "handle_add"}
    assert controller.calls == [
        (
            "handle_add",
            {"node_id": "node-2", "node_info": {}, "model_name": None},
        )
    ]


@pytest.mark.asyncio
async def test_auto_dead_dispatches_dead_with_grace_period_metadata():
    controller = FakeController()
    event = SpotEvent(
        time=1.0,
        event="preempt",
        node_id="node-0",
        model_name="model-a",
        grace_period_s=30.0,
    )

    result = await _dispatch_auto_dead_after(
        controller,
        event,
        resolved_instance_id=None,
        sleep_s=0.0,
        notice_time_s=100.0,
        deadline_time_s=130.0,
        trace_deadline_time_s=31.0,
        grace_period_s=30.0,
    )

    assert result["status"] == "dispatched"
    assert controller.calls == [
        (
            "handle_instance_dead",
            {
                "node_id": "node-0",
                "instance_id": None,
                "model_name": "model-a",
                "notice_time_s": 100.0,
                "deadline_time_s": 130.0,
                "trace_event_time_s": 31.0,
                "trace_deadline_time_s": 31.0,
                "grace_period_s": 30.0,
                "auto_deadline": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_recover_cancels_pending_auto_deadline():
    controller = FakeController()
    preempt_event = SpotEvent(
        time=1.0,
        event="preempt",
        node_id="node-0",
        model_name="model-a",
        grace_period_s=30.0,
    )
    task = asyncio.create_task(
        _dispatch_auto_dead_after(
            controller,
            preempt_event,
            resolved_instance_id=None,
            sleep_s=30.0,
            notice_time_s=100.0,
            deadline_time_s=130.0,
            trace_deadline_time_s=31.0,
            grace_period_s=30.0,
        )
    )
    pending = {
        key: task
        for key in _deadline_keys_for_event(preempt_event, None)
    }
    recover_event = SpotEvent(
        time=2.0,
        event="recover",
        node_id="node-0",
        model_name="model-a",
    )

    cancelled = await _cancel_pending_deadlines(
        pending, recover_event, resolved_instance_id=None
    )

    assert cancelled == 1
    assert pending == {}
    assert controller.calls == []
