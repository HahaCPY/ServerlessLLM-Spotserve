import pytest

from sllm.spot.preemption_simulator import _dispatch_event
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
