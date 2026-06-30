import pytest

from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import NodeState


@pytest.mark.asyncio
async def test_scheduler_marks_node_health_states():
    scheduler = FcfsScheduler({})

    await scheduler.mark_node_preempting("node-0")
    assert (
        scheduler.worker_nodes["node-0"]["state"]
        == NodeState.PREEMPTING.value
    )
    assert not scheduler._node_is_ready(scheduler.worker_nodes["node-0"])

    await scheduler.mark_node_recovered("node-0")
    assert scheduler.worker_nodes["node-0"]["state"] == NodeState.READY.value
    assert scheduler._node_is_ready(scheduler.worker_nodes["node-0"])

    await scheduler.mark_node_dead("node-0")
    assert scheduler.worker_nodes["node-0"]["state"] == NodeState.DEAD.value
    assert not scheduler._node_is_ready(scheduler.worker_nodes["node-0"])


@pytest.mark.asyncio
async def test_scheduler_preserves_node_state_during_worker_update():
    scheduler = FcfsScheduler({})
    scheduler.worker_nodes = {
        "node-0": {
            "ray_node_id": "old",
            "address": "old-address",
            "free_gpu": 1,
            "total_gpu": 1,
            "state": NodeState.PREEMPTING.value,
        }
    }
    worker_nodes = {
        "node-0": {
            "ray_node_id": "new",
            "address": "new-address",
            "free_gpu": 2,
            "total_gpu": 2,
        }
    }

    await scheduler._update_worker_nodes(worker_nodes)

    assert scheduler.worker_nodes["node-0"]["ray_node_id"] == "new"
    assert scheduler.worker_nodes["node-0"]["free_gpu"] == 2
    assert (
        scheduler.worker_nodes["node-0"]["state"]
        == NodeState.PREEMPTING.value
    )
