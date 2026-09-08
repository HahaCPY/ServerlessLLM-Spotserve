import pytest

import sllm.schedulers.fcfs_scheduler as fcfs_scheduler_module
from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.utils import NodeState


class FakeRuntimeMetadataActor:
    async def get_runtime_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ):
        return {
            "instance_id": instance_id,
            "node_id": node_id,
            "loading_cost": 9.0,
            "spot_risk": 0.7,
            "remaining_lifetime_s": 600,
            "model_resource_profile": {
                "model_name": "risk-model",
                "num_gpus": 1,
            },
        }


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
async def test_scheduler_collects_backend_runtime_metadata(monkeypatch):
    scheduler = FcfsScheduler({"enable_backend_runtime_metadata": True})
    scheduler.model_instance = {
        "risk-model": {
            "instance-0": "node-0",
        }
    }

    monkeypatch.setattr(
        fcfs_scheduler_module.ray,
        "get_actor",
        lambda _name: FakeRuntimeMetadataActor(),
    )

    metadata_rows = await scheduler._collect_backend_runtime_metadata()

    assert len(metadata_rows) == 1
    assert metadata_rows[0]["instance_id"] == "instance-0"
    assert metadata_rows[0]["node_id"] == "node-0"
    assert metadata_rows[0]["loading_cost"] == 9.0
    assert metadata_rows[0]["risk_metadata_source"] == "backend_runtime"
    assert metadata_rows[0]["risk_provider"] == "backend_runtime"


@pytest.mark.asyncio
async def test_scheduler_collects_backend_runtime_metadata_from_models_namespace(
    monkeypatch,
):
    scheduler = FcfsScheduler({"enable_backend_runtime_metadata": True})
    scheduler.model_instance = {
        "risk-model": {
            "instance-0": "node-0",
        }
    }
    calls = []

    def fake_get_actor(name, namespace=None):
        calls.append((name, namespace))
        if namespace == "models":
            return FakeRuntimeMetadataActor()
        raise ValueError("wrong namespace")

    monkeypatch.setattr(fcfs_scheduler_module.ray, "get_actor", fake_get_actor)

    metadata_rows = await scheduler._collect_backend_runtime_metadata()

    assert len(metadata_rows) == 1
    assert calls == [("instance-0", None), ("instance-0", "models")]


@pytest.mark.asyncio
async def test_scheduler_merges_backend_runtime_metadata_into_nodes():
    scheduler = FcfsScheduler({})
    worker_nodes = {
        "node-0": {
            "ray_node_id": "ray-node-0",
            "address": "10.0.0.1",
            "free_gpu": 1,
            "total_gpu": 1,
            "state": NodeState.READY.value,
        }
    }

    merged = scheduler._merge_backend_runtime_metadata(
        worker_nodes,
        [
            {
                "instance_id": "instance-0",
                "node_id": "node-0",
                "loading_cost": 9.0,
                "spot_risk": 0.7,
                "remaining_lifetime_s": 600,
                "model_resource_profile": {
                    "model_name": "risk-model",
                    "num_gpus": 1,
                },
            },
            {
                "instance_id": "instance-1",
                "node_id": "node-0",
                "loading_cost": 5.0,
                "spot_risk": 0.2,
                "remaining_lifetime_s": 1200,
            },
        ],
    )

    node_info = merged["node-0"]
    assert node_info["spot_risk"] == 0.7
    assert node_info["remaining_lifetime_s"] == 600
    assert node_info["loading_cost"] == 9.0
    assert len(node_info["backend_runtime_metadata"]) == 2
    assert (
        node_info["model_resource_profiles"][0]["model_name"]
        == "risk-model"
    )


@pytest.mark.asyncio
async def test_scheduler_keeps_ray_capacity_authoritative_when_merging_backend_metadata():
    scheduler = FcfsScheduler({})
    worker_nodes = {
        "node-0": {
            "ray_node_id": "ray-node-0",
            "address": "10.0.0.1",
            "free_gpu": 0,
            "total_gpu": 1,
            "state": NodeState.READY.value,
        }
    }

    merged = scheduler._merge_backend_runtime_metadata(
        worker_nodes,
        [
            {
                "instance_id": "instance-0",
                "node_id": "node-0",
                "loading_cost": 9.0,
                "spot_risk": 0.7,
                "remaining_lifetime_s": 600,
                "free_gpu": 4,
                "total_gpu": 4,
                "risk_metadata_source": "backend_runtime",
                "risk_provider": "vllm",
            }
        ],
    )

    node_info = merged["node-0"]
    assert node_info["free_gpu"] == 0
    assert node_info["total_gpu"] == 1
    assert node_info["backend_reported_free_gpu"] == 4
    assert node_info["backend_reported_total_gpu"] == 4
    assert node_info["risk_metadata_source"] == "backend_runtime"
    assert node_info["risk_provider"] == "vllm"


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


@pytest.mark.asyncio
async def test_scheduler_preserves_allocated_capacity_during_worker_refresh(
    monkeypatch,
):
    scheduler = FcfsScheduler({})
    scheduler.model_instance = {
        "test-model": {
            "instance-0": "0",
        }
    }
    scheduler.model_instance_resources = {
        "test-model": {
            "instance-0": {
                "node_id": "0",
                "num_gpus": 1,
            }
        }
    }
    scheduler.worker_nodes = {
        "0": {
            "ray_node_id": "ray-node-0",
            "address": "10.0.0.1",
            "free_gpu": 0,
            "total_gpu": 1,
            "state": NodeState.READY.value,
        },
        "1": {
            "ray_node_id": "ray-node-1",
            "address": "10.0.0.2",
            "free_gpu": 1,
            "total_gpu": 1,
            "state": NodeState.READY.value,
        },
    }

    monkeypatch.setattr(
        fcfs_scheduler_module,
        "get_worker_nodes",
        lambda: {
            "0": {
                "ray_node_id": "ray-node-0-new",
                "address": "10.0.0.1",
                "free_gpu": 1,
                "total_gpu": 1,
            },
            "1": {
                "ray_node_id": "ray-node-1-new",
                "address": "10.0.0.2",
                "free_gpu": 1,
                "total_gpu": 1,
            },
        },
    )

    refreshed = await scheduler._get_worker_nodes()

    assert refreshed["0"]["ray_node_id"] == "ray-node-0-new"
    assert refreshed["0"]["free_gpu"] == 0
    assert refreshed["1"]["free_gpu"] == 1


@pytest.mark.asyncio
async def test_scheduler_reconciles_single_gpu_worker_with_ray_available_resources(
    monkeypatch,
):
    scheduler = FcfsScheduler({})

    monkeypatch.setattr(
        fcfs_scheduler_module,
        "get_worker_nodes",
        lambda: {
            "0": {
                "ray_node_id": "ray-node-0",
                "address": "10.0.0.1",
                "free_gpu": 1,
                "total_gpu": 1,
            },
            "1": {
                "ray_node_id": "ray-node-1",
                "address": "10.0.0.2",
                "free_gpu": 1,
                "total_gpu": 1,
            },
        },
    )
    monkeypatch.setattr(
        fcfs_scheduler_module.ray,
        "available_resources",
        lambda: {
            "GPU": 1.0,
            "worker_id_0": 1.0,
            "worker_id_1": 0.9,
        },
    )

    refreshed = await scheduler._get_worker_nodes()

    assert refreshed["0"]["free_gpu"] == 1
    assert refreshed["1"]["free_gpu"] == 0
    assert refreshed["1"]["ray_available_worker_resource"] == 0.9


@pytest.mark.asyncio
async def test_scheduler_clear_model_cancels_pending_and_releases_allocations():
    scheduler = FcfsScheduler({})
    pending = scheduler.loop.create_future()
    scheduler.model_loading_queues = {
        "test-model": [
            (0.0, "instance-pending", 1, pending, None),
        ]
    }
    scheduler.model_instance = {
        "test-model": {
            "instance-ready": "0",
        }
    }
    scheduler.model_instance_resources = {
        "test-model": {
            "instance-ready": {
                "node_id": "0",
                "num_gpus": 1,
            }
        }
    }
    scheduler.worker_nodes = {
        "0": {
            "ray_node_id": "ray-node-0",
            "address": "10.0.0.1",
            "free_gpu": 0,
            "total_gpu": 1,
            "state": NodeState.READY.value,
        }
    }

    result = await scheduler.clear_model("test-model")

    assert result["cleared_pending"] == 1
    assert result["released_allocations"] == 1
    assert scheduler.model_loading_queues == {}
    assert scheduler.model_instance == {}
    assert scheduler.model_instance_resources == {}
    assert scheduler.worker_nodes["0"]["free_gpu"] == 1
    assert pending.done()
    with pytest.raises(RuntimeError):
        pending.result()
