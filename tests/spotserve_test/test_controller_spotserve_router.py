import asyncio

import pytest

import sllm.controller as controller_module
from sllm.controller import SllmController
from sllm.spot.controller_config import spotserve_scheduler_config


class FakeRemoteClearModel:
    def __init__(self):
        self.calls = []

    async def remote(self, model_name):
        self.calls.append(model_name)


class FakeScheduler:
    def __init__(self):
        self.clear_model = FakeRemoteClearModel()


def test_spotserve_controller_ignores_legacy_enable_migration():
    scheduler_config, legacy_requested = spotserve_scheduler_config(
        {
            "enable_migration": True,
            "scheduler_config": {
                "enable_migration": True,
                "enable_spot_risk_aware": True,
            },
        }
    )

    assert legacy_requested is True
    assert scheduler_config["enable_migration"] is False
    assert scheduler_config["enable_spot_risk_aware"] is True


def test_spotserve_controller_keeps_migration_disabled_by_default():
    scheduler_config, legacy_requested = spotserve_scheduler_config({})

    assert legacy_requested is False
    assert scheduler_config["enable_migration"] is False


@pytest.mark.asyncio
async def test_controller_delete_cleans_stale_model_actors(monkeypatch):
    controller = SllmController.__new__(SllmController)
    controller.metadata_lock = asyncio.Lock()
    controller.request_routers = {}
    controller.registered_models = {}
    controller.scheduler = FakeScheduler()
    controller.config = {
        "stale_actor_cleanup_timeout_s": 0.0,
        "stale_actor_cleanup_quiet_s": 0.0,
    }
    actor_handle = object()
    killed = []

    import ray._private.state as ray_state

    monkeypatch.setattr(
        ray_state,
        "actors",
        lambda: {
            "actor-live": {
                "Name": "stale-model_instance-0",
                "State": "ALIVE",
            },
            "actor-dead": {
                "Name": "stale-model_instance-1",
                "State": "DEAD",
            },
            "actor-other": {
                "Name": "other-model_instance-0",
                "State": "ALIVE",
            },
        },
    )
    monkeypatch.setattr(
        controller_module.ray,
        "get_actor",
        lambda name, namespace=None: actor_handle,
    )
    monkeypatch.setattr(
        controller_module.ray,
        "kill",
        lambda actor, no_restart=True: killed.append((actor, no_restart)),
    )

    await controller.delete("stale-model")

    assert controller.scheduler.clear_model.calls == ["stale-model"]
    assert killed == [(actor_handle, True)]
