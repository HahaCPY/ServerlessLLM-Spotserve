import pytest

from sllm.spot.reparallelization import ParallelPlan
from sllm.spot.reparallelization_executor import ReparallelizationExecutor


@pytest.mark.asyncio
async def test_executor_ready_then_switches_and_stops_old():
    events = []
    async def create(plan): events.append(("create", plan.tensor_parallel_size)); return "new"
    async def ready(worker, plan): events.append(("ready", worker)); return True
    async def switch(worker, plan): events.append(("switch", worker))
    async def drain(worker): events.append(("drain", worker))
    async def stop(worker): events.append(("stop", worker))
    executor = ReparallelizationExecutor(create, ready, switch, drain, stop, "old")
    await executor.apply(ParallelPlan("m", "vllm", 4, 1, num_gpus=4))
    assert events == [("create", 4), ("ready", "new"), ("drain", "old"),
                      ("switch", "new"), ("stop", "old")]


@pytest.mark.asyncio
async def test_executor_stops_unready_target_and_keeps_old():
    stopped = []
    async def create(plan): return "new"
    async def ready(worker, plan): return False
    async def stop(worker): stopped.append(worker)
    executor = ReparallelizationExecutor(create, ready, lambda *_: None,
                                          lambda *_: None, stop, "old")
    with pytest.raises(RuntimeError, match="target_workers_not_ready"):
        await executor.apply(ParallelPlan("m", "vllm", 2, 1, num_gpus=2))
    assert stopped == ["new"]
    assert executor.current == "old"


@pytest.mark.asyncio
async def test_executor_can_stop_current_before_create_when_configured():
    events = []
    async def create(plan): events.append(("create", plan.num_gpus)); return "new"
    async def ready(worker, plan): events.append(("ready", worker)); return True
    async def switch(worker, plan): events.append(("switch", worker))
    async def drain(worker): events.append(("drain", worker))
    async def stop(worker): events.append(("stop", worker))
    executor = ReparallelizationExecutor(
        create,
        ready,
        switch,
        drain,
        stop,
        "old",
        stop_current_before_create=True,
    )
    await executor.apply(ParallelPlan("m", "vllm", 1, 1, num_gpus=1))
    assert events == [("drain", "old"), ("stop", "old"), ("create", 1),
                      ("ready", "new"), ("switch", "new")]
    assert executor.current == "new"


@pytest.mark.asyncio
async def test_live_source_migration_waits_before_stopping_source():
    events = []

    async def create(plan):
        events.append(("create", plan.num_gpus))
        return "new"

    async def ready(worker, plan):
        events.append(("ready", worker))
        return True

    async def switch(worker, plan):
        events.append(("switch", worker))

    async def drain(worker):
        events.append(("drain", worker))

    async def wait(worker):
        events.append(("wait", worker))

    async def stop(worker):
        events.append(("stop", worker))

    executor = ReparallelizationExecutor(
        create,
        ready,
        switch,
        drain,
        stop,
        "old",
        wait_for_migration=wait,
        migrate_before_create=True,
        transition_mode="make_before_break",
        migration_requires_live_source=True,
    )

    await executor.apply(ParallelPlan("m", "vllm", 1, 1, num_gpus=1))

    assert events == [
        ("drain", "old"),
        ("create", 1),
        ("ready", "new"),
        ("switch", "new"),
        ("wait", "old"),
        ("stop", "old"),
    ]
    assert executor.last_transition == {
        "status": "applied",
        "transition_mode": "make_before_break",
        "migration_requested_before_create": True,
        "migration_requires_live_source": True,
        "source_stopped_before_target_create": False,
        "source_kept_alive_until_target_ready": True,
    }


def test_live_source_migration_rejects_break_before_make():
    executor = ReparallelizationExecutor(
        lambda *_: "new",
        lambda *_: True,
        lambda *_: None,
        lambda *_: None,
        lambda *_: None,
        "old",
        stop_current_before_create=True,
        wait_for_migration=lambda *_: None,
        migrate_before_create=True,
        transition_mode="break_before_make",
        migration_requires_live_source=True,
    )

    with pytest.raises(
        ValueError,
        match="live_source_migration_requires_make_before_break",
    ):
        executor.resolved_transition_mode()
