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
