"""Runtime-neutral executor for applying a :class:`ParallelPlan`.

The planner deliberately has no knowledge of Ray or vLLM worker construction.
This small controller provides the missing V6 boundary while keeping those
deployment details injectable (and therefore easy to test).
"""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .reparallelization import ParallelPlan


MaybeAsync = Callable[..., Any]


async def _call(fn: MaybeAsync, *args: Any, **kwargs: Any) -> Any:
    value = fn(*args, **kwargs)
    if isinstance(value, Awaitable):
        return await value
    return value


@dataclass
class ReparallelizationExecutor:
    """Drain old workers, start a planned shape, then atomically switch traffic.

    ``create_workers`` must return a worker handle; ``ready`` may perform a
    health check.  The old handle is only stopped after the target is ready,
    so a failed replan leaves serving traffic untouched.
    """

    create_workers: MaybeAsync
    ready: MaybeAsync
    switch_traffic: MaybeAsync
    drain: MaybeAsync
    stop: MaybeAsync
    current: Optional[Any] = None

    async def apply(self, plan: ParallelPlan) -> Any:
        target = await _call(self.create_workers, plan)
        try:
            if not await _call(self.ready, target, plan):
                raise RuntimeError("target_workers_not_ready")
        except Exception:
            await _call(self.stop, target)
            raise

        old = self.current
        if old is not None:
            await _call(self.drain, old)
        try:
            await _call(self.switch_traffic, target, plan)
        except Exception:
            # Traffic was not switched; keep the old workers available.
            await _call(self.stop, target)
            raise
        self.current = target
        if old is not None:
            await _call(self.stop, old)
        return target
