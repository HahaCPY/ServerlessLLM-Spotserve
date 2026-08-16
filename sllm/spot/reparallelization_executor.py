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

    Set ``stop_current_before_create`` only for deployments where the planned
    target must reuse the exact GPU resources currently held by ``current``.
    """

    create_workers: MaybeAsync
    ready: MaybeAsync
    switch_traffic: MaybeAsync
    drain: MaybeAsync
    stop: MaybeAsync
    current: Optional[Any] = None
    stop_current_before_create: bool = False
    wait_for_migration: Optional[MaybeAsync] = None
    migrate_before_create: bool = False

    async def apply(self, plan: ParallelPlan) -> Any:
        prepared_old = None
        if self.migrate_before_create and self.current is not None:
            prepared_old = self.current
            await _call(self.drain, prepared_old)
        if self.stop_current_before_create and self.current is not None:
            old = self.current
            await _call(self.drain, old)
            await _call(self.stop, old)
            self.current = None

        target = await _call(self.create_workers, plan)
        try:
            if not await _call(self.ready, target, plan):
                raise RuntimeError("target_workers_not_ready")
        except Exception:
            await _call(self.stop, target)
            raise

        old = self.current
        if old is not None and old is not prepared_old:
            try:
                await _call(self.drain, old)
            except Exception:
                # A failed drain/migration must not leak the newly created
                # target deployment or leave it serving without a switch.
                await _call(self.stop, target)
                raise
        try:
            await _call(self.switch_traffic, target, plan)
        except Exception:
            # Traffic was not switched; keep the old workers available.
            await _call(self.stop, target)
            raise
        self.current = target
        if old is not None:
            if self.wait_for_migration is not None:
                await _call(self.wait_for_migration, old)
            await _call(self.stop, old)
        return target
