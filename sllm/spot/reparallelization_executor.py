"""Runtime-neutral executor for applying a :class:`ParallelPlan`.

The planner deliberately has no knowledge of Ray or vLLM worker construction.
This small controller provides the missing V6 boundary while keeping those
deployment details injectable (and therefore easy to test).
"""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .reparallelization import ParallelPlan


MaybeAsync = Callable[..., Any]
MAKE_BEFORE_BREAK = "make_before_break"
BREAK_BEFORE_MAKE = "break_before_make"
_TRANSITION_MODES = {MAKE_BEFORE_BREAK, BREAK_BEFORE_MAKE}


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

    Set ``stop_current_before_create`` when the target must reuse current GPUs,
    or for forced revocation after a recoverable host snapshot has been saved.
    These paths cannot promise that old serving survives target creation failure.
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
    # Keep the transition ordering explicit in metrics/configuration.  When
    # omitted, retain the legacy behaviour selected by
    # ``stop_current_before_create``.
    transition_mode: Optional[str] = None
    # NIXL active-request exports are leases over source-owned allocations.
    # Such a lease is invalid as soon as the source actor is stopped.
    migration_requires_live_source: bool = False
    # Optional actual-runtime evidence check, not another planner assertion.
    # Failure cleans up only the newly created target before traffic switches.
    verify_runtime: Optional[MaybeAsync] = None
    last_transition: Optional[dict[str, Any]] = None

    def resolved_transition_mode(self) -> str:
        mode = self.transition_mode or (
            BREAK_BEFORE_MAKE
            if self.stop_current_before_create
            else MAKE_BEFORE_BREAK
        )
        if mode not in _TRANSITION_MODES:
            raise ValueError(f"unsupported_transition_mode:{mode}")
        if self.stop_current_before_create != (mode == BREAK_BEFORE_MAKE):
            raise ValueError(
                "transition_mode_conflicts_with_stop_current_before_create"
            )
        if (
            mode == BREAK_BEFORE_MAKE
            and self.migrate_before_create
            and self.migration_requires_live_source
        ):
            raise ValueError(
                "live_source_migration_requires_make_before_break"
            )
        return mode

    async def apply(self, plan: ParallelPlan) -> Any:
        mode = self.resolved_transition_mode()
        self.last_transition = {
            "status": "started",
            "transition_mode": mode,
            "migration_requested_before_create": bool(
                self.migrate_before_create
            ),
            "migration_requires_live_source": bool(
                self.migration_requires_live_source
            ),
            "source_stopped_before_target_create": False,
            "source_kept_alive_until_target_ready": False,
        }
        prepared_old = None
        if self.migrate_before_create and self.current is not None:
            prepared_old = self.current
            await _call(self.drain, prepared_old)
        if self.stop_current_before_create and self.current is not None:
            old = self.current
            await _call(self.drain, old)
            await _call(self.stop, old)
            self.current = None
            self.last_transition[
                "source_stopped_before_target_create"
            ] = True

        try:
            target = await _call(self.create_workers, plan)
        except Exception:
            self.last_transition["status"] = "target_create_failed"
            raise
        try:
            if not await _call(self.ready, target, plan):
                raise RuntimeError("target_workers_not_ready")
            if (self.verify_runtime is not None
                    and not await _call(self.verify_runtime, target, plan)):
                raise RuntimeError("target_runtime_not_verified")
        except Exception:
            await _call(self.stop, target)
            self.last_transition["status"] = "target_not_ready"
            raise

        old = self.current
        self.last_transition["source_kept_alive_until_target_ready"] = bool(
            old is not None
        )
        if old is not None and old is not prepared_old:
            try:
                await _call(self.drain, old)
            except Exception:
                # A failed drain/migration must not leak the newly created
                # target deployment or leave it serving without a switch.
                await _call(self.stop, target)
                self.last_transition["status"] = "source_drain_failed"
                raise
        try:
            await _call(self.switch_traffic, target, plan)
        except Exception:
            # Traffic was not switched; keep the old workers available.
            await _call(self.stop, target)
            self.last_transition["status"] = "traffic_switch_failed"
            raise
        self.current = target
        if old is not None:
            if self.wait_for_migration is not None:
                await _call(self.wait_for_migration, old)
            await _call(self.stop, old)
        self.last_transition["status"] = "applied"
        return target
