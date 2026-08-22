# trace 模擬 spot instance，照時間播放事件，通知 Controller

import argparse
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple

try:
    import ray
except ModuleNotFoundError:
    ray = None

from sllm.spot.trace_reader import SpotEvent, load_spot_trace


logger = logging.getLogger(__name__)


def _is_ready_instance_state(state: dict) -> bool:
    return state.get("pool") == "ready" and state.get("state") == "ready"


def _is_preempting_instance_state(state: dict) -> bool:
    return state.get("state") == "preempting"


def _instance_concurrency(state: dict) -> int:
    try:
        return int(state.get("concurrency", 0) or 0)
    except (TypeError, ValueError):
        return 0


async def _resolve_instance_id(event: SpotEvent):
    if event.instance_id is not None:
        return event.instance_id
    if event.instance_index is None and event.instance_selector is None:
        return None
    if ray is None:
        raise RuntimeError(
            "Ray is required to resolve trace instance selection"
        )
    if event.model_name is None:
        raise ValueError("instance-selected trace event requires model_name")

    router = ray.get_actor(event.model_name, namespace="models")
    states = await router.get_instance_states.remote()
    if event.instance_selector in ("active", "active_context", "busy"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if _is_ready_instance_state(state)
        ]
        ready_instances = [
            (instance_id, state)
            for instance_id, state in ready_instances
            if _instance_concurrency(state) > 0
        ]
        ready_instances = sorted(
            ready_instances,
            key=lambda item: (-_instance_concurrency(item[1]), item[0]),
        )
    elif event.instance_selector in ("preempting", "preempted"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if _is_preempting_instance_state(state)
        ]
        ready_instances = sorted(ready_instances, key=lambda item: item[0])
    elif event.instance_selector in (None, "ready"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in states.items()
            if _is_ready_instance_state(state)
        ]
        ready_instances = sorted(ready_instances, key=lambda item: item[0])
    else:
        raise RuntimeError(
            f"Unsupported instance_selector: {event.instance_selector}"
        )

    ready_instance_ids = [instance_id for instance_id, _ in ready_instances]
    index = int(event.instance_index or 0)
    if index < 0:
        index += len(ready_instance_ids)
    if index < 0 or index >= len(ready_instance_ids):
        raise RuntimeError(
            f"instance_index {event.instance_index} is out of range for "
            f"{event.model_name}; selector={event.instance_selector}; "
            f"ready instances={ready_instance_ids}; states={states}"
        )
    return ready_instance_ids[index]


DeadlineKey = Tuple[str, str, str]


def _deadline_keys_for_event(
    event: SpotEvent,
    resolved_instance_id: Optional[str] = None,
) -> Set[DeadlineKey]:
    model = str(event.model_name or "")
    keys: Set[DeadlineKey] = set()
    instance_id = resolved_instance_id or event.instance_id
    if instance_id is not None:
        keys.add(("instance", model, str(instance_id)))
    if event.node_id is not None:
        keys.add(("node", model, str(event.node_id)))
    if event.instance_selector is not None or event.instance_index is not None:
        selector = str(event.instance_selector or "ready")
        index = str(int(event.instance_index or 0))
        keys.add(("selector", model, f"{selector}:{index}"))
    return keys


async def _dispatch_event(
    controller,
    event: SpotEvent,
    *,
    resolved_instance_id: Optional[str] = None,
    notice_time_s: Optional[float] = None,
    deadline_time_s: Optional[float] = None,
    trace_deadline_time_s: Optional[float] = None,
    grace_period_s: Optional[float] = None,
    auto_deadline: bool = False,
):
    instance_id = (
        resolved_instance_id
        if resolved_instance_id is not None
        else await _resolve_instance_id(event)
    )
    if event.event == "add":
        return await controller.handle_add.remote(
            node_id=event.node_id,
            node_info=event.node_info or {},
            model_name=event.model_name,
        )
    if event.event == "remove":
        return await controller.handle_remove.remote(
            node_id=event.node_id,
            model_name=event.model_name,
        )
    if event.event == "preempt":
        return await controller.handle_preemption.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
            notice_time_s=notice_time_s,
            deadline_time_s=deadline_time_s,
            trace_event_time_s=event.time,
            trace_deadline_time_s=trace_deadline_time_s,
            grace_period_s=grace_period_s,
        )
    if event.event == "dead":
        return await controller.handle_instance_dead.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
            notice_time_s=notice_time_s,
            deadline_time_s=deadline_time_s,
            trace_event_time_s=event.time,
            trace_deadline_time_s=trace_deadline_time_s,
            grace_period_s=grace_period_s,
            auto_deadline=auto_deadline,
        )
    if event.event == "recover":
        return await controller.handle_recover.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
        )

    raise ValueError(f"Unsupported spot event: {event.event}")


async def _dispatch_auto_dead_after(
    controller,
    event: SpotEvent,
    resolved_instance_id: Optional[str],
    sleep_s: float,
    *,
    notice_time_s: float,
    deadline_time_s: float,
    trace_deadline_time_s: float,
    grace_period_s: float,
) -> Dict[str, Any]:
    try:
        await asyncio.sleep(max(0.0, sleep_s))
    except asyncio.CancelledError:
        return {
            "status": "cancelled",
            "event": "dead",
            "trace_deadline_time_s": trace_deadline_time_s,
        }

    dead_event = SpotEvent(
        time=trace_deadline_time_s,
        event="dead",
        node_id=event.node_id,
        model_name=event.model_name,
        instance_id=resolved_instance_id or event.instance_id,
        auto_generated=True,
    )
    logger.info(
        "Auto-replaying spot dead event after grace period: %s",
        dead_event,
    )
    result = await _dispatch_event(
        controller,
        dead_event,
        resolved_instance_id=resolved_instance_id,
        notice_time_s=notice_time_s,
        deadline_time_s=deadline_time_s,
        trace_deadline_time_s=trace_deadline_time_s,
        grace_period_s=grace_period_s,
        auto_deadline=True,
    )
    return {
        "status": "dispatched",
        "event": "dead",
        "trace_deadline_time_s": trace_deadline_time_s,
        "result": result,
    }


async def _cancel_pending_deadlines(
    pending_deadlines: Dict[DeadlineKey, asyncio.Task],
    event: SpotEvent,
    resolved_instance_id: Optional[str],
) -> int:
    keys = _deadline_keys_for_event(event, resolved_instance_id)
    tasks = {
        task
        for key in keys
        for task in [pending_deadlines.pop(key, None)]
        if task is not None
    }
    tasks = {task for task in tasks if not task.done()}
    if not tasks:
        return 0
    for key, task in list(pending_deadlines.items()):
        if task in tasks:
            pending_deadlines.pop(key, None)
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    return len(tasks)


async def replay_trace(
    trace_path: str,
    speedup: float = 1.0,
    controller_name: str = "controller",
    model_name: str | None = None,
):
    if ray is None:
        raise RuntimeError(
            "Ray is required to replay spot traces. Install ray or run the "
            "benchmark with --skip-trace."
        )
    if speedup <= 0:
        raise ValueError("speedup must be positive")

    events = load_spot_trace(trace_path, default_model_name=model_name)
    controller = ray.get_actor(controller_name)
    replay_started_at = time.monotonic()
    last_event_time = 0.0
    pending_deadlines: Dict[DeadlineKey, asyncio.Task] = {}
    auto_dead_scheduled = 0
    auto_dead_cancelled = 0
    auto_dead_dispatched = 0

    for event in events:
        sleep_time = max(event.time - last_event_time, 0.0) / speedup
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        logger.info("Replaying spot event: %s", event)
        resolved_instance_id = await _resolve_instance_id(event)
        notice_time_s = time.time() if event.event == "preempt" else None
        grace_period_s = event.grace_period_s
        trace_deadline_time_s = (
            event.time + grace_period_s
            if event.event == "preempt" and grace_period_s is not None
            else None
        )
        deadline_time_s = (
            notice_time_s + (grace_period_s / speedup)
            if notice_time_s is not None and grace_period_s is not None
            else None
        )
        await _dispatch_event(
            controller,
            event,
            resolved_instance_id=resolved_instance_id,
            notice_time_s=notice_time_s,
            deadline_time_s=deadline_time_s,
            trace_deadline_time_s=trace_deadline_time_s,
            grace_period_s=grace_period_s,
        )
        if event.event in {"recover", "dead", "remove"}:
            auto_dead_cancelled += await _cancel_pending_deadlines(
                pending_deadlines,
                event,
                resolved_instance_id,
            )
        if event.event == "preempt" and grace_period_s is not None:
            task = asyncio.create_task(
                _dispatch_auto_dead_after(
                    controller,
                    event,
                    resolved_instance_id,
                    grace_period_s / speedup,
                    notice_time_s=notice_time_s or time.time(),
                    deadline_time_s=deadline_time_s or time.time(),
                    trace_deadline_time_s=trace_deadline_time_s
                    if trace_deadline_time_s is not None
                    else event.time,
                    grace_period_s=grace_period_s,
                )
            )
            for key in _deadline_keys_for_event(event, resolved_instance_id):
                pending_deadlines[key] = task
            auto_dead_scheduled += 1
        last_event_time = event.time

    remaining_tasks = set(pending_deadlines.values())
    if remaining_tasks:
        results = await asyncio.gather(*remaining_tasks)
        auto_dead_dispatched += sum(
            1 for result in results if result.get("status") == "dispatched"
        )

    return {
        "trace": trace_path,
        "events": len(events),
        "auto_dead_scheduled": auto_dead_scheduled,
        "auto_dead_cancelled": auto_dead_cancelled,
        "auto_dead_dispatched": auto_dead_dispatched,
        "elapsed_s": time.monotonic() - replay_started_at,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay SpotServe trace events")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    parser.add_argument("--controller-name", default="controller")
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Default model name for trace events that select instances but "
            "do not include model_name."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if ray is None:
        parser.exit(
            2,
            "error: Ray is required to replay spot traces. Install ray or run "
            "the benchmark with --skip-trace.\n",
        )
    ray.init(
        address=args.ray_address,
        namespace=args.ray_namespace,
        ignore_reinit_error=True,
    )
    result = asyncio.run(
        replay_trace(
            trace_path=args.trace,
            speedup=args.speedup,
            controller_name=args.controller_name,
            model_name=args.model_name,
        )
    )
    logger.info("Trace replay finished: %s", result)


if __name__ == "__main__":
    main()
