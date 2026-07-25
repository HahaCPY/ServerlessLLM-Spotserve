# trace 模擬 spot instance，照時間播放事件，通知 Controller

import argparse
import asyncio
import logging
import time

try:
    import ray
except ModuleNotFoundError:
    ray = None

from sllm.spot.trace_reader import SpotEvent, load_spot_trace


logger = logging.getLogger(__name__)


def _is_ready_instance_state(state: dict) -> bool:
    return state.get("pool") == "ready" and state.get("state") == "ready"


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
    ready_instances = [
        (instance_id, state)
        for instance_id, state in states.items()
        if _is_ready_instance_state(state)
    ]
    if event.instance_selector in ("active", "active_context", "busy"):
        ready_instances = [
            (instance_id, state)
            for instance_id, state in ready_instances
            if _instance_concurrency(state) > 0
        ]
        ready_instances = sorted(
            ready_instances,
            key=lambda item: (-_instance_concurrency(item[1]), item[0]),
        )
    elif event.instance_selector in (None, "ready"):
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


async def _dispatch_event(controller, event: SpotEvent):
    instance_id = await _resolve_instance_id(event)
    if event.event == "preempt":
        return await controller.handle_preemption.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
        )
    if event.event == "dead":
        return await controller.handle_instance_dead.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
        )
    if event.event == "recover":
        return await controller.handle_recover.remote(
            node_id=event.node_id,
            instance_id=instance_id,
            model_name=event.model_name,
        )

    raise ValueError(f"Unsupported spot event: {event.event}")


async def replay_trace(
    trace_path: str,
    speedup: float = 1.0,
    controller_name: str = "controller",
):
    if ray is None:
        raise RuntimeError(
            "Ray is required to replay spot traces. Install ray or run the "
            "benchmark with --skip-trace."
        )
    if speedup <= 0:
        raise ValueError("speedup must be positive")

    events = load_spot_trace(trace_path)
    controller = ray.get_actor(controller_name)
    replay_started_at = time.monotonic()
    last_event_time = 0.0

    for event in events:
        sleep_time = max(event.time - last_event_time, 0.0) / speedup
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        logger.info("Replaying spot event: %s", event)
        await _dispatch_event(controller, event)
        last_event_time = event.time

    return {
        "trace": trace_path,
        "events": len(events),
        "elapsed_s": time.monotonic() - replay_started_at,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay SpotServe trace events")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    parser.add_argument("--controller-name", default="controller")
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
        )
    )
    logger.info("Trace replay finished: %s", result)


if __name__ == "__main__":
    main()
