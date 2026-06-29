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


async def _dispatch_event(controller, event: SpotEvent):
    if event.event == "preempt":
        return await controller.handle_preemption.remote(
            node_id=event.node_id,
            instance_id=event.instance_id,
            model_name=event.model_name,
        )
    if event.event == "dead":
        return await controller.handle_instance_dead.remote(
            node_id=event.node_id,
            instance_id=event.instance_id,
            model_name=event.model_name,
        )

    logger.info(
        "Spot event %s is parsed but not dispatched in v1: %s",
        event.event,
        event,
    )
    return {"event": event.event, "dispatched": False}


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
