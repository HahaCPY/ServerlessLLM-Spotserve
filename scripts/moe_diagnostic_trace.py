"""Opt-in process-local timing; never changes model execution policy."""

from collections import defaultdict
from contextlib import contextmanager
import functools
import json
import os
import resource
import sys
import time


EVENTS = []
STACK = []
TOTALS = defaultdict(lambda: {"calls": 0, "wall_s": 0.0})
GPU_EVENTS = []
MEASUREMENT = None


def emit(event, **fields):
    row = {"event": event, "pid": os.getpid(),
           "monotonic_s": time.monotonic(), **fields}
    EVENTS.append(row)
    print("MOE_DIAGNOSTIC_EVENT=" + json.dumps(row), flush=True)
    return row


@contextmanager
def span(name):
    token = len(EVENTS)
    start = emit("span_start", name=name, span_id=token,
                 parent_id=STACK[-1] if STACK else None)
    STACK.append(token)
    try:
        yield
    finally:
        STACK.pop()
        emit("span_end", name=name, span_id=token,
             duration_s=time.monotonic() - start["monotonic_s"])


def io_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {"major_faults": usage.ru_majflt, "minor_faults": usage.ru_minflt,
              "input_blocks": usage.ru_inblock}
    try:
        with open("/proc/self/io") as stream:
            values.update({k: int(v) for k, v in
                           (line.split(":", 1) for line in stream)})
    except OSError:
        pass
    return values


def timed(owner, attribute, name, gpu=False, startup=False):
    original = getattr(owner, attribute)
    if getattr(original, "_moe_diagnostic", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if MEASUREMENT is None and not startup:
            return original(*args, **kwargs)
        before = time.monotonic()
        begin = end = None
        if gpu and MEASUREMENT is not None:
            import torch
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
        try:
            return original(*args, **kwargs)
        finally:
            if end is not None:
                end.record()
                GPU_EVENTS.append((name, begin, end))
            key = (MEASUREMENT or "startup") + ":" + name
            TOTALS[key]["calls"] += 1
            TOTALS[key]["wall_s"] += time.monotonic() - before

    wrapped._moe_diagnostic = True
    setattr(owner, attribute, wrapped)


def marked(owner, attribute, name):
    original = getattr(owner, attribute)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with span(name):
            return original(*args, **kwargs)

    setattr(owner, attribute, wrapped)


def snapshot():
    result = {"events": list(EVENTS), "cpu_totals": dict(TOTALS),
              "gpu_totals": {}, "io": io_snapshot()}
    for name, begin, end in GPU_EVENTS:
        end.synchronize()
        value = result["gpu_totals"].setdefault(name, {"calls": 0, "elapsed_ms": 0.0})
        value["calls"] += 1
        value["elapsed_ms"] += begin.elapsed_time(end)
    return result


def set_measurement(label):
    global MEASUREMENT
    GPU_EVENTS.clear()
    TOTALS.clear()
    MEASUREMENT = label
    return {"label": label, "pid": os.getpid()}
