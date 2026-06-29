# trace 模擬 spot instance

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


SUPPORTED_EVENTS = {"preempt", "recover", "dead"}


@dataclass(frozen=True)
class SpotEvent:
    time: float
    event: str
    node_id: Optional[str] = None
    model_name: Optional[str] = None
    instance_id: Optional[str] = None

    def __post_init__(self):
        if self.time < 0:
            raise ValueError("Spot event time must be non-negative")
        if self.event not in SUPPORTED_EVENTS:
            raise ValueError(
                f"Unsupported spot event '{self.event}'. "
                f"Expected one of {sorted(SUPPORTED_EVENTS)}"
            )
        if self.node_id is None and self.instance_id is None:
            raise ValueError(
                "Spot event must target either node_id or instance_id"
            )


def _event_from_dict(raw_event: dict, source: str) -> SpotEvent:
    try:
        event_time = float(raw_event["time"])
        event_type = str(raw_event["event"])
    except KeyError as exc:
        raise ValueError(f"{source}: missing required field {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid event time") from exc

    return SpotEvent(
        time=event_time,
        event=event_type,
        node_id=raw_event.get("node_id"),
        model_name=raw_event.get("model_name"),
        instance_id=raw_event.get("instance_id"),
    )


def load_spot_trace(trace_path: str | Path) -> List[SpotEvent]:
    path = Path(trace_path)
    if path.suffix != ".jsonl":
        raise ValueError("Only JSONL spot traces are supported in v1")

    events: List[SpotEvent] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw_event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL event"
                ) from exc
            events.append(
                _event_from_dict(raw_event, f"{path}:{line_number}")
            )

    return sorted(events, key=lambda event: event.time)


def iter_spot_trace(trace_path: str | Path) -> Iterable[SpotEvent]:
    return iter(load_spot_trace(trace_path))
