"""CPU-only scheduler logic; live EngineCore checks are required separately."""

from types import SimpleNamespace

import pytest

from sllm.spot.benchmark_freeze import (
    FREEZE_KEY, NativeFreezeMixin, freeze_sampling_extra_args,
)


class FakeScheduler:
    freeze_pause_state = "paused"

    def __init__(self, count, threshold=256, placeholders=0):
        self.state = "running"
        self.running = [SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={FREEZE_KEY: threshold}),
            max_tokens=512, output_token_ids=[1] * count,
            num_output_placeholders=placeholders)]

    def update_from_output(self):
        return {"outputs": "unchanged"}

    def set_pause_state(self, state):
        self.state = state


class ControlledScheduler(NativeFreezeMixin, FakeScheduler):
    pass


def test_native_boundary_pauses_an_unfinished_request_without_truncation():
    scheduler = ControlledScheduler(256)
    assert scheduler.update_from_output() == {"outputs": "unchanged"}
    assert scheduler.state == "paused"
    assert len(scheduler.running[0].output_token_ids) == 256


def test_before_boundary_does_not_pause():
    scheduler = ControlledScheduler(255)
    scheduler.update_from_output()
    assert scheduler.state == "running"


def test_public_resume_can_continue_after_a_one_shot_barrier():
    scheduler = ControlledScheduler(256)
    scheduler.update_from_output()
    scheduler.state = "running"
    scheduler.running[0].output_token_ids.append(1)
    scheduler.update_from_output()
    assert scheduler.state == "running"


def test_ordinary_warmup_without_freeze_metadata_is_unchanged():
    scheduler = ControlledScheduler(64)
    scheduler.running[0].sampling_params.extra_args = None
    scheduler.update_from_output()
    assert scheduler.state == "running"


def test_overshoot_is_rejected_instead_of_slicing_output_tokens():
    scheduler = ControlledScheduler(258)
    with pytest.raises(RuntimeError, match="overshoot"):
        scheduler.update_from_output()
    assert len(scheduler.running[0].output_token_ids) == 258


def test_in_flight_async_tokens_cannot_be_reported_as_frozen():
    with pytest.raises(RuntimeError, match="async_tokens"):
        ControlledScheduler(256, placeholders=1).update_from_output()


@pytest.mark.parametrize("threshold", [0, 512, 513, True, 256.5])
def test_invalid_boundaries_are_rejected(threshold):
    with pytest.raises(ValueError):
        freeze_sampling_extra_args(threshold, 512)
