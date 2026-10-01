"""Control-plane tests; real worker correctness is checked separately on GPU."""

from types import SimpleNamespace
import threading

import pytest

from scripts.moe_gpu_runtime import GenerationPreempted, GPUWorker, validate_stream_update
from scripts.moe_recovery_actor import actual_runtime_evidence, execute_recovery


def worker_with_events(events):
    worker = GPUWorker.__new__(GPUWorker)
    worker.pool = SimpleNamespace(timeout=0.05)
    worker.events, worker.cancelled_requests = list(events), set()
    worker.event_condition = threading.Condition()
    worker.io_error = None
    return worker


def test_control_ack_cannot_swallow_foreground_output():
    worker = worker_with_events([{"event": "output", "request_id": "a", "token_ids": [7]},
                                 {"event": "generation_paused"}])
    assert worker.wait(lambda e: e["event"] == "generation_paused")["event"] == "generation_paused"
    assert worker.wait(lambda e: e["event"] == "output")["token_ids"] == [7]


def test_one_request_cannot_swallow_another_request_event():
    worker = worker_with_events([{"event": "output", "request_id": "a"},
                                 {"event": "output", "request_id": "b"}])
    assert worker.wait(lambda e: e["request_id"] == "b")["request_id"] == "b"
    assert worker.wait(lambda e: e["request_id"] == "a")["request_id"] == "a"


def test_preempted_consumer_is_woken_instead_of_hanging():
    worker = worker_with_events([])
    worker.cancel_measurement("a")
    with pytest.raises(GenerationPreempted):
        worker.wait(lambda e: False, request_id="a")


def test_final_buffered_event_is_not_lost_when_connection_closes():
    worker = worker_with_events([{"event": "output", "finished": True}])
    worker.io_error = "EOF"
    assert worker.wait(lambda e: e["event"] == "output")["finished"]
    with pytest.raises(RuntimeError, match="closed"):
        worker.wait(lambda e: False)


def test_stream_update_preserves_previously_delivered_prefix_and_allows_batched_tokens():
    assert validate_stream_update([7], {"generated_tokens": 3, "cumulative_token_ids": [7, 8, 9]}) == [7, 8, 9]
    assert validate_stream_update([7], {"generated_tokens": 1, "cumulative_token_ids": [7]}) == [7]


@pytest.mark.parametrize("event", [
    {"generated_tokens": 0, "cumulative_token_ids": []},
    {"generated_tokens": 2, "cumulative_token_ids": [7]},
    {"generated_tokens": 2, "cumulative_token_ids": [8, 7]},
    {"generated_tokens": True, "cumulative_token_ids": [7]},
])
def test_stream_update_rejects_missing_counts_and_revised_already_delivered_tokens(event):
    with pytest.raises(ValueError, match="streamed"):
        validate_stream_update([7], event)


def test_runtime_evidence_uses_worker_observation_and_host_handoff_not_kv_claim():
    worker = SimpleNamespace(runtime={"tensor_parallel_size": 2,
        "pipeline_parallel_size": 1, "data_parallel_size": 1, "enable_expert_parallel": False},
        runtime_signature="observed-signature", ranks=[{"physical_gpu_index": 3}, {"physical_gpu_index": 1}])
    evidence = actual_runtime_evidence(worker, {"frozen_prefix_sha256": "observed-token-hash",
        "install_mode": "host_token_handoff_excluding_gpu_prefill"}, "artifact")
    assert evidence["physical_gpu_indices"] == [1, 3]
    assert evidence["frozen_prefix_sha256"] == "observed-token-hash"
    assert evidence["direct_gpu_kv_restore"] is False


def test_forced_recovery_stops_our_source_before_target_handoff():
    from copy import deepcopy
    from tests.spotserve_test.test_moe_route_cost import calibration

    data = calibration()
    case = deepcopy(data["source_cases"]["validation-a"])
    case.update({"client_prefix": [9] * 256, "reference": {"output_token_ids": [9] * 512},
                 "notice_monotonic_s": 1.0, "source_last_token_monotonic_s": 0.9})
    events = []

    class Source:
        label, group, closed = "own-source", [0], False

        def close(self):
            events.append("source_stopped")
            self.closed = True

        def evidence(self):
            return {"worker_exit_code": 0, "physical_gpu_indices": self.group}

    source = Source()

    class Target:
        label, group, closed = "own-target", [1], False
        runtime = {"tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                   "data_parallel_size": 1, "enable_expert_parallel": False}
        runtime_signature = data["candidates"]["gpu1-tp1"]["runtime_signature"]
        ranks = [{"rank": 0, "physical_gpu_index": 1,
                  "granite_moe_model": True, "expert_modules": True}]

        def command_event(self, command, event):
            assert source.closed
            events.append("target_handoff")
            return {"install_mode": "host_token_handoff_excluding_gpu_prefill",
                    "gpu_kv_restored": False, "token_count": 4352,
                    "frozen_prefix_sha256": case["frozen"]["full_prefix_sha256"]}

        def measure(self, *args):
            assert source.closed
            events.append("target_serving")
            return {"output_token_ids": [9] * 256, "timed_jit_warnings": [],
                    "started_monotonic_s": 2.0, "first_token_monotonic_s": 2.1,
                    "completed_monotonic_s": 3.0, "first_four_tokens_s": 0.2,
                    "max_delivery_gap_s": 0.1}

        def evidence(self):
            return {"physical_gpu_indices": self.group}

        def close(self):
            self.closed = True

    target = Target()
    result, selected = execute_recovery(SimpleNamespace(timeout=1), data, source, case,
                                        "migration", True, "CPU-test", [target])
    assert selected is target
    assert events == ["source_stopped", "target_handoff", "target_serving"]
    assert result["configuration_applied"]
    assert result["source_termination"]["worker_exit_code"] == 0
    assert result["maximum_delivery_gap_including_handoff_s"] == pytest.approx(1.2)
