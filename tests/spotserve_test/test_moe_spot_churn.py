"""CPU guards for the wall-clock protocol; not GPU experiment evidence."""

import pytest

from scripts.moe_spot_churn import serializable_live, trace_from_pilot
from scripts.run_moe_experiment_pipeline import report_markdown, summarize_runs
from tests.spotserve_test.test_moe_gpu_runtime import worker_with_events


def test_trace_is_fixed_from_both_pilot_versions_and_has_two_revocations():
    pilot = {"runs": [{"notice_to_first_token_s": 90.2}, {"notice_to_first_token_s": 92.1}]}
    trace = trace_from_pilot(pilot)
    assert trace["first_revoke_s"] == 5.0
    assert trace["second_revoke_s"] == 135.5
    assert trace["second_return_s"] == 140.5
    assert trace["second_revoked_gpus"] == [1, 2, 3]
    assert trace["request_arrivals_s"] == list(map(float, range(12)))
    assert trace["not_adjusted_per_version_or_run"]
    assert trace_from_pilot({"runs": list(reversed(pilot["runs"]))}) == trace


def test_live_executor_handles_are_not_written_as_experiment_artifacts():
    assert serializable_live({"future": object(), "delivery_executor": object(),
                              "request_id": "actual-request", "timeline": []}) == {
                                  "request_id": "actual-request", "timeline": []}


def test_summary_requires_three_complete_independent_runs_and_uses_paired_repeats():
    rows = [{"version": version, "repeat": i, "status": "passed", "formal_run": True,
             "latency": 10 + i + (1 if version == "moe_aware" else 0)}
            for version in ("original", "moe_aware") for i in range(3)]
    summary = summarize_runs(list(reversed(rows)), "latency")
    assert summary["original"]["mean"] == 11
    assert summary["original"]["sample_sd"] == 1
    assert summary["paired_moe_minus_original"] == [1, 1, 1]
    with pytest.raises(ValueError, match="independent"):
        summarize_runs(rows[:-1], "latency")
    rows[0]["formal_run"] = False
    with pytest.raises(ValueError, match="formal"):
        summarize_runs(rows, "latency")


def test_failed_gate_report_does_not_present_fabricated_formal_means():
    report = report_markdown({"status": "failed", "stages": [
        {"stage": "pilot", "status": "failed", "artifact": "pilot/pilot.json"}]})
    assert "尚未全部通過" in report
    assert "mean ±" not in report


def snapshot_worker(monkeypatch, generated=197, carried=100):
    from scripts import moe_gpu_runtime

    worker = worker_with_events([])
    prompt, reference = [11, 12], {"output_token_ids": list(range(512))}
    old = reference["output_token_ids"][:carried]
    new = reference["output_token_ids"][carried:generated]
    metadata = {"found": True, "prompt_tokens": prompt + old, "output_tokens": new,
                "tokens": prompt + reference["output_token_ids"][:generated],
                "allocated_kv_block_count": 14, "computed_tokens": len(prompt) + generated - 1}
    worker.client_visible = {"live": new[:80]}
    worker.live_measurements = {"live": {"started_monotonic_s": 1.0, "deliveries": []}}
    worker.command_event = lambda command, event: ({"result": metadata} if event == "metadata" else {})
    monkeypatch.setattr(moe_gpu_runtime, "validate_route_histogram", lambda *args: {"origin": "test_fixture"})
    return worker, prompt, old, reference, metadata


def test_wall_clock_freeze_keeps_actual_boundary_and_previous_handoff_prefix(monkeypatch):
    worker, prompt, old, reference, _ = snapshot_worker(monkeypatch)
    case = worker.snapshot_running("live", prompt, old, reference)
    assert case["frozen"]["completed_tokens"] == 197
    assert case["frozen"]["remaining_tokens"] == 315
    assert case["frozen"]["tokens"] == prompt + reference["output_token_ids"][:197]
    assert case["frozen"]["requested_token_boundary"] is None
    assert case["client_prefix"] == reference["output_token_ids"][:197]
    assert "live" in worker.cancelled_requests


@pytest.mark.parametrize("invalid", ["complete", "wrong_prompt", "wrong_output", "no_kv"])
def test_wall_clock_snapshot_rejects_invalid_or_finished_source(monkeypatch, invalid):
    worker, prompt, old, reference, metadata = snapshot_worker(monkeypatch)
    if invalid == "complete":
        metadata["output_tokens"] = reference["output_token_ids"][len(old):]
        metadata["tokens"] = prompt + reference["output_token_ids"]
    elif invalid == "wrong_prompt":
        metadata["prompt_tokens"] = prompt
    elif invalid == "wrong_output":
        metadata["output_tokens"][0] = 10000
    else:
        metadata["allocated_kv_block_count"] = 0
    with pytest.raises(ValueError, match="snapshot"):
        worker.snapshot_running("live", prompt, old, reference)
