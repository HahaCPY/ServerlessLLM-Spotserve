"""CPU accounting/gating checks; no fixture is GPU EP evidence."""

from copy import deepcopy
import json

import pytest

from scripts.run_moe_architecture_diagnostics import (
    MAJOR_WORKER_SPANS, cold_breakdown, diagnostic_log_events, validate_placement,
)


def placement(ep=True):
    ranks = []
    for rank in range(2):
        ranks.append({"actual_gpu_uuid": f"00000000-0000-0000-0000-00000000000{rank}",
            "model_class": "GraniteMoeForCausalLM", "expert_modules": [
                {"layer": layer, "checkpoint_samples_match": True, "weight_device": f"cuda:{rank}",
                 "parallel": {"use_ep": ep, "ep_size": 2 if ep else 1, "tp_size": 1 if ep else 2},
                 "w13_shape": [20 if ep else 40, 1024 if ep else 512, 1536],
                 "intermediate_partition": 512 if ep else 256,
                 "global_expert_ids": list(range(rank * 20, rank * 20 + 20)) if ep else list(range(40))}
                for layer in range(32)]})
    return {"status": "passed", "placement": ranks, "ep_flag": ep}


UUIDS = {"GPU-00000000-0000-0000-0000-000000000000", "GPU-00000000-0000-0000-0000-000000000001"}


def test_ep_requires_real_disjoint_checkpoint_verified_weights():
    assert validate_placement(placement(), UUIDS)
    assert validate_placement(placement(False), UUIDS)


@pytest.mark.parametrize("change", ("overlap", "wrong_weights", "tp_sharded", "wrong_gpu", "missing_layer"))
def test_flag_only_or_incomplete_placement_cannot_pass(change):
    report = placement()
    module = report["placement"][0]["expert_modules"][0]
    if change == "overlap":
        module["global_expert_ids"] = list(range(20, 40))
    elif change == "wrong_weights":
        module["checkpoint_samples_match"] = False
    elif change == "tp_sharded":
        module["intermediate_partition"] = 256
    elif change == "wrong_gpu":
        report["placement"][0]["actual_gpu_uuid"] = "00000000-0000-0000-0000-000000000009"
    else:
        report["placement"][0]["expert_modules"].pop()
    with pytest.raises(ValueError):
        validate_placement(report, UUIDS)


def test_interleaved_rank_logs_do_not_abort_probe():
    a = {"event": "span_end", "pid": 1}
    b = {"event": "span_end", "pid": 2}
    line = "(Worker 1) MOE_DIAGNOSTIC_EVENT=" + json.dumps(a) + "MOE_DIAGNOSTIC_EVENT=" + json.dumps(b)
    assert diagnostic_log_events(line) == [a, b]
    assert diagnostic_log_events("MOE_DIAGNOSTIC_EVENT={broken") == []


def cold_fixture():
    events = [{"event": "span_end", "name": name, "duration_s": 1.0}
              for name in MAJOR_WORKER_SPANS]
    events.append({"event": "span_end", "name": "checkpoint_and_weight_loading", "duration_s": 0.5})
    frontend = [{"event": "recovery_frontend_entry", "monotonic_s": 1.0},
                {"event": "span_start", "name": "recovery_engine_constructor", "monotonic_s": 3.0},
                {"event": "span_end", "name": "recovery_engine_constructor", "monotonic_s": 10.0}]
    return {"startup": {"events": events, "cpu_totals": {}}}, frontend


def test_cold_accounting_excludes_nested_double_counting():
    diagnostic, frontend = cold_fixture()
    result = cold_breakdown({}, diagnostic, frontend, 0.0, 11.0, 15.0)
    assert result["total_s"] == 15
    assert sum(result["major_nonoverlapping_stages_s"].values()) == 15
    assert result["weight_loading_share_percent"] == pytest.approx(100 * 0.5 / 15)


def test_overlapping_major_stages_cannot_create_false_breakdown():
    diagnostic, frontend = cold_fixture()
    diagnostic["startup"]["events"][0]["duration_s"] = 4.0
    with pytest.raises(ValueError, match="overlapping"):
        cold_breakdown({}, diagnostic, frontend, 0.0, 11.0, 15.0)


def test_spawn_child_entry_does_not_replace_frontend_timestamp():
    diagnostic, frontend = cold_fixture()
    for event in frontend:
        event["pid"] = 1
    frontend.append({"event": "recovery_frontend_entry", "pid": 2, "monotonic_s": 8.0})
    result = cold_breakdown({}, diagnostic, frontend, 0.0, 11.0, 15.0)
    stages = result["major_nonoverlapping_stages_s"]
    assert stages["container_python_startup"] == 1
    assert stages["frontend_imports_and_configuration"] == 2
