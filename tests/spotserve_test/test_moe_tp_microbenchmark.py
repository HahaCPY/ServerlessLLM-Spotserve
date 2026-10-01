from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.profile_granite_moe_tp import output_counts, summarize_samples
from scripts.run_moe_tp_microbenchmark import crossover_gate, curves_svg, summarize_stage


def runs():
    result = []
    for block in range(3):
        for tp, wall in ((1, 1.0), (2, 1.2)):
            cells = []
            for output in (1, 64):
                cells.append({"status": "passed", "prompt_tokens": 512, "concurrency": 1,
                              "max_new_tokens": output, "samples": [
                                  {"batch_wall_s": wall + 0.001 * block, "requests": [
                                      {"latency_s": wall + 0.001 * block, "ttft_s": 0.1,
                                       "output_sha256": "same", "prompt_sha256": "same"}]
                                   } for _ in range(3)]})
            result.append({"status": "passed", "block": block,
                           "profile": {"status": "passed", "tp": tp, "cells": cells}})
    return result


def test_output_sweep_is_opt_in_and_preserves_single_length():
    assert output_counts(SimpleNamespace(max_new_tokens=256)) == [256]
    assert output_counts(SimpleNamespace(max_new_tokens=512, output_tokens=[1, 64, 256])) == [1, 64, 256]


@pytest.mark.parametrize("lengths", [[1, 1], [0, 64], [True, 64]])
def test_invalid_output_sweeps_are_rejected(lengths):
    with pytest.raises(ValueError):
        output_counts(SimpleNamespace(max_new_tokens=256, output_tokens=lengths))


def test_single_token_decode_rate_is_unavailable_not_divided_by_zero():
    samples = [{"batch_wall_s": 1.0, "requests": [{"generated_tokens": 1,
                "finish_reason": "length", "latency_s": 1.0, "ttft_s": 0.9}]}]
    summary = summarize_samples(samples, 1, 1)
    assert summary["per_remaining_token_mean_s"] is None
    assert summary["decode_after_first_mean_s"] == pytest.approx(0.1)
    assert "not_pure" in summary["ttft_definition"]


def test_first_token_after_finish_is_invalid():
    samples = [{"batch_wall_s": 1.0, "requests": [{"generated_tokens": 64,
                "finish_reason": "length", "latency_s": 1.0, "ttft_s": 1.1}]}]
    with pytest.raises(ValueError, match="precede"):
        summarize_samples(samples, 1, 64)


def test_three_engine_block_means_not_nine_independent_deployments():
    rows = summarize_stage(runs())
    assert rows[1]["profiles"]["1"]["independent_engine_blocks"] == 3
    assert len(rows[1]["paired_tp2_minus_tp1_s"]) == 3
    assert rows[1]["repeatable_winner"] == "tp1"
    assert rows[1]["outputs_identical_across_tp"]


def test_missing_block_and_partial_batch_do_not_create_average():
    with pytest.raises(ValueError, match="three independent"):
        summarize_stage(runs()[:-1])
    values = runs()
    values[0]["profile"]["cells"][0]["samples"].pop()
    with pytest.raises(ValueError, match="three complete"):
        summarize_stage(values)


def test_input_mismatch_is_rejected():
    values = runs()
    values[1]["profile"]["cells"][0]["samples"][0]["requests"][0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="identical inputs"):
        summarize_stage(values)


def test_single_token_win_cannot_qualify_multi_token_crossover():
    rows = summarize_stage(runs())
    rows[0]["repeatable_winner"] = "tp2"
    gate = crossover_gate(rows)
    assert gate["single_token_only_tp2_advantage"]
    assert not gate["multi_token_crossover_observed"]
    assert not gate["fleet_experiment_eligible"]


def test_crossover_requires_both_signs_and_matching_outputs():
    rows = summarize_stage(runs())
    win = deepcopy(rows[1])
    win.update(prompt_tokens=4096, repeatable_winner="tp2")
    assert crossover_gate(rows + [win])["multi_token_crossover_observed"]
    win["outputs_identical_across_tp"] = False
    assert not crossover_gate(rows + [win])["multi_token_crossover_observed"]


def test_cost_plot_is_repo_native_svg():
    svg = curves_svg(summarize_stage(runs()), 1)
    assert svg.startswith("<svg")
    assert "polyline" in svg
    assert "engine-block sample SD" in svg
