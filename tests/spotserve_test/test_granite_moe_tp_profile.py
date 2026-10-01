from copy import deepcopy

import pytest

from scripts.profile_granite_moe_tp import (
    canonical_gpu_uuid, digest, summarize_samples, validate_frozen_batch, validate_workload,
)


def samples():
    return [{"batch_wall_s": wall, "requests": [
        {"generated_tokens": 512, "finish_reason": "length",
         "latency_s": wall, "ttft_s": 0.1} for _ in range(2)
    ]} for wall in (2.0, 4.0)]


def test_throughput_uses_total_time_not_mean_of_batch_ratios():
    summary = summarize_samples(samples(), 2, 512)
    assert summary["throughput_req_s"] == pytest.approx(4 / 6)
    assert summary["throughput_output_tokens_s"] == pytest.approx(4 * 512 / 6)
    assert summary["latency_p95_nearest_rank_s"] == 4
    assert summary["capacity_scope"].startswith("closed_batch")


@pytest.mark.parametrize("change", ["partial", "short_output", "bad_finish", "nan", "no_ttft"])
def test_incomplete_or_invalid_samples_fail_closed(change):
    value = deepcopy(samples())
    if change == "partial":
        value[0]["requests"].pop()
    elif change == "short_output":
        value[0]["requests"][0]["generated_tokens"] = 511
    elif change == "bad_finish":
        value[0]["requests"][0]["finish_reason"] = "abort"
    elif change == "nan":
        value[0]["batch_wall_s"] = float("nan")
    else:
        value[0]["requests"][0]["ttft_s"] = None
    with pytest.raises(ValueError):
        summarize_samples(value, 2, 512)


def test_empty_profile_is_not_a_zero_cost_candidate():
    with pytest.raises(ValueError):
        summarize_samples([], 2, 512)


def test_8k_plus_512_context_is_valid():
    validate_workload([4096, 8192], [1, 2, 4], 512, 8704, 3)


@pytest.mark.parametrize("prompts,concurrency,output,length,repeats", [
    ([8192], [2], 512, 8192, 3),
    ([4096], [0], 512, 8704, 3),
    ([4096, 4096], [1], 512, 8704, 3),
    ([4096], [1, 1], 512, 8704, 3),
    ([4096], [1], 512, 8704, 1),
])
def test_invalid_workload_is_not_silently_shrunk(prompts, concurrency, output,
                                               length, repeats):
    with pytest.raises(ValueError):
        validate_workload(prompts, concurrency, output, length, repeats)


def frozen_batch():
    return {"prompt_token_ids": [[1, 2, 3]], "remaining_tokens": 2,
            "prompt_sha256s": [digest([1, 2, 3])],
            "frozen_prefix_sha256": digest([1, 2, 3]),
            "source_artifact": "cpu-fixture-not-live-freeze",
            "source_artifact_sha256": "fixture-hash",
            "strict_requested_boundary_verified": False}


def test_prior_exact_tokens_can_be_profiled_without_claiming_a_new_freeze():
    batch = frozen_batch()
    assert validate_frozen_batch(batch, [3], [1], 2) == [[1, 2, 3]]
    assert batch["strict_requested_boundary_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("remaining_tokens", 3), ("prompt_token_ids", [[1, 2, 4]]),
    ("prompt_sha256s", ["wrong"]), ("source_artifact_sha256", None),
    ("frozen_prefix_sha256", "wrong"),
])
def test_wrong_frozen_batch_is_not_substituted_with_synthetic_tokens(field, value):
    batch = frozen_batch()
    batch[field] = value
    with pytest.raises(ValueError):
        validate_frozen_batch(batch, [3], [1], 2)


def test_nvml_and_torch_uuid_formats_preserve_the_same_gpu_identity():
    raw = "a90a5521-c5fc-807b-be8c-4479014ed079"
    assert canonical_gpu_uuid(raw) == canonical_gpu_uuid("GPU-" + raw)


def test_uuid_normalization_does_not_merge_different_devices():
    assert canonical_gpu_uuid("GPU-a90a5521-c5fc-807b-be8c-4479014ed079") != (
        canonical_gpu_uuid("GPU-f8166666-3067-b764-e524-ecc41b05ea38"))


def test_invalid_uuid_is_rejected_not_used_as_a_label():
    with pytest.raises(ValueError):
        canonical_gpu_uuid("gpu-3")
