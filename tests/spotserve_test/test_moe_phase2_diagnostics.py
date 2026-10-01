"""CPU analysis tests, not substitutes for GPU correctness evidence."""

import pytest

from scripts.analyze_moe_phase2_diagnostics import first_difference,logprob_comparison,refined_cache_stages,kernel_category,vector_relative_l2
from scripts.run_moe_cache_diagnostics import cache_inventory


def probs(a,b):
    return {"1":{"logprob":a},"2":{"logprob":b}}


def test_first_divergence_and_length_mismatch():
    assert first_difference([1,2],[1,2]) is None
    assert first_difference([1,2],[1,3])==1
    assert first_difference([1],[1,2])==1


def test_common_prefix_tie_is_bounded_not_hidden():
    result=logprob_comparison([probs(-1,-1.01)],[probs(-1.02,-1)])
    assert result["top1_disjoint_positions"]==1
    assert result["abs_tolerance_failures"]==0
    assert result["unexplained_top1_changes"]==0


def test_large_numeric_error_does_not_pass():
    result=logprob_comparison([probs(-1,-2)],[probs(-2,-1)])
    assert result["abs_tolerance_failures"]==1


def test_no_shared_probabilities_is_not_correctness_evidence():
    result=logprob_comparison([{"1":{"logprob":-1}}],[{"2":{"logprob":-1}}])
    assert result["abs_tolerance_failures"]==1


def test_cache_inventory_detects_library_changes(tmp_path):
    assert cache_inventory(tmp_path)==[]
    (tmp_path/"sampling.so").write_bytes(b"first")
    initial=cache_inventory(tmp_path)
    assert initial==cache_inventory(tmp_path)
    (tmp_path/"sampling.so").write_bytes(b"second")
    assert initial!=cache_inventory(tmp_path)


def test_refined_cuda_accounting_does_not_double_count_nested_init():
    events=[{"event":"diagnostic_python_entry","monotonic_s":12},
        {"event":"span_start","name":"worker_constructor","monotonic_s":17,"span_id":8},
        {"event":"span_start","name":"early_cuda_lazy_initialization","monotonic_s":14,"span_id":2,"parent_id":None},
        {"event":"span_start","name":"early_cuda_lazy_initialization","monotonic_s":14.5,"span_id":3,"parent_id":2},
        {"event":"span_end","name":"early_cuda_lazy_initialization","monotonic_s":15,"span_id":3,"duration_s":0.5},
        {"event":"span_end","name":"early_cuda_lazy_initialization","monotonic_s":16,"span_id":2,"duration_s":2}]
    row={"clock_provenance":{"engine_start":10},"diagnostics":{"startup":{"events":events}},
         "major_nonoverlapping_stages_s":{"engine_bootstrap_and_other":10,"other":4},"total_s":14}
    stages=refined_cache_stages(row)
    assert stages["cuda_lazy_initialization_all_processes_outermost"]==2
    assert sum(stages.values())==14


def test_frontend_cuda_init_is_removed_from_bootstrap_not_added_twice():
    events=[{"event":"diagnostic_python_entry","pid":8,"monotonic_s":12},
        {"event":"span_start","name":"worker_constructor","pid":8,"monotonic_s":17,"span_id":8}]
    parent=[{"event":"span_start","name":"early_cuda_lazy_initialization","pid":1,"monotonic_s":10.5,"span_id":2,"parent_id":None},
            {"event":"span_end","name":"early_cuda_lazy_initialization","pid":1,"monotonic_s":11.5,"span_id":2,"duration_s":1}]
    row={"clock_provenance":{"engine_start":10},"diagnostics":{"startup":{"events":events}},
         "major_nonoverlapping_stages_s":{"engine_bootstrap_and_other":10,"other":4},"total_s":14}
    stages=refined_cache_stages(row,parent)
    assert stages["cuda_lazy_initialization_all_processes_outermost"]==1
    assert stages["engine_core_process_spawn_and_preworker_bootstrap"]==1
    assert sum(stages.values())==14


def test_gpu_annotations_are_not_counted_as_collective_kernels():
    assert kernel_category("nccl:_all_gather_base") is None
    assert kernel_category("MoEDiag:expert_apply") is None
    assert kernel_category("ncclDevKernel_AllGather_RING_LL(args)")=="collective_AllGather"
    assert kernel_category("fused_moe_kernel")=="expert_gemm"
    assert kernel_category("moe_align_block_size_small_batch_expert_kernel")=="dispatch_alignment"


def test_layer_relative_l2_preserves_reference_denominator():
    assert vector_relative_l2([1.1,0],[1,0])==pytest.approx(0.1)
