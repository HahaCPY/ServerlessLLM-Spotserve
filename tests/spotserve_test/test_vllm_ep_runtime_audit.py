from pathlib import Path

from sllm.spot.vllm_ep_runtime_audit import (
    audit_source_tree,
    classify_audit_report,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_vllm_ep_runtime_audit_detects_observe_only_patch(tmp_path: Path):
    vllm_root = tmp_path / "vllm"
    write(
        vllm_root / "spotserve_moe.py",
        """
def record_moe_routing():
    pass
def apply_expert_placement_plan():
    return {"reason": "physical_expert_placement_migration_not_supported",
            "hook_kind": "spotserve_observation_only"}
def verify_expert_placement_plan():
    return {"reason": "physical_expert_placement_verification_not_supported",
            "hook_kind": "spotserve_observation_only"}
def inspect_runtime_expert_placement():
    pass
def _compare_runtime_expert_placement():
    pass
""",
    )
    write(
        vllm_root / "v1/worker/gpu_model_runner.py",
        """
from vllm.spotserve_moe import clear_moe_request_metadata, moe_request_context
with moe_request_context(req_ids, num_scheduled_tokens_np):
    pass
clear_moe_request_metadata([])
""",
    )
    write(
        vllm_root
        / "model_executor/layers/fused_moe/fused_moe_modular_method.py",
        "record_moe_routing(layer_name='x', topk_ids=topk_ids)\n",
    )
    write(
        vllm_root
        / "model_executor/layers/fused_moe/unquantized_fused_moe_method.py",
        "record_moe_routing(layer_name='x', topk_ids=topk_ids)\n",
    )
    write(
        vllm_root / "v1/worker/worker_base.py",
        """
def get_request_moe_metadata(): pass
def get_moe_runtime_metadata(): pass
def _spotserve_runtime_expert_placement_snapshot(): pass
def apply_expert_placement_plan(): pass
def verify_expert_placement_plan(): pass
""",
    )
    write(
        vllm_root / "v1/engine/core.py",
        """
def get_request_moe_metadata(): pass
def get_moe_runtime_metadata(): pass
def apply_expert_placement_plan(): pass
def verify_expert_placement_plan(): pass
""",
    )
    write(
        vllm_root / "v1/engine/async_llm.py",
        """
async def get_request_moe_metadata(): pass
async def get_moe_runtime_metadata(): pass
async def apply_expert_placement_plan(): pass
async def verify_expert_placement_plan(): pass
""",
    )

    source_checks = audit_source_tree(tmp_path)

    assert source_checks["available"] is True
    assert source_checks["forward_path_has_moe_request_context"] is True
    assert source_checks["route_recording_hooks"] == 2
    assert source_checks["apply_verify_boundary_present"] is True
    assert source_checks["runtime_layout_inspector_present"] is True
    assert source_checks["observe_only_markers_present"] is True


def test_vllm_ep_runtime_audit_classifies_observe_only_contract():
    gate = classify_audit_report({
        "source_checks": {
            "apply_verify_boundary_present": True,
            "observe_only_markers_present": True,
        },
        "runtime_hook_probe": {
            "apply_callable": True,
            "verify_callable": True,
            "apply_result": {
                "applied": False,
                "reason": "physical_expert_placement_migration_not_supported",
                "hook_kind": "spotserve_observation_only",
            },
            "verify_result": {
                "verified": False,
                "reason": (
                    "physical_expert_placement_verification_not_supported"
                ),
                "hook_kind": "spotserve_observation_only",
                "contract_seen_by_runtime": True,
            },
        },
    })

    assert gate["classification"] == "observe_only_expert_placement_contract"
    assert gate["runtime_verification_level"] == "contract_seen_only"
    assert gate["can_claim_physical_expert_migration"] is False
    assert gate["runtime_verified_placement"] is False
    assert gate["runtime_can_verify_physical_placement"] is False
    assert gate["runtime_can_remap_live_ep_rank"] is False
    assert gate["runtime_can_measure_all_to_all"] is False
    assert gate["runtime_contract_seen_by_runtime"] is True
    assert gate["runtime_physical_weight_migration"] is False
    assert gate["recommended_execution_model"] == "expert_aware_actor_recreate"


def test_vllm_ep_runtime_audit_classifies_physical_migration_support():
    gate = classify_audit_report({
        "source_checks": {
            "apply_verify_boundary_present": True,
            "observe_only_markers_present": False,
        },
        "runtime_hook_probe": {
            "apply_callable": True,
            "verify_callable": True,
            "apply_result": {
                "applied": True,
                "physical_weight_migration": True,
            },
            "verify_result": {
                "verified": True,
                "physical_weight_migration": True,
            },
        },
    })

    assert gate["classification"] == "physical_expert_migration_supported"
    assert gate["runtime_verification_level"] == "physical_migration_verified"
    assert gate["can_claim_physical_expert_migration"] is True
    assert gate["runtime_verified_placement"] is True
    assert gate["runtime_can_verify_physical_placement"] is True
    assert gate["runtime_can_remap_live_ep_rank"] is False
    assert gate["runtime_can_measure_all_to_all"] is False
    assert gate["runtime_contract_seen_by_runtime"] is False
    assert gate["runtime_physical_weight_migration"] is True
    assert gate["recommended_execution_model"] == "live_expert_weight_migration"
