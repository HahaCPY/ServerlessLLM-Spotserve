"""Audit vLLM MoE expert-placement runtime capabilities.

Phase 5 starts by deciding whether an ExpertPlacementPlan can be applied to the
live vLLM runtime, or whether SpotServe should keep using actor recreate as the
execution mechanism.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SOURCE_MARKERS = {
    "spotserve_moe": {
        "path": "spotserve_moe.py",
        "markers": {
            "record_moe_routing": "def record_moe_routing",
            "apply_hook": "def apply_expert_placement_plan",
            "verify_hook": "def verify_expert_placement_plan",
            "observe_only_reason": (
                "physical_expert_placement_migration_not_supported"
            ),
            "observe_only_kind": "spotserve_observation_only",
        },
    },
    "gpu_model_runner": {
        "path": "v1/worker/gpu_model_runner.py",
        "markers": {
            "imports_moe_context": "moe_request_context",
            "forward_path_context": "moe_request_context(",
            "clears_request_metadata": "clear_moe_request_metadata",
        },
    },
    "fused_moe_modular_method": {
        "path": (
            "model_executor/layers/fused_moe/"
            "fused_moe_modular_method.py"
        ),
        "markers": {
            "records_topk_ids": "record_moe_routing(",
        },
    },
    "unquantized_fused_moe_method": {
        "path": (
            "model_executor/layers/fused_moe/"
            "unquantized_fused_moe_method.py"
        ),
        "markers": {
            "records_topk_ids": "record_moe_routing(",
        },
    },
    "worker_base": {
        "path": "v1/worker/worker_base.py",
        "markers": {
            "request_moe_metadata_hook": "def get_request_moe_metadata",
            "runtime_moe_metadata_hook": "def get_moe_runtime_metadata",
            "apply_hook": "def apply_expert_placement_plan",
            "verify_hook": "def verify_expert_placement_plan",
        },
    },
    "engine_core": {
        "path": "v1/engine/core.py",
        "markers": {
            "request_moe_metadata_rpc": "def get_request_moe_metadata",
            "runtime_moe_metadata_rpc": "def get_moe_runtime_metadata",
            "apply_rpc": "def apply_expert_placement_plan",
            "verify_rpc": "def verify_expert_placement_plan",
        },
    },
    "async_llm": {
        "path": "v1/engine/async_llm.py",
        "markers": {
            "request_moe_metadata_client": (
                "async def get_request_moe_metadata"
            ),
            "runtime_moe_metadata_client": (
                "async def get_moe_runtime_metadata"
            ),
            "apply_client": "async def apply_expert_placement_plan",
            "verify_client": "async def verify_expert_placement_plan",
        },
    },
}


def _safe_import(module_name: str) -> tuple[Optional[Any], str]:
    try:
        return importlib.import_module(module_name), ""
    except Exception as exc:
        return None, repr(exc)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return repr(value)
    return value


def _normalize_vllm_package_root(source_root: Optional[str | Path]) -> Optional[Path]:
    if source_root is None:
        return None
    root = Path(source_root).expanduser().resolve()
    if (root / "spotserve_moe.py").exists() or (root / "v1").exists():
        return root
    if (root / "vllm").is_dir():
        return root / "vllm"
    return root


def _source_root_from_imported_vllm(vllm_module: Any) -> Optional[Path]:
    module_file = getattr(vllm_module, "__file__", None)
    if not module_file:
        return None
    return Path(module_file).resolve().parent


def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def audit_source_tree(package_root: Optional[str | Path]) -> Dict[str, Any]:
    root = _normalize_vllm_package_root(package_root)
    if root is None:
        return {
            "package_root": "",
            "available": False,
            "files": {},
            "forward_path_has_moe_request_context": False,
            "route_recording_hooks": 0,
            "apply_verify_boundary_present": False,
            "observe_only_markers_present": False,
        }

    files: Dict[str, Any] = {}
    route_recording_hooks = 0
    for name, spec in SOURCE_MARKERS.items():
        relative_path = str(spec["path"])
        path = root / relative_path
        text = _read_source(path)
        markers = {
            marker_name: bool(marker in text)
            for marker_name, marker in spec["markers"].items()
        }
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "markers": markers,
            "all_markers_present": bool(markers and all(markers.values())),
        }
        if name in {
            "fused_moe_modular_method",
            "unquantized_fused_moe_method",
        } and markers.get("records_topk_ids"):
            route_recording_hooks += 1

    spotserve_moe = files.get("spotserve_moe", {}).get("markers", {})
    worker_base = files.get("worker_base", {}).get("markers", {})
    engine_core = files.get("engine_core", {}).get("markers", {})
    async_llm = files.get("async_llm", {}).get("markers", {})
    gpu_model_runner = files.get("gpu_model_runner", {}).get("markers", {})
    return {
        "package_root": str(root),
        "available": any(file_info["exists"] for file_info in files.values()),
        "files": files,
        "forward_path_has_moe_request_context": bool(
            gpu_model_runner.get("forward_path_context")
        ),
        "route_recording_hooks": route_recording_hooks,
        "apply_verify_boundary_present": bool(
            spotserve_moe.get("apply_hook")
            and spotserve_moe.get("verify_hook")
            and worker_base.get("apply_hook")
            and worker_base.get("verify_hook")
            and engine_core.get("apply_rpc")
            and engine_core.get("verify_rpc")
            and async_llm.get("apply_client")
            and async_llm.get("verify_client")
        ),
        "observe_only_markers_present": bool(
            spotserve_moe.get("observe_only_reason")
            and spotserve_moe.get("observe_only_kind")
        ),
    }


def _sample_expert_placement_plan() -> Dict[str, Any]:
    return {
        "model_name": "spotserve-phase5-audit",
        "placement_fingerprint": "phase5-audit-sample",
        "placement_epoch": 1,
        "expert_placement_available": True,
        "physical_weight_migration": False,
        "target_parallel_plan": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "replica_count": 1,
            "enable_expert_parallel": False,
        },
        "expert_to_target_rank": {
            "layer:0/expert:0": "replica:0/ep-rank:0",
        },
        "expert_placement_snapshot": {
            "layer:0/expert:0": {
                "layer_id": 0,
                "expert_id": 0,
                "rank_id": "replica:0/ep-rank:0",
                "node_id": "phase5-audit-node",
                "gpu_id": "0",
            },
        },
    }


def probe_spotserve_moe_runtime() -> Dict[str, Any]:
    module, error = _safe_import("vllm.spotserve_moe")
    if module is None:
        return {
            "spotserve_moe_importable": False,
            "error": error,
            "apply_callable": False,
            "verify_callable": False,
            "apply_result": {},
            "verify_result": {},
        }

    apply_hook = getattr(module, "apply_expert_placement_plan", None)
    verify_hook = getattr(module, "verify_expert_placement_plan", None)
    plan = _sample_expert_placement_plan()
    apply_result: Any = {}
    verify_result: Any = {}
    apply_error = ""
    verify_error = ""
    if callable(apply_hook):
        try:
            apply_result = apply_hook(plan, worker_rank=0)
        except Exception as exc:
            apply_error = repr(exc)
    if callable(verify_hook):
        try:
            verify_result = verify_hook(plan, worker_rank=0)
        except Exception as exc:
            verify_error = repr(exc)

    return {
        "spotserve_moe_importable": True,
        "module_path": str(getattr(module, "__file__", "") or ""),
        "apply_callable": callable(apply_hook),
        "verify_callable": callable(verify_hook),
        "apply_error": apply_error,
        "verify_error": verify_error,
        "apply_result": _jsonable(apply_result),
        "verify_result": _jsonable(verify_result),
    }


def _truthy_result(result: Any, *keys: str) -> bool:
    if not isinstance(result, Mapping):
        return False
    return any(bool(result.get(key)) for key in keys)


def classify_audit_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    source_checks = report.get("source_checks")
    if not isinstance(source_checks, Mapping):
        source_checks = {}
    runtime_probe = report.get("runtime_hook_probe")
    if not isinstance(runtime_probe, Mapping):
        runtime_probe = {}

    apply_result = runtime_probe.get("apply_result", {})
    verify_result = runtime_probe.get("verify_result", {})
    apply_success = _truthy_result(
        apply_result,
        "applied",
        "success",
        "ok",
        "expert_placement_plan_applied",
    )
    verify_success = _truthy_result(
        verify_result,
        "verified",
        "success",
        "ok",
        "expert_placement_plan_verified",
    )
    physical_flag = False
    if isinstance(apply_result, Mapping):
        physical_flag = physical_flag or bool(
            apply_result.get("physical_weight_migration")
        )
    if isinstance(verify_result, Mapping):
        physical_flag = physical_flag or bool(
            verify_result.get("physical_weight_migration")
        )

    source_observe_only = bool(
        source_checks.get("observe_only_markers_present")
    )
    runtime_observe_only = False
    if isinstance(apply_result, Mapping):
        runtime_observe_only = runtime_observe_only or (
            apply_result.get("hook_kind") == "spotserve_observation_only"
            or apply_result.get("reason")
            == "physical_expert_placement_migration_not_supported"
        )
    if isinstance(verify_result, Mapping):
        runtime_observe_only = runtime_observe_only or (
            verify_result.get("hook_kind") == "spotserve_observation_only"
            or verify_result.get("reason")
            == "physical_expert_placement_verification_not_supported"
        )

    physical_supported = bool(
        apply_success and verify_success and physical_flag
    )
    boundary_present = bool(
        source_checks.get("apply_verify_boundary_present")
        or (
            runtime_probe.get("apply_callable")
            and runtime_probe.get("verify_callable")
        )
    )
    if physical_supported:
        classification = "physical_expert_migration_supported"
        recommended_execution_model = "live_expert_weight_migration"
        blocking_gaps: list[str] = []
    elif source_observe_only or runtime_observe_only:
        classification = "observe_only_expert_placement_contract"
        recommended_execution_model = "expert_aware_actor_recreate"
        blocking_gaps = [
            "apply_expert_placement_plan returns applied=false",
            "verify_expert_placement_plan returns verified=false",
            "runtime does not update live EP rank mapping or expert weights",
        ]
    elif boundary_present:
        classification = "runtime_boundary_present_but_not_verified"
        recommended_execution_model = "expert_aware_actor_recreate"
        blocking_gaps = [
            "runtime apply/verify hooks are present but do not prove physical "
            "weight movement",
        ]
    else:
        classification = "runtime_boundary_unavailable"
        recommended_execution_model = "actor_recreate_only"
        blocking_gaps = [
            "patched vLLM apply/verify expert placement hooks are unavailable",
        ]

    return {
        "classification": classification,
        "can_claim_physical_expert_migration": physical_supported,
        "recommended_execution_model": recommended_execution_model,
        "blocking_gaps": blocking_gaps,
    }


def audit_vllm_ep_runtime(
    source_root: Optional[str | Path] = None,
    probe_runtime: bool = True,
) -> Dict[str, Any]:
    vllm_module, vllm_error = _safe_import("vllm")
    imported_root = (
        _source_root_from_imported_vllm(vllm_module)
        if vllm_module is not None
        else None
    )
    package_root = _normalize_vllm_package_root(source_root) or imported_root
    source_checks = audit_source_tree(package_root)
    runtime_probe = (
        probe_spotserve_moe_runtime()
        if probe_runtime
        else {
            "spotserve_moe_importable": False,
            "skipped": True,
        }
    )
    report: Dict[str, Any] = {
        "vllm": {
            "importable": vllm_module is not None,
            "version": str(getattr(vllm_module, "__version__", "") or ""),
            "module_path": str(getattr(vllm_module, "__file__", "") or ""),
            "import_error": vllm_error,
        },
        "source_checks": source_checks,
        "runtime_hook_probe": runtime_probe,
    }
    report["phase5_gate"] = classify_audit_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit patched vLLM MoE EP runtime migration capability."
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Optional vLLM package/source root. Accepts either .../vllm or a "
            "repo root containing vllm/."
        ),
    )
    parser.add_argument(
        "--no-runtime-probe",
        action="store_true",
        help="Only inspect source files; do not import vllm.spotserve_moe.",
    )
    args = parser.parse_args()

    report = audit_vllm_ep_runtime(
        source_root=args.source_root,
        probe_runtime=not args.no_runtime_probe,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
