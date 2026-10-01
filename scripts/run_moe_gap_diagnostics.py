"""CPU-only diagnosis of the recorded Original/MoE experiment gap.

Reads formal artifacts without modifying them. Counterfactual fixtures and
planner profiles are diagnostic evidence, never additional GPU experiment runs.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    estimate_expert_dispatch_cost,
    plan_low_cost_migration,
)
from sllm.spot.reparallelization import ParallelPlan, plan_dynamic_reparallelization


TEST_FILES = [
    "test_context_migration_planner.py",
    "test_context_migration_phase2_ablation.py",
    "test_reparallelization_planner.py",
    "test_reparallelization_phase4_ablation.py",
    "test_reparallelization_executor.py",
    "test_vllm_deployment_adapter.py",
    "test_moe_placement.py",
    "test_stateful_recovery.py",
    "test_vllm_ep_runtime_audit.py",
    "test_moe_gap_diagnostics.py",
]
PLAN_KEYS = (
    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    "replica_count", "enable_expert_parallel", "num_gpus", "target_nodes",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_rows(run: Mapping[str, Any]) -> list[dict]:
    """Normalize run-specific request IDs by recorded prompt token hash."""
    recovery = run["recovery"]
    rows = []
    for request_id, audit in run["prompt_audit"].items():
        profile = recovery.get("moe_route_profiles", {}).get(request_id, {})
        histogram = profile.get("per_request_expert_route_histogram")
        sequence = recovery.get("sequence_checks", {}).get(request_id, {})
        rows.append({
            "prompt_sha256": audit["token_ids_sha256"],
            "prompt_tokens": audit["token_count"],
            "completed_tokens": recovery["source_completed_tokens"][request_id],
            "prefix_tokens": recovery["source_prefix_tokens"][request_id],
            "remaining_tokens": recovery["target_requested_tokens"][request_id],
            "route_histogram_sha256": digest(histogram) if histogram else None,
            "reference_sha256": sequence.get("reference_sha256"),
            "actual_sha256": sequence.get("actual_sha256"),
        })
    return sorted(rows, key=lambda row: row["prompt_sha256"])


def normalized_migration_plans(run: Mapping[str, Any]) -> list[dict]:
    plans = (run.get("migration_planner_decision") or {}).get("plans", [])
    return sorted([
        {
            "prompt_sha256": run["prompt_audit"][plan["request_id"]]["token_ids_sha256"],
            **{key: plan.get(key) for key in (
                "old_node_id", "new_node_id", "new_instance_id",
                "kv_migration_cost", "expert_dispatch_cost", "queue_penalty_cost",
                "reusable_tokens", "reusable_context_blocks", "estimated_cost",
            )},
        }
        for plan in plans
    ], key=lambda row: row["prompt_sha256"])


def physical_plan(run: Mapping[str, Any]) -> dict:
    plan = (run.get("planner_decision") or {}).get("parallel_plan") or {}
    return {key: plan.get(key) for key in PLAN_KEYS}


def artifact_row(path: Path, run: Mapping[str, Any]) -> dict:
    recovery = run["recovery"]
    decision = run.get("planner_decision") or {}
    migration = run.get("migration_planner_decision") or {}
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_digest(path),
        "mode": run["mode"],
        "status": run.get("status"),
        "outcome": run.get("outcome"),
        "requests": request_rows(run),
        "migration_plans": normalized_migration_plans(run),
        "migration_target_count": migration.get("moe_target_placement_available_count"),
        "expert_locality": [p.get("hot_expert_locality_ratio") for p in migration.get("plans", [])],
        "expert_score_available": [p.get("expert_locality_available") for p in migration.get("plans", [])],
        "physical_dispatch_observed": migration.get("moe_physical_dispatch_traffic_available_count"),
        "route_sources": sorted({p.get("source", "unavailable") for p in recovery.get("moe_route_audit", {}).values()}),
        "candidate_count": decision.get("candidate_count"),
        "parallel_plan": physical_plan(run),
        "expert_placement_covered": decision.get("expert_placement_plan_covered_experts"),
        "movement_observed": decision.get("expert_placement_plan_movement_observation_available"),
        "unknown_movement_experts": decision.get("expert_placement_plan_unknown_movement_experts"),
        "physical_weight_migration": decision.get("expert_placement_plan_physical_weight_migration"),
        "workload_cost_model_enabled": (decision.get("workload_cost_model") or {}).get("enabled"),
        "configuration_applied_raw": recovery.get("configuration_applied"),
        "target_preexisting": recovery.get("target_preexisting"),
        "runtime_parallel_configuration_verified": None,
        "actual_transfer_bytes": None,
        "acknowledged_kv_blocks": None,
        "gpu_recomputed_tokens_observed": None,
        "expected_blocks_not_acknowledgements": recovery.get("restored_blocks"),
        "metadata_recompute_counter_not_gpu_observation": recovery.get("recomputed_tokens"),
        "all_sequences_equal_reference": recovery.get("all_sequences_equal_reference"),
        "times_s": {key: recovery.get(key) for key in (
            "recovery_s", "downtime_s", "migration_operation_s", "migration_planner_s",
            "reparallelization_planner_s", "reparallelization_s", "target_startup_s",
        )},
        "elapsed_s": run.get("elapsed_s"),
    }


def summarize_times(runs: list[dict]) -> dict:
    keys = ["elapsed_s", *runs[0]["times_s"]]
    return {
        key: {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
        for key in keys
        if (values := [r["elapsed_s"] if key == "elapsed_s" else r["times_s"][key] for r in runs])
        and all(isinstance(v, (int, float)) for v in values)
    }


def compare_pair(original: Mapping[str, Any], moe: Mapping[str, Any]) -> dict:
    is_migration = original["mode"] == "original_migration"
    return {
        "original": original["path"], "moe": moe["path"],
        "matched_recorded_requests": original["requests"] == moe["requests"],
        "migration_mapping_and_costs_equal": (
            original["migration_plans"] == moe["migration_plans"] if is_migration else None
        ),
        "selected_parallel_shape_and_nodes_equal": (
            original["parallel_plan"] == moe["parallel_plan"] if not is_migration else None
        ),
        "expected_blocks_equal": original["expected_blocks_not_acknowledgements"] == moe["expected_blocks_not_acknowledgements"],
        "metadata_recompute_counter_equal": original["metadata_recompute_counter_not_gpu_observation"] == moe["metadata_recompute_counter_not_gpu_observation"],
        "measured_physical_work_reduction": None,
        "runtime_configuration_difference_verified": None,
    }


def load_functions(source: str, names: set[str], bindings: dict, filename: str) -> dict:
    """Execute only named function definitions, never the module/main/engine."""
    tree = ast.parse(source, filename=filename)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(functions) != len(names) or {n.name for n in functions} != names:
        raise ValueError(f"Missing or ambiguous diagnostic functions in {filename}")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *functions,
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(bindings)
    exec(compile(module, filename, "exec"), namespace)
    return {name: namespace[name] for name in names}


def patch_hook_probe() -> dict:
    patch = REPO_ROOT / "sllm_store/vllm_patch/runtime_moe_metadata.patch"
    first_file = patch.read_text(encoding="utf-8").split("diff --git ")[1]
    source = "\n".join(line[1:] for line in first_file.splitlines() if line.startswith("+") and not line.startswith("+++"))
    functions = load_functions(source, {
        "_placement_plan_payload", "_placement_plan_fingerprint",
        "apply_expert_placement_plan", "verify_expert_placement_plan",
    }, {"Any": Any, "Mapping": Mapping, "_LOCK": threading.Lock(), "_LAST_EXPERT_PLACEMENT_PLAN": {}}, str(patch))
    plan = {"placement_fingerprint": "gap-diagnostic-contract", "expert_to_target_rank": {"layer:0/expert:0": "rank-1"}}
    applied = functions["apply_expert_placement_plan"](expert_placement_plan=plan)
    verified = functions["verify_expert_placement_plan"](expert_placement_plan=plan)
    return {
        "evidence_kind": "CPU execution of function definitions in the repository patch; not a live container probe",
        "patch_sha256": file_digest(patch), "apply_result": applied, "verify_result": verified,
        "passed": applied["applied"] is False and verified["verified"] is False and verified["contract_seen_by_runtime"] is True,
    }


def migration_score_probe(recorded_run: Mapping[str, Any]) -> dict:
    profile = next(iter(recorded_run["recovery"]["moe_route_profiles"].values()))
    histogram = profile["per_request_expert_route_histogram"]
    source = ContextMetadata(request_id="score-probe", instance_id="source", node_id="probe", metadata={
        "per_request_expert_route_histogram": histogram,
        "moe_route_histogram_source": "recorded_histogram_counterfactual_fixture",
        "moe_route_histogram_kind": "offline_diagnostic",
    })
    def target(name: str, rank: str, missing: str | None = None) -> MigrationTarget:
        return MigrationTarget(instance_id=name, node_id="probe", metadata={
            "expert_placement_snapshot": {key: {"rank_id": rank} for key in histogram if key != missing},
        })
    full = target("all-experts", "rank-0")
    swapped = target("all-experts-other-rank", "rank-1")
    missing_key = max(histogram, key=histogram.get)
    partial = target("missing-one-expert", "rank-0", missing_key)
    on = {"enable_moe_expert_locality": True, "expert_dispatch_weight": 10.0, "queue_penalty_weight": 0.0}
    off = {**on, "enable_moe_expert_locality": False, "expert_dispatch_weight": 0.0}
    results = {name: estimate_expert_dispatch_cost(source, instance, config) for name, instance, config in (
        ("full_coverage_on", full, on), ("same_keys_different_rank_on", swapped, on),
        ("missing_one_expert_on", partial, on), ("missing_one_expert_off", partial, off),
    )}
    selections = {
        name: plan_low_cost_migration([source], [partial, full], config).plans[0].new_instance_id
        for name, config in (("off", off), ("on", on))
    }
    return {
        "evidence_kind": "counterfactual metadata fixture using one recorded request histogram; not a deployable target or GPU experiment",
        "histogram_sha256": digest(histogram), "removed_expert": missing_key,
        "covered_expert_keys": len(histogram), "selected_targets": selections,
        "scores": {name: {key: value.get(key) for key in (
            "available", "locality_ratio", "cost", "routed_tokens", "remote_routed_tokens",
            "rank_locality_available", "physical_dispatch_traffic_available",
        )} for name, value in results.items()},
        "passed": results["full_coverage_on"]["cost"] == 0.0
        and results["same_keys_different_rank_on"]["cost"] == 0.0
        and results["missing_one_expert_on"]["cost"] > 0.0
        and results["missing_one_expert_off"]["available"] is False
        and selections["off"] != selections["on"],
    }


def repara_cpu_replay(model: Path, output_dir: Path, repetitions: int) -> dict:
    harness = REPO_ROOT / "tests/spotserve_test/run_tiny_batch_recovery.py"
    plan_for = load_functions(harness.read_text(encoding="utf-8"), {"plan_for"}, {
        "Path": Path, "ParallelPlan": ParallelPlan,
        "plan_dynamic_reparallelization": plan_dynamic_reparallelization,
        "args": SimpleNamespace(model=str(model), gpus=[0, 1, 2, 3]), "tensor_parallel_size": 2,
    }, str(harness))["plan_for"]
    rows = []
    decisions = {}
    # Warm import/cache once, then counterbalance Original/MoE call order.
    for flag in (False, True):
        plan_for({2, 3}, flag)
    for repetition in range(repetitions):
        for flag in ((False, True) if repetition % 2 == 0 else (True, False)):
            started = time.perf_counter()
            decision, plan = plan_for({2, 3}, flag)
            name = "moe" if flag else "original"
            rows.append({"pair": repetition + 1, "mode": name, "seconds": time.perf_counter() - started})
            decisions[name] = {
                "candidate_count": decision["candidate_count"],
                "parallel_plan": {key: plan.to_dict().get(key) for key in PLAN_KEYS},
                "movement_observed": decision.get("expert_placement_plan_movement_observation_available"),
                "unknown_movement_experts": decision.get("expert_placement_plan_unknown_movement_experts"),
            }
    profiles = {}
    for flag in (False, True):
        profiler = cProfile.Profile()
        profiler.runcall(plan_for, {2, 3}, flag)
        name = "moe" if flag else "original"
        profiler.dump_stats(str(output_dir / f"repara-{name}.prof"))
        stats = pstats.Stats(profiler).stats
        profiles[name] = [
            {"file": key[0], "line": key[1], "function": key[2], "primitive_calls": value[0],
             "total_calls": value[1], "total_seconds": value[2], "cumulative_seconds": value[3]}
            for key, value in sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:25]
        ]
    return {
        "evidence_kind": "CPU-only replay of the actual nested harness plan_for function; warm timings are not formal GPU timings",
        "harness_sha256": file_digest(harness), "model_config_sha256": file_digest(model / "config.json"),
        "decisions": decisions, "warm_cpu_timings": rows, "profiles": profiles,
        "same_shape_and_nodes": decisions["original"]["parallel_plan"] == decisions["moe"]["parallel_plan"],
        "passed": all(d["candidate_count"] == 1 for d in decisions.values())
        and decisions["original"]["parallel_plan"] == decisions["moe"]["parallel_plan"],
    }


def run_tests(output_dir: Path) -> dict:
    command = [sys.executable, "-m", "pytest", *[f"tests/spotserve_test/{name}" for name in TEST_FILES],
               "-q", "-p", "no:cacheprovider", f"--junitxml={output_dir / 'pytest.xml'}"]
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    (output_dir / "pytest.log").write_text(result.stdout, encoding="utf-8")
    return {"command": command, "returncode": result.returncode, "passed": result.returncode == 0,
            "log": str(output_dir / "pytest.log"), "junit": str(output_dir / "pytest.xml")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "results/moe_spotserve_qwen_a27b_4k_v2")
    parser.add_argument("--model", type=Path, default=Path("/work/spotserve-models/Qwen1.5-MoE-A2.7B"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory, outside the formal artifact tree")
    parser.add_argument("--cpu-repetitions", type=int, default=6)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == artifacts or artifacts in output_dir.parents:
        parser.error("Diagnostic output must not change the formal artifact tree")
    if args.cpu_repetitions < 1:
        parser.error("--cpu-repetitions must be positive")
    if not (args.model / "config.json").is_file():
        parser.error("A local model config.json is required; no downloading is performed")
    modes = ("original_migration", "moe_migration", "original_reparallelization", "moe_reparallelization")
    sources = {str(path): file_digest(path) for path in artifacts.rglob("*.json")}
    raw = {mode: [read_json(artifacts / mode / f"run-{n}.json") for n in range(1, 4)] for mode in modes}
    rows = {mode: [artifact_row(artifacts / mode / f"run-{n}.json", run) for n, run in enumerate(runs, 1)] for mode, runs in raw.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    from scripts.run_context_migration_phase2_ablation import run_ablation as migration_ablation
    from scripts.run_reparallelization_phase4_movement_ablation import run_ablation as repara_ablation
    migration = migration_ablation(REPO_ROOT / "benchmarks/spotserve/context_migration_phase2_ablation.json", output_dir / "migration-fixture")
    repara = repara_ablation(REPO_ROOT / "benchmarks/spotserve/reparallelization_phase4_movement_ablation.json", output_dir / "repara-fixture")
    report = {
        "scope": "offline artifact audit + CPU diagnostics only; no GPU engines, deployment changes, source/target changes, or formal re-runs",
        "artifact_runs": rows,
        "summaries": {mode: summarize_times(runs) for mode, runs in rows.items()},
        "paired_artifact_comparisons": {
            "migration": [compare_pair(a, b) for a, b in zip(rows["original_migration"], rows["moe_migration"])],
            "reparallelization": [compare_pair(a, b) for a, b in zip(rows["original_reparallelization"], rows["moe_reparallelization"])],
        },
        "migration_score_probe": migration_score_probe(raw["moe_migration"][0]),
        "patch_hook_probe": patch_hook_probe(),
        "repara_cpu_replay": repara_cpu_replay(args.model, output_dir, args.cpu_repetitions),
        "synthetic_fixture_checks": {
            "evidence_kind": "existing synthetic planner ablations, not GPU performance evidence",
            "migration_passed": migration["passed"], "migration_comparisons": migration["comparisons"],
            "reparallelization_passed": repara["passed"], "reparallelization_comparisons": repara["comparisons"],
        },
        "source_sha256": {str(path.relative_to(REPO_ROOT)): file_digest(path) for path in (
            REPO_ROOT / "sllm/spot/context_migration.py", REPO_ROOT / "sllm/spot/reparallelization.py",
            REPO_ROOT / "sllm/spot/moe_placement.py", REPO_ROOT / "tests/spotserve_test/run_tiny_batch_recovery.py",
            Path(__file__),
        )},
        "limits": [
            "Formal artifacts do not contain actual transferred bytes, block acknowledgements, GPU recompute counters, or verified runtime parallel shape.",
            "Formal runtimes and the current diagnostic source have no recorded source-version equivalence proof; current source hashes are preserved here.",
            "configuration_applied/placement_changed/restore_success are harness labels, not independent execution observations.",
            "Recorded routing calibration is not same-run NIXL routing capture.",
            "No correction or estimate of old lifecycle-confounded timings is attempted.",
        ],
    }
    if args.run_tests:
        report["tests"] = run_tests(output_dir)
    report["formal_json_artifacts_unchanged"] = all(Path(path).is_file() and file_digest(Path(path)) == value for path, value in sources.items())
    passed = all((report["migration_score_probe"]["passed"], report["patch_hook_probe"]["passed"],
                  report["repara_cpu_replay"]["passed"], migration["passed"], repara["passed"], report["formal_json_artifacts_unchanged"]))
    report["diagnostic_probes_passed"] = passed
    (output_dir / "diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "diagnosis.json"), "diagnostic_probes_passed": passed,
                      "tests": report.get("tests"), "formal_json_artifacts_unchanged": report["formal_json_artifacts_unchanged"]}, indent=2))
    return 0 if passed and report.get("tests", {"passed": True})["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
