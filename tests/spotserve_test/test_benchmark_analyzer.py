import importlib.util
from pathlib import Path


def load_analyzer():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "analyze_spotserve_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "spotserve_benchmark_analyzer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_summary_exposes_state_restore_evidence():
    analyzer = load_analyzer()

    summary = analyzer.summarize_router_metrics(
        [
            {
                "type": "request",
                "request_id": "req-1",
                "state_restore_attempts": 1,
                "state_restore_successes": 1,
                "state_restore_fallback": False,
                "state_restored_tokens": 16,
                "supports_state_restore": True,
                "state_kind": "vllm_kv_snapshot",
                "state_restore_reason": "nixl_kv_attach_completed",
                "state_restored_blocks": 6,
                "state_restore_staged": True,
            }
        ]
    )

    assert summary["state_restore_attempts_total"] == 1
    assert summary["state_restore_successes_total"] == 1
    assert summary["state_restore_fallback_count"] == 0
    assert summary["state_restored_tokens_total"] == 16
    assert summary["state_restored_blocks_total"] == 6
    assert summary["supports_state_restore_requests"] == 1
    assert summary["state_restore_staged_count"] == 1
    assert summary["true_kv_restore_successes_total"] == 1
    assert summary["state_kinds"] == "vllm_kv_snapshot"
    assert summary["state_restore_reasons"] == "nixl_kv_attach_completed"


def test_raw_response_summary_exposes_kv_restore_evidence():
    analyzer = load_analyzer()

    summary = analyzer.summarize_response_kv_restore(
        [
            {
                "response": {
                    "_spotserve_kv_restore": {
                        "restored": True,
                        "restored_blocks": 6,
                        "cached_tokens": 82,
                        "reason": "nixl_kv_attach_completed",
                    }
                }
            }
        ]
    )

    assert summary["response_kv_restore_events"] == 1
    assert summary["response_kv_restore_successes"] == 1
    assert summary["response_kv_restore_restored_blocks"] == 6
    assert summary["response_kv_restore_cached_tokens"] == 82
    assert summary["response_kv_restore_reasons"] == "nixl_kv_attach_completed"


def test_true_kv_summary_uses_response_evidence_for_legacy_router_rows():
    analyzer = load_analyzer()

    summary = analyzer.summarize_true_kv_restore_evidence(
        request_rows=[
            {
                "request_id": "req-1",
                "response": {
                    "_spotserve_kv_restore": {
                        "restored": True,
                        "restored_blocks": 6,
                        "cached_tokens": 82,
                        "reason": "nixl_kv_attach_completed",
                    }
                },
            }
        ],
        router_request_rows=[
            {
                "type": "request",
                "request_id": "req-1",
                "state_restore_attempts": 1,
                "state_restore_successes": 1,
                "state_restored_tokens": 16,
                "state_restored_blocks": 0,
            }
        ],
    )

    assert summary["true_kv_restore_successes_total"] == 1
    assert summary["true_kv_restored_blocks_total"] == 6
    assert summary["true_kv_restore_evidence_sources"] == "raw_response"


def test_replanning_summary_exposes_workload_cost_metrics():
    analyzer = load_analyzer()

    summary = analyzer.summarize_replanning_metrics(
        [
            {
                "type": "reparallelization",
                "action": "reparallelize",
                "selected_total_gpus": 2,
                "parallel_plan": {"target_nodes": ["node-b"]},
                "execution_status": "applied",
                "execution": {"status": "applied"},
                "execution_duration_ms": 2500,
                "workload_cost_model_enabled": True,
                "selected_replan_window_cost_ms": 5200,
                "selected_load_time_estimate_ms": 5000,
                "selected_migration_cost_estimate_ms": 200,
                "selected_queue_penalty_ms": 0,
                "selected_throughput_estimate_req_s": 2,
                "cross_node_target": True,
                "multi_worker_target": False,
                "target_worker_node_count": 1,
                "ready_worker_node_count": 2,
                "synthetic_worker_node_count": 0,
                "runtime_worker_node_count": 2,
                "physical_worker_node_count": 2,
            }
        ]
    )

    assert summary["replanning_events"] == 1
    assert summary["replanning_execution_applied"] == 1
    assert summary["replanning_workload_cost_model_events"] == 1
    assert summary["replanning_avg_execution_duration_ms"] == 2500
    assert summary["replanning_avg_selected_replan_window_cost_ms"] == 5200
    assert summary["replanning_avg_selected_load_time_estimate_ms"] == 5000
    assert summary["replanning_avg_selected_migration_cost_estimate_ms"] == 200
    assert summary["replanning_cross_node_targets"] == 1
    assert summary["replanning_max_ready_worker_node_count"] == 2
    assert summary["replanning_max_runtime_worker_node_count"] == 2
    assert summary["replanning_max_physical_worker_node_count"] == 2
