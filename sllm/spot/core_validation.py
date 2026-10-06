"""Evidence gates for a live-KV/dynamic-target probe, not full paper parity."""


def audit_live_kv_core(baseline_rows, recovery_rows, metrics, initial, final,
                       preempted_worker, output_tokens):
    def requests(rows):
        return {row["request_id"]: row for row in rows
                if row.get("benchmark_phase") == "measured"}

    baseline, recovery = requests(baseline_rows), requests(recovery_rows)
    same_inputs = (
        bool(baseline) and set(baseline) == set(recovery)
        and len(baseline) == sum(row.get("benchmark_phase") == "measured"
                                 for row in baseline_rows)
        and len(recovery) == sum(row.get("benchmark_phase") == "measured"
                                 for row in recovery_rows)
    )
    greedy_equal = same_inputs
    for key, row in recovery.items():
        left = baseline.get(key, {})
        expected, actual = left.get("response", {}), row.get("response", {})
        greedy_equal &= bool(
            left.get("success") and row.get("success")
            and isinstance(expected.get("_spotserve_prompt_token_ids"), list)
            and bool(expected.get("_spotserve_prompt_token_ids"))
            and len(expected.get("_spotserve_token_ids", [])) == output_tokens
            and actual.get("_spotserve_token_ids") == expected.get("_spotserve_token_ids")
            and actual.get("_spotserve_prompt_token_ids") == expected.get("_spotserve_prompt_token_ids")
            and actual.get("usage", {}).get("completion_tokens") == output_tokens
        )

    replans = [row for row in metrics if row.get("type") == "reparallelization"
               and row.get("event") == "preempt"]
    plan = replans[0] if len(replans) == 1 else {}
    targets = set(str(value) for value in plan.get("target_nodes", []))
    old_ids = set(initial)
    new_ready = {key: row for key, row in final.items() if key not in old_ids
                 and row.get("pool") == "ready" and row.get("state") == "ready"}
    actual_workers = {str(value) for row in new_ready.values()
                      for value in row.get("member_node_ids", [])}
    start, notice = plan.get("planner_started_at_s"), plan.get("event_state_marked_at_s")
    dynamic = bool(
        plan.get("execution_status") == "applied" and targets
        and str(preempted_worker) not in targets and targets == actual_workers
        and isinstance(start, (int, float)) and isinstance(notice, (int, float))
        and 0 < notice <= start
    )
    receipts = [row for row in metrics if row.get("type") == "native_kv_receipt"
                and row.get("request_id") in recovery]
    native = bool(receipts) and all(
        int(row.get("cached_tokens", 0)) > 0 and int(row.get("restored_blocks", 0)) > 0
        and row.get("source_instance_id") in old_ids
        and row.get("target_instance_id") in new_ready
        and recovery[row["request_id"]].get("response", {}).get("_spotserve_kv_restore", {}).get("restored")
        and int(recovery[row["request_id"]]["response"]["_spotserve_kv_restore"].get("cached_tokens", 0)) > 0
        for row in receipts
    )
    request_metrics = [row for row in metrics if row.get("type") == "request"
                       and row.get("request_id") in recovery]
    no_replay = (
        {row.get("request_id") for row in request_metrics} == set(recovery)
        and len(request_metrics) == len(recovery) and all(
            row.get("success") and row.get("recovery_fallback") is False
            and row.get("state_restore_fallback") is False for row in request_metrics
        )
    )
    releases = [row for row in metrics
                if row.get("type") == "native_kv_source_release_authorized"]
    release_order = bool(receipts) and all(any(
        receipt.get("request_id") in row.get("request_ids", [])
        and receipt.get("source_instance_id") in row.get("source_instance_ids", [])
        and isinstance(row.get("deadline_time_s"), (int, float))
        and 0 < receipt.get("timestamp", 0) <= row.get("timestamp", 0) <= row["deadline_time_s"]
        for row in releases
    ) for receipt in receipts)
    checks = {"paired_requests_and_full_greedy_output_equal": bool(greedy_equal),
              "dynamic_new_target_members_match_plan": dynamic,
              "native_cached_prefix_observed": native,
              "no_token_replay_fallback": no_replay,
              "source_release_after_native_receipt_before_deadline": release_order}
    return {
        "status": "live_kv_core_verified" if all(checks.values()) else "blocked",
        "checks": checks, "target_workers": sorted(targets),
        "actual_new_target_workers": sorted(actual_workers),
        "full_spotserve_verified": False,
        "not_verified": ["GPU_rank_to_pod_identity", "persistent_context_owner",
                         "KV_aware_KM_mapping", "progressive_weight_migration",
                         "JIT_interruption", "early_connector_transfer_ACK"],
        "receipt_kind": "completed_native_request_not_early_connector_ACK",
    }
