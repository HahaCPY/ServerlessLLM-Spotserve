# SpotServe Performance Summary

Last updated: 2026-09-16

This page is intentionally short. It keeps only the commands needed to rerun
each version's benchmark path and a compact performance summary. V6, V7, V8,
and the combined V7-V9 core run were refreshed after the latest backend switch.

## Performance Commands

All container-side runs assume the compose stack is already using the intended
`MODEL_FOLDER`. Use `--skip-build` for benchmark reruns unless backend code,
dependencies, or the image itself changed.


V1 benchmark harness smoke:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set standard

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V2 long-running availability benchmark:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set standard

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_long.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V3 recovery correctness:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set correctness

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
'
```

V4 dense vLLM black-box benchmark:

```bash
MODEL_FOLDER="$PWD/model" scripts/prepare_spotserve.sh --skip-build --deploy-set vllm-dense

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 120 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V5 MoE vLLM black-box benchmark:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set vllm-moe

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V5 dense-vs-MoE comparison, requires `/models` to contain both dense and MoE
snapshots:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set vllm-blackbox

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V6 dynamic reparallelization smoke:

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
scripts/prepare_spotserve.sh --skip-build --deploy-set reparallelization

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 120 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V6 dynamic reparallelization performance:

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS=1 \
scripts/prepare_spotserve.sh --skip-build --deploy-set reparallelization-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

For V6, `--request-timeout` controls normal chat completion requests, while
`--trace-event-timeout` controls each `/spot/event` replay request. Keep the
trace timeout longer than the request timeout because the preemption event may
wait for re-planning, actor recreation, model loading, and metric emission.
Do not claim a valid V6 re-plan result unless `trace_replay_success=1` and
`replanning_events > 0`.

The single-worker V6 performance config uses explicit same-node recreate mode:
`allow_preempting_target_recreate=true` plus
`allow_stop_before_recreate=true`. It validates the controller/apply/logical
placement metric path on one real worker node, but it is not a cross-failure
domain relocation experiment. Use the multi-worker performance matrix for
multi-worker relocation validation.

After changing `runtime_moe_metadata.patch`, rebuild the image once by omitting
`--skip-build`. With the observe-only vLLM placement hook patch present, the
applied V6 summary should show runtime hook availability/attempts but still no
physical apply/verify success:

```text
runtime_apply_hooks=1
runtime_apply_success=0
runtime_verify_hooks=1
runtime_verify_success=0
```

V7 context migration performance:

The MoE-side run uses the 1.5s `busy` preemption trace so Qwen2-MoE is still
serving the warm-prefix requests when preemption is injected.
If backend/router code changed, omit `--skip-build` for the first run so the
container image includes the latest context migration observability hooks. For
plain benchmark reruns, keep `--skip-build`.

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
scripts/prepare_spotserve.sh --skip-build --deploy-set context-migration-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

After the run, the summary should print the selected target and selected-plan
cost breakdown:

```text
selected=...
kv_cost=...
expert_cost=...
queue_cost=...
route_source=...
route_kind=...
```

For the MoE-aware claim, `route_source=vllm_runtime_topk` and
`route_kind=runtime_observed_topk` mean the planner consumed routing data from
the patched vLLM fused MoE top-k path. If those fields show
`request_instrumentation/request_fixture`, the run is still useful as a
deterministic planner check, but it is not runtime routing instrumentation.

The applied router metrics should also include candidate-level cost breakdown:

```bash
grep -R '"candidate_component_costs"' -n \
  results/spotserve_context_migration_performance
```

V8 stateful recovery correctness:

```bash
scripts/prepare_spotserve.sh --skip-build --deploy-set correctness

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
'
```

V8 stateful recovery performance:

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
scripts/prepare_spotserve.sh --skip-build --deploy-set stateful-recovery-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

After the Phase 3 update, the applied V8 summary should also print recovery
compatibility/locality fields:

```text
recovery_kv_compatible=...
recovery_ep_required=...
recovery_ep_mismatch=...
recovery_locality=...
recovery_remote_tokens=...
recovery_expert_cost=...
```

V9 risk-aware scheduling synthetic benchmark:

```bash
python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_test
```

V7-V9 SpotServe core live combined benchmark:

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CORE_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CORE_LOAD_FORMAT=auto \
scripts/prepare_spotserve.sh --skip-build --deploy-set spotserve-core-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

V7-V9 SpotServe core multi-trace sweep:

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CORE_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CORE_LOAD_FORMAT=auto \
scripts/prepare_spotserve.sh --skip-build --deploy-set spotserve-core-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_trace_sweep.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

`spotserve-core-performance` now starts both `sllm_worker_0` and
`sllm_worker_1` by default because the core matrices deploy two ready vLLM
instances (`min_instances=2`, `min_ready_instances=2`). If only
`sllm_worker_0` is running, the second model instance can remain in `starting`
and the sweep will fail during setup with a Ray scheduling warning for
`worker_id_0`.

## Performance Summary

| Version | Benchmark | Current claim | Refresh status |
|---|---|---|---|
| V1 | `benchmark_matrix.yaml` | Harness/reporting validation only; no latency speedup claim. | Stable unless harness changes. |
| V2 | `benchmark_matrix_long.yaml` | Availability/control-plane validation; no direct latency speedup claim. | Stable unless scheduler/control-plane changes. |
| V3 | `benchmark_matrix_recovery_correctness.yaml` | Recovery policies turn forced failures into successes; latency is not the main metric. | Stable correctness check. |
| V4 | `benchmark_matrix_vllm_dense.yaml` | Dense vLLM compatibility milestone; black-box recovery smoke only. | Rerun if backend image/model changed. |
| V5 | `benchmark_matrix_vllm_moe.yaml`, `benchmark_matrix_vllm_dense_vs_moe.yaml` | MoE compatibility milestone; no MoE-specific speedup claim. | Rerun after backend/model changes. |
| V6 | `benchmark_matrix_reparallelization_performance.yaml` | Shared Qwen2 MoE single-worker same-node recreate run improves success from 37.50% to 100.00%; trace replay succeeds, one replan is applied, the logical expert placement plan is emitted with 8 shards/full coverage, movement diff/runtime contract observability is reported, and the execution model is expert-aware actor recreate. This validates the control-plane/logical placement/runtime-hook plumbing path, not physical expert weight movement. | Phase 5B actor-recreate run refreshed 2026-09-15. |
| V7 | `benchmark_matrix_context_migration_performance.yaml` | Shared Qwen2 MoE run keeps both versions at 100% success; preemption is injected, one context-migration plan executes, selected-plan KV/expert/queue costs are reported, and route metadata is consumed from patched vLLM runtime top-k instrumentation. V7 verifies low-cost target selection and prefix warmup/context planning, not physical expert migration or true remote expert-dispatch traffic. | Runtime top-k MoE run refreshed 2026-08-30. |
| V8 | `benchmark_matrix_stateful_recovery_performance.yaml` | Shared Qwen2 MoE run keeps both versions at 100% success; stateful recovery restores once, restores 16 tokens and 6 KV blocks, has no fallback, and reports separated KV compatibility vs EP/locality signals. Latest run verifies the patched vLLM true KV restore path with `true_kv_rate=100.00%`. | Phase 3 MoE-aware recovery run refreshed 2026-09-16. |
| V9 | `risk_aware_scheduling_synthetic.json` | Placement-quality improvement: lower-risk / longer-lived node selection. | Synthetic result; live latency/SLO impact still requires a real workload. |
| V7-V9 core | `benchmark_matrix_spotserve_core_performance.yaml` | One live matrix deploys baseline and applied variants of the same model. Applied enables context migration, stateful recovery policy, and risk-aware scheduling together, with runtime MoE top-k route metadata and Phase 3 recovery compatibility/locality signals. The core run validates integrated control flow; use the standalone V8 matrix for the cleanest true-KV-restore evidence. | Refreshed 2026-09-16. |
| V7-V9 trace sweep | `benchmark_matrix_spotserve_core_trace_sweep.yaml` | Runs the same baseline/applied core pair across busy-recover, fast-recover, slow-recover, no-recover, dead-after-preempt, and double-preempt traces. Use it as a robustness sweep after the single core matrix succeeds; keep scenario-level claims tied to `latest_comparisons.json` or the pasted benchmark summary. | Validation path available; primary numbers are not duplicated here. |

| Metric | Baseline | Applied | Result | Status |
|---|---|---|---|---|
| V6 success rate | Reparallelization disabled | Reparallelization applied | `37.50% -> 100.00%` | `3/8 -> 8/8`; clean success also improves from `37.50%` to `100.00%`, with `0` fallbacks. |
| V6 trace replay | Reparallelization disabled | Reparallelization applied | `1 success, 0 failed -> 1 success, 0 failed` | The preemption event replay completed on both runs, so the replan metrics are valid. |
| V6 overall p95 | Reparallelization disabled | Reparallelization applied | `180102.20ms -> 14069.87ms` | Latest applied run reduces overall p95 by `166032.33ms`. |
| V6 replan-window p95 | Reparallelization disabled | Reparallelization applied | `180102.20ms -> 14069.87ms` | Replan-window success improves from `0.00%` to `100.00%`. |
| V6 post-replan p95 | Reparallelization disabled | Reparallelization applied | `180045.50ms -> 1036.97ms` | Post-replan success improves from `0.00%` to `100.00%`. |
| V6 replan execution | Reparallelization disabled | Reparallelization applied | `0 replans -> 1 replan, 1 applied, 0 failed` | Lifecycle verified; latest average execution duration was `16340.36ms`. |
| V6 expert execution model | Reparallelization disabled | Reparallelization applied | `0 -> actor_recreate=1, live_migration=0, runtime_workers=1` | Phase 5B confirms expert-aware actor recreate with observe-only placement/runtime contract. |
| V6 workload-aware cost model | Reparallelization disabled | Reparallelization applied | `0 -> 1 cost-model event` | Selected estimates: replan window `5800.00ms`, model load `5500.00ms`, migration `300.00ms`. |
| V6 logical expert placement | Reparallelization disabled | Reparallelization applied | `0 -> 1 plan, 8 shards, coverage=1.00` | Phase 4 logical placement was emitted; physical expert migration remains `0`. |
| V6 expert movement diff | Reparallelization disabled | Reparallelization applied | `0 -> 1 movement observation, moved=0, moved_bytes=0, move_cost=0.00ms` | Phase 4E compared selected placement against the current runtime snapshot. `moved=0` is expected for this single-worker same-node recreate run. |
| V6 runtime placement hooks | Reparallelization disabled | Reparallelization applied | `0 -> apply_hooks=1, apply_attempted=1, verify_hooks=1, verify_attempted=1` | Observe-only hook plumbing verified; apply/verify success and plan applied/verified remain `0`. |
| V7 MoE success rate | Context migration disabled | Context migration applied | `100.00% -> 100.00%` | New backend, `8/8 -> 8/8`. |
| V7 MoE overall p95 | Context migration disabled | Context migration applied | `48238.92ms -> 4010.09ms` | Applied reduces overall p95 by `44228.82ms`. |
| V7 MoE migration-window p95 | Context migration disabled | Context migration applied | `37079.91ms -> 1030.30ms` | Migration window improved by `36049.61ms`. |
| V7 MoE post-migration p95 | Context migration disabled | Context migration applied | `1023.59ms -> 1033.12ms` | Near parity after migration. |
| V7 MoE migration signals | Context migration disabled | Context migration applied | `0 -> 1 context migration, 1 plan, 1 prefix-warmup/KV success, 228 migrated/warmed tokens` | Selected-plan costs were `kv=293.00`, `expert=0.00`, `queue=0.00`; route metadata was `vllm_runtime_topk/runtime_observed_topk`. Reusable blocks stayed `0`, so do not claim true KV block transfer from this V7 row. |
| V8 MoE success rate | Token replay | Stateful recovery | `100.00% -> 100.00%` | New backend, `3/3 -> 3/3`. |
| V8 MoE overall p95 | Token replay | Stateful recovery | `48739.13ms -> 5157.64ms` | Latest stateful recovery run reduces overall p95 by `43581.49ms`. |
| V8 MoE failure-window p95 | Token replay | Stateful recovery | `48739.13ms -> 5157.64ms` | Failure-window p95 improves by `89.42%`. |
| V8 MoE post-recovery p95 | Token replay | Stateful recovery | Near steady-state parity | The main win is avoiding token replay during the failure window. |
| V8 MoE restore signals | Token replay | Stateful recovery | `0 -> 1 restore, 16 restored tokens, 6 restored blocks, 0 fallback, true_kv_rate=100.00%` | Latest run reported `state_restores=1/1`, `state_blocks=6`, `response_blocks=6`, `true_kv_restores=1`, and `supports_state_restore=1`. |
| V8 MoE recovery compatibility | Token replay | Stateful recovery | `0 -> 1 KV-compatible, 0 EP-required, 1 EP mismatch, locality=1.00, remote_tokens=0, expert_cost=0.00` | Confirms EP mismatch was reported as topology/locality information, not used as a hard KV restore rejection. |
| V7-V9 live combined overall p95 | Token replay + no context migration + health-only scheduling | Stateful recovery + context migration + risk-aware scheduling | Latest core run completed `8/8 -> 8/8` | Integrated control flow succeeded; standalone V7/V8 matrices remain the cleaner source for individual speedup claims. |
| V7-V9 live migration-window p95 | Context migration disabled | Context/KV migration applied | `37195.15ms -> 1026.63ms` | Migration window improves by `36168.52ms`. |
| V7-V9 live failure-window p95 | Token replay | Stateful recovery | `2226.07ms -> 2429.70ms` | In this combined timing, stateful failure window is slower by `203.62ms`; standalone V8 remains the cleaner stateful-recovery speedup result. |
| V7-V9 live post-recovery p95 | Token replay | Stateful recovery | `1042.62ms -> 1041.89ms` | Post-recovery steady-state latency is near parity. |
| V7-V9 live core signals | Baseline policies | Applied policies | `0 -> 1 context migration, route=vllm_runtime_topk/runtime_observed_topk, 1 state event, 1 state fallback, 3 risk scheduling events` | Latest core run proves the integrated V7/V8/V9 control paths execute together, but this matrix fell back for stateful restore; use standalone V8 for true KV restore evidence. |
| V7-V9 live recovery compatibility | Token replay | Stateful recovery | `0 -> 1 KV-compatible, 0 EP-required, 1 EP mismatch, locality=1.00, remote_tokens=0, expert_cost=0.00` | Combined run also preserves the Phase 3 separation between KV restore correctness and EP/locality signals. |
| V7-V9 live selected spot risk | Health-only scheduling | Risk-aware scheduling | `0.9000 -> 0.9000` | Single worker node, so V9 decision path ran but had no alternate lower-risk placement. |
| V9 avg selected spot risk | Health-only scheduling | Risk-aware scheduling | `0.6333 -> 0.1500` | `76.32%` lower synthetic placement risk. |

Notes:

- Keep latency claims tied to the matrix that produced them.
- For V7, require `context_migration_plan_count > 0`,
  `context_migration_selected_target_ids`, and selected-plan cost fields before
  claiming the target-selection path worked. `kv_cache_migration_successes > 0`
  means the prefix warmup/token replay path ran; it is not true vLLM KV block
  serialization. Only claim prefix/block reuse when
  `context_migration_reusable_context_blocks > 0`.
- For V8, require `state_restore_successes_total > 0`,
  `state_restored_blocks_total > 0`, `true_kv_restore_rate > 0`, and
  `state_restore_fallback_count = 0` before claiming true KV stateful restore.
  For Phase 3 MoE-aware recovery, also check
  `state_recovery_kv_restore_compatible_count`,
  `state_recovery_ep_layout_required_count`,
  `state_recovery_expert_placement_mismatch_count`, and recovery locality
  fields.
- The 2026-09-15 V6 table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto`. The applied run reported
  `trace_success=1`, `replans=1`, `applied=1`, `cost_model=1`,
  `expert_plan=1`, `expert_plan_shards=8`, and
  `replanning_avg_expert_placement_plan_coverage_ratio=1.0`. After Phase 4E,
  it also reported
  `replanning_expert_placement_plan_movement_observation_events=1`,
  `replanning_max_expert_placement_plan_moved_experts=0`,
  `replanning_total_expert_placement_plan_moved_weight_bytes=0`, and
  `replanning_avg_expert_placement_plan_weight_movement_cost_ms=0.0`.
  After Phase 4D observe-only hook plumbing, it reported
  `runtime_apply_hooks=1`, `runtime_apply_success=0`,
  `runtime_verify_hooks=1`, and `runtime_verify_success=0`.
  After the runtime capability split, a correct observe-only run should report
  either `runtime_level=contract_seen_only` when only the contract was seen, or
  `runtime_level=runtime_placement_verified` when patched vLLM can inspect the
  loaded model and match resident experts against the plan. In both cases,
  `runtime_physical_migration=0`, `runtime_remap_ep=0`, and
  `runtime_a2a_counters=0` must remain true until live migration/all-to-all
  instrumentation exists.
  After Phase 5B, the single-worker same-node recreate capacity entry is marked
  `_spotserve_counts_as_runtime_worker=true`, so the expected summary is
  `runtime_workers=1`, `exec_model=expert_aware_actor_recreate`,
  `actor_recreate=1`, and `live_migration=0`. The refreshed run reported those
  values, plus `expert_plan=1` and `expert_plan_shards=8`. This still
  represents same-node actor recreate, not multi-worker runtime relocation.
  `replanning_expert_placement_plan_physical_migration_events=0`, so this row
  must not be used as evidence of physical expert weight movement.
- A 2026-07-28 V7 dense-side run used `/models/vllm/vllm-dense-baseline`;
  do not use it as the V7 MoE result.
- The 2026-08-30 V7 MoE table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto`. The applied run reported
  `route_source=vllm_runtime_topk` and
  `route_kind=runtime_observed_topk`, so runtime MoE routing instrumentation
  reached the V7 planner/summary. It still warmed/replayed prefix tokens and
  did not report reusable context blocks.
- The 2026-09-16 V8 MoE table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT=auto`. The applied run reported
  `state_restores=1/1`, `state_tokens=16`, `state_blocks=6`,
  `response_blocks=6`, `true_kv_restores=1`, `true_kv_rate=100.00%`,
  `true_kv_blocks=6`, `supports_state_restore=1`,
  `recovery_kv_compatible=1`, `recovery_ep_required=0`,
  `recovery_ep_mismatch=1`, and `recovery_locality=1.00`.
- For the live V7-V9 core benchmark, require the applied summary to show
  `context_migration_events > 0`, stateful recovery policy evidence
  (`state_recovery_events > 0` or `state_events > 0`), and
  `risk_scheduling_events > 0` before claiming that all three control paths ran
  together. Require `state_restore_successes_total > 0`,
  `true_kv_restore_successes_total > 0`, and `state_restore_fallback_count = 0`
  only when claiming true KV restore inside the core matrix. The standalone V8
  matrix is the primary true-KV-restore evidence.
- For Phase 4 pre-work, use `context_migration_moe_routed_tokens`,
  `context_migration_moe_local_routed_tokens`,
  `context_migration_moe_remote_routed_tokens`,
  `context_migration_moe_avg_remote_routing_ratio`,
  `state_recovery_moe_routed_tokens`,
  `state_recovery_moe_local_routed_tokens`,
  `state_recovery_moe_remote_routed_tokens`, and
  `state_recovery_moe_avg_remote_routing_ratio` as routing + placement-derived
  expert dispatch observability. These fields are not physical network traffic
  counters. Require `context_migration_moe_locality_definitions` and
  `state_recovery_moe_locality_definitions` to show
  `target_placement_coverage`; require the corresponding
  `*_moe_rank_locality_available_count` and
  `*_moe_physical_dispatch_traffic_available_count` to remain `0` until
  per-rank locality or real dispatch traffic instrumentation exists.
- For the Phase 4 logical re-parallelization planner, require
  `replanning_expert_placement_plan_available_events > 0`,
  `replanning_max_expert_placement_plan_shards > 0`, and
  `replanning_avg_expert_placement_plan_coverage_ratio = 1.0` on MoE runs.
  After Phase 4E, also check
  `replanning_expert_placement_plan_movement_observation_events`,
  `replanning_max_expert_placement_plan_moved_experts`,
  `replanning_total_expert_placement_plan_moved_weight_bytes`, and
  `replanning_avg_expert_placement_plan_weight_movement_cost_ms` to confirm
  that the planner compared the selected placement against the current runtime
  placement snapshot. Use
  `scripts/run_reparallelization_phase4_movement_ablation.py` for a controlled
  Phase 4F non-zero movement check: the unpenalized synthetic run should report
  `moved_experts=2`, `moved_weight_bytes=2097152`, and `movement_cost=20ms`,
  while the penalized run should select the stationary placement with
  `moved_experts=0`.
  `replanning_expert_placement_plan_physical_migration_events` should remain
  `0` until physical expert weight movement is implemented.
  Replan comparison must treat a changed `ExpertPlacementPlan` fingerprint as
  a changed deployment plan even when TP/DP/PP/replica shape is unchanged.
- For the Phase 4 runtime placement contract pre-work, require
  `context_migration_selected_target_expert_placement_contracts > 0` or
  `state_recovery_target_expert_placement_contracts > 0` only when a target was
  created from a logical `ExpertPlacementPlan`. The corresponding
  `*_expert_placement_plan_applied` and
  `*_expert_placement_plan_verified` values should remain `0` until vLLM EP
  rank mapping / weight loading can explicitly apply and verify the plan.
  The hook-level fields
  `*_expert_placement_apply_hook_available`,
  `*_expert_placement_apply_attempted`,
  `*_expert_placement_apply_success`,
  `*_expert_placement_verify_hook_available`,
  `*_expert_placement_verify_attempted`, and
  `*_expert_placement_verify_success` show whether the backend actually found
  and called a patched vLLM runtime hook. After the observe-only vLLM hook
  patch is rebuilt into the image, availability/attempt counts may be non-zero,
  and `*_contract_seen*` counts can become non-zero, but apply/verify success
  and plan applied/verified counts should still remain `0`. Also require the
  target/runtime capability fields to show `*_runtime_verified=0`,
  `*_runtime_can_verify_physical=0`, `*_runtime_can_remap_ep=0`, and
  `*_runtime_can_measure_a2a=0` with physical expert migration unsupported
  reasons.
- For Phase 5A, run `python -m sllm.spot.vllm_ep_runtime_audit` inside the
  worker runtime. The current expected gate is
  `observe_only_expert_placement_contract` with
  `runtime_verification_level=contract_seen_only`,
  `can_claim_physical_expert_migration=false`,
  `runtime_verified_placement=false`,
  `runtime_can_remap_live_ep_rank=false`, and
  `runtime_can_measure_all_to_all=false`. Only treat Phase 5 as physical
  migration when `apply` returns `applied=true`, `verify` returns
  `verified=true`, both report `physical_weight_migration=true`, and the audit
  reports `runtime_verification_level=physical_migration_verified`. If the running container cannot import the
  audit module, sync local source with `SPOTSERVE_SYNC_SOURCE=1` or rebuild
  before running the audit. The 2026-09-06 `sllm_worker_0` audit reported vLLM
  `0.11.2`, source markers present, `contract_seen_by_runtime=true`, but
  `applied=false`, `verified=false`, and `physical_weight_migration=false`.
- For Phase 5C, after rebuilding the image with the updated
  `runtime_moe_metadata.patch`, runtime metadata may include
  `runtime_expert_placement_available`,
  `runtime_expert_placement_worker_count`,
  `runtime_expert_placement_shard_count`, and
  `runtime_expert_placement_shards`. Context migration and recovery should use
  this actual runtime placement before falling back to logical
  `expert_placement_snapshot`. This verifies resident expert placement, not
  live expert migration; `runtime_physical_migration`, `runtime_remap_ep`, and
  `runtime_a2a_counters` should still be `0`.
- For Phase 5B, the current execution claim should be actor recreate, not live
  weight movement. A valid re-parallelization run should report
  `replanning_expert_placement_actor_recreate_events > 0`,
  `replanning_expert_placement_live_migration_events = 0`, and
  `replanning_expert_placement_physical_migration_required_events = 0`. The
  string fields in `latest_summary.json` should include
  `replanning_execution_models=actor_recreate`,
  `replanning_expert_placement_execution_models=expert_aware_actor_recreate`,
  and `replanning_expert_placement_contract_modes=observe_only_contract`.
- For the active-request fixed-EP remap path, prepare the environment with
  `VLLM_SPOTSERVE_EXPERT_REMAP=1` and
  `VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1`, then use
  `--deploy-set expert-remap-active-performance` and run
  `benchmark_matrix_expert_remap_active_request_performance.yaml`. A valid run
  should report `replanning_execution_models=in_place_expert_remap`,
  `replanning_expert_placement_execution_models=active_fixed_ep_remap`,
  `replanning_expert_placement_runtime_active_request_remap > 0`,
  `replanning_expert_placement_runtime_step_boundary_barrier > 0`,
  `replanning_expert_placement_runtime_physical_weight_migration > 0`,
  and `replanning_expert_placement_runtime_verified_placement > 0`.
  The 2026-09-22 `02-33-05` run had 3/3 successful requests and verified
  physical movement of four expert shards (0.75 MiB), but it used actor
  recreate: `runtime_active_remap=0` and `runtime_step_barrier=0`.
  It does not pass the active-request gate. The controller now initializes
  its vLLM deployment adapter before comparing the active and selected plans;
  the active benchmark trace uses an `add` event so the original actor remains
  READY. For a preemption event,
  the target actor is PREEMPTING and is excluded from in-place remap.
  The next attempt on 2026-09-22 failed during vLLM engine startup, before
  trace replay or remap: the worker log reported only 1.25 GiB free on a
  15.47 GiB GPU, below the 6.96 GiB required by
  `gpu_memory_utilization=0.45`. At inspection time, external Python
  processes held about 14 GiB each on GPUs 1-3 while GPU 0 was free;
  `/v1/models` was empty. This is an environment capacity failure and does
  not validate or invalidate the active-remap implementation. Re-run after
  two GPUs have sufficient free memory; do not count this run in latency or
  correctness comparisons.
- The 2026-09-23 `01-08-25` active-request rerun passes the full gate: requests
  `3/3`, trace replay `1/1`, replan applied `1/1`,
  `execution_model=active_fixed_ep_remap`, `actor_recreate=0`, runtime
  apply/verify `1/1`, `runtime_active_remap=1`,
  `runtime_step_barrier=1`, `runtime_physical_migration=1`, and runtime moved
  four expert shards / 786432 bytes in 168.12 ms. The runtime reported two
  workers and eight actual shards. This validates synchronous step-boundary
  active-request fixed-EP remap only; EP-size remap, cross-node movement, and
  all-to-all counters remain unverified.
- The active fixture was tightened again on 2026-09-24. The previous warmup
  could delay dispatch until after the trace event, and test token pacing is
  applied per DELTA output chunk rather than per token. The workload now sends
  the long request at `t=0`, applies a one-second per-chunk hold, and replays
  `add` at `t=1`. Run `12-27-04` passed with requests `2/2`, trace `1/1`,
  replan applied `1/1`, `runtime_active_remap=1`,
  `runtime_step_barrier=1`, runtime apply/verify `1/1`, four moved shards
  (786432 bytes), and a 117.99 ms runtime remap. An earlier startup-only
  failure that day did not reach trace replay; Ray also warned that its session
  filesystem was over 95% used, so that attempt is classified as an
  environment/startup failure rather than a remap result.
- On 2026-09-25, the first actor failed during startup while its replacement
  was already `starting`. The benchmark runner previously aborted on any
  historical failed actor. It now keeps waiting while a non-failed startup
  candidate exists and fails early only after three failed-only polls. Run
  `08-55-34` then passed with requests `2/2`, trace `1/1`, replan applied
  `1/1`, `execution_model=active_fixed_ep_remap`, `actor_recreate=0`, runtime
  apply/verify `1/1`, `runtime_active_remap=1`, `runtime_step_barrier=1`, four
  moved shards (0.75 MiB), a 133.18 ms runtime remap, and 3192.94 ms p95
  request latency. This remains fixed-EP active remap evidence only.
- The next runtime boundary is now explicit. EP-size changes use controlled
  actor recreation and are labeled `actor_recreate_ep_resize`; they are not
  reported as in-place remaps. Cross-node remap can be required with
  `require_cross_node_expert_migration` and fails unless distinct physical host
  IDs and actual cross-host moved shards are observed. The vLLM CUDA
  communicator patch records real dispatch/combine invocations and observed
  tensor payload bytes when `VLLM_SPOTSERVE_A2A_TRACE=1`. A rebuild and new
  runtime runs are still required before recording EP2-to-EP4, cross-node, or
  all-to-all reduction as experimental results.
- The 2026-09-25 `11-01-35` active-request run passed fixed-EP physical remap
  (`2/2`, trace `1/1`, four moved shards, runtime apply/verify `1/1`, active
  remap and step barrier both `1`). Its A2A instrumentation was enabled, but
  calls and payload were zero. This is expected for that run's `TP=2, DP=1`
  shape: vLLM 0.11.2 enables its MoE all-to-all kernels only when `DP > 1`.
  It is therefore neither evidence of A2A reduction nor an instrumentation
  failure. Use `benchmark_matrix_expert_remap_dp2_a2a_performance.yaml` for
  the new `TP=1, DP=2, EP=2` inference gate, then run
  `verify_spotserve_all_to_all_traffic.py`. The runtime gate passes only when
  actual dispatch/combine counter deltas and payload bytes are greater than
  zero. DP2 physical remap remains rejected because its transfer collective
  requires coordination across independent DP EngineCores.
- The 2026-09-25 DP2 A2A gate passed. The inference benchmark completed `2/2`
  requests with 1149.20 ms p95. Four additional verifier requests increased
  real runtime collective calls by 512, input payload by 486080 bytes, and
  output payload by 488128 bytes, for 974208 observed payload bytes. Internode
  calls remained zero as expected on one physical host. This validates A2A
  observability, not A2A traffic reduction.
- The first DP2 physical-remap attempt deadlocked because a utility request
  entered the expert-transfer collective through only one DP EngineCore. A
  subsequent all-DP broadcast run at `2026-09-26_03-34-13` also failed: one
  EngineCore could reject an active request while an idle peer entered the
  collective. The runtime patch now uses a global two-phase protocol: every
  DP EngineCore completes a side-effect-free preflight, and apply starts only
  if all engines agree. Runtime MoE metadata and verification are aggregated
  across DP engines. The new sequential verifier waits for router concurrency
  to reach zero before sending the event; a fixed trace timestamp is retained
  only as a stress test, not as proof of quiescence.
- The 2026-09-26 sequential DP2 gate passed after that fix. Both pre-remap and
  post-remap inference completed, the router was idle before the event, and
  runtime apply/verify each succeeded. Two DP EngineCores reported a verified
  physical migration of four expert shards (786432 bytes) in 123.13 ms. This
  validates same-host `TP1 x DP2 x EP2` coordinated physical remap; it does not
  validate cross-host movement, live EP resizing, or A2A traffic reduction.
- For the placement ordering guard, require
  `context_migration_placement_handshake_stale = 0` and
  `state_recovery_placement_handshake_stale = 0` before claiming that migration
  or recovery used a stable target placement view. The corresponding
  `*_placement_handshake_attempts` and `*_placement_handshake_successes` fields
  show whether the runtime could verify the target epoch/fingerprint.
- The 2026-09-16 V7-V9 core table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_CORE_LOAD_FORMAT=auto`. The applied run reported
  `context_migrations=1`, `route_source=vllm_runtime_topk`,
  `route_kind=runtime_observed_topk`, `kv_successes=1`,
  `state_events=1`, `state_fallbacks=1`, `true_kv_restores=0`,
  `supports_state_restore=0`, `recovery_kv_compatible=1`,
  `recovery_ep_required=0`, `recovery_ep_mismatch=1`, and
  `risk_scheduling_events=3`. This is an integrated control-flow result, not
  the primary true-KV-restore result.
- For the V7-V9 trace sweep, baseline and applied use the same trace file
  inside each scenario. The benchmark runner fills in each run's model name at
  replay time. This sweep expects two ready vLLM instances, so rerun
  `prepare_spotserve.sh --skip-build --deploy-set spotserve-core-performance`
  after pulling this change and confirm that both `sllm_worker_0` and
  `sllm_worker_1` are active. After the sweep, compare scenarios in
  `results/spotserve_core_trace_sweep/latest_comparisons.json`.
- V9 scheduling affects model placement/loading, not per-token decoding. On a
  single worker node it can emit the scheduling decision but may not improve
  placement risk until multiple worker nodes are available.
