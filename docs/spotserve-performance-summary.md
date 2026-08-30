# SpotServe Performance Summary

Last updated: 2026-08-30

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
scripts/prepare_spotserve.sh --skip-build --deploy-set reparallelization-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
'
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

## Performance Summary

| Version | Benchmark | Current claim | Refresh status |
|---|---|---|---|
| V1 | `benchmark_matrix.yaml` | Harness/reporting validation only; no latency speedup claim. | Stable unless harness changes. |
| V2 | `benchmark_matrix_long.yaml` | Availability/control-plane validation; no direct latency speedup claim. | Stable unless scheduler/control-plane changes. |
| V3 | `benchmark_matrix_recovery_correctness.yaml` | Recovery policies turn forced failures into successes; latency is not the main metric. | Stable correctness check. |
| V4 | `benchmark_matrix_vllm_dense.yaml` | Dense vLLM compatibility milestone; black-box recovery smoke only. | Rerun if backend image/model changed. |
| V5 | `benchmark_matrix_vllm_moe.yaml`, `benchmark_matrix_vllm_dense_vs_moe.yaml` | MoE compatibility milestone; no MoE-specific speedup claim. | Rerun after backend/model changes. |
| V6 | `benchmark_matrix_reparallelization_performance.yaml` | Shared Qwen2 run keeps both versions at 100% success; reparallelization executes once, succeeds, and lowers overall p95. Post-replan p95 stays near parity. | Refreshed 2026-07-27. |
| V7 | `benchmark_matrix_context_migration_performance.yaml` | Shared Qwen2 MoE run keeps both versions at 100% success; preemption is injected, one context-migration plan executes, selected-plan KV/expert/queue costs are reported, and route metadata is consumed from patched vLLM runtime top-k instrumentation. V7 verifies low-cost target selection and prefix warmup/context planning, not physical expert migration or true remote expert-dispatch traffic. | Runtime top-k MoE run refreshed 2026-08-30. |
| V8 | `benchmark_matrix_stateful_recovery_performance.yaml` | Shared Qwen2 MoE run keeps both versions at 100% success; stateful recovery restores once, restores 16 tokens and 6 KV blocks, has no fallback, and reports separated KV compatibility vs EP/locality signals. | Phase 3 MoE-aware recovery run refreshed 2026-08-30. |
| V9 | `risk_aware_scheduling_synthetic.json` | Placement-quality improvement: lower-risk / longer-lived node selection. | Synthetic result; live latency/SLO impact still requires a real workload. |
| V7-V9 core | `benchmark_matrix_spotserve_core_performance.yaml` | One live matrix deploys baseline and applied variants of the same model. Applied enables context/KV migration, stateful recovery, and risk-aware scheduling together. | Refreshed 2026-08-04. |
| V7-V9 trace sweep | `benchmark_matrix_spotserve_core_trace_sweep.yaml` | Runs the same baseline/applied core pair across busy-recover, fast-recover, slow-recover, no-recover, dead-after-preempt, and double-preempt traces. | Pending run. |

| Metric | Baseline | Applied | Result | Status |
|---|---|---|---|---|
| V6 success rate | Reparallelization disabled | Reparallelization applied | `100.00% -> 100.00%` | New backend, `8/8 -> 8/8`. |
| V6 overall p95 | Reparallelization disabled | Reparallelization applied | `53261.47ms -> 13095.55ms` | Applied reduces overall p95 by `40165.93ms`. |
| V6 replan-window p95 | Reparallelization disabled | Reparallelization applied | `14226.90ms -> 13095.55ms` | Replan window improved. |
| V6 post-replan p95 | Reparallelization disabled | Reparallelization applied | `1048.39ms -> 1094.16ms` | Near parity after replan. |
| V6 replan execution | Reparallelization disabled | Reparallelization applied | `0 replans -> 1 replan, 1 applied, 0 failed` | Lifecycle verified. |
| V7 MoE success rate | Context migration disabled | Context migration applied | `100.00% -> 100.00%` | New backend, `8/8 -> 8/8`. |
| V7 MoE overall p95 | Context migration disabled | Context migration applied | `48238.92ms -> 4010.09ms` | Applied reduces overall p95 by `44228.82ms`. |
| V7 MoE migration-window p95 | Context migration disabled | Context migration applied | `37079.91ms -> 1030.30ms` | Migration window improved by `36049.61ms`. |
| V7 MoE post-migration p95 | Context migration disabled | Context migration applied | `1023.59ms -> 1033.12ms` | Near parity after migration. |
| V7 MoE migration signals | Context migration disabled | Context migration applied | `0 -> 1 context migration, 1 plan, 1 prefix-warmup/KV success, 228 migrated/warmed tokens` | Selected-plan costs were `kv=293.00`, `expert=0.00`, `queue=0.00`; route metadata was `vllm_runtime_topk/runtime_observed_topk`. Reusable blocks stayed `0`, so do not claim true KV block transfer from this V7 row. |
| V8 MoE success rate | Token replay | Stateful recovery | `100.00% -> 100.00%` | New backend, `3/3 -> 3/3`. |
| V8 MoE overall p95 | Token replay | Stateful recovery | `49201.94ms -> 2374.77ms` | Stateful recovery reduces overall p95 by `46827.17ms`. |
| V8 MoE failure-window p95 | Token replay | Stateful recovery | `49201.94ms -> 2374.77ms` | Failure-window p95 improves by `95.17%`. |
| V8 MoE post-recovery p95 | Token replay | Stateful recovery | `1082.20ms -> 1070.37ms` | Post-recovery latency stays near parity. |
| V8 MoE restore signals | Token replay | Stateful recovery | `0 -> 1 restore, 16 restored tokens, 6 restored blocks, 0 fallback, true_kv_rate=100.00%` | True KV restore path verified for this run. |
| V8 MoE recovery compatibility | Token replay | Stateful recovery | `0 -> 1 KV-compatible, 0 EP-required, 1 EP mismatch, locality=1.00, remote_tokens=0, expert_cost=0.00` | Confirms EP mismatch was reported as topology/locality information, not used as a hard KV restore rejection. |
| V7-V9 live combined success rate | Token replay + no context migration + health-only scheduling | Stateful recovery + context/KV migration + risk-aware scheduling | `100.00% -> 100.00%` | New live combined run, `8/8 -> 8/8`. |
| V7-V9 live combined overall p95 | Token replay + no context migration + health-only scheduling | Stateful recovery + context/KV migration + risk-aware scheduling | `48793.75ms -> 3109.22ms` | Overall p95 improves by `93.63%`, about `15.69x` faster. |
| V7-V9 live migration-window p95 | Context migration disabled | Context/KV migration applied | `37078.66ms -> 1024.55ms` | Migration window improves by `97.24%`. |
| V7-V9 live failure-window p95 | Token replay | Stateful recovery | `2222.02ms -> 3109.22ms` | In this combined timing, stateful failure window is slower by `887.20ms`; standalone V8 remains the cleaner stateful-recovery speedup result. |
| V7-V9 live core signals | Baseline policies | Applied policies | `0 -> 1 context migration, 1 KV success, 1 state restore, 2 risk scheduling events` | All three core code paths ran in one applied benchmark. |
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
- The 2026-07-27 V6 table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto`.
- A 2026-07-28 V7 dense-side run used `/models/vllm/vllm-dense-baseline`;
  do not use it as the V7 MoE result.
- The 2026-08-30 V7 MoE table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto`. The applied run reported
  `route_source=vllm_runtime_topk` and
  `route_kind=runtime_observed_topk`, so runtime MoE routing instrumentation
  reached the V7 planner/summary. It still warmed/replayed prefix tokens and
  did not report reusable context blocks.
- The 2026-08-30 V8 MoE table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT=auto`. The applied run reported
  `state_restores=1/1`, `state_blocks=6`, `true_kv_restores=1`,
  `recovery_kv_compatible=1`, `recovery_ep_required=0`,
  `recovery_ep_mismatch=1`, and `recovery_locality=1.00`.
- For the live V7-V9 core benchmark, require the applied summary to show
  `context_migration_events > 0`, `state_restore_successes_total > 0`, and
  `risk_scheduling_events > 0` before claiming that all three core paths ran
  together.
- For the V7-V9 trace sweep, baseline and applied use the same trace file
  inside each scenario. The benchmark runner fills in each run's model name at
  replay time. After the sweep, compare scenarios in
  `results/spotserve_core_trace_sweep/latest_comparisons.json`.
- V9 scheduling affects model placement/loading, not per-token decoding. On a
  single worker node it can emit the scheduling decision but may not improve
  placement risk until multiple worker nodes are available.
