# SpotServe Performance Summary

Last updated: 2026-07-28

This page is intentionally short. It keeps only the commands needed to rerun
each version's benchmark path and a compact performance summary. V6 and V7 were
refreshed after the latest backend switch; V8 still needs a target new-backend
performance rerun.

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

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto \
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
MODEL_FOLDER="$PWD/model" scripts/prepare_spotserve.sh --skip-build --deploy-set stateful-recovery-performance

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

V9 risk-aware scheduling synthetic benchmark:

```bash
python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_test
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
| V7 | `benchmark_matrix_context_migration_performance.yaml` | Shared Qwen2 MoE run keeps both versions at 100% success; preemption is injected, one context-migration plan executes, and one KV migration succeeds. | Refreshed 2026-07-28. |
| V8 | `benchmark_matrix_stateful_recovery_performance.yaml` | Old backend record showed stateful recovery beating token replay with no fallback. | Needs rerun with new backend. |
| V9 | `risk_aware_scheduling_synthetic.json` | Placement-quality improvement: lower-risk / longer-lived node selection. | Synthetic result; live latency/SLO impact still requires a real workload. |

| Metric | Baseline | Applied | Result | Status |
|---|---|---|---|---|
| V6 success rate | Reparallelization disabled | Reparallelization applied | `100.00% -> 100.00%` | New backend, `8/8 -> 8/8`. |
| V6 overall p95 | Reparallelization disabled | Reparallelization applied | `53261.47ms -> 13095.55ms` | Applied reduces overall p95 by `40165.93ms`. |
| V6 replan-window p95 | Reparallelization disabled | Reparallelization applied | `14226.90ms -> 13095.55ms` | Replan window improved. |
| V6 post-replan p95 | Reparallelization disabled | Reparallelization applied | `1048.39ms -> 1094.16ms` | Near parity after replan. |
| V6 replan execution | Reparallelization disabled | Reparallelization applied | `0 replans -> 1 replan, 1 applied, 0 failed` | Lifecycle verified. |
| V7 MoE success rate | Context migration disabled | Context migration applied | `100.00% -> 100.00%` | New backend, `8/8 -> 8/8`. |
| V7 MoE overall p95 | Context migration disabled | Context migration applied | `45668.11ms -> 2541.99ms` | Applied reduces overall p95 by `43126.12ms`. |
| V7 MoE migration-window p95 | Context migration disabled | Context migration applied | `34090.02ms -> 1023.38ms` | Migration window improved by `33066.64ms`. |
| V7 MoE post-migration p95 | Context migration disabled | Context migration applied | `1021.80ms -> 1042.07ms` | Near parity after migration. |
| V7 MoE migration signals | Context migration disabled | Context migration applied | `0 -> 1 context migration, 1 plan, 1 KV success, 245 KV tokens` | KV migration path verified; reusable blocks stayed `0`. |
| V8 overall p95 | Token replay | Stateful recovery | `63540.97ms -> 2904.41ms` | Old backend; replace after rerun. |
| V8 failure-window p95 | Token replay | Stateful recovery | `63540.97ms -> 2904.41ms` | Old backend; replace after rerun. |

Notes:

- Keep latency claims tied to the matrix that produced them.
- For V7, require `context_migration_plan_count > 0` and
  `kv_cache_migration_successes > 0` before claiming the migration path worked.
  Only claim prefix/block reuse when
  `context_migration_reusable_context_blocks > 0`.
- For V8, require `state_restore_successes_total > 0` and
  `state_restore_fallback_count = 0` before claiming true stateful restore.
- The 2026-07-27 V6 table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto`.
- A 2026-07-28 V7 dense-side run used `/models/vllm/vllm-dense-baseline`;
  do not use it as the V7 MoE result.
- The 2026-07-28 V7 MoE table uses shared `Qwen2-MoE-Tiny` with
  `SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto`; this run migrated KV tokens
  but did not report reusable context blocks.
