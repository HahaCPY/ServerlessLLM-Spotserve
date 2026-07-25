# SpotServe Performance Summary

Last updated: 2026-07-25

This document summarizes the performance and benchmark evidence for each
SpotServe version. The numbers below should be read with their benchmark scope:
some versions measure request latency directly, while others validate
correctness, availability, or scheduling quality.

## High-level Result

The clearest latency improvement so far is Version 7:

```text
V7 context migration performance benchmark
success_rate: 100% -> 100%
overall p95: 62489.00ms -> 4616.59ms
migration-window p95: 51255.32ms -> 1027.54ms
post-migration p95: 5233.93ms -> 1051.30ms
```

The applied V7 run also produced the required context-migration signals:

```text
context_migration_plan_count = 1
context_migration_reusable_tokens = 160
context_migration_reusable_context_blocks = 10
context_migration_reuse_ratio = 0.9090909090909091
kv_cache_migration_attempts = 1
kv_cache_migration_successes = 1
```

This supports the claim that the live vLLM V7 runtime-metadata-driven context
migration and target warmup path can reduce preemption-window latency in the
same-host benchmark. It does not claim full cross-node KV block serialization
or direct KV transfer.

## Version Summary

| Version | Feature | Benchmark Evidence | Performance Claim |
|---|---|---|---|
| V1 | Benchmark harness + initial dummy recovery policies | Full trace completed: `dummy-no-preemption`, `dummy-naive-retry`, and `dummy-token-replay` all reached `46/46` successes | No speedup claim. Validates benchmark/report plumbing and trace replay. |
| V2 | Recover dispatch + scheduler node health | Recover/preempt/dead events are dispatched; non-ready nodes are filtered | Availability/control-plane improvement only. No direct latency benchmark. |
| V3 | Recovery correctness validation | Forced-failure dummy benchmark: `none=0/2`, `naive_retry=2/2`, `token_replay=2/2`; token replay recovered 2 tokens with 1 fallback | Correctness improvement: recovery policies turn failures into successes. Tail latency is higher than failed `none` because retry/replay work is actually performed. |
| V4 | Dense vLLM black-box integration | vLLM dense aliases deploy and serve through SLLM; HTTP spot trace replay works | Compatibility milestone. No controlled latency speedup claim. |
| V5 | MoE vLLM black-box integration | MoE aliases deploy and serve through SLLM; dense-vs-MoE matrix exists | MoE compatibility milestone. No expert-aware or MoE-specific latency speedup claim. |
| V6 | Dynamic reparallelization planner + vLLM deployment adapter | Live vLLM performance matrix: `success_rate=1.0` both; applied has `replanning_events=1`, `replanning_execution_applied=1` | On the recorded single-worker run, overall p95 improved by 74.25%, but the safe claim is lifecycle correctness and post-replan steady-state parity. A production capacity-loss speedup needs multi-worker evidence. |
| V7 | Low-cost context migration planner + live vLLM metadata | Latest live vLLM matrix: `plan_count=1`, `reuse_ratio=0.909`, `kv_cache_migration_successes=1` | Strong same-host latency result: migration-window p95 reduced by 98.00% and post-migration p95 by 79.91%. This is target warmup/reuse, not full cross-node KV direct transfer. |
| V8 | Stateful inference recovery | Correctness matrix: `none=0/2`, `naive_retry=2/2`, `token_replay=2/2`, `stateful_recovery=2/2`; state restore attempts/successes/tokens are positive | Recovery correctness improvement. Dummy p95 is similar to retry/replay because the benchmark validates restore behavior, not a latency speedup. Same-node NIXL restore is validated separately from production latency. |
| V9 | Spot-risk-aware scheduling | Synthetic scheduler benchmark selects lower-risk / longer-lived nodes instead of the first health-only candidate | Scheduling-quality improvement. No request-latency speedup claim yet; benefit is reduced expected preemption risk / placement cost. |

## Direct Latency Comparisons

### V6 Dynamic Reparallelization

Benchmark:

```text
benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml
```

Recorded comparison:

| Metric | Disabled | Applied | Delta | Reduction |
|---|---:|---:|---:|---:|
| `success_rate` | 1.0 | 1.0 | 0.0 | 0.00% |
| `latency_p95_ms` | 54753.39 | 14099.99 | -40653.40 | 74.25% |
| `phase_replan_window_latency_p95_ms` | 15333.89 | 14099.99 | -1233.91 | 8.05% |
| `phase_post_replan_latency_p95_ms` | 1047.34 | 1047.15 | -0.19 | 0.02% |

Interpretation:

- The applied run did execute one `ParallelPlan`.
- Post-replan steady-state latency is effectively the same as baseline.
- The single-worker setup mostly measures adapter/recreate behavior.
- Do not claim production V6 latency improvement without a multi-worker run
  where the applied path moves traffic to another live worker.

### V7 Low-cost Context Migration

Benchmark:

```text
benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml
```

Latest recorded comparison:

| Metric | Disabled | Applied | Delta | Reduction |
|---|---:|---:|---:|---:|
| `success_rate` | 1.0 | 1.0 | 0.0 | 0.00% |
| `latency_p95_ms` | 62489.00 | 4616.59 | -57872.42 | 92.61% |
| `phase_migration_window_latency_p95_ms` | 51255.32 | 1027.54 | -50227.78 | 98.00% |
| `phase_post_migration_latency_p95_ms` | 5233.93 | 1051.30 | -4182.63 | 79.91% |

Context migration counters:

| Metric | Disabled | Applied |
|---|---:|---:|
| `context_migration_events` | 0 | 1 |
| `context_migration_plan_count` | 0 | 1 |
| `context_migration_reusable_tokens` | 0 | 160 |
| `context_migration_reusable_context_blocks` | 0 | 10 |
| `context_migration_reuse_ratio` | 0.0 | 0.9090909090909091 |
| `kv_cache_migration_attempts` | 0 | 1 |
| `kv_cache_migration_successes` | 0 | 1 |
| `kv_cache_migration_tokens` | 0 | 164 |

Interpretation:

- The benchmark preempted a busy source replica.
- The router collected live source context metadata.
- The router found a target with compatible runtime prefix/KV metadata.
- The planner produced one migration plan.
- The target warmup path succeeded through `resume_kv_cache()`.

Safe wording:

```text
V7 reduced preemption-window p95 latency from 51.3s to 1.03s in the live
same-host vLLM context-migration benchmark, while maintaining 100% success
rate and producing one real context-migration plan with 10 reusable context
blocks.
```

Boundary:

```text
This validates metadata-driven context reuse and target warmup. It does not
prove full cross-node KV block serialization or direct KV transfer.
```

## Correctness / Availability Comparisons

### V3 Recovery Correctness

Benchmark:

```text
benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml
```

Expected successful shape:

```text
dummy-correctness-none: successes=0/2
dummy-correctness-naive-retry: successes=2/2, retries=2
dummy-correctness-token-replay: successes=2/2, retries=2, recovered_tokens=2
```

Performance interpretation:

- `none` has lower apparent latency only because the failed requests stop early.
- `naive_retry` and `token_replay` increase work but recover the request.
- The improvement is success rate and recovered work, not lower p95 latency.

### V8 Stateful Recovery

Benchmark:

```text
benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml
```

Recorded shape:

```text
dummy-correctness-none: successes=0/2
dummy-correctness-naive-retry: successes=2/2
dummy-correctness-token-replay: successes=2/2
dummy-correctness-stateful-recovery: successes=2/2
```

Stateful recovery metrics:

```text
state_recovery_events = 1
state_recovery_restore_events = 1
state_recovery_fallback_events = 0
state_restore_attempts_total = 1
state_restore_successes_total = 1
state_restored_tokens_total = 2
```

Performance interpretation:

- V8 proves backend state export/restore integration and no-fallback restore in
  the correctness benchmark.
- The dummy benchmark does not show a latency win over retry/token replay.
- A latency claim for V8 should use a live vLLM KV restore benchmark with
  restored KV blocks and no fallback.

## Scheduling / Placement Comparisons

### V9 Risk-aware Scheduling

Benchmark:

```text
benchmarks/spotserve/risk_aware_scheduling_synthetic.json
scripts/run_scheduler_benchmark.py
```

Expected shape:

```text
health_only selects the first ready node with enough GPUs
risk_aware selects the lower-risk / longer-lived ranked node
```

Performance interpretation:

- V9 does not directly reduce per-request latency in the synthetic benchmark.
- The improvement is placement quality: lower selected risk, longer expected
  lifetime, and lower expected interruption cost.
- Production latency/SLO improvement requires a live spot-risk predictor and a
  workload where bad placements cause measurable preemption or reload cost.

## Run Commands

V6 performance:

```bash
scripts/prepare_spotserve.sh --deploy-set reparallelization-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180
'
```

V7 performance:

```bash
scripts/prepare_spotserve.sh --deploy-set context-migration-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240
'
```

V8 recovery correctness:

```bash
scripts/prepare_spotserve.sh --deploy-set correctness

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
'
```

V9 scheduling:

```bash
python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_test
```

## Claim Checklist

Use these checks before writing performance claims:

- A latency claim must compare a disabled baseline and an applied run in the
  same benchmark matrix.
- `success_rate` should remain equal or improve.
- V6 claims require `replanning_execution_applied > 0`.
- V7 claims require `context_migration_plan_count > 0`,
  `context_migration_reusable_context_blocks > 0`, and
  `kv_cache_migration_successes > 0` when describing the warmup path.
- V8 true KV restore claims require `state_restore_successes_total > 0`,
  `state_restore_fallback_count = 0`, and a live vLLM restore benchmark if
  latency is mentioned.
- V9 production claims require real provider/runtime risk metadata, not only
  synthetic risk input.
