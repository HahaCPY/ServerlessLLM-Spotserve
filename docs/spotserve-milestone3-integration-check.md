# SpotServe Milestone 3 Integration Check

Date: 2026-07-07

This check validates the CPY V6-V9 control-plane work against the backend
metadata contracts added for Milestone 3.

## Summary

| Area | Backend metadata status | CPY integration status | Result |
|---|---|---|---|
| V6 Dynamic Reparallelization | `BackendCapability` and vLLM capability helpers exist | Planner now filters candidates through backend supported configs | Integrated |
| V7 Context Migration | vLLM context metadata helper and backend hook exist | Planner accepts the metadata schema; live router migration path is not wired yet | Schema integrated, runtime path pending |
| V8 Stateful Recovery | vLLM state metadata helper and backend hook exist | Router calls backend state hooks and falls back when restore is unsupported | Integrated with conservative vLLM fallback |
| V9 Risk-aware Scheduling | vLLM runtime metadata/resource profile helper exists | Scheduler can consume risk/loading metadata from config/synthetic node metadata | Planner integrated, live runtime feed pending |

## V6 Check

Backend files:

```text
sllm/backends/capability.py
sllm/backends/vllm_capability.py
```

CPY files:

```text
sllm/spot/reparallelization.py
tests/spotserve_test/test_reparallelization_planner.py
```

Finding:

`BackendCapability` existed, but `plan_dynamic_reparallelization()` was still
generating unconstrained candidates. A dense vLLM config advertising TP=2 could
still be replanned as TP=4.

Fix applied:

```text
plan_dynamic_reparallelization()
-> get_backend_capability(model_config)
-> use BackendCapability.supported_configs when available
-> return no_capacity if backend-supported configs do not fit
```

Regression tests added:

```text
test_reparallelization_respects_backend_capability_supported_shape
test_reparallelization_does_not_fallback_when_capability_has_no_capacity
```

Validated shape:

```text
capability: TP=2, DP=1, PP=1, num_gpus=4
planner:    TP=2, DP=1, PP=1, num_gpus=4
```

## V7 Check

Backend files:

```text
sllm/backends/vllm_context_metadata.py
sllm/backends/vllm_backend.py
```

CPY files:

```text
sllm/spot/context_migration.py
scripts/run_context_migration_benchmark.py
```

Status:

- `get_vllm_context_metadata()` emits fields accepted by
  `ContextMetadata.from_dict()`.
- `VllmBackend.get_context_metadata()` exists and returns token-level context
  metadata for ongoing vLLM requests.
- Synthetic context migration benchmark still uses
  `benchmarks/spotserve/context_migration_synthetic.json`.

Remaining integration gap:

The live router / controller does not yet call `get_context_metadata()` and feed
it into `plan_low_cost_migration()`. Therefore V7 is schema-integrated, but
live vLLM context migration planning is still pending.

Synthetic validation:

```text
migration_plan_count=3
total_estimated_cost=36.0
reuse_ratio=0.9444
```

## V8 Check

Backend files:

```text
sllm/backends/vllm_state_metadata.py
sllm/backends/vllm_backend.py
sllm/backends/backend_utils.py
```

CPY files:

```text
sllm/spot/stateful_recovery.py
sllm/routers/roundrobin_router.py
```

Status:

- `VllmBackend.export_inference_state()` returns token snapshot metadata.
- `VllmBackend.supports_state_restore()` returns `False`.
- `VllmBackend.restore_inference_state()` returns conservative
  `restored=false`.
- Router calls backend state hooks for `stateful_recovery` and falls back when
  restore is unsupported.

This is the correct integration state until true vLLM KV restore exists.

Existing dummy benchmark validation:

```text
state_recovery_events=1
state_recovery_restore_events=1
state_restore_attempts_total=1
state_restore_successes_total=1
state_restored_tokens_total=2
state_restore_fallback_count=0
```

## V9 Check

Backend files:

```text
sllm/backends/vllm_runtime_metadata.py
sllm/backends/vllm_backend.py
```

CPY files:

```text
sllm/spot/risk_aware_scheduling.py
sllm/schedulers/fcfs_scheduler.py
scripts/run_scheduler_benchmark.py
```

Status:

- `get_vllm_runtime_metadata()` can emit `loading_cost`, GPU capacity, spot
  risk, and remaining lifetime fields.
- `node_risk_score()` accepts that metadata shape.
- FCFS scheduler can opt into risk-aware ranking via
  `scheduler_config.enable_spot_risk_aware`.

Remaining integration gap:

The live scheduler currently consumes configured/synthetic node risk metadata.
It does not yet query running backend actors for `get_runtime_metadata()`.

Synthetic validation:

```text
health_only avg_selected_spot_risk = 0.6333
risk_aware  avg_selected_spot_risk = 0.15
```

## Commands Run

Compile:

```bash
python -m py_compile \
  sllm/backends/capability.py \
  sllm/backends/vllm_capability.py \
  sllm/backends/vllm_context_metadata.py \
  sllm/backends/vllm_state_metadata.py \
  sllm/backends/vllm_runtime_metadata.py \
  sllm/backends/vllm_backend.py \
  sllm/spot/reparallelization.py \
  sllm/spot/context_migration.py \
  sllm/spot/stateful_recovery.py \
  sllm/spot/risk_aware_scheduling.py \
  scripts/run_context_migration_benchmark.py \
  scripts/run_scheduler_benchmark.py \
  scripts/analyze_spotserve_benchmark.py \
  scripts/plot_spotserve_benchmark.py
```

Synthetic benchmarks:

```bash
python scripts/run_context_migration_benchmark.py \
  --input benchmarks/spotserve/context_migration_synthetic.json \
  --output-dir /tmp/spotserve_context_migration_integration_check

python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_integration_check
```

Direct function tests were also run for:

```text
backend capability
vLLM context metadata
vLLM state metadata
vLLM runtime metadata
reparallelization planner
context migration planner
risk-aware scheduling planner
```

`pytest` is not installed in the host environment, so pytest-native tests that
import `pytest` were not run through pytest in this check.

## Next Integration Work

1. Wire live V7:
   collect `VllmBackend.get_context_metadata()` from active instances and feed
   it to `plan_low_cost_migration()`.

2. Wire live V9 metadata feed:
   decide whether scheduler gets node risk/loading metadata from config,
   controller, backend actors, or a lightweight metadata registry.

3. Run container benchmarks:
   - V6 reparallelization benchmark
   - V8 recovery correctness benchmark
   - V7/V9 synthetic benchmarks inside `sllm_head`

4. Only after these pass, move to Version 10 expert-aware recovery.
