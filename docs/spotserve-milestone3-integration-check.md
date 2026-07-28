# SpotServe Milestone 3 Integration Check

Date: 2026-07-07

This check validates the CPY V6-V9 control-plane work against the backend
metadata contracts added for Milestone 3.

## Summary

| Area | Backend metadata status | CPY integration status | Result |
|---|---|---|---|
| V6 Dynamic Reparallelization | `BackendCapability` and vLLM capability helpers exist | Planner now filters candidates through backend supported configs | Integrated |
| V7 Context Migration | vLLM context metadata helper and backend hook exist | Router now collects live context metadata on spot events and feeds the planner | Integrated planning path |
| V8 Stateful Recovery | vLLM state metadata helper and backend hook exist | Router calls backend state hooks and falls back when restore is unsupported | Integrated with conservative vLLM fallback |
| V9 Risk-aware Scheduling | vLLM runtime metadata/resource profile helper exists | Scheduler can consume config/synthetic metadata and opt into backend actor runtime metadata | Integrated planning path |

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
sllm/routers/roundrobin_router.py
scripts/run_context_migration_benchmark.py
```

Status:

- `get_vllm_context_metadata()` emits fields accepted by
  `ContextMetadata.from_dict()`.
- `VllmBackend.get_context_metadata()` exists and returns token-level context
  metadata for ongoing vLLM requests.
- `RoundRobinRouter` can opt into `enable_context_migration`.
- On `handle_preemption()` and `handle_dead()`, the router calls
  `get_context_metadata(instance_id, node_id)` on affected backend instances.
- The router builds `MigrationTarget` entries from READY inference instances,
  calls `plan_low_cost_migration()`, emits a `context_migration` metric, and
  returns the decision in the spot-event response.
- Synthetic context migration benchmark still uses
  `benchmarks/spotserve/context_migration_synthetic.json`.

Remaining integration gap:

This is still a planning path. It chooses where contexts should migrate, but
does not execute true KV cache export, transfer, restore, or production request
resume.

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

## V7 KV Cache Warmup Check

CPY router now has an opt-in cache warmup path for context migration:

```text
enable_context_migration = true
enable_kv_cache_migration = true
source get_current_tokens()
-> target resume_kv_cache(request_datas=tokens)
-> kv_cache_migration metrics in context_migration event
```

This validates the live control-plane path and target prefix/cache warmup. It is
not true vLLM KV block restore; `supports_state_restore=false` still causes V8
stateful recovery to use fallback behavior for real request restoration.

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
- FCFS scheduler can opt into live backend actor metadata refresh via
  `scheduler_config.enable_backend_runtime_metadata`.
- The scheduler queries `ray.get_actor(instance_id).get_runtime_metadata()`
  for allocated instances, merges metadata into worker-node info, and then
  feeds the result to `node_risk_score()`.
- Backend actor lookup falls back to the `models` namespace and runtime rows
  are normalized with `risk_metadata_source=backend_runtime`.
- Ray worker-node `free_gpu` remains authoritative; backend GPU counts are
  preserved as observations and do not create phantom allocation capacity.

Remaining integration gap:

There is still no real cloud spot provider integration or production risk
prediction model. Backend actor metadata can expose loading cost and resource
profile today; spot risk / lifetime are omitted unless a provider, config, or
backend runtime source supplies them.

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
  sllm/routers/roundrobin_router.py \
  sllm/schedulers/fcfs_scheduler.py \
  sllm/spot/reparallelization.py \
  sllm/spot/context_migration.py \
  sllm/spot/stateful_recovery.py \
  sllm/spot/risk_aware_scheduling.py \
  tests/spotserve_test/test_router_state.py \
  tests/spotserve_test/test_scheduler_node_health.py \
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

Direct router smoke validation was also run for V7 live planning:

```text
handle_preemption()
-> fake backend get_context_metadata()
-> context_migration action=migrate
-> context_migration metric row emitted
```

Direct scheduler smoke validation was also run for V9 live metadata:

```text
fake backend actors
-> get_runtime_metadata()
-> merge into worker node info
-> risk-aware ranking selects lower-risk node
```

`pytest` is not installed in the host environment, so pytest-native tests that
import `pytest` were not run through pytest in this check.

## Next Integration Work

1. Run container benchmarks:
   - V6 reparallelization benchmark
   - V8 recovery correctness benchmark
   - V7/V9 synthetic benchmarks inside `sllm_head`

2. Add the true backend V7 executor:
   KV cache export, transfer, restore, and request resume.

3. Add a real spot provider / lifetime feed if the project needs production
   risk values rather than synthetic/config metadata.

4. Only after these pass, move to Version 10 expert-aware recovery.
