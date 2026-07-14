# Bignose TODO

This document is the handoff between CPY's ServerlessLLM/SpotServe
control-plane work and Bignose's vLLM/runtime integration work for Milestone 3.

The short version:

```text
V6: CPY planner is done; vLLM still needs an executor that applies ParallelPlan.
V7: CPY live context-migration path calls vLLM backend; true KV reuse needs runtime metadata.
V8: CPY recovery path is done; dummy validates token restore; true vLLM KV restore needs runtime export/attach.
V9: CPY risk-aware scheduler is done; real risk quality depends on runtime/provider metadata.
```

## Status Matrix

| Version | CPY / ServerlessLLM status | Bignose / runtime status | What can be claimed now |
|---|---|---|---|
| V6 Dynamic Reparallelization | Planner, `ParallelPlan`, spot-event replanning, metrics | vLLM does not yet apply selected `ParallelPlan` to live workers | Control-plane replanning only |
| V7 Low-cost Context Migration | Router calls backend `get_context_metadata()`, plans migration, optional `resume_kv_cache()` warmup | Runtime must provide real KV block metadata and target-specific reuse | Live planning and token/prefix warmup |
| V8 Stateful Restore | Router recovery path calls export/restore and falls back safely | Runtime must export non-empty KV `runtime_state` and attach it on target | Dummy token snapshot restore; vLLM contract only unless hooks work |
| V9 Risk-aware Scheduling | Scheduler can rank by risk and query backend actor metadata | Runtime/provider should provide accurate risk/capacity/lifetime metadata | Risk-aware ranking with synthetic/config or available runtime metadata |

## Current Backend Contract

`VllmBackend` is now written as a capability-negotiated runtime contract. A vLLM
engine or KV connector may expose:

```text
get_request_kv_metadata(request_id)
get_all_request_kv_metadata()
export_inference_state(request_id, request_data, runtime_metadata)
restore_inference_state(state, request_id, request_data)
supports_state_restore()
```

Accepted aliases:

```text
get_kv_cache_metadata
export_kv_cache_state
restore_kv_cache_state
```

The backend must report true restore support only when export and restore hooks
exist, the optional capability probe succeeds, export returns
`supports_restore=true`, and export includes a non-empty `runtime_state`.
Unpatched upstream vLLM remains on the safe token snapshot / prefix warmup path.

## V6 Dynamic Reparallelization Boundary

CPY has implemented:

```text
spot event
-> update worker GPU availability
-> select backend-capability-aware ParallelPlan
-> return replanning decision
-> emit replanning metrics
```

The selected `ParallelPlan` is currently a decision artifact. The vLLM runtime
does not yet consume it to rebuild or reconfigure live actors:

```text
ParallelPlan
-> stop / drain old vLLM workers
-> create target vLLM workers with new TP / DP / PP / EP
-> migrate or restore request state when possible
-> switch traffic to the new parallel layout
```

Bignose/runtime TODO for V6:

- implement an executor/controller that consumes selected `ParallelPlan`.
- create or reconfigure vLLM workers with the selected TP / DP / PP / EP shape.
- drain or stop old workers safely.
- switch router traffic only after target workers are ready.
- coordinate with V7/V8 if in-flight request state must be migrated/restored.

Until this executor exists, V6 benchmarks can validate replanning metrics and
backend-capability-aware plan selection, but not vLLM runtime
reparallelization latency speedup.

## V7 Context Migration Boundary

V7 is more connected to the backend than V6. The router already calls live
backend metadata:

```text
spot preempt/dead event
-> source VllmBackend.get_context_metadata()
-> ContextMetadata.from_dict()
-> plan_low_cost_migration()
-> optional target resume_kv_cache() warmup
-> context_migration metrics
```

`VllmBackend.get_context_metadata()` first tries runtime-level hooks such as
`get_all_request_kv_metadata()`. If those hooks are unavailable, it falls back
to active `RequestOutput` / request trace metadata.

Bignose/runtime TODO for V7:

- expose real per-request `kv_block_count` / `context_blocks`.
- expose block IDs or block table when available.
- expose target-specific `reusable_tokens_by_target`.
- expose target-specific `reusable_blocks_by_target`.
- keep reuse maps empty when target-specific reuse cannot be proven.
- do not estimate `context_blocks` from token count.

Safe conservative values:

```text
context_blocks = 0
reusable_tokens_by_target = {}
reusable_blocks_by_target = {}
```

With only conservative values, V7 can validate live planning and token/prefix
warmup. It should not be reported as true low-cost KV context migration latency
speedup.

## V8 Stateful Restore Boundary

CPY has implemented the recovery control path:

```text
request fails on preempted/dead instance
-> capture/export inference state when available
-> ask target backend whether state restore is supported
-> call restore_inference_state()
-> fall back to token replay / retry when restore is unsupported
-> emit state_restore metrics
```

The dummy backend benchmark validates token-level state:

```text
state_kind = token_snapshot
tokens = generated/prompt token snapshot
completed_tokens = restored token progress
```

That dummy success is not vLLM KV cache restore.

Bignose/runtime TODO for V8 true KV restore:

- export real restorable KV state from the source runtime.
- return `supports_restore=true` only for restorable state.
- return `state_kind=vllm_kv_snapshot`.
- return a non-empty `runtime_state` such as a snapshot handle, lease, or
  connector transfer payload.
- attach or rebind that KV state on the target runtime.
- return `restored=true` only after attach succeeds.
- return `restored_blocks > 0` when KV blocks were restored.
- validate model, parallelism, cache block size, dtype, layout, device placement,
  same-node support, and cross-node support before claiming success.

Safe fallback when true KV restore is unavailable:

```text
state_kind = token_snapshot
supports_restore = false
reason = vllm_kv_restore_not_available or vllm_kv_export_failed
```

Capability flags must only become true when the runtime hooks are real:

```text
VllmBackend.supports_state_restore() -> true
BackendCapability.supports_state_restore -> true
exported state supports_restore -> true
```

If the environment only supports token replay, prefix warmup, or
`resume_kv_cache()`, keep restore capability false.

## V9 Risk-aware Scheduling Boundary

CPY has implemented:

```text
worker node metadata
-> optional backend actor get_runtime_metadata()
-> node_risk_score()
-> risk-aware ranking
-> scheduler allocation decision
-> risk_aware_scheduling metrics
```

The scheduler can use config/synthetic metadata today. It can also query backend
actors for runtime metadata when that path is enabled.

Bignose/runtime TODO for V9:

- expose `VllmBackend.get_runtime_metadata(instance_id, node_id)` fields that
  match CPY's risk-aware scheduler shape.
- provide accurate GPU/resource profile when runtime can know it.
- provide loading cost / model load time when known.
- provide spot risk / remaining lifetime only when a real predictor or provider
  integration exists.
- keep unknown risk/lifetime conservative instead of fabricating values.

Useful metadata fields:

```text
node_id
instance_id
backend
model_name
total_gpu
free_gpu
gpu_utilization
loading_cost
load_time_s
spot_risk
remaining_lifetime_s
```

If no cloud provider or risk predictor is available, CPY can continue using
`scheduler_config.node_risk` or synthetic benchmark metadata. Bignose does not
need to implement scheduler ranking.

## Runtime Data Stability

Do not treat all runtime metadata as fixed. The data has three lifetimes:

| Data layer | Examples | When it changes | Rule |
|---|---|---|---|
| Engine lifetime | model/revision, TP/PP, cache group/block size, dtype, layout, worker rank/device, connector | engine restart, worker replacement, reconfiguration | collect after cache initialization; do not guess from unresolved config |
| Request/step dynamic | tokens, status, computed tokens, block IDs/table/count, `kv_transfer_params` | every decode/preemption/reallocation/free | query live by request ID |
| Transfer dynamic | snapshot/lease handle, block pin status, handle expiry, connector health, target reachability | every export/transfer | never hard-code; validate at export/restore time |

Block IDs only have meaning inside the allocator and current request ownership.
After preemption, free, or engine restart, the same ID may refer to different
state. Do not write one execution's block table into a static config.

## Required Runtime Hook Behavior

### Per-request Metadata

Runtime metadata should expose, per active request:

```text
request_id
prompt tokens
generated tokens
completed_tokens / computed_tokens
kv_block_count / context_blocks
block_ids or kv_block_ids
block_table
cache block size
cache dtype
cache layout / engine identifier
source node id
source device ids
sequence id / sequence group id if needed for restore
```

### Export

Expected successful export shape:

```json
{
  "request_id": "req-1",
  "state_kind": "vllm_kv_snapshot",
  "supports_restore": true,
  "runtime_state": {
    "handle": "opaque-runtime-owned-handle"
  },
  "metadata": {
    "cache_engine": "vllm",
    "kv_block_count": 2,
    "block_ids": [10, 11],
    "block_table": {"req-1": [10, 11]},
    "can_restore_same_node": true,
    "can_restore_cross_node": false
  }
}
```

If export cannot create a real restorable handle, return fallback metadata with
`supports_restore=false`.

### Restore

Expected successful restore result:

```json
{
  "restored": true,
  "state_kind": "vllm_kv_snapshot",
  "recovered_tokens": 128,
  "restored_blocks": 8,
  "restore_scope": "same_node"
}
```

Expected failure result:

```json
{
  "restored": false,
  "reason": "incompatible_cache_config"
}
```

Same-node and cross-node support must be explicit:

```text
can_restore_same_node
can_restore_cross_node
```

Same-node restore may be possible before cross-node transfer. Cross-node restore
must remain disabled unless the connector really provides transport.

## Validation Checklist

Minimum tests or live checks before claiming vLLM true KV restore:

- active vLLM request exposes real non-zero KV block metadata when blocks exist.
- `get_context_metadata()` returns real `context_blocks` from runtime data.
- target-specific reusable maps are non-empty only with evidence.
- `export_inference_state()` returns `supports_restore=true` only with non-empty
  `runtime_state`.
- `restore_inference_state()` returns `restored=true` only after target attach.
- incompatible cache config returns `restored=false`.
- unsupported cross-node restore returns `cross_node_restore_unsupported`.
- router stateful recovery uses `restore_state` without token replay fallback.
- benchmark summary shows `state_restore_successes_total > 0`.
- benchmark summary shows `state_restore_fallback_count = 0` for the restore
  path being claimed.
- restored vLLM path reports `restored_blocks > 0`.

Useful validation commands depend on the deployment, but the expected signals
are:

```text
context_migration_events > 0
context_migration_reusable_context_blocks > 0   # only when true reuse exists
kv_cache_migration_successes > 0                # warmup path only
state_restore_successes_total > 0               # restore path
state_restore_fallback_count = 0                # no fallback for true restore
```

## Reporting Guidance

Safe claims:

- V6 ServerlessLLM planner can select a new `ParallelPlan` after spot events.
- V7 ServerlessLLM router can call vLLM backend metadata and plan context
  migration.
- V8 ServerlessLLM recovery path supports backend state export/restore and safe
  fallback.
- V9 ServerlessLLM scheduler can rank nodes with risk metadata.
- Dummy benchmarks validate control-plane behavior and token-level recovery.

Do not claim until runtime validation exists:

- vLLM runtime dynamically reapplies V6 `ParallelPlan`.
- V7 achieves true low-cost KV block context migration.
- V8 restores real vLLM KV cache state.
- V9 uses production cloud spot-risk prediction.
- latency improvements from KV migration/restore, unless the benchmark shows
  real runtime hooks, no fallback, and restored/reused KV blocks.

## Definition Of Done

Bignose runtime work is done when:

- V6 executor applies selected `ParallelPlan` to vLLM workers, or the limitation
  is explicitly out of scope.
- V7 live metadata includes real block counts and target-specific reuse when the
  runtime can prove reuse.
- V8 export returns non-empty `runtime_state` and restore attaches it on a
  compatible target.
- V9 metadata exposes real provider/runtime risk fields when available and
  conservative defaults otherwise.
- same-node and cross-node capabilities are reported separately.
- capability flags are true only for actually working runtime paths.
- live GPU benchmark results confirm the claimed path without fallback.
