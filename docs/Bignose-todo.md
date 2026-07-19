# Bignose TODO

This document is the handoff between CPY's ServerlessLLM/SpotServe
control-plane work and Bignose's vLLM/runtime integration work for Milestone 3.

The short version:

```text
V6: CPY planner and the concrete vLLM deployment adapter are done; a live
    container replan has applied a ParallelPlan across two GPU workers.
V7: CPY live context-migration path calls vLLM backend; target-specific reuse is
    derived only when matching runtime prefix evidence is available.
V8: CPY recovery path and same-node NIXL export/attach are validated; GPU
    container deployment smoke is now complete, while cross-node transport
    remains pending.
V9: CPY risk-aware scheduler and the provider metadata adapter are done; real
    production risk quality remains provider/deployment-specific.
```

## Status Matrix

| Version | CPY / ServerlessLLM status | Bignose / runtime status | What can be claimed now |
|---|---|---|---|
| V6 Dynamic Reparallelization | Planner, `ParallelPlan`, spot-event replanning, metrics | `VllmDeploymentAdapter` creates target-node vLLM Ray actors, waits for readiness, switches traffic, drains/stops old actors | Live two-worker GPU container smoke applied TP1/PP1 plan after preemption |
| V7 Low-cost Context Migration | Router calls backend `get_context_metadata()`, plans migration, optional `resume_kv_cache()` warmup | Router derives target-specific maps from matching target token/block metadata; empty when unproven | Live same-host GPU target-reuse benchmarks pass for tiny MoE and Qwen1.5-MoE TP2 |
| V8 Stateful Restore | Router recovery path calls export/restore and falls back safely | Same-node vLLM runtime export/attach and ID/lease tracking validated in dual-engine harness | Tiny and Qwen1.5-MoE TP2 restore pass; GPU container/head/worker/NIXL smoke pass; cross-node validation pending |
| V9 Risk-aware Scheduling | Scheduler can rank by risk and query backend actor metadata | `RiskMetadataProvider` supports callable provider, JSON/env integration, provenance, normalization, and conservative fallback | Real provider-shaped fields are preserved when available; unknown nodes remain conservative |

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

The selected `ParallelPlan` is now consumed by the vLLM deployment adapter. The
adapter rebuilds live actors on the planner-selected target nodes:

```text
ParallelPlan
-> stop / drain old vLLM workers
-> create target vLLM workers with new TP / DP / PP / EP
-> migrate or restore request state when possible
-> switch traffic to the new parallel layout
```

Bignose/runtime implementation:

- ~~implement an executor/controller that consumes selected `ParallelPlan`.~~ `sllm.spot.ReparallelizationExecutor` implements this lifecycle.
- ~~create or reconfigure vLLM workers with the selected TP / DP / PP / EP shape.~~ `VllmDeploymentAdapter` starts real `start_instance` actors with the selected shape.
- ~~drain or stop old workers safely.~~ The adapter drains active requests, then stops and deallocates old actors.
- ~~switch router traffic only after target workers are ready.~~ Router traffic is switched only after readiness checks pass.
- coordinate with V7/V8 if in-flight request state must be migrated/restored.

The live smoke validates the deployment lifecycle. It is a control-plane
replan benchmark, not a claim about end-to-end latency improvement.

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
- ~~expose target-specific `reusable_tokens_by_target`.~~ Router now derives the
  map from an aligned matching token prefix reported by the target runtime.
- ~~expose target-specific `reusable_blocks_by_target`.~~ Router derives the
  block map only when block size, dtype, layout, node, and prefix evidence match.
- ~~keep reuse maps empty when target-specific reuse cannot be proven.~~ This is
  enforced by the conservative target compatibility checks.
- do not estimate `context_blocks` from token count.

Safe conservative values:

```text
context_blocks = 0
reusable_tokens_by_target = {}
reusable_blocks_by_target = {}
```

The live target-reuse harness now proves non-empty maps on real CUDA engines:

```text
Qwen2-MoE-Tiny: reusable_tokens_by_target={vllm-target: 64}
              reusable_blocks_by_target={vllm-target: 4}
              source blocks=5, reuse_ratio=0.80
Qwen1.5-MoE-A2.7B TP2: same target-specific result (64 tokens / 4 blocks)
```

This proves target-specific KV reuse and low-cost planning. It is not a
cross-node latency claim; the measured runs are same-host GPU runs.

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

The real same-node path is now exercised by
`tests/v1/kv_connector/nixl_integration/spotserve_dual_engine_harness.py`.
The harness exports non-empty KV blocks, stages target metadata, completes the
NIXL read, and reports `state_restore_successes_total=1` and
`state_restore_fallback_count=0` for both the tiny model and the Qwen1.5-MoE
TP2 run. This is still a same-host result, not cross-node validation.

Bignose/runtime TODO for V8 true KV restore:

- ~~export real restorable KV state from the source runtime.~~
- ~~return `supports_restore=true` only for restorable state.~~
- ~~return `state_kind=vllm_kv_snapshot`.~~
- ~~return a non-empty `runtime_state` such as a snapshot handle, lease, or
  connector transfer payload.~~
- ~~attach or rebind that KV state on the target runtime.~~
- ~~return `restored=true` only after attach succeeds.~~ Same-node target
  generation now waits for the staged NIXL read to complete.
- ~~return `restored_blocks > 0` when KV blocks were restored.~~
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

The scheduler can use config/synthetic metadata today. `RiskMetadataProvider`
also supports a production callable (`module:callable`) plus JSON/env adapters;
all values are normalized, bounded, and tagged with source/provider/time/
confidence. A conservative provider is used when no authoritative source is
available.

Bignose/runtime implementation:

- ~~expose `VllmBackend.get_runtime_metadata(instance_id, node_id)` fields that
  match CPY's risk-aware scheduler shape.~~ Runtime metadata is merged with
  provider output by the scheduler.
- ~~provide accurate GPU/resource profile when runtime can know it.~~ Existing
  Ray capacity is preserved as authoritative when available.
- ~~provide loading cost / model load time when known.~~ Provider aliases are
  normalized into the scheduler schema.
- ~~keep unknown risk/lifetime conservative instead of fabricating values.~~
  Unknown values use conservative defaults with confidence `0.0`.

The host does not expose a cloud spot-risk service, so production predictor
quality is intentionally not claimed here. A deployment supplies one through
`risk_provider` / `SLLM_RISK_PROVIDER`; its observed quality must be benchmarked
with that provider's own live data.

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

Current validation status (2026-07-19):

- ServerlessLLM router/backend/state-restore suite: 31 passed.
- Same-node tiny dual-engine NIXL harness: passed.
- Same-node Qwen1.5-MoE-A2.7B dual-engine TP2 harness: passed with five source
  blocks, 65 source computed tokens, 5 restored blocks,
  `state_restore_successes_total=1`, and zero fallback.
- Live target-specific reuse (2026-07-19): both real two-engine GPU runs
  returned `reusable_tokens_by_target={"vllm-target": 64}` and
  `reusable_blocks_by_target={"vllm-target": 4}` (`reuse_ratio=0.80`).
- GPU deployment smoke (2026-07-19): image `sha256:93a6f5d8d5ea...` (vLLM
  0.11.2, torch 2.9.0, Ray 2.48.0) ran on `cupid1.inter.lsa` with four RTX
  5070 Ti GPUs (driver 595.80).  Host-network head `/health` returned 200;
  the worker joined Ray with one visible GPU and CUDA was available.  A live
  `/v1/chat/completions` request returned HTTP 200.  The deployed engine log
  confirms NIXL availability, UCX agent creation, CUDA KV-cache registration,
  and NIXL HND layout.  The router request itself had no preemption event, so
  its per-request restore counters correctly stayed at zero; restore success
  is measured by the dual-engine harness above.
- V6 live deployment-adapter smoke (2026-07-19): a real ServerlessLLM head
  plus two GPU workers (each one visible RTX 5070 Ti) deployed the direct-vLLM
  model. A preempt event on worker `1` selected TP1/PP1 target node `0` and
  returned `execution.status="applied"`; the router then reported exactly one
  ready instance on node `0`. The adapter created the target actor, waited for
  readiness, switched traffic, drained, and stopped the old actor.
- V9 provider metadata validation (2026-07-19): callable/config-shaped provider
  fields were normalized and bounded, provenance/confidence were retained, and
  an unknown node (or empty provider response) selected the conservative
  provider (`confidence=0.0`). No cloud provider endpoint is available on this
  host, so live predictor accuracy remains deployment-specific rather than a
  synthetic production claim.
- V9 scheduler integration smoke (2026-07-19): the real ServerlessLLM Python
  environment passed provider metadata through `FcfsScheduler`; node `node-1`
  retained provider risk `0.12`, while unknown `node-x` retained conservative
  provenance and confidence `0.0`.
- Cross-node positive restore is not complete because a second node is not
  available; the capability remains explicitly `can_restore_cross_node=false`.

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

- V6 ServerlessLLM planner and concrete vLLM deployment adapter can apply a new
  `ParallelPlan` after a spot event on a live same-host GPU cluster.
- V7 ServerlessLLM router can call vLLM backend metadata and plan context
  migration.
- V8 ServerlessLLM recovery path supports backend state export/restore and safe
  fallback.
- V9 ServerlessLLM scheduler can rank nodes with risk metadata.
- Dummy benchmarks validate control-plane behavior and token-level recovery.

Do not claim without additional deployment evidence:

- V6 reparallelization latency improvement; the live result validates lifecycle
  correctness, not a production latency SLO.
- V7 achieves true low-cost KV block context migration in a live target-reuse
  benchmark (same-host GPU evidence is complete; cross-node latency remains
  unmeasured).
- V8 restores real vLLM KV cache state through the containerized dual-engine
  NIXL path.  A deployed-router recovery event and cross-node restore still
  require a second live worker/node.
- V9 production cloud spot-risk prediction quality; the provider hook is live,
  but this host has no cloud predictor endpoint.
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

## Completed Core Work

The following Bignose runtime tasks are completed and validated as of
2026-07-19:

- **V6 executor:** `ReparallelizationExecutor` is connected to the concrete
  `VllmDeploymentAdapter`. A live ServerlessLLM GPU smoke applied a selected
  `ParallelPlan` (TP1/PP1) after preemption, created the target vLLM worker,
  switched traffic, drained/stopped the old worker, and served a successful
  inference request.
- **V7 target-specific KV reuse:** live CUDA engines reported real block
  metadata and target-specific reuse:
  `reusable_tokens_by_target={"vllm-target": 64}` and
  `reusable_blocks_by_target={"vllm-target": 4}`. Reuse maps remain empty when
  compatibility evidence is unavailable.
- **V8 same-node KV restore:** source export returned non-empty
  `runtime_state`; target attach completed through NIXL; restored blocks were
  reported with `state_restore_successes_total=1` and
  `state_restore_fallback_count=0` in the validated same-node harness.
- **V9 risk metadata:** callable provider, JSON/file, and environment inputs
  are normalized, bounded, and tagged with provider/source/time/confidence.
  `FcfsScheduler` preserves authoritative Ray capacity and falls back to
  conservative metadata (`confidence=0.0`) when provider data is unavailable.
- **Capability reporting:** same-node and cross-node restore capabilities are
  reported independently. Same-node capability is enabled only on the tested
  path; `can_restore_cross_node` remains false until a real cross-node test
  succeeds.
- **Validation artifacts:** router/backend/state-restore tests, same-node
  tiny and Qwen1.5-MoE dual-engine NIXL harnesses, target-reuse GPU harnesses,
  V6 deployment smoke, and V9 provider/scheduler smoke have passed.

The remaining non-completed items are external deployment validations: a
positive cross-node NIXL restore, production cloud risk-provider quality, and
production latency/SLO benchmarking. These are intentionally not marked as
completed by the runtime implementation.
