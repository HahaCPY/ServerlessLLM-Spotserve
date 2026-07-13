# SpotServe Version 7: Low-cost Context Migration Planner

Version 7 implements the CPY control-plane side of SpotServe low-cost context
migration:

```text
context metadata
-> cost matrix
-> fixed-warmup-cost assignment
-> MigrationPlan
-> migration metrics
```

This version does not implement true KV cache migration. It produces the
mapping decision and consumes conservative backend metadata; true KV transfer
still requires a later vLLM/runtime implementation.

## Scope

Implemented:

- `ContextMetadata`
- `MigrationTarget`
- `MigrationPlan`
- cost estimation using reusable tokens / reusable context blocks
- target capacity handling
- minimum-cost assignment with fixed per-target warmup cost
- unassigned-context reporting when target capacity is insufficient
- context migration metrics
- benchmark analyzer/report fields for migration metrics
- synthetic context migration benchmark script and input
- live router spot-event planning path

Out of scope:

- true KV cache export
- true KV cache transfer
- true KV cache restore
- vLLM scheduler / block manager modification
- MoE expert migration
- production request resume

## Planner

File:

```text
sllm/spot/context_migration.py
```

The planner takes source contexts:

```python
@dataclass(frozen=True)
class ContextMetadata:
    request_id: str | None
    instance_id: str
    node_id: str
    num_tokens: int = 0
    context_blocks: int = 0
    reusable_tokens_by_target: Mapping[str, int] = field(default_factory=dict)
    reusable_blocks_by_target: Mapping[str, int] = field(default_factory=dict)
```

and migration targets:

```python
@dataclass(frozen=True)
class MigrationTarget:
    instance_id: str
    node_id: str
    capacity: int = 1
    warmup_cost: float = 0.0
```

It returns:

```python
@dataclass(frozen=True)
class MigrationPlan:
    request_id: str | None
    old_instance_id: str
    new_instance_id: str
    old_node_id: str
    new_node_id: str
    estimated_cost: float
    reusable_tokens: int = 0
    reusable_context_blocks: int = 0
    reason: str = "low_cost_mapping"
```

## Cost Model

The default cost model is:

```text
base_migration_cost
+ non_reusable_tokens * token_transfer_cost
+ non_reusable_blocks * context_block_transfer_cost
+ target warmup_cost once per used target
+ cross_node_penalty when old_node_id != new_node_id
```

`warmup_cost` is charged once when a target is used, not once per assignment.
For example, if two requests migrate to the same target with `capacity=2`, that
target contributes one warmup cost to the total estimate.

If backend metadata has target-specific reuse information, the planner uses it:

```text
reusable_tokens_by_target[target_instance_id or target_node_id]
reusable_blocks_by_target[target_instance_id or target_node_id]
```

If target-specific reuse is not available, the planner falls back to configured
same-node and cross-node reuse ratios.

## Assignment

The planner respects target capacity:

```text
target capacity=2 -> up to two source contexts can use that target
```

It then solves a minimum-cost assignment where target warmup is a fixed opening
cost. If there are more source contexts than target capacity, the excess
contexts are reported as unassigned.

This is intentionally not a plain Hungarian/KM assignment when `warmup_cost` is
non-zero, because warmup is a fixed cost for opening a target rather than an
independent per-source edge cost.

`MigrationDecision.cost_matrix` records the marginal source-to-target cost.
Fixed target warmup is reflected in `estimated_cost` and
`total_estimated_cost`.

The output is a `MigrationDecision` containing:

```text
action
plans
unassigned_contexts
total_estimated_cost
total_reusable_tokens
total_context_tokens
total_reusable_context_blocks
total_context_blocks
reuse_ratio
cost_matrix
```

## Metrics

Files:

```text
sllm/spot/metrics.py
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

Version 7 adds `type=context_migration` metrics:

```json
{
  "type": "context_migration",
  "model": "context-migration-synthetic",
  "action": "migrate",
  "migration_plan_count": 3,
  "unassigned_context_count": 0,
  "total_estimated_cost": 27.0,
  "total_reusable_tokens": 272,
  "total_context_tokens": 288,
  "total_reusable_context_blocks": 17,
  "total_context_blocks": 18,
  "reuse_ratio": 0.9444
}
```

Benchmark summaries now include:

```text
context_migration_events
context_migration_plan_count
context_migration_unassigned_count
context_migration_total_estimated_cost
context_migration_avg_estimated_cost
context_migration_reusable_tokens
context_migration_reusable_context_blocks
context_migration_reuse_ratio
context_migration_latest_plans
kv_cache_migration_attempts
kv_cache_migration_successes
kv_cache_migration_tokens
kv_cache_migration_latest
```

## Live Router Path

File:

```text
sllm/routers/roundrobin_router.py
```

The router can now opt into live V7 planning with:

```json
{
  "router_config": {
    "enable_context_migration": true,
    "enable_kv_cache_migration": true,
    "context_migration_config": {
      "target_warmup_cost": 1.0,
      "planner_config": {
        "cross_node_penalty": 10.0
      }
    }
  }
}
```

When a spot preemption or dead-node event is handled, the router:

```text
affected backend instances
-> get_context_metadata(instance_id, node_id)
-> ContextMetadata
-> READY inference instances as MigrationTarget
-> plan_low_cost_migration()
-> optional target resume_kv_cache() cache warmup
-> context_migration metric
-> context_migration decision in event response
```

Target capacity defaults to:

```text
instance.max_queue_length - instance.concurrency
```

It can be overridden for test or synthetic runs with:

```json
{
  "context_migration_config": {
    "target_capacity": 2
  }
}
```

Warmup cost is configured per target, not per assignment. The router accepts:

```text
warmup_cost_by_instance[instance_id]
warmup_cost_by_node[node_id]
default_target_warmup_cost
target_warmup_cost
```

By default this live path is planning-only. It does not move KV cache bytes or
resume production requests by itself. The backend/runtime still needs a true KV
export, transfer, and restore executor before V7 can claim real serving-latency
gains.

When `enable_kv_cache_migration=true`, the router also executes a conservative
cache-warmup path after a migration plan is produced:

```text
source backend get_current_tokens()
-> target backend resume_kv_cache(request_datas=tokens)
-> kv_cache_migration result attached to context_migration decision/metric
```

For vLLM this uses the existing backend `resume_kv_cache()` hook, which
replays/prefills token batches to warm target cache state. This is useful for
control-plane validation and prefix/cache warmup, but it is not true vLLM KV
block serialization, transfer, or request binding.

## Synthetic Benchmark

Files:

```text
benchmarks/spotserve/context_migration_synthetic.json
scripts/run_context_migration_benchmark.py
```

Run:

```bash
python scripts/run_context_migration_benchmark.py \
  --input benchmarks/spotserve/context_migration_synthetic.json \
  --output-dir results/spotserve_context_migration
```

Outputs:

```text
results/spotserve_context_migration/migration_plan.json
results/spotserve_context_migration/migration_metrics.jsonl
results/spotserve_context_migration/summary.json
```

This synthetic benchmark validates the planner and metrics only. It does not
claim real KV cache migration or serving latency improvement.

## Backend Handoff

大鼻 backend-side 提供 vLLM context metadata 的保守第一版：

```text
sllm/backends/vllm_context_metadata.py
Backend.get_context_metadata()
VllmBackend.get_context_metadata()
```

這個 hook 只回傳 CPY `ContextMetadata.from_dict()` 可讀的 payload，不做
matching，也不決定要把 request 搬到哪個 target。CPY planner 仍然使用
`sllm/spot/context_migration.py` 做 assignment。

目前 vLLM 第一版 metadata 的語意是：

```text
request_id = RequestOutput.request_id when available
instance_id = backend caller provided instance id
node_id = backend caller provided node id
num_tokens = prompt token count + generated token count when available
context_blocks = 0 when vLLM KV block metadata is not safely exposed
reusable_tokens_by_target = {} unless backend can prove target-specific reuse
reusable_blocks_by_target = {} unless backend can prove target-specific reuse
```

因此 V7 backend 目前支援 token-level estimated migration input，但不宣稱 true
KV cache migration、KV block transfer、或 production request resume。若之後
vLLM 能安全 expose block table / KV cache metadata，只需要擴充這個 helper
輸出的 `context_blocks` 和 reusable maps，CPY assignment algorithm 不需要改。

Backend state export / restore capability 由後續 state metadata hook 明確回報；
在 V7 context metadata 中，未知或未驗證的 KV restore 不會被假設為可用。

完整 backend handoff 細節記在：

```text
docs/Bignose-milestone3-backend-capability.md
```

Because backend context metadata now exists, CPY can feed real vLLM/MoE context
metadata into the planner without changing the assignment algorithm. The live
router path performs that planner integration for spot preemption and dead-node
events.

## Definition Of Done

Version 7 CPY side is complete when:

- source context metadata can be represented.
- target migration capacity can be represented.
- a cost matrix can be built.
- minimum-cost worker mapping is produced with per-target warmup charged once.
- unassigned contexts are reported.
- migration metrics are emitted and summarized.
- synthetic benchmark can produce a migration plan and summary.
- router spot-event path can collect live backend context metadata and return a
  context migration decision.
- tests validate cost, assignment, target capacity, per-target warmup,
  unassigned contexts, live router planning, and metric shape.
- Bignose backend hook can provide conservative vLLM `ContextMetadata` payloads
  from active request traces without changing CPY router/scheduler/controller
  main flow.
