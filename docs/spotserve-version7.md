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
mapping decision that a later backend/runtime implementation can execute.

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
```

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

大鼻 needs to provide backend/runtime metadata later:

```text
active request ids
instance id
node id
token count
context block count
reusable tokens / blocks per target
state export support
state restore support
```

The backend handoff is documented in:

```text
docs/Bignose-milestone3-backend-capability.md
```

Once that metadata exists, CPY can replace synthetic context input with real
vLLM/MoE context metadata without changing the assignment algorithm.

## Definition Of Done

Version 7 CPY side is complete when:

- source context metadata can be represented.
- target migration capacity can be represented.
- a cost matrix can be built.
- minimum-cost worker mapping is produced with per-target warmup charged once.
- unassigned contexts are reported.
- migration metrics are emitted and summarized.
- synthetic benchmark can produce a migration plan and summary.
- tests validate cost, assignment, target capacity, per-target warmup,
  unassigned contexts, and metric shape.
