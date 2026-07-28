# SpotServe Version 9: Spot-risk-aware Scheduling

Version 9 implements the CPY control-plane side of spot-risk-aware model
placement:

```text
worker node metadata
-> spot risk / remaining lifetime / loading cost score
-> ranked ready nodes
-> allocation decision
-> scheduler benchmark metrics
```

This version does not integrate a real cloud spot provider. Risk metadata can
come from config, synthetic benchmark input, or running backend actor runtime
metadata.

## Scope

Implemented:

- `NodeRiskScore`
- `SchedulingDecision`
- risk-aware node ranking
- opt-in FCFS scheduler ranking with `enable_spot_risk_aware`
- opt-in backend actor runtime metadata query with
  `enable_backend_runtime_metadata`
- scheduler risk metadata preservation across worker-node refreshes
- controller pass-through for `scheduler_config`
- synthetic health-only vs risk-aware scheduler benchmark
- scheduler metrics summary/report fields
- Version 9 tests

Out of scope:

- real cloud spot provider
- autoscaling policy changes
- MoE-aware placement
- production risk prediction model

## Planner

File:

```text
sllm/spot/risk_aware_scheduling.py
```

The planner scores ready nodes with enough GPU capacity:

```text
score =
  risk_weight * spot_risk
+ lifetime_weight * lifetime_penalty
+ loading_cost_weight * loading_penalty
- free_gpu_weight * free_gpu_bonus
```

Lower score is better.

Node metadata fields:

```text
spot_risk
remaining_lifetime_s
loading_cost
free_gpu
total_gpu
state
```

Aliases are accepted:

```text
risk_score / preemption_risk
expected_remaining_lifetime_s
model_loading_cost / load_cost
```

## Scheduler Integration

File:

```text
sllm/schedulers/fcfs_scheduler.py
```

Risk-aware ranking is opt-in:

```json
{
  "scheduler_config": {
    "enable_spot_risk_aware": true,
    "enable_backend_runtime_metadata": true,
    "runtime_metadata_timeout_s": 1.0,
    "metrics_path": "results/spotserve/scheduler-risk.jsonl",
    "risk_weight": 1.0,
    "lifetime_weight": 0.8,
    "loading_cost_weight": 0.6,
    "free_gpu_weight": 0.05,
    "node_risk": {
      "0": {
        "spot_risk": 0.8,
        "remaining_lifetime_s": 300,
        "loading_cost": 8
      }
    }
  }
}
```

When disabled, the FCFS scheduler keeps the old health-only behavior.

## Live Backend Runtime Metadata

File:

```text
sllm/schedulers/fcfs_scheduler.py
```

When `enable_backend_runtime_metadata=true`, the scheduler refresh path uses
currently allocated model instances to query backend actors:

```text
model_instance[model_name][instance_id] -> node_id
ray.get_actor(instance_id)
-> get_runtime_metadata(instance_id, node_id)
-> normalize as risk_metadata_source=backend_runtime
-> merge metadata into worker node info
-> node_risk_score()
```

Backend actors are resolved first in the scheduler's current namespace and then
in the `models` namespace, matching detached model actors created by
`start_instance`. Deployments can override that search order with:

```json
{
  "scheduler_config": {
    "backend_actor_namespaces": [null, "models"]
  }
}
```

The merge is conservative across multiple instances on the same node:

```text
spot_risk: max observed risk
remaining_lifetime_s: min observed lifetime
loading_cost: max observed loading cost
model_resource_profiles: preserved as a list
backend_runtime_metadata: preserved as raw rows
risk_metadata_source / risk_provider / confidence: preserved for metrics
```

Ray worker-node GPU accounting remains authoritative for placement capacity.
If backend runtime metadata reports `free_gpu` or `total_gpu`, those values are
kept as `backend_reported_free_gpu` and `backend_reported_total_gpu`; they do
not overwrite the scheduler's live `free_gpu` count.

Configured `scheduler_config.node_risk[node_id]` still takes precedence. This
keeps synthetic/config benchmarks reproducible while allowing live backend
metadata to fill in missing fields.

If an actor is missing, stale, or does not implement `get_runtime_metadata()`,
the scheduler logs the miss and continues with Ray worker-node metadata plus
configured defaults.

## Benchmark

Files:

```text
benchmarks/spotserve/risk_aware_scheduling_synthetic.json
scripts/run_scheduler_benchmark.py
```

Run:

```bash
python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_test
```

Outputs:

```text
/tmp/spotserve_risk_aware_scheduling_test/summary.json
/tmp/spotserve_risk_aware_scheduling_test/allocations.json
/tmp/spotserve_risk_aware_scheduling_test/scheduler_metrics.jsonl
```

Expected shape:

```text
health_only selects the first ready node with enough GPUs
risk_aware selects the lower-risk / longer-lived ranked node
```

## Metrics

Files:

```text
sllm/spot/metrics.py
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

Version 9 adds `type=risk_aware_scheduling` metrics:

```json
{
  "type": "risk_aware_scheduling",
  "model": "scheduler-risk-model-a",
  "policy": "risk_aware",
  "action": "allocate",
  "selected_node_id": "node-1",
  "selected_spot_risk": 0.1,
  "selected_remaining_lifetime_s": 3200,
  "selected_loading_cost": 15
}
```

Benchmark summaries include:

```text
risk_scheduling_events
risk_scheduling_allocations
risk_scheduling_avg_selected_risk
risk_scheduling_avg_selected_score
risk_scheduling_latest_node
risk_scheduling_latest_metadata_source
risk_scheduling_latest_provider
risk_scheduling_avg_selected_confidence
risk_scheduling_latest_decision
```

## Backend Handoff

大鼻 does not need to implement the scheduler decision. Backend/runtime work is
to expose conservative runtime metadata:

```text
sllm/backends/vllm_runtime_metadata.py
Backend.get_runtime_metadata()
VllmBackend.get_runtime_metadata()
```

The current vLLM metadata includes:

```text
model loading cost from VllmBackend.init_backend()
model resource profile
tensor / pipeline / data parallel sizes
expert parallel enabled flag
GPU usage / free capacity when supplied by runtime metadata
optional spot risk / lifetime estimate when supplied by runtime metadata
```

If spot risk or remaining lifetime are unavailable, the backend omits those
fields instead of fabricating a prediction; CPY can still run with scheduler
defaults, `scheduler_config.node_risk`, or synthetic metadata. Bignose does
not implement scheduler ranking or a real cloud risk predictor in backend code.

The scheduler can now consume those backend fields directly from running backend
actors when `enable_backend_runtime_metadata=true`.

## Definition Of Done

Version 9 CPY side is complete when:

- risk-aware score and ranking are implemented.
- unhealthy nodes are still filtered out.
- scheduler can opt into risk-aware ranking.
- scheduler can opt into backend actor `get_runtime_metadata()` refresh.
- backend runtime rows are normalized with provenance before ranking.
- Ray-reported capacity remains authoritative when backend metadata also
  reports GPU counts.
- health-only and risk-aware synthetic benchmark can be compared.
- metrics/report fields expose selected risk and latest decision.
- backend handoff exposes conservative vLLM runtime metadata without claiming
  a real cloud risk predictor.
