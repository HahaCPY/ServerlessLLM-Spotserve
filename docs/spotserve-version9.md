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
come from config or synthetic benchmark input.

## Scope

Implemented:

- `NodeRiskScore`
- `SchedulingDecision`
- risk-aware node ranking
- opt-in FCFS scheduler ranking with `enable_spot_risk_aware`
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
risk_scheduling_latest_decision
```

## Backend Handoff

大鼻 does not need to implement the scheduler decision. Backend/runtime work is
only to expose better metadata later:

```text
model loading cost
GPU usage / free capacity
model resource profile
optional spot risk / lifetime estimate if available
```

If those fields are unavailable, CPY can still run with conservative defaults
or synthetic metadata.

## Definition Of Done

Version 9 CPY side is complete when:

- risk-aware score and ranking are implemented.
- unhealthy nodes are still filtered out.
- scheduler can opt into risk-aware ranking.
- health-only and risk-aware synthetic benchmark can be compared.
- metrics/report fields expose selected risk and latest decision.
- backend handoff is documented without claiming a real cloud risk predictor.
