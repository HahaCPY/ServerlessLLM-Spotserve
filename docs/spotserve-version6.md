# SpotServe Version 6: Dynamic Reparallelization Planner

Version 6 implements the first SpotServe core idea in the ServerlessLLM
control plane:

```text
spot event
-> GPU availability changes
-> replanning heuristic
-> ParallelPlan
-> replanning metrics
-> benchmark report
```

This version produces a backend-independent deployment plan after spot GPU
resources change. It does not modify vLLM internals and does not execute true
MoE repartitioning yet.

## Scope

Implemented:

- shared `ParallelPlan` schema
- GPU availability summary by node state
- candidate TP / DP / PP configuration generation
- target-node selection for the selected plan
- router-side replanning after `preempt`, `recover`, and `dead` events
- JSONL replanning metrics
- benchmark summary fields for replanning decisions
- planner and router tests for the shared interface and metric path

Out of scope:

- vLLM internal repartitioning
- MoE expert placement optimization
- backend capability validation
- CUDA kernels
- KV cache migration
- request-state migration

## Shared Interface

The control plane now exposes a `ParallelPlan`:

```python
@dataclass(frozen=True)
class ParallelPlan:
    model_name: str
    backend: str
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    num_replicas: int = 1
    num_gpus: int = 1
    target_nodes: List[str] = field(default_factory=list)
    reason: str = "replan"
```

File:

```text
sllm/spot/reparallelization.py
```

CPY owns producing this plan. Backend owners can later validate whether the
selected TP / DP / PP / EP shape is executable by vLLM or a MoE runtime.

For now, `expert_parallel_size` defaults to `1`. Version 6 is a decision-layer
implementation, not an expert-aware runtime implementation.

## Planner

Files:

```text
sllm/spot/reparallelization.py
sllm/spot/__init__.py
```

The planner tracks GPU availability using node states:

```text
ready
preempting
dead
```

Only `ready` nodes contribute to `available_gpus`. `preempting` and `dead`
nodes remain in the availability summary, but are not selected as target nodes.

Candidate generation considers:

```text
tensor_parallel_size
pipeline_parallel_size
data_parallel_size
```

The selected candidate favors:

1. higher GPU utilization
2. higher data-parallel replica count
3. replica GPU shape close to the configured target
4. fewer unused GPUs

The selected candidate is converted into a `ParallelPlan` with:

```text
num_replicas = data_parallel_size
num_gpus     = selected total GPUs
target_nodes = ready nodes that can cover num_gpus
reason       = "<event>_replan"
```

The planner also keeps the older `selected_*` fields in the returned decision
dict so existing tests, logs, and benchmark tools remain compatible.

## Router Integration

File:

```text
sllm/routers/roundrobin_router.py
```

Router replanning is opt-in:

```json
{
  "router_config": {
    "enable_reparallelization": true,
    "reparallelization_config": {
      "model_gpu_requirement": 2,
      "max_tensor_parallel_size": 4,
      "max_pipeline_parallel_size": 2,
      "min_data_parallel_size": 1
    }
  }
}
```

When enabled, the router replans after:

```text
handle_preemption()
handle_recover()
handle_dead()
```

The event handler still returns the original instance-state result, but now
also includes:

```json
{
  "reparallelization": {
    "action": "reparallelize",
    "parallel_plan": {
      "model_name": "dummy-reparallelization",
      "backend": "dummy",
      "tensor_parallel_size": 2,
      "data_parallel_size": 1,
      "pipeline_parallel_size": 1,
      "expert_parallel_size": 1,
      "num_replicas": 1,
      "num_gpus": 2,
      "target_nodes": ["1"],
      "reason": "preempt_replan"
    }
  }
}
```

For synthetic benchmarks, the router can use configured worker nodes:

```json
"synthetic_worker_nodes": {
  "0": {
    "ray_node_id": "synthetic-node-0",
    "address": "synthetic-0",
    "free_gpu": 2,
    "total_gpu": 2,
    "state": "ready"
  },
  "1": {
    "ray_node_id": "synthetic-node-1",
    "address": "synthetic-1",
    "free_gpu": 2,
    "total_gpu": 2,
    "state": "ready"
  }
}
```

If synthetic nodes are not provided, the router builds a best-effort node
snapshot from current instance metadata.

## Metrics

Files:

```text
sllm/spot/metrics.py
scripts/analyze_spotserve_benchmark.py
```

Each replanning decision emits a JSONL metric:

```json
{
  "type": "reparallelization",
  "model": "dummy-reparallelization",
  "event": "preempt",
  "action": "reparallelize",
  "available_gpus": 2,
  "unavailable_gpus": 2,
  "candidate_count": 3,
  "selected_total_gpus": 2,
  "selected_tensor_parallel_size": 2,
  "selected_pipeline_parallel_size": 1,
  "selected_data_parallel_size": 1,
  "target_nodes": ["1"],
  "parallel_plan": {
    "model_name": "dummy-reparallelization",
    "backend": "dummy",
    "tensor_parallel_size": 2,
    "data_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 1,
    "num_replicas": 1,
    "num_gpus": 2,
    "target_nodes": ["1"],
    "reason": "preempt_replan"
  }
}
```

Benchmark summaries now include:

```text
replanning_events
replanning_no_capacity_events
replanning_max_selected_gpus
replanning_latest_plan
```

## Benchmark

Files:

```text
examples/spotserve/config-dummy-reparallelization.json
examples/spotserve/spot_trace_reparallelization.jsonl
benchmarks/spotserve/benchmark_matrix_reparallelization.yaml
```

The synthetic trace is:

```json
{"time": 1.0, "event": "preempt", "node_id": "0", "model_name": "dummy-reparallelization"}
{"time": 3.0, "event": "recover", "node_id": "0", "model_name": "dummy-reparallelization"}
{"time": 5.0, "event": "dead", "node_id": "1", "model_name": "dummy-reparallelization"}
```

Expected planner behavior:

- after `preempt(node=0)`, node 0 is unavailable and the plan targets node 1
- after `recover(node=0)`, both nodes are available again
- after `dead(node=1)`, node 1 is unavailable and the plan targets node 0
- if no ready GPU remains, action becomes `no_capacity`

## How To Run

Prepare the SpotServe environment and copy benchmark artifacts:

```bash
scripts/prepare_spotserve.sh --skip-deploy
```

Then deploy the dummy replanning config:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/sllm deploy \
  --config examples/spotserve/config-dummy-reparallelization.json
'
```

Then run the replanning benchmark:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30
'
```

The matrix writes router metrics to:

```text
/tmp/spotserve-work/results/spotserve_reparallelization/dummy-reparallelization-router.jsonl
```

The combined summary is written under:

```text
results/spotserve_reparallelization/latest_summary.json
results/spotserve_reparallelization/latest_summary.csv
```

## Tests

Relevant tests:

```text
tests/spotserve_test/test_reparallelization_planner.py
tests/spotserve_test/test_router_state.py
tests/spotserve_test/test_scheduler_node_health.py
```

The planner tests cover:

- GPU availability after preempt and recover
- candidate selection
- `ParallelPlan` serialization
- `parallel_plan` content after GPU loss

The router tests cover:

- preempting instances stop accepting new requests
- dead instances are not revived
- recover only revives preempting instances
- replanning decisions are written to router metrics

## Interpretation

Version 6 proves that ServerlessLLM can make a new deployment decision after
spot GPU availability changes. It does not prove that a backend can execute the
selected plan.

For vLLM / MoE integration, backend work still needs to confirm:

- which TP / DP / PP / EP configurations are supported
- maximum GPU count per model
- whether state export and restore are supported
- whether a selected `ParallelPlan` is legal for a specific backend

That later backend contract should be represented by `BackendCapability` and
used to filter planner candidates before a plan is accepted.

## Definition Of Done

Version 6 is complete when:

- spot events update GPU availability used by the planner
- the planner generates a `ParallelPlan`
- router spot handlers return the replanning decision
- replanning metrics are written to JSONL
- benchmark summaries include replanning fields
- tests validate the planner and router metric path
