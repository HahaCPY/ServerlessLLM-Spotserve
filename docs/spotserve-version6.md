# SpotServe Version 6: Dynamic Reparallelization Planner

Version 6 implements the first SpotServe core idea in the ServerlessLLM
control plane:

```text
spot event
-> GPU availability changes
-> replanning heuristic
-> ParallelPlan
-> vLLM deployment adapter
-> replanning metrics
-> benchmark report
```

This version produces a backend-independent deployment plan after spot GPU
resources change. For `backend=vllm`, the selected plan is handed to the vLLM
deployment adapter, which rebuilds target Ray actors and switches router traffic
after readiness checks pass. It does not modify vLLM internals or execute
in-place MoE expert repartitioning.

## Scope

Implemented:

- shared `ParallelPlan` schema
- GPU availability summary by node state
- candidate TP / DP / PP configuration generation
- target-node selection for the selected plan
- router-side replanning after `preempt`, `recover`, and `dead` events
- JSONL replanning metrics
- benchmark summary fields for replanning decisions and execution status
- vLLM deployment adapter path for applying a selected `ParallelPlan`
- planner and router tests for the shared interface and metric path

Out of scope:

- in-place vLLM internal repartitioning
- MoE expert placement optimization
- CUDA kernels
- KV cache migration during V6 replan
- request-state migration during V6 replan

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

CPY owns producing this plan. Backend capability metadata and the deployment
executor validate whether the selected TP / DP / PP / EP shape is executable by
vLLM or another runtime.

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
  "model": "vllm-reparallelization-gpu-smoke",
  "backend": "vllm",
  "router_config": {
    "enable_reparallelization": true,
    "reparallelization_config": {
      "target_replica_gpus": 1,
      "max_tensor_parallel_size": 1,
      "max_pipeline_parallel_size": 1,
      "drain_timeout_s": 30,
      "allow_stop_before_recreate": true
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
      "model_name": "vllm-reparallelization-gpu-smoke",
      "backend": "vllm",
      "tensor_parallel_size": 1,
      "data_parallel_size": 1,
      "pipeline_parallel_size": 1,
      "expert_parallel_size": 1,
      "num_replicas": 1,
      "num_gpus": 1,
      "target_nodes": ["0"],
      "reason": "preempt_replan"
    },
    "execution": {
      "status": "applied",
      "instance_ids": ["vllm-reparallelization-gpu-smoke_reparallel_..."]
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
  "model": "vllm-reparallelization-gpu-smoke",
  "event": "preempt",
  "action": "reparallelize",
  "available_gpus": 1,
  "unavailable_gpus": 1,
  "candidate_count": 1,
  "selected_total_gpus": 1,
  "selected_tensor_parallel_size": 1,
  "selected_pipeline_parallel_size": 1,
  "selected_data_parallel_size": 1,
  "target_nodes": ["0"],
  "parallel_plan": {
    "model_name": "vllm-reparallelization-gpu-smoke",
    "backend": "vllm",
    "tensor_parallel_size": 1,
    "data_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 1,
    "num_replicas": 1,
    "num_gpus": 1,
    "target_nodes": ["0"],
    "reason": "preempt_replan"
  },
  "execution_status": "applied",
  "execution": {
    "status": "applied",
    "instance_ids": ["vllm-reparallelization-gpu-smoke_reparallel_..."]
  }
}
```

Benchmark summaries now include:

```text
replanning_events
replanning_no_capacity_events
replanning_max_selected_gpus
replanning_latest_plan
replanning_execution_applied
replanning_execution_failed
replanning_latest_execution
```

## Benchmark

Files:

```text
examples/spotserve/config-vllm-reparallelization-gpu-smoke.json
examples/spotserve/spot_trace_vllm_reparallelization.jsonl
benchmarks/spotserve/workloads/reparallelization_vllm_smoke.jsonl
benchmarks/spotserve/benchmark_matrix_reparallelization.yaml
```

The benchmark matrix now uses the vLLM reparallelization smoke model:

```text
vllm-reparallelization-applied: enable_reparallelization=true
```

This run replays a spot event against a deployed vLLM model and expects the
router metrics to report `replanning_execution_applied > 0`. The old dummy
configs remain useful for planner-only tests, but they are no longer the default
V6 benchmark path because dummy cannot execute a selected `ParallelPlan`.

The default smoke model path is `/models/vllm/vllm-dense-baseline`, matching the
local ServerlessLLM-store layout prepared by the dense vLLM benchmark fixtures:

```text
/models/vllm/vllm-dense-baseline/rank_0/tensor.data_0
/models/vllm/vllm-dense-baseline/rank_0/tensor_index.json
```

Because that layout is not a Hugging Face safetensors/bin snapshot, the smoke
config uses the patched vLLM `load_format="serverless_llm"` loader. Override
the defaults when using a different container-local path or a normal Hugging
Face model id:

```bash
export SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/vllm/vllm-dense-baseline
export SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=serverless_llm

# or, for a standard Hugging Face id/snapshot:
export SPOTSERVE_REPARALLELIZATION_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
export SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto
```

The vLLM trace is:

```json
{"time": 1.0, "event": "preempt", "node_id": "1", "model_name": "vllm-reparallelization-gpu-smoke"}
```

Expected planner behavior:

- after `preempt(node=1)`, node 1 is unavailable and the plan targets node 0
- for vLLM, the deployment adapter creates the target actor and switches traffic
- metrics show `execution.status = applied`
- if no ready GPU remains, action becomes `no_capacity`

The smoke config enables `allow_stop_before_recreate=true` because the default
root compose file starts one GPU worker (`sllm_worker_0`). This lets the adapter
release the current actor before recreating it on the same target GPU. Production
or multi-worker deployments should keep the default create-before-stop behavior
unless they explicitly accept that disruptive re-create window.

## How To Run

Prepare the SpotServe environment and copy benchmark artifacts:

```bash
scripts/prepare_spotserve.sh --deploy-set reparallelization
```

If the container is already running and you only need to deploy manually, deploy
the vLLM smoke config:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/sllm deploy \
  --config examples/spotserve/config-vllm-reparallelization-gpu-smoke.json
'
```

Then run the replanning benchmark:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 120
'
```

To compare the V6 applied path against a no-reparallelization baseline, run the
performance matrix:

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

This matrix deploys and compares:

```text
vllm-reparallelization-disabled: enable_reparallelization=false
vllm-reparallelization-applied:  enable_reparallelization=true
```

The applied performance run uses the model alias
`vllm-reparallelization-applied-perf` so its router metrics path is independent
from the correctness smoke model.

The performance workload labels requests by phase:

```text
warmup
pre_replan
replan_window
post_replan
```

Benchmark summaries include phase-specific fields such as:

```text
phase_replan_window_latency_p95_ms
phase_post_replan_latency_p95_ms
phase_post_replan_throughput_req_s
```

The runner also writes:

```text
results/spotserve_reparallelization_performance/latest_comparisons.json
```

On the default root compose setup, only `sllm_worker_0` is a real worker id.
This performance matrix can measure V6 adapter/recreate overhead and
post-replan steady-state behavior. A true latency-improvement or capacity-loss
recovery claim requires at least two real worker nodes, so the baseline can
lose the active node while the applied run moves traffic to another live node.

In the local root-compose validation on 2026-07-23, the performance matrix
passed with both runs at `successes=8/8`; the applied run reported
`replanning_events=1`, `replanning_execution_applied=1`, and
`replanning_execution_failed=0`. The most useful single-worker performance
signal was the post-replan phase (`1044.20ms` baseline p95 vs `1047.13ms`
applied p95), while the replan-window phase captured recreate/model-load
overhead.

The matrix writes router metrics to:

```text
/tmp/spotserve-work/results/spotserve_reparallelization/vllm-reparallelization-router.jsonl
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
tests/spotserve_test/test_reparallelization_executor.py
tests/spotserve_test/test_router_state.py
tests/spotserve_test/test_vllm_deployment_adapter.py
tests/spotserve_test/run_vllm_deployment_adapter_smoke.py
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
- vLLM replanning decisions can be applied by the deployment adapter

## Interpretation

Version 6 proves that ServerlessLLM can make a new deployment decision after
spot GPU availability changes. When backend capability metadata is available,
the planner now filters replanning candidates through `BackendCapability` before
accepting a plan.

The vLLM benchmark now verifies that the selected `ParallelPlan` is consumed by
the vLLM deployment adapter. The meaningful V6 signals are:

```text
replanning_events
replanning_latest_plan
replanning_no_capacity_events
replanning_max_selected_gpus
replanning_execution_applied
replanning_latest_execution
```

This is still a deployment-lifecycle validation, not a production latency
speedup claim. Latency gains require a benchmark that shows request traffic
benefits from the new layout after the plan has been applied.

That backend contract is represented by:

```text
sllm/backends/capability.py
sllm/backends/vllm_capability.py
```

If a backend advertises supported configs and none fit the available GPUs, the
planner returns `no_capacity` instead of falling back to an unconstrained plan.

## Definition Of Done

Version 6 is complete when:

- spot events update GPU availability used by the planner
- the planner generates a `ParallelPlan`
- router spot handlers return the replanning decision
- replanning metrics are written to JSONL
- benchmark summaries include replanning fields
- tests validate the planner and router metric path
- backend capability configs are respected when available
