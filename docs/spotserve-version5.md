# SpotServe Version 5: Dynamic Reparallelization Planner

Version 5 begins implementing SpotServe-style dynamic reparallelization at the
planning layer. It reacts to spot preempt / recover / dead events by generating
a new parallel deployment configuration from the GPUs that remain available.

## Scope

Implemented:

- GPU availability summary from scheduler worker-node state.
- Parallel configuration schema:
  - tensor parallel size
  - pipeline parallel size
  - data parallel size
  - selected GPU count
  - unused GPU count
- Candidate configuration generation.
- Replanning heuristic that prefers:
  - maximum GPU utilization,
  - then higher data parallelism,
  - then a per-replica GPU shape close to the current model requirement.
- Replanning metrics:
  - `replanning_events`
  - `replanning_successes`
  - `replanning_no_capacity`
  - `replanning_available_gpus_min`
  - `replanning_selected_gpus_last`
  - `replanning_selected_tp_last`
  - `replanning_selected_pp_last`
  - `replanning_selected_dp_last`
- Synthetic replanning benchmark:
  - `examples/spotserve/config-dummy-reparallelization.json`
  - `examples/spotserve/spot_trace_reparallelization.jsonl`
  - `benchmarks/spotserve/benchmark_matrix_reparallelization.yaml`

Not implemented in Version 5:

- actual worker rebuild,
- actual vLLM engine restart with the new plan,
- MoE-specific expert optimization,
- KV cache or context migration.

## Design

The planner lives in:

```text
sllm/spot/reparallelization.py
```

On a spot event, the controller:

```text
spot event
  -> scheduler node health update
  -> router instance state update
  -> dynamic replanning planner
  -> JSONL metrics
```

Models opt into planner metrics with:

```json
"router_config": {
  "enable_reparallelization": true,
  "reparallelization_config": { ... }
}
```

For the synthetic benchmark, `reparallelization_config.synthetic_worker_nodes`
provides a 4-GPU planner view, so the planner can be validated without a real
multi-GPU worker rebuild.

## Run

Prepare:

```bash
scripts/prepare_spotserve.sh --deploy-set reparallelization
```

Run:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30
'
```

Expected summary shape:

```text
dummy-reparallelization-planner: successes=2/2, ..., replans=3,
available_gpus_min=2, selected_gpus_last=2, tp/pp/dp=2/1/1
```

Exact latency is not important. The important checks are that trace replay
finishes, replanning events are recorded, and the selected plan changes when
available GPUs change.

## Interpretation

This version proves that the control plane can compute a new deployment plan
after GPU availability changes. It does not prove that workers are rebuilt or
that vLLM accepts a new parallel configuration online. Those belong to later
versions.
