# SpotServe First-version Implementation Notes

## Goal

This first version turns the ServerlessLLM control plane into a SpotServe-style
prototype without touching vLLM internals, MoE expert dispatch, CUDA kernels, or
the scheduler placement policy.

The vLLM worker is still treated as a black-box backend. CPY's changes focus on:

- worker state
- preemption-aware routing
- synthetic spot trace replay
- bounded retry
- best-effort generated-token replay
- metrics
- benchmark/report workflow

## Scope

Implemented in v1:

- single-model / multi-replica control-plane behavior
- `PREEMPTING`, `DRAINING`, and `DEAD` worker states
- router avoids assigning new requests to unhealthy instances
- controller-level preemption broadcast
- synthetic trace reader and trace simulator
- `none`, `naive_retry`, and `generated_token_replay` policies
- JSONL metrics writer
- dummy backend benchmark configs
- benchmark runner, analyzer, and HTML/Markdown report generator

Not implemented in v1:

- true cloud spot provider integration
- spot-aware scheduler placement
- true KV cache migration
- vLLM/MoE internal modification
- expert-aware scheduling
- production dashboard

## Main Code Changes

### Worker State

File:

```text
sllm/utils.py
```

Added `InstanceState`:

```python
STARTING
READY
BUSY
DRAINING
PREEMPTING
DEAD
```

Added helper methods on `InstanceHandle`:

- `can_accept_request()`
- `mark_ready()`
- `mark_draining()`
- `mark_preempting()`
- `mark_dead()`

Important behavior:

- `BUSY` does not automatically mean the worker cannot accept requests.
- Routing uses `can_accept_request()` so concurrency and state are checked
  together.
- `PREEMPTING`, `DRAINING`, and `DEAD` workers do not receive new requests.
- `mark_ready()` does not revive an instance that is already `PREEMPTING`,
  `DRAINING`, or `DEAD`.

### Router

File:

```text
sllm/routers/roundrobin_router.py
```

Added:

- `handle_preemption()`
- `handle_dead()`
- `mark_instance_preempting()`
- `mark_instance_dead()`
- `drain_instance()`
- `get_instance_states()`
- retry-aware `inference()`

Routing change:

```python
await instance.can_accept_request()
```

is now the only way to decide whether an instance can take a new request.

Recovery policies:

```json
{
  "router_config": {
    "recovery_policy": "none | naive_retry | generated_token_replay",
    "max_retries": 2
  }
}
```

Policy behavior:

- `none`: return the first result/error.
- `naive_retry`: retry failed/preempted requests on another instance.
- `generated_token_replay`: best-effort replay using current/generated tokens,
  then fallback to naive retry if tokens/resume are unavailable.

Generated-token replay is best-effort only. It is not true KV cache migration.

### Controller

File:

```text
sllm/controller.py
```

Added:

- `handle_preemption()`
- `handle_instance_dead()`

These methods broadcast node-level or instance-level events to matching
per-model routers.

### Spot Modules

New directory:

```text
sllm/spot/
```

Files:

- `trace_reader.py`
- `preemption_simulator.py`
- `recovery_policy.py`
- `metrics.py`

Trace reader supports JSONL:

```jsonl
{"time": 5.0, "event": "preempt", "node_id": "0"}
{"time": 20.0, "event": "preempt", "instance_id": "model_instance"}
{"time": 30.0, "event": "dead", "node_id": "0"}
```

Supported events:

- `preempt`
- `dead`
- `recover`

In v1, `recover` is parsed but not dispatched.

### Backend Compatibility

File:

```text
sllm/backends/dummy_backend.py
```

Updated `DummyBackend.__init__()` to accept the same constructor shape used by
`start_instance()`:

```python
DummyBackend(model_name, backend_config)
```

This lets the dummy backend run benchmark/control-plane smoke tests without GPU.

Dummy registration and dummy instance startup also bypass model store download
and worker-node scheduler allocation, so the dummy benchmark can run on the head
node only.

### Docker Packaging

File:

```text
Dockerfile
```

Added:

```dockerfile
COPY sllm/spot /app/sllm/spot
```

This makes `python -m sllm.spot.preemption_simulator` available inside the
container.

## Example Configs

Directory:

```text
examples/spotserve/
```

Added:

- `config-dummy-none.json`
- `config-dummy-naive-retry.json`
- `config-dummy-token-replay.json`
- `config-vllm-spot.json`
- `spot_trace_sample.jsonl`
- `README.md`

Dummy configs use separate model names:

- `dummy-none`
- `dummy-naive-retry`
- `dummy-token-replay`

This allows benchmark comparison across policies in one run.

## Benchmark Workflow

Directory:

```text
benchmarks/spotserve/
```

Added:

- `run_benchmark.py`
- `benchmark_matrix.yaml`
- `workloads/steady_low.jsonl`
- `workloads/steady_high.jsonl`
- `workloads/burst.jsonl`
- `README.md`

Scripts:

```text
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

Generated outputs:

```text
results/spotserve/<run_id>/
  run_metadata.json
  raw_requests.jsonl
  summary.json
  summary.csv
  report.html
  report.md
```

## How To Run

### 1. Start ServerlessLLM

```bash
export MODEL_FOLDER=/path/to/models
docker compose up -d --build
```

Check status:

```bash
docker compose exec -T sllm_head /opt/venvs/head/bin/ray status
docker compose exec -T sllm_head /opt/venvs/head/bin/sllm status
```

### 2. Deploy Dummy Policies

From the host:

```bash
curl http://127.0.0.1:8344/register \
  -H "Content-Type: application/json" \
  --data-binary @examples/spotserve/config-dummy-none.json

curl http://127.0.0.1:8344/register \
  -H "Content-Type: application/json" \
  --data-binary @examples/spotserve/config-dummy-naive-retry.json

curl http://127.0.0.1:8344/register \
  -H "Content-Type: application/json" \
  --data-binary @examples/spotserve/config-dummy-token-replay.json
```

### 3. Run Benchmark Without Trace

This validates the request/report pipeline first. The benchmark runner now
generates summary and visual reports automatically:

```bash
python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --endpoint http://127.0.0.1:8344/v1/chat/completions \
  --skip-trace
```

Expected terminal output:

```text
Produced benchmark runs:
results/spotserve/<timestamp>_dummy-no-preemption
  report: results/spotserve/<timestamp>_dummy-no-preemption/report.html
results/spotserve/<timestamp>_dummy-naive-retry
  report: results/spotserve/<timestamp>_dummy-naive-retry/report.html
results/spotserve/<timestamp>_dummy-token-replay
  report: results/spotserve/<timestamp>_dummy-token-replay/report.html

Benchmark summary:
  dummy-no-preemption: successes=2/2, success_rate=100.00%, p95=...
```

Each benchmark matrix entry gets its own run directory. For the default matrix,
that means three directories: no-preemption, naive-retry, and token-replay.

If every request fails, the runner prints:

```text
Warning: every benchmark request failed. Check the deployed model states before trusting these results.
```

That usually means the models are registered in `/v1/models`, but their router
instances are still `STARTING` or the backend actor failed before becoming
`READY`.

### 4. Re-analyze Existing Runs

```bash
for d in results/spotserve/*; do
  [ -d "$d" ] || continue
  python scripts/analyze_spotserve_benchmark.py "$d"
  python scripts/plot_spotserve_benchmark.py "$d"
done
```

This is only needed for older runs, or if you used `--no-report`.

Open:

```text
results/spotserve/<run_id>/report.html
```

### 5. Run Benchmark With Trace Replay

Trace replay needs Ray access, so run it from the head container or another
environment that can connect to the Ray cluster.

Copy benchmark artifacts into the head container:

```bash
docker cp benchmarks/spotserve sllm_head:/app/benchmarks/
docker cp scripts sllm_head:/app/scripts
docker cp examples/spotserve sllm_head:/app/examples/
```

Run inside the head container:

```bash
docker compose exec -T sllm_head /opt/venvs/head/bin/python \
  benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions
```

Copy results back:

```bash
docker cp sllm_head:/app/results/spotserve ./results/spotserve
```

### 6. Full Trace Benchmark Result

A long-form dummy benchmark was run with the sample trace replaying to
completion. The run completed all four trace events and all three policies
completed all workload requests:

```text
dummy-no-preemption: successes=46/46, success_rate=100.00%
dummy-naive-retry: successes=46/46, success_rate=100.00%
dummy-token-replay: successes=46/46, success_rate=100.00%
```

This validates the benchmark and control-plane plumbing for v1: the benchmark
runner can drive requests, the trace simulator can connect to Ray, the trace can
replay to completion, and the routers remain functional across the trace.

This result should not be interpreted as proof that generated-token replay is
faster or better than naive retry. In v1, `recover` is parsed but not dispatched,
and the token-replay run did not show recovered tokens. The run is therefore a
full-trace integration validation, not a final performance comparison.

## Tests And Validation

Added tests:

```text
tests/spotserve_test/test_trace_reader.py
tests/spotserve_test/test_router_state.py
tests/spotserve_test/test_recovery_policy.py
```

Checks run during implementation:

- `python -m compileall ...`
- trace reader / recovery policy smoke test
- benchmark analyzer + report smoke test

Not fully run locally:

- `pytest tests/spotserve_test -q`
- end-to-end Ray benchmark

Reason:

- local Python environment did not have `pytest`
- local Python environment did not have `ray`

## Known Limitations

- `generated_token_replay` is best-effort only.
- vLLM path uses `input_tokens` replay when possible; this is not true KV
  migration.
- `recover` trace event is parsed but not used to restore a worker.
- scheduler placement is not spot-aware in v1.
- dead/preempting nodes are handled at router/instance level, not scheduler
  allocation level.
- benchmark runner assumes the target models are already deployed.

## Next Steps

Recommended next PRs:

1. Run `tests/spotserve_test` inside the project environment with Ray/pytest.
2. Run dummy benchmark with `--skip-trace`.
3. Run dummy benchmark with trace replay from the head container.
4. Validate retry behavior under forced backend failure.
5. Validate generated-token replay with transformers backend.
6. Integrate 大鼻's vLLM + MoE config as black-box backend.
7. Only after v1 is stable, consider scheduler-level spot-aware allocation.
