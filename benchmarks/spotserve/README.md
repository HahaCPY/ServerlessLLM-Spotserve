# SpotServe Benchmark Harness

This benchmark harness replays fixed request workloads against a running
ServerlessLLM endpoint and optionally starts a synthetic spot preemption trace.

## Run

Start ServerlessLLM first. The dummy configs can run head-only; they do not need
a worker container or GPU after the latest local code is rebuilt into the image.

Then deploy the dummy policy configs. They use separate model names so one
benchmark run can compare all policies:

```bash
sllm deploy --config examples/spotserve/config-dummy-none.json
sllm deploy --config examples/spotserve/config-dummy-naive-retry.json
sllm deploy --config examples/spotserve/config-dummy-token-replay.json
```

Then run the benchmark:

```bash
python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions
```

The default matrix uses `workloads/smoke.jsonl` and a 15-second request timeout
so broken endpoints fail quickly instead of waiting for several minutes.

The runner checks `/health` and `/v1/models` before sending workload requests.
If any model in `benchmark_matrix.yaml` is missing, the benchmark stops before
writing misleading failed request rows.

If the machine running the benchmark does not have Ray installed, skip automatic
trace replay first:

```bash
python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --skip-trace \
  --request-timeout 5
```

For longer experiments, change `benchmark_matrix.yaml` to use
`steady_low.jsonl`, `steady_high.jsonl`, or `burst.jsonl`.

For recovery correctness validation, deploy the `dummy-correctness-*` configs
and run:

```bash
scripts/prepare_spotserve.sh --deploy-set correctness

python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
```

This matrix forces mid-generation failures and reports request-level
`failed_attempts`, `retry_count`, `recovered_tokens`, and `recovery_fallback`
from router metrics. It also includes `dummy-correctness-stateful-recovery`,
which reports `state_restore_attempts_total`,
`state_restore_successes_total`, and `state_restored_tokens_total`.

On head-only dummy setups, avoid deploying both standard and correctness dummy
model sets at the same time; the extra Ray actors can exceed the container
thread limit.

`scripts/prepare_spotserve.sh` prunes dangling images, build cache, and stopped
SpotServe containers after build/recreate by default so repeated benchmark
iterations do not keep filling local container storage. Use `--no-cleanup` to
preserve cache for debugging.

For dynamic reparallelization validation, deploy only the
reparallelization configs and run:

```bash
scripts/prepare_spotserve.sh --deploy-set reparallelization

python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --request-timeout 120
```

This matrix deploys the vLLM reparallelization smoke model with
`enable_reparallelization=true`. It validates that the selected `ParallelPlan`
is consumed by the vLLM deployment adapter and that the router metrics report
`replanning_execution_applied > 0`.

By default the smoke config loads `/models/vllm/vllm-dense-baseline` with the
patched `serverless_llm` load format. That local fixture is a
ServerlessLLM-store layout (`rank_0/tensor.data_0`), not a standard HF
safetensors/bin snapshot. Set both overrides before running
`prepare_spotserve.sh` if your worker exposes a different model path or should
load a Hugging Face model id:

```bash
export SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/vllm/vllm-dense-baseline
export SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=serverless_llm

# For a normal Hugging Face id or snapshot:
export SPOTSERVE_REPARALLELIZATION_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
export SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto
```

The smoke config enables `allow_stop_before_recreate=true` so the root
single-worker compose setup can release and recreate the vLLM actor on
`sllm_worker_0`. Multi-worker deployments can leave that option disabled to keep
the safer create-before-stop flow.

The old dummy reparallelization configs remain useful for planner-only unit
validation, but they are no longer the default V6 benchmark path because dummy
does not execute a selected `ParallelPlan`.

For a V6 performance comparison, run the separate performance matrix:

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

This matrix compares:

```text
vllm-reparallelization-disabled: enable_reparallelization=false
vllm-reparallelization-applied:  enable_reparallelization=true
```

The workload labels each request as `warmup`, `pre_replan`,
`replan_window`, or `post_replan`; the summary writes phase-specific fields
such as `phase_post_replan_latency_p95_ms`. It also writes
`latest_comparisons.json` in the performance output directory.

On the default root compose setup there is only one real worker id
(`sllm_worker_0`). The performance matrix is therefore useful for measuring
adapter/recreate overhead and post-replan steady-state latency, but a true
latency-improvement claim needs at least two real worker nodes so the baseline
can lose the active node while the applied run moves to a different live node.
The applied performance run uses the model alias
`vllm-reparallelization-applied-perf` so its router metrics do not collide with
the correctness smoke.

For a V7 context-migration performance comparison, run:

```bash
scripts/prepare_spotserve.sh --deploy-set context-migration-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240
'
```

This matrix compares:

```text
vllm-context-migration-disabled: enable_context_migration=false
vllm-context-migration-applied:  enable_context_migration=true
```

Both configs keep two vLLM replicas alive. The trace targets
`instance_selector=busy`; the benchmark runner resolves that to a live router
instance id with active concurrency immediately before replaying the spot
event. This avoids hardcoding deploy-time UUIDs in the trace and prevents a
preemption from landing on an idle replica with no source context.

The workload overlaps two same-prefix long requests before the preemption so
the applied run can observe source and target runtime KV metadata. The
warm-prefix requests intentionally use a larger decode length, and the trace
preempts at 3 seconds, so `get_context_metadata()` runs while the source
replica still has active context. A true low-cost context-reuse result should
show `context_migration_events > 0`,
`context_migration_reusable_context_blocks > 0`, and
`context_migration_reuse_ratio > 0`. `kv_cache_migration_successes > 0` means
the conservative target warmup path ran; it is still prefill/replay based, not
vLLM KV block serialization.

On clusters where the two replicas land on different worker nodes, the
proof-only V7 router may report migration with zero reusable blocks because it
does not assume cross-node cache reuse without explicit runtime proof. For a
same-node reuse benchmark, run on a worker with at least two GPUs or target a
specific same-node replica via the trace.

For risk-aware scheduling validation, run the synthetic scheduler benchmark.
This does not require a deployed model:

```bash
python scripts/run_scheduler_benchmark.py \
  --input benchmarks/spotserve/risk_aware_scheduling_synthetic.json \
  --output-dir /tmp/spotserve_risk_aware_scheduling_test
```

This compares health-only node selection against risk-aware ranking over
synthetic `spot_risk`, `remaining_lifetime_s`, and `loading_cost` metadata.

For vLLM dense black-box validation, deploy the dense configs and run the
dedicated matrix:

```bash
export MODEL_FOLDER=$PWD/model
scripts/prepare_spotserve.sh --deploy-set vllm-dense

python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --request-timeout 120 \
  --ray-address auto \
  --ray-namespace sllm
```

The vLLM dense matrix validates control-plane behavior around synthetic
preemption with a black-box vLLM worker. Treat generated-token replay as
best-effort prompt-token replay, not true KV cache recovery. Trace runs also
surface instance-state event counts in the summary and report.

For vLLM MoE black-box validation, deploy the MoE configs and run the MoE
matrix:

```bash
export MODEL_FOLDER=$PWD/model
export SPOTSERVE_VLLM_MOE_MODEL=Qwen/Qwen1.5-MoE-A2.7B
export SPOTSERVE_VLLM_MOE_LOAD_FORMAT=auto
scripts/prepare_spotserve.sh --deploy-set vllm-moe

python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
```

If the vLLM team provides a local snapshot or different load format, keep the
SpotServe model aliases unchanged and override only the backend target:

```bash
export SPOTSERVE_VLLM_MOE_MODEL=/models/hf/qwen3-moe
export SPOTSERVE_VLLM_MOE_LOAD_FORMAT=auto
export SPOTSERVE_VLLM_MOE_TP=2
scripts/prepare_spotserve.sh --deploy-set vllm-moe
```

To produce a dense-vs-MoE report under `none`, `naive_retry`, and
`generated_token_replay`, deploy both vLLM sets and run the combined matrix:

```bash
scripts/prepare_spotserve.sh --deploy-set vllm-blackbox

python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml \
  --endpoint http://127.0.0.1:8345/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
```

The MoE path intentionally treats vLLM as a black box. `load_format` is set so
ServerlessLLM skips SLLM-store conversion and lets vLLM load the MoE model
directly from the provided Hugging Face id or local path.

If old results contain `{"error": "timed out"}` or
`{"error": "Internal Server Error"}`, treat those runs as invalid setup
failures, not as recovery-policy results.

The target models must already be deployed with the intended router config, for
example `recovery_policy=none`, `naive_retry`, or `generated_token_replay`.

Important: the `policy` field in `benchmark_matrix.yaml` is metadata for the
report. The actual policy is selected when the model is deployed through
`router_config.recovery_policy`.

## Analyze

```bash
python scripts/analyze_spotserve_benchmark.py results/spotserve/<run_id>
python scripts/plot_spotserve_benchmark.py results/spotserve/<run_id>
```

Each run writes:

- `run_metadata.json`
- `raw_requests.jsonl`
- `summary.json`
- `summary.csv`
- `report.html`
- `report.md`
- `router_request_metrics.jsonl` when matching router metrics are available
