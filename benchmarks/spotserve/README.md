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
