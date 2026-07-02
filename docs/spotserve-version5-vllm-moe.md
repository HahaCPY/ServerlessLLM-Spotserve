# SpotServe Version 5: vLLM MoE Black-box Integration

Version 5 validates whether the existing SpotServe-style control plane still
works when the vLLM backend serves a single MoE model instead of a dense model.
The MoE backend is treated as a black-box vLLM service.

## Scope

Implemented:

- vLLM MoE deployment configs:
  - `examples/spotserve/config-vllm-moe-baseline.json`
  - `examples/spotserve/config-vllm-moe-none.json`
  - `examples/spotserve/config-vllm-moe-naive-retry.json`
  - `examples/spotserve/config-vllm-moe-token-replay.json`
- Direct vLLM load path for configs with `backend_config.load_format`, which
  skips ServerlessLLM store conversion and passes the configured model path to
  vLLM.
- Synthetic MoE workload and spot traces.
- MoE benchmark matrix:
  - `benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml`
- Dense-vs-MoE benchmark matrix:
  - `benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml`
- Prepare-script deploy sets:
  - `vllm-moe`
  - `vllm-blackbox`

Out of scope:

- expert routing
- expert dispatch
- CUDA kernels
- expert-aware scheduling
- expert migration
- MoE-specific recovery optimization

## Pipeline

```text
Spot trace
  -> Controller
  -> Router
  -> vLLM MoE worker
  -> retry / generated-token replay
  -> router metrics
  -> benchmark report
```

## Run

Use the vLLM team's MoE model id or container-local snapshot path:

```bash
export SPOTSERVE_VLLM_MOE_MODEL=Qwen/Qwen1.5-MoE-A2.7B
export SPOTSERVE_VLLM_MOE_LOAD_FORMAT=auto
export SPOTSERVE_VLLM_MOE_TP=1
scripts/prepare_spotserve.sh --deploy-set vllm-moe
```

Run the MoE-only matrix:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180
'
```

Run the dense-vs-MoE comparison:

```bash
scripts/prepare_spotserve.sh --deploy-set vllm-blackbox

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180
'
```

## Validation

Definition of done checks:

- MoE model appears in `/v1/models` under the SpotServe aliases.
- Requests complete against `vllm-moe-*`.
- HTTP spot trace replay completes.
- `none`, `naive_retry`, and `generated_token_replay` runs write router metrics.
- Benchmark reports are generated under `results/spotserve_vllm_moe` and
  `results/spotserve_vllm_dense_vs_moe`.

Generated-token replay remains best-effort prompt/token replay. It is not KV
cache migration and does not use MoE expert state.
