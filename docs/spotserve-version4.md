# SpotServe Version 4: vLLM Dense Black-box Integration

Version 4 moves the recovery benchmark path from dummy / transformers backends
to a dense vLLM backend, while keeping vLLM itself as a black box.

## Scope

Implemented:

- vLLM dense policy configs:
  - `examples/spotserve/config-vllm-dense-baseline.json`
  - `examples/spotserve/config-vllm-dense-none.json`
  - `examples/spotserve/config-vllm-dense-naive-retry.json`
  - `examples/spotserve/config-vllm-dense-token-replay.json`
- vLLM dense benchmark matrix:
  - `benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml`
- vLLM dense trace workload:
  - `benchmarks/spotserve/workloads/vllm_dense_trace.jsonl`
- per-policy targeted traces:
  - `examples/spotserve/spot_trace_vllm_dense_none.jsonl`
  - `examples/spotserve/spot_trace_vllm_dense_naive_retry.jsonl`
  - `examples/spotserve/spot_trace_vllm_dense_token_replay.jsonl`
- `scripts/prepare_spotserve.sh --deploy-set vllm-dense`
- HTTP spot-event replay through `POST /spot/event`
- report integration through the existing router metrics fields:
  - `failed_attempts`
  - `retry_count`
  - `recovered_tokens`
  - `recovery_fallback`
- instance-state metrics in the benchmark summary/report:
  - `instance_state_rows`
  - `instances_marked_preempting`
  - `instances_marked_ready`
  - `instances_marked_dead`

Not implemented in Version 4:

- MoE serving
- expert-aware scheduling
- vLLM scheduler changes
- PagedAttention changes
- true KV cache migration

## Model

The dense configs use:

```text
Qwen/Qwen3-0.6B
```

Each config uses the existing ServerlessLLM vLLM backend with one GPU per
replica and a fixed one-replica autoscaling cap. This is meant as a black-box
dense-model smoke benchmark, not as a tuned performance profile.

The vLLM dense configs intentionally use conservative runtime settings:

```text
gpu_memory_utilization=0.35
max_model_len=2048
max_num_seqs=4
enforce_eager=true
enable_prefix_caching=false
```

This keeps four policy aliases deployable on a four-GPU worker without
triggering extra replicas or heavy CUDA graph/prefix-cache allocation during
the first benchmark request.

The vLLM dense router configs also set:

```text
count_preempting_toward_capacity=true
```

With one GPU per policy alias, a synthetic preemption should not cause the
router to start a replacement vLLM actor while the original actor is still
alive. Without this cap, the trace run can briefly create a second actor for a
policy and crash the Ray head with native thread exhaustion.

The three deployed model names are policy-specific aliases, but the vLLM
downloader uses `backend_config.pretrained_model_name_or_path` to download or
load the real Hugging Face model. A local ServerlessLLM transformers store such
as `/models/transformers/Qwen/Qwen3-0.6B` is not a vLLM source snapshot.

## Replay Semantics

vLLM dense generated-token replay is conservative:

- `VllmBackend.get_current_tokens()` reads the latest prompt plus generated
  token ids from vLLM `RequestOutput` records when they are available.
- The router can pass those ids back through `input_tokens`.
- This is best-effort generated-token replay.
- This is not true KV cache recovery.

If no current tokens are available when a request fails, the generated-token
replay policy records `recovery_fallback=true` and behaves like a retry.

Synthetic trace preemption does not necessarily interrupt an already-running
vLLM generation. Therefore the vLLM dense trace benchmark validates
control-plane behavior and black-box compatibility. Use Version 3
recovery-correctness results to validate forced mid-generation retry/replay
triggering.

## Run

Prepare and deploy the vLLM dense policy models:

```bash
scripts/prepare_spotserve.sh --deploy-set vllm-dense
```

This starts `sllm_head` and `sllm_worker_0`, then checks that Ray reports a
worker GPU before deploy.

Then run:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 120 \
  --ray-address auto \
  --ray-namespace sllm
'
```

`run_benchmark.py` replays traces with HTTP by default. The benchmark posts
each trace event to `POST /spot/event`, and the existing SLLM API process
dispatches it to the Ray controller. This avoids starting an extra Ray driver
for each trace run, which can fail under vLLM dense load with
`pthread_create failed: Resource temporarily unavailable`.

The legacy Ray subprocess trace path is still available for comparison:

```bash
--trace-transport ray
```

The matrix includes:

- `vllm-dense-no-preemption`
- `vllm-dense-preemption-none`
- `vllm-dense-naive-retry`
- `vllm-dense-token-replay`

The no-preemption baseline uses a separate model alias
`vllm-dense-baseline` so that preemption state from the `none` policy run does
not pollute the baseline router.

If the benchmark command returns immediately with no summary, check
`podman ps -a --filter name=sllm`. An `sllm_head` exit code of 139 means the
head Ray driver crashed before `raw_requests.jsonl` was written. In the first
vLLM dense attempt this showed up as Ray native logs containing
`thread: Resource temporarily unavailable [system:11]`, followed by
SIGABRT/SIGSEGV. The likely cause is resource pressure from multiple vLLM
actors/threads, not a normal benchmark failure.

If only `trace_replayer.log` shows `pthread_create failed` while the benchmark
summary is produced, the old Ray subprocess trace transport is exhausting
thread quota. Use the default HTTP trace transport instead.

## Expected Interpretation

For normal synthetic trace runs, it is valid to see:

```text
failed_attempts=0
retry_count=0
recovered_tokens=0
fallbacks=0
```

That means the trace changed routing state without killing an in-flight vLLM
request. The important checks are:

- the vLLM dense models deploy through SLLM,
- the benchmark reaches `/v1/chat/completions`,
- trace replay completes,
- router metrics are available,
- instance-state metrics show preempt/recover/dead events when trace replay is
  enabled,
- preempted instances stop receiving new requests,
- generated-token replay either uses captured tokens or reports fallback.

If `failed_attempts > 0` appears in the vLLM dense run, then compare:

- `naive_retry`: should show `retry_count > 0`.
- `generated_token_replay`: should show either `recovered_tokens > 0` or
  `recovery_fallback_count > 0`.

Do not read this benchmark as evidence of KV cache migration or MoE recovery.
Those belong to later versions.
