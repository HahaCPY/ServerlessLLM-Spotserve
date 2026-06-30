# SpotServe Version 3: Recovery Correctness Validation

## Goal

Version 3 validates that retry and generated-token replay are actually
triggered when a request fails in the middle of generation.

This version does not try to prove performance superiority. It answers the
correctness question:

```text
request starts
-> backend generates partial tokens
-> backend is forced to fail
-> router retries or replays
-> report shows recovery metrics
```

## Implemented

### Forced Failure Workload

Files:

```text
benchmarks/spotserve/workloads/recovery_correctness.jsonl
benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml
```

The workload has two cases:

- `recovery-replay-0001`: forced preemption after two generated tokens. This
  should let token replay report `recovered_tokens_total > 0`.
- `recovery-fallback-0001`: forced backend error after two generated tokens,
  with current tokens cleared. This should let token replay report
  `recovery_fallback_count > 0`.

### Dummy Backend Recovery Hooks

File:

```text
sllm/backends/dummy_backend.py
```

The dummy backend now supports:

- token-by-token simulated generation
- `get_current_tokens`
- forced failure after `force_fail_after_tokens`
- preempted partial responses with `current_output`
- exception failures for fallback validation
- continuation from router-provided `input_tokens`

### Correctness Configs

Files:

```text
examples/spotserve/config-dummy-correctness-none.json
examples/spotserve/config-dummy-correctness-naive-retry.json
examples/spotserve/config-dummy-correctness-token-replay.json
```

These use separate model names from the long benchmark:

```text
dummy-correctness-none
dummy-correctness-naive-retry
dummy-correctness-token-replay
```

### Recovery Metrics Report

Files:

```text
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
benchmarks/spotserve/run_benchmark.py
```

The report now includes:

```text
failed_attempts_total
retry_count_total
recovered_tokens_total
recovery_fallback_count
recovery_triggered_requests
replay_succeeded_requests
replay_not_needed_requests
```

When router metrics are found, each benchmark run also stores:

```text
router_request_metrics.jsonl
```

### Transformers Backend Smoke Hook

File:

```text
sllm/backends/transformers_backend.py
```

Transformers generation now understands the same forced-failure request fields,
so a small-model smoke test can validate the replay path after the dummy backend
passes.

This remains best-effort generated-token replay. It does not claim true KV cache
migration.

## How To Run

Prepare the environment and deploy both normal and correctness dummy models:

```bash
scripts/prepare_spotserve.sh --deploy-set correctness
```

Because Version 3 changes backend code, use the full script the first time so
the image is rebuilt and `sllm_head` is recreated. After that,
`--skip-build --skip-recreate` is only safe for benchmark/config-only edits.

Do not deploy the standard dummy models and correctness dummy models together on
small Ray head-only setups. Six router actors can exceed the container thread
limit and leave some model names registered while their router actors are dead.
For Version 3, use only the `correctness` deploy set.

Run the recovery correctness benchmark:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
'
```

Expected interpretation:

- `dummy-correctness-none` should fail the forced-failure requests.
- `dummy-correctness-naive-retry` should show `retry_count_total > 0`.
- `dummy-correctness-token-replay` should show `recovered_tokens_total > 0`.
- the fallback request should show `recovery_fallback_count > 0`.

Example successful Version 3 shape:

```text
dummy-correctness-none: successes=0/2, failed_attempts=2, retries=0, recovered_tokens=0, fallbacks=0
dummy-correctness-naive-retry: successes=2/2, failed_attempts=2, retries=2, recovered_tokens=0, fallbacks=0
dummy-correctness-token-replay: successes=2/2, failed_attempts=2, retries=2, recovered_tokens=2, fallbacks=1
```

## Definition Of Done

Version 3 is complete when:

- dummy backend correctness tests pass
- naive retry benchmark reports retry activity
- generated-token replay benchmark reports recovered tokens
- fallback is recorded when current tokens are unavailable
- transformers has a forced-failure smoke hook available
