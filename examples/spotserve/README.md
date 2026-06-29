# SpotServe-style Controller Example

This directory contains small artifacts for the first-version SpotServe-style
control-plane prototype.

## Sample Trace

```bash
python -m sllm.spot.preemption_simulator \
  --trace examples/spotserve/spot_trace_sample.jsonl \
  --speedup 10
```

Trace events:

- `preempt`: mark matching worker instances as `PREEMPTING`.
- `dead`: mark matching worker instances as `DEAD`.
- `recover`: parsed in v1, but not dispatched yet.

## Router Config

Use `router_config.recovery_policy` to choose behavior:

```json
{
  "router_config": {
    "recovery_policy": "naive_retry",
    "max_retries": 2,
    "metrics_path": "results/spotserve/requests.jsonl"
  }
}
```

Supported first-version policies:

- `none`
- `naive_retry`
- `generated_token_replay`

`generated_token_replay` is best-effort and is not true KV cache migration.
