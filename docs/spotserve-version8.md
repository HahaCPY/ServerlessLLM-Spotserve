# SpotServe Version 8: Stateful Inference Recovery

Version 8 implements the CPY control-plane side of stateful inference
recovery:

```text
request failure / preemption
-> export backend-independent InferenceState
-> decide restore_state / fallback_token_replay / retry
-> restore state on target backend when supported
-> record state recovery metrics
```

This version validates the flow with the dummy backend. It does not implement
production vLLM KV cache export/restore.

## Scope

Implemented:

- `stateful_recovery` router policy
- backend-independent `InferenceState`
- `StateRecoveryPlan`
- `StateRecoveryDecision`
- backend default hooks for state restore capability
- dummy backend state export / restore
- router fallback to generated-token replay / retry
- `type=state_recovery` metrics
- benchmark analyzer/report fields for stateful recovery
- recovery correctness benchmark matrix entry
- Version 8 tests

Out of scope:

- production vLLM KV cache export
- production vLLM KV cache restore
- CUDA / PagedAttention block movement
- MoE expert state migration

## Shared Interface

File:

```text
sllm/spot/stateful_recovery.py
```

The control plane represents recoverable inference state as:

```python
@dataclass(frozen=True)
class InferenceState:
    request_id: str | None
    instance_id: str = ""
    node_id: str = ""
    backend: str = ""
    model_name: str = ""
    tokens: list[int] = field(default_factory=list)
    completed_tokens: int = 0
    state_kind: str = "token_snapshot"
    supports_restore: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

The decision layer returns:

```python
@dataclass(frozen=True)
class StateRecoveryPlan:
    request_id: str | None
    action: str
    source_instance_id: str = ""
    target_instance_id: str = ""
    recovered_tokens: int = 0
    state_kind: str = ""
    fallback_policy: str = ""
    reason: str = "stateful_recovery"
```

Actions:

```text
restore_state
fallback_token_replay
retry
```

## Backend Hooks

File:

```text
sllm/backends/backend_utils.py
```

Backends can expose:

```python
async def supports_state_restore(self) -> bool

async def export_inference_state(
    self,
    request_data: dict | None = None,
    current_output: list[list[int]] | None = None,
    completed_tokens: int | None = None,
) -> dict

async def restore_inference_state(
    self,
    state: dict,
    request_data: dict | None = None,
) -> dict
```

The base backend returns conservative unsupported values. This keeps vLLM and
transformers safe until their real runtime hooks are implemented.

## vLLM Metadata

File:

```text
sllm/backends/vllm_state_metadata.py
sllm/backends/vllm_backend.py
```

vLLM state export remains conservative:

```text
supports_restore = false
state_kind = token_snapshot
```

However, when vLLM `RequestOutput` or `kv_transfer_params` exposes cache
metadata, the exported state now preserves it for debugging/planning:

```text
metadata.kv_block_count
metadata.block_ids
metadata.block_table
metadata.cache_engine = vllm
```

This does not make `restore_inference_state()` a true KV restore. It only means
CPY no longer discards visible KV/cache metadata while falling back safely.

## Dummy Backend

File:

```text
sllm/backends/dummy_backend.py
```

The dummy backend now supports:

- export current generated tokens as `dummy_token_state`
- restore that state on another dummy backend instance
- continue generation from the restored state

This validates CPY control logic. It is still token-state simulation, not real
GPU KV cache movement.

## Router Integration

File:

```text
sllm/routers/roundrobin_router.py
```

Configure:

```json
{
  "router_config": {
    "recovery_policy": "stateful_recovery",
    "max_retries": 2
  }
}
```

When a request fails:

```text
stateful_recovery:
  export InferenceState
  retry on a target instance
  if target supports restore:
    restore_inference_state
    resume generation from restored state
  else:
    fallback to generated-token replay / retry
```

## Metrics

Files:

```text
sllm/spot/metrics.py
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

Version 8 adds `type=state_recovery` metrics:

```json
{
  "type": "state_recovery",
  "model": "dummy-correctness-stateful-recovery",
  "request_id": "recovery-replay-0001",
  "action": "restore_state",
  "state_available": true,
  "restore_supported": true,
  "fallback_used": false,
  "recovered_tokens": 2
}
```

Benchmark summaries include:

```text
state_restore_attempts_total
state_restore_successes_total
state_restore_fallback_count
state_restored_tokens_total
state_recovery_events
state_recovery_restore_events
state_recovery_fallback_events
state_recovery_recovered_tokens
state_recovery_latest_plan
```

## Benchmark

Files:

```text
examples/spotserve/config-dummy-correctness-stateful-recovery.json
benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml
```

After rebuilding/deploying the correctness set:

```bash
scripts/prepare_spotserve.sh --deploy-set correctness
```

Run:

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

Expected Version 8 shape for the stateful run:

```text
policy=stateful_recovery
failed_attempts_total > 0
retry_count_total > 0
state_restore_attempts_total > 0
state_restore_successes_total > 0
state_restored_tokens_total > 0
```

## vLLM Feasibility

大鼻 backend-side 目前提供 vLLM state metadata 的保守第一版：

```text
sllm/backends/vllm_state_metadata.py
Backend.supports_state_restore()
Backend.export_inference_state()
Backend.restore_inference_state()
VllmBackend.supports_state_restore()
VllmBackend.export_inference_state()
VllmBackend.restore_inference_state()
```

vLLM is not marked as true state-restore capable yet. The backend explicitly
reports:

```text
supports_state_restore = false
```

and CPY falls back to token replay / retry.

The current vLLM export payload is a token snapshot that CPY
`InferenceState.from_dict()` can parse:

```text
state_kind = token_snapshot
supports_restore = false
metadata.cache_engine = vllm
metadata.can_restore_same_node = false
metadata.can_restore_cross_node = false
metadata.reason = vllm_kv_restore_not_available
```

這個 hook 讓 CPY 有明確 fallback input，但不宣稱 true KV cache restore。若之後
vLLM 可以安全 expose 下列資訊，才可以把 restore capability 從 false 改成
true：

- active request ids
- prompt + generated token ids
- KV block table / block ids
- per-request KV block ownership
- same-node KV reuse semantics
- cross-node state transfer semantics
- restore hook that can bind state to a new request

## Definition Of Done

Version 8 CPY side is complete when:

- `stateful_recovery` policy is accepted.
- backend-independent state schema exists.
- dummy backend exports and restores inference state.
- router tries state restore before fallback.
- state recovery metrics are emitted and summarized.
- recovery correctness benchmark includes the stateful policy.
- vLLM backend reports explicit conservative state metadata without claiming
  true KV restore.
