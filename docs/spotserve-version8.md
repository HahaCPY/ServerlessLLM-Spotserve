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

This version validates the control-plane flow with the dummy backend and adds a
live vLLM same-node NIXL state-restore path when the patched runtime hooks are
available.

## Scope

Implemented:

- `stateful_recovery` router policy
- backend-independent `InferenceState`
- `StateRecoveryPlan`
- `StateRecoveryDecision`
- backend default hooks for state restore capability
- dummy backend state export / restore
- vLLM runtime hook discovery for KV state export / restore
- vLLM forced-preemption benchmark hook for repeatable live recovery tests
- router fallback to generated-token replay / retry
- `type=state_recovery` metrics
- benchmark analyzer/report fields for stateful recovery
- recovery correctness benchmark matrix entry
- live vLLM stateful-recovery performance matrix entry
- Version 8 tests

Out of scope:

- unpatched upstream vLLM state restore
- cross-node KV cache restore without runtime support
- MoE expert-state-aware restore policy

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
    runtime_state: dict[str, Any] = field(default_factory=dict)
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

The base backend returns conservative unsupported values. Backends opt into
state restore only when they can return a non-empty restorable state payload.

## vLLM Live Restore

File:

```text
sllm/backends/vllm_state_metadata.py
sllm/backends/vllm_backend.py
```

The vLLM backend first builds a conservative token snapshot from visible
`RequestOutput` / runtime metadata. If the patched runtime exposes all of the
following hooks, the snapshot is upgraded to a restorable KV snapshot:

```text
supports_state_restore
export_inference_state
restore_inference_state
get_request_kv_metadata
get_all_request_kv_metadata
```

Restorable vLLM state has this shape:

```text
state_kind = vllm_kv_snapshot
supports_restore = true
runtime_state.kv_transfer_params.remote_block_ids
metadata.can_restore_same_node = true
```

If those hooks are missing or return an unsupported payload, CPY keeps the safe
fallback:

```text
state_kind = token_snapshot
supports_restore = false
fallback_policy = generated_token_replay
```

The live benchmark uses a synthetic request field to force one vLLM preemption
after a known number of generated tokens:

```json
{
  "force_failure": "preempted",
  "force_fail_after_tokens": 16,
  "force_fail_once": true
}
```

These keys are removed before creating vLLM `SamplingParams`, so normal vLLM
requests are unaffected. On the forced-preempt path, the backend exports
`_spotserve_inference_state` before returning the preempted response. The router
uses that embedded snapshot first, then falls back to calling
`export_inference_state()` if no embedded state is present.

When multiple same-node vLLM/NIXL replicas run at once, each backend actor also
derives its own `VLLM_NIXL_SIDE_CHANNEL_PORT` before engine startup. Without
this, every NIXL engine tries to bind the default `127.0.0.1:5600`, and the
second replica remains stuck in `starting` with:

```text
ZMQError: Address already in use (addr='tcp://127.0.0.1:5600')
```

The derived port can be controlled with:

```text
nixl_side_channel_base_port
nixl_side_channel_port_span
nixl_side_channel_port
```

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
    "max_retries": 2,
    "enable_stateful_target_planner": true
  }
}
```

Before a stateful retry, the router now asks the recovery target planner to
reserve an already-ready target whose model, TP/PP/EP and KV-cache metadata are
compatible with the exported state.  The selected target is then passed to
`restore_inference_state`; the planner does not create an engine or change the
parallel shape.  If no compatible target is available, the normal allocator is
used and the existing token-replay fallback remains explicit.  Changing TP/EP
still belongs to the separate re-parallelization path and is not claimed as a
direct NIXL restore.

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

## Benchmarks

Correctness files:

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

Live vLLM performance files:

```text
examples/spotserve/config-vllm-stateful-recovery-token-replay-performance.json
examples/spotserve/config-vllm-stateful-recovery-applied-performance.json
benchmarks/spotserve/workloads/stateful_recovery_vllm_performance.jsonl
benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml
```

Deploy:

```bash
scripts/prepare_spotserve.sh --deploy-set stateful-recovery-performance
```

This deploy set prepares the container and copies the benchmark/config files.
The benchmark runner then deploys and deletes each policy model one run at a
time, so only one policy's two replicas are alive concurrently.

Run:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240
'
```

Expected applied-run shape:

```text
policy = stateful_recovery
failed_attempts_total > 0
retry_count_total > 0
state_recovery_events > 0
state_restore_attempts_total > 0
state_restore_successes_total > 0
state_restore_fallback_count = 0
state_restored_tokens_total > 0
```

The baseline run uses `generated_token_replay`; the applied run uses
`stateful_recovery`. A latency claim should compare the generated
`latest_comparisons.json` fields from the same benchmark invocation.

Recorded live vLLM result:

```text
vllm-stateful-recovery-token-replay:
  successes=3/3
  p95=63540.97ms
  failed_attempts=1
  retries=1
  recovered_tokens=16

vllm-stateful-recovery-applied:
  successes=3/3
  p95=2904.41ms
  failed_attempts=1
  retries=1
  recovered_tokens=16
  state_restores=1/1
  state_tokens=16
```

Recorded comparison:

| Metric | Token Replay | Stateful Recovery | Result |
|---|---:|---:|---:|
| `latency_p95_ms` | 63540.97 | 2904.41 | 95.43% lower |
| `phase_failure_window_latency_p95_ms` | 63540.97 | 2904.41 | 95.43% lower |
| `phase_post_recovery_latency_p95_ms` | 24243.46 | 1073.46 | 95.57% lower |
| `throughput_req_s` | 0.04721 | 0.10332 | 2.19x |
| `phase_post_recovery_throughput_req_s` | 0.07993 | 0.22152 | 2.77x |
| `state_restore_successes_total` | 0 | 1 | restore path active |
| `state_restore_fallback_count` | 0 | 0 | no fallback |
| `state_restored_tokens_total` | 0 | 16 | 16 tokens restored |

Safe claim:

```text
V8 reduced failure-window p95 latency from 63.5s to 2.90s in the live
same-node vLLM/NIXL stateful-recovery benchmark, while maintaining 100%
success rate and restoring 16 tokens with zero state-restore fallback.
```

## Definition Of Done

Version 8 CPY side is complete when:

- `stateful_recovery` policy is accepted.
- backend-independent state schema exists.
- dummy backend exports and restores inference state.
- router tries state restore before fallback.
- state recovery metrics are emitted and summarized.
- recovery correctness benchmark includes the stateful policy.
- vLLM backend consumes patched runtime hooks when present and falls back safely
  when they are not available.
- vLLM performance benchmark compares token replay against stateful recovery
  and reports state-restore counters.
