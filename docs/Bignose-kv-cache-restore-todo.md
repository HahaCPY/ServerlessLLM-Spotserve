# Bignose KV Cache Restore TODO

This document lists the backend work Bignose still needs to add before CPY can
claim true vLLM KV cache migration / stateful restore.

## Current CPY Status

CPY has implemented the safe control-plane pieces:

```text
vLLM RequestOutput / kv_transfer_params metadata extraction
-> ContextMetadata / InferenceState payloads
-> context migration planning
-> optional target resume_kv_cache() warmup
-> benchmark metrics for kv_cache_migration
```

This is not true KV block restore yet. In the current checkout:

```text
VllmBackend.supports_state_restore() = False
VllmBackend.restore_inference_state() -> restored=false,
                                      reason=vllm_kv_restore_not_available
```

The current `resume_kv_cache()` path only replays token batches to warm target
prefix/cache state. It does not serialize, transfer, attach, or rebind live vLLM
KV blocks to the resumed request.

## Backend Hooks To Implement

### 1. Expose real per-request KV metadata

Bignose should make the vLLM backend able to expose, per active request:

```text
request_id
prompt tokens
generated tokens
completed_tokens
kv_block_count / context_blocks
block_ids or kv_block_ids
block_table
cache block size
cache dtype
cache layout / engine identifier
source node id
source device ids
sequence id / sequence group id if vLLM needs it for restore
```

This can come from `RequestOutput.kv_transfer_params`, vLLM scheduler state,
block manager state, or another runtime object. CPY only needs a stable dict-like
payload.

Do not set `context_blocks` by estimating from token count. Use real block
metadata, or return `context_blocks = 0`.

### 2. Fill target-specific reuse only with evidence

For V7 context migration, CPY can consume:

```text
reusable_tokens_by_target
reusable_blocks_by_target
```

These must be target-specific. Do not fill every target with the same value
unless the backend can prove that target actually has reusable KV/prefix state.

Safe conservative default:

```json
{
  "reusable_tokens_by_target": {},
  "reusable_blocks_by_target": {}
}
```

Useful future inputs:

```text
target instance id
target node id
target cache registry
same-node KV ownership info
cross-node transfer availability
```

### 3. Implement true export_inference_state()

`VllmBackend.export_inference_state()` should return a CPY-readable payload that
represents real restorable state:

```json
{
  "request_id": "req-1",
  "instance_id": "old-vllm-0",
  "node_id": "node-0",
  "backend": "vllm",
  "model_name": "model",
  "tokens": [1, 2, 3],
  "completed_tokens": 3,
  "state_kind": "vllm_kv_snapshot",
  "supports_restore": true,
  "metadata": {
    "cache_engine": "vllm",
    "kv_block_count": 2,
    "block_ids": [10, 11],
    "block_table": {"req-1": [10, 11]},
    "can_restore_same_node": true,
    "can_restore_cross_node": false
  }
}
```

If true KV restore is not available, keep the current conservative behavior:

```text
state_kind = token_snapshot
supports_restore = false
metadata.reason = vllm_kv_restore_not_available
```

### 4. Implement true restore_inference_state()

`VllmBackend.restore_inference_state(state, request_data)` should restore or
attach the exported KV state to the target backend, then return:

```json
{
  "restored": true,
  "state_kind": "vllm_kv_snapshot",
  "recovered_tokens": 128,
  "restored_blocks": 8,
  "restore_scope": "same_node"
}
```

If restore cannot be done, return a clear false result:

```json
{
  "restored": false,
  "reason": "incompatible_cache_config"
}
```

The backend should validate compatibility before claiming success:

```text
model name / revision
tensor parallel size
pipeline parallel size
cache block size
cache dtype
KV cache layout
device placement
same-node vs cross-node transfer support
```

### 5. Update capability flags only when real

Only after true restore works:

```text
VllmBackend.supports_state_restore() -> True
BackendCapability.supports_state_restore -> True
supports_restore in exported state -> True
```

If only token replay / prefix warmup works, keep these false.

### 6. Decide same-node and cross-node support separately

Please expose separate metadata for:

```text
can_restore_same_node
can_restore_cross_node
```

Same-node restore may be possible before cross-node transfer. CPY can use these
flags to choose a safe target and avoid pretending cross-node KV migration works.

### 7. Add tests

Minimum tests Bignose should add:

```text
vLLM metadata exposes real kv block count from runtime state
export_inference_state returns supports_restore=true only for restorable state
restore_inference_state returns restored=true after attaching KV state
restore_inference_state returns restored=false for incompatible target
router stateful_recovery path uses restore_state without fallback when supported
benchmark summary shows state_restore_successes_total > 0
```

## Definition Of Done

True KV cache restore is done when:

- active vLLM requests expose real KV block metadata.
- CPY context metadata reports non-zero `context_blocks` when blocks exist.
- `export_inference_state()` returns a real restorable vLLM KV state.
- `restore_inference_state()` can attach that state on a compatible target.
- `supports_state_restore()` returns true only after the restore path works.
- stateful recovery benchmark shows restore success without token replay fallback.
- docs clearly state whether support is same-node only or cross-node capable.

Until then, CPY can benchmark planning and cache warmup, but should not claim
true KV block migration latency gains.
