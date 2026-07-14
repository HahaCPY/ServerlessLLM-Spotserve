# Bignose KV Cache Restore TODO

## Implementation status

The Bignose backend integration is implemented as a capability-negotiated
runtime contract. A vLLM engine or KV connector may expose:

```text
get_request_kv_metadata(request_id)
export_inference_state(request_id, request_data, runtime_metadata)
restore_inference_state(state, request_id, request_data)
supports_state_restore()
```

The aliases `get_kv_cache_metadata`, `export_kv_cache_state`, and
`restore_kv_cache_state` are also accepted. The backend reports true restore
support only when both export and restore hooks exist and the optional
capability probe succeeds. Unpatched upstream runtimes remain on the safe token
snapshot path.

Same-node restore is supported when the runtime's exported metadata declares
`can_restore_same_node=true`. Cross-node restore is separate and is attempted
only with `can_restore_cross_node=true`; no cross-node transport is implied by
this integration. Model, parallelism, cache block size, dtype, and layout are
checked when both the source snapshot and target configuration provide them.

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

---

## Bignose 實作內容逐項說明

本次完成的是 **ServerlessLLM/CPY 與 vLLM runtime 之間的 KV state
restore backend contract**。backend 已能偵測、呼叫並驗證 runtime 提供的真實
KV metadata、export 與 restore hook；未提供這些 hook 的原生 upstream vLLM
則繼續使用保守的 token snapshot，不會宣稱已完成 KV block restore。

主要修改檔案：

- `sllm/backends/vllm_backend.py`
- `sllm/backends/vllm_context_metadata.py`
- `sllm/backends/vllm_state_metadata.py`
- `sllm/backends/vllm_capability.py`
- `tests/spotserve_test/test_vllm_backend_state_restore.py`

### 1. Expose real per-request KV metadata

已完成的 backend 行為如下：

1. `runtime_metadata_from_request_output()` 從每次 `RequestOutput` 保留：
   `request_id`、prompt token、generated token、全部 token、
   `num_cached_tokens`，以及原始 `kv_transfer_params`。
2. `_request_runtime_metadata()` 會以 `request_id` 呼叫 runtime 的
   `get_request_kv_metadata()`；為了相容不同 connector，也接受
   `get_kv_cache_metadata()` 這個名稱。
3. runtime 回傳的 dict 會合併進該 request 的 metadata，因此可攜帶
   `kv_block_count`、`block_ids`/`kv_block_ids`、`block_table`、cache block
   size、dtype、layout、engine ID、source node/device IDs，以及 sequence ID。
4. `context_block_count_from_runtime()` 只接受 runtime 明確提供的 block
   count、block ID 或 block table。沒有真實 block 資料時固定回傳 `0`，不會用
   token 數量推估 block 數量。
5. `get_context_metadata()` 會將上述真實數量轉成 CPY 使用的
   `context_blocks`。

這一層不直接讀取不穩定的 vLLM scheduler 私有欄位；實際使用哪一個
scheduler/block manager 物件取得 block table，由 vLLM runtime hook 負責。

### 2. Fill target-specific reuse only with evidence

`get_vllm_context_metadata()` 只有在 runtime 明確提供
`reusable_tokens_by_target` 或 `reusable_blocks_by_target` 時才保留對應資料，
並將 key 正規化成 target ID 字串、value 正規化成非負整數。

若 runtime 沒有 target cache registry 或其他可證明重用狀態的資訊，輸出維持：

```json
{
  "reusable_tokens_by_target": {},
  "reusable_blocks_by_target": {}
}
```

backend 不會將來源端的 cached token/block 數複製到所有 target。

### 3. Implement true export_inference_state()

`VllmBackend.export_inference_state()` 現在有兩條明確路徑：

1. 先建立安全的 `token_snapshot` fallback，確保 runtime 沒有真實 KV export
   能力或 export 失敗時，回傳 `supports_restore=false` 和具體 reason。
2. 只有當 `supports_state_restore()` 證實 runtime 同時存在 export 與 restore
   hook 後，才呼叫 `export_inference_state()`；相容別名為
   `export_kv_cache_state()`。
3. runtime 必須回傳 dict、明確宣告 `supports_restore=true`，並提供非空的
   `runtime_state`。通過後 backend 才將 state 標示為 `vllm_kv_snapshot` 與
   `supports_restore=true`。
4. `runtime_state` 中的 snapshot handle/transfer payload 會原樣保留；backend 同時補上
   `backend=vllm`、model、request、token progress、block metadata、cache config
   與 same-node/cross-node flags。
5. runtime hook 發生 exception、回傳非 dict 或主動拒絕 export 時，backend
   會回到 `token_snapshot`，reason 為 `vllm_kv_export_failed`，不會產生假成功。

支援的 runtime hook contract 為：

```text
export_inference_state(request_id, request_data, runtime_metadata) -> dict
```

hook 可以是同步或 async function，也可以位於 async engine、engine core 或其
內層 engine 物件。

### 4. Implement true restore_inference_state()

`VllmBackend.restore_inference_state()` 會先完成下列檢查，再允許 runtime
掛載 KV state：

1. exported state 必須為 `supports_restore=true`，且 target runtime 必須仍然
   同時提供 export/restore 能力。
2. backend 必須為 `vllm`，model name 必須與 target backend 相同。
3. 比對 snapshot 與 target 的 tensor parallel size、pipeline parallel size、
   cache block size、cache dtype 與 cache layout；雙方都有值且不一致時，回傳
   `restored=false`、`reason=incompatible_cache_config`。
4. 依 source/target node 判斷 restore scope。跨節點但 snapshot 未宣告
   `can_restore_cross_node=true` 時會拒絕；同節點但未宣告
   `can_restore_same_node=true` 時也會拒絕。
5. 通過後呼叫 runtime 的 `restore_inference_state()`；相容別名為
   `restore_kv_cache_state()`。真正的 block attachment/rebinding 在這個 runtime
   hook 中執行。
6. runtime 成功時，backend 補齊 `state_kind`、`recovered_tokens`、
   `restored_blocks` 和 `restore_scope`。runtime 回傳 false、非 dict 或拋出
   exception 時，統一回傳明確的失敗結果，不會進入假成功路徑。

model revision 與精確 device placement 若存在於 snapshot，仍應由 runtime hook
在實際 attach 前做最後驗證，因為只有 runtime 知道 worker rank、實體 GPU 與
KV tensor ownership。backend 已負責 node scope 與可比較的 cache/parallel
設定檢查。

### 5. Update capability flags only when real

能力旗標改為分層判斷：

- `VllmBackend.supports_state_restore()`：engine 尚未初始化時為 false；runtime
  必須同時具有 export 與 restore hook。若 runtime 額外提供
  `supports_state_restore()` probe，其結果也必須為 true。
- exported state 的 `supports_restore`：只有 capability 通過且 export hook
  成功回傳可恢復 state 時才為 true。
- context metadata 的 `supports_state_export` 與
  `supports_state_restore`：使用相同的 runtime capability 結果。
- `BackendCapability`：預設仍為 false；只有部署設定明確設置
  `backend_config.supports_state_restore=true` 時，才對 controller 宣告 export
  與 restore 能力。部署端應只在所使用的 vLLM runtime hooks 已驗證後開啟。

因此只有 prefix warmup 或 `resume_kv_cache()` token replay 的環境，所有 restore
能力仍維持 false。

### 6. Decide same-node and cross-node support separately

exported metadata 分別保留：

```text
can_restore_same_node
can_restore_cross_node
```

預設成功 export contract 為 same-node 可用、cross-node 不可用。若 connector
確實具有跨節點 KV transport，必須由其 exported state 明確宣告
`can_restore_cross_node=true`。restore 前會比較 source 與 target node，並回報
`restore_scope=same_node` 或 `restore_scope=cross_node`；不支援時分別回傳
`same_node_restore_unsupported` 或 `cross_node_restore_unsupported`。

### 7. Tests

本次新增與既有測試所涵蓋的項目如下：

- `test_vllm_context_metadata.py`：驗證真實 explicit block count、
  `kv_transfer_params.block_ids`、token count 與保守的 target reuse maps。
- `test_vllm_state_metadata.py`：驗證 token fallback、block IDs、block table 與
  `supports_restore=false` 的安全預設。
- `test_vllm_backend_state_restore.py`：驗證沒有 runtime hooks 時不可恢復；
  hooks 齊全時 export 為 `vllm_kv_snapshot`；成功 attach 後回報 restored blocks；
  cache config 不相容時拒絕；未支援 cross-node transport 時拒絕跨節點 restore。
- `test_stateful_recovery.py`：驗證 capability/state 都支援時 router recovery
  decision 使用 `restore_state` 且 `fallback_used=false`，並驗證 restore metrics。

相關測試執行結果為 `24 passed`。完整 `tests/spotserve_test` 在目前開發環境因
缺少 `ray` package，有兩個 router/scheduler test module 無法完成 collection；
這不是上述 KV restore 測試失敗。

### Definition of Done 對照與目前邊界

backend contract、metadata propagation、能力保護、相容性檢查、restore result
與單元測試已完成。是否能在實際 GPU 上宣稱「true KV cache restore」，則取決於
部署的 vLLM runtime/connector 是否真的實作上述 export/restore hooks，並能在
worker 中搬移或重新綁定 KV tensors。

因此目前狀態應解讀為：

- **已完成**：ServerlessLLM backend integration 與安全 fallback。
- **已完成**：具 hooks 的 runtime 可走 restore path，且不會進行 token replay
  fallback。
- **需由部署 runtime 驗證**：實體 vLLM scheduler/block manager 的 KV attach、
  model revision/device ownership 相容性與 GPU restore latency。
- **尚不可由預設 upstream vLLM 宣稱完成**：沒有實作 hooks 的環境仍只支援
  token snapshot/prefix warmup。
- **跨節點支援依 connector 而定**：只有 runtime 明確宣告且實際提供 transport
  時才會開啟。

正式 benchmark 應在已實作 hooks 的 vLLM runtime 上確認
`state_restore_successes_total > 0`、`state_restore_fallback=false`，再宣稱已達成
端到端真實 KV block migration。

---

## Runtime 資料穩定性與即時取得實作

### 結論：runtime 資料不是全部固定

這些資料必須依生命週期分成三層；不能只啟動一次、記下所有數字後永久重用。

| 資料層級 | 代表欄位 | 變動時機 | 本次處理方式 |
|---|---|---|---|
| Engine lifetime 固定 | model/revision、TP/PP、resolved dtype、cache group/block size、NHD/HND layout、worker rank/device、engine ID、connector | engine 重啟、重新配置或 worker replacement 時改變 | cache 初始化完成後由每個 worker 取得一次，不再使用未解析的 `backend_config` 猜測 |
| Request/step 動態 | tokens、`num_computed_tokens`、status、block IDs/table/count、null blocks、`kv_transfer_params` | 每次 decode、preemption、reallocation、offload/free 都可能改變 | 每次查詢依 request ID 進入 EngineCore，直接讀當下 Scheduler/`KVCacheManager` |
| Transfer 動態 | snapshot/lease handle、block pin 狀態、handle expiry、connector health、target reachability | 每次 export/transfer 都不同 | 尚無通用 upstream API；維持 restore capability=false，不能由 block metadata 推論可恢復 |

特別是 block ID 只在 allocator 與當下 request ownership 中有意義。request 被
preempt、free 或 engine restart 後，block ID 可能重新分配給其他 request，因此
不能將一次執行取得的 block table 寫死到設定檔。

### 固定資料：cache 初始化後從 worker 取得一次

本次在 vLLM worker 增加 `get_kv_runtime_metadata()`。EngineCore 完成
`initialize_cache` 後，透過 worker collective RPC 收集一次：

```text
rank
local_rank
device
physical_device_ids
resolved cache_layout (NHD/HND)
```

cache dtype 不使用可能仍為 `auto` 的 `CacheConfig.cache_dtype` 作為實際值，而是
讀取初始化後各 `KVCacheGroupSpec.kv_cache_spec.dtype`。同時保留
`configured_cache_dtype` 供診斷。每個 cache group 分別回報 kind、block size、
resolved dtype 與 sliding-window 設定，因此 heterogeneous KV cache 不會被強迫
壓成一個錯誤的共同設定。

若所有 group 的 block size/dtype 相同，另外提供方便比較的 scalar
`cache_block_size`/`cache_dtype`；不同時 scalar 為 null，呼叫端必須使用
`cache_groups`。

### 動態資料：每次由 EngineCore 取得

新增的 runtime API：

```text
AsyncLLM.get_request_kv_metadata(request_id)
AsyncLLM.get_all_request_kv_metadata()
EngineCore.get_request_kv_metadata(request_id)
EngineCore.get_all_request_kv_metadata()
```

EngineCore 的資料來源是目前 vLLM V1 真實物件：

```text
Scheduler.requests[request_id]
Scheduler.kv_cache_manager.get_blocks(request_id)
Request.prompt_token_ids / output_token_ids / all_token_ids
Request.num_computed_tokens / status / kv_transfer_params
```

utility RPC 在 EngineCore event loop 中取得 point-in-time snapshot，不再只依賴
`RequestOutput`。因此尚未輸出第一個 token 的 active request 也能被列出；
`request_trace` 只保留為舊 runtime 的 fallback。

目前 snapshot 包含：

```text
request/sequence ID
prompt/output/all tokens
completed/computed tokens
request status
raw block table by cache group
null block mask by cache group
non-null/restorable block IDs by cache group
per-group block count
logical context block count
physical allocated block count
resolved cache group/block size/dtype/layout
engine/worker/device/model/parallel/connector metadata
```

`block_table` 與 `raw_block_ids_by_group` 保留 null block 的原始位置；
`kv_block_ids_by_group` 則排除 null blocks。相容用的扁平 `block_ids` 不應用來
重建 heterogeneous cache，因為不同 group 的 ID namespace 可能重複。

新版 vLLM 會把 external request ID 改成隨機化 internal scheduler ID。本次查詢
先經 `OutputProcessor.external_req_ids` 做映射，再以 internal ID 路由到正確的
DP EngineCore。若 parallel sampling (`n > 1`) 對應多個 child sequences，目前
明確回傳 `parallel_sampling_metadata_not_supported`，不會拿第一個 child 冒充
完整 request state。部署使用的 vLLM 0.11.2 尚未做 ID 隨機化，則直接使用原
request ID。

### ServerlessLLM backend 的使用方式

`VllmBackend.get_context_metadata()` 現在優先呼叫
`get_all_request_kv_metadata()`，直接建立 active contexts。指定 request export
在 `request_trace` 尚未有 output 時，也會直接呼叫
`get_request_kv_metadata(request_id)`。

下列資訊會完整傳入 CPY context/state metadata：

- grouped/raw/non-null block table 與 count；
- resolved cache block size、dtype、layout 和 group 規格；
- worker/device、model revision、TP/PP 與 connector；
- request status、sequence IDs 與 completed tokens。

target reuse maps 仍預設為空。metadata getter 只提供觀察資訊，不會 pin block，
所以 runtime patch 明確回傳：

```text
supports_state_export = false
supports_state_restore = false
can_restore_same_node = false
can_restore_cross_node = false
```

此外，export hook 現在必須**明確**回傳 `supports_restore=true`，並提供非空的
`runtime_state`（例如具有期限的 snapshot/lease handle），才能進入 restore
path。`InferenceState` 已新增 `runtime_state` round-trip，opaque handle 不會在
router 序列化時遺失。缺少欄位不再被當成 true，same-node capability 也不再
預設為 true；source/target node 無法確認時回傳 `unknown_restore_scope`。

### 實際部署版本與 patch

SpotServe worker 的 `requirements-worker.txt` 固定使用 `vllm==0.11.2`。為避免只
修改開發用的 `/Qwen3/vllm` checkout、但容器仍載入未修改 wheel，本次新增：

```text
sllm_store/vllm_patch/runtime_kv_metadata.patch
```

Docker build 會在既有 `sllm_load.patch` 後套用這份 runtime patch。
`patch.sh`、`check_patch.sh` 與 `remove_patch.sh` 已改為同時管理兩份 patch。

`scripts/prepare_spotserve.sh` 也會在部署 vLLM workload 前，於 worker container
輸出並驗證：

```text
installed vLLM version
AsyncLLMEngine import path
get_request_kv_metadata 是否存在
get_all_request_kv_metadata 是否存在
```

任一 hook 缺少時部署會 fail closed，不會靜默回到一個看似支援 runtime
metadata、實際卻仍使用原始 vLLM wheel 的環境。

### 本次執行與驗證結果

本次嘗試查詢 Podman runtime，但 `podman ps -a` 沒有可用 container；本機
PyTorch 也回報無法初始化 NVML/沒有 active CUDA driver，因此無法在這個工作
環境產生真實 GPU request 的 block ID 樣本。由於 block IDs 本來就是動態且
不可跨 execution 重用，本次沒有將 synthetic ID 當成正式 runtime 數據寫入設定。

已完成的可重現驗證：

- vLLM EngineCore grouped blocks、null mask、resolved dtype/layout 的直接測試：
  `5 passed`（包含 external/internal ID、DP routing 與 parallel sampling 的
  fail-closed 行為，以無 GPU 的 runtime object fixture 執行）。
- ServerlessLLM metadata/export/capability/recovery 相關測試：`24 passed`。
- vLLM 0.11.2 deployment patch 對 tag 原始碼執行 `patch --dry-run`：通過。
- 修改檔案 `py_compile`、shell `bash -n` 與 `git diff --check`：通過。

啟動具 GPU 的 worker 後，`prepare_spotserve.sh` 會先輸出實際 runtime 版本與
hook probe；送出 active request 後呼叫 backend `get_context_metadata()`，即可
取得該 scheduling step 的真實 block table，而不是使用本文件中的範例數字。

### 仍然不能宣稱 true KV restore 的原因

這次完成的是 **live runtime metadata acquisition**，不是 GPU KV snapshot
transport。point-in-time block IDs 在下一次 scheduling/preemption 後可能失效；
真正 export 必須在同一個 runtime transaction 中：

1. fence/等待 in-flight worker writes；
2. pin blocks 或建立具有 lifetime/expiry 的 connector lease；
3. 保存完整 compatibility fingerprint 與 opaque transfer handle；
4. target 在建立 resumed request 時消費 handle 並跳過已恢復 token 的 prefill；
5. 成功後才釋放 source blocks。

upstream vLLM 0.11.2 沒有任意 active request 的通用 snapshot/attach API；現有
NIXL、LMCache、Mooncake 等 KV connectors 是特定 P/D 或 external-cache 工作流，
不能僅憑 block table 當作通用 restore。完成 connector-specific export/attach
與 GPU benchmark 前，capability flags 必須維持 false。
