# Milestone 3 Backend Capability and Metadata Handoff

這份文件是給大鼻的 backend-side 工作說明。

Version 6 已經完成 CPY 這邊的 control-plane planner：

```text
spot event
-> GPU availability changes
-> CPY planner
-> ParallelPlan
-> replanning metrics
```

大鼻接下來要做的是：

```text
backend / vLLM / MoE runtime
-> BackendCapability
-> tell planner which ParallelPlan is executable
-> runtime context metadata
-> tell migration planner which context can be reused
```

重點是：大鼻不需要決定「要選哪個 plan」。大鼻只需要告訴 CPY：

```text
這個 backend 對這個 model，到底支援哪些 TP / DP / PP / EP config。
```

CPY 仍然負責根據 spot event 和 GPU availability 選 plan。

Milestone 3 目前分兩條 backend contract：

```text
V6 Dynamic Reparallelization:
  BackendCapability tells CPY which ParallelPlan is executable.

V7 Low-cost Context Migration:
  Context metadata tells CPY which request context can be reused.
```

## V6 Backend Capability

V6 這邊要完成的是 backend capability contract。CPY planner 仍然負責根據
spot event 和 GPU availability 選 plan；大鼻負責提供 backend 對某個 model
實際可執行的 `ParallelPlan` allowlist。

V6 的資料流是：

```text
model config
-> get_backend_capability(model_config)
-> BackendCapability.supported_configs
-> CPY planner filters illegal ParallelPlan
```

這裡的原則是：只宣告已知可跑的 config。不要因為 vLLM 或 MoE 理論上可能支援
某種 parallel mode，就讓 planner 選到還沒驗證過的 plan。

### V6 Dense vLLM Config

dense vLLM 第一版保持保守，只宣告目前 config 已經設定的 TP shape：

```text
TP = backend_config.tensor_parallel_size
DP = 1
PP = 1
EP = 1
num_replicas = 1
num_gpus = max(model_config.num_gpus, TP)
reason = current_vllm_config
```

這代表：

```text
vLLM can execute the currently configured TP shape.
SLLM control plane can still represent DP by launching replicas.
PP / EP are not claimed for dense vLLM in this first V6 contract.
```

### V6 MoE vLLM Config

MoE vLLM 這邊使用已驗證的 allowlist。現在可宣告的 config 是：

| TP | DP | PP | EP | num_gpus | reason |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 1 | 1 | 1 | 4 | `verified_vllm_moe_config` |
| 4 | 1 | 1 | 2 | 4 | `verified_vllm_moe_config` |
| 2 | 1 | 1 | 1 | 2 | `verified_vllm_moe_config` |
| 2 | 2 | 1 | 1 | 4 | `verified_vllm_moe_config` |
| 2 | 1 | 1 | 2 | 2 | `verified_vllm_moe_config` |
| 2 | 2 | 1 | 2 | 4 | `verified_vllm_moe_config` |
| 1 | 4 | 1 | 1 | 4 | `verified_vllm_moe_config` |
| 1 | 4 | 1 | 2 | 4 | `verified_vllm_moe_config` |
| 2 | 1 | 2 | 1 | 4 | `verified_vllm_moe_config` |
| 2 | 1 | 2 | 2 | 4 | `verified_vllm_moe_config` |

這裡的 `DP=N` 表示 SLLM control plane 的 N 個 replicas，不是單一 vLLM
engine 內部的 DP resharding。`PP=2` 則是已用 vLLM pipeline parallel 實測過的
4-GPU config。對應到 `ParallelPlan` 時：

```text
data_parallel_size = DP
num_replicas = DP
num_gpus = TP * DP * PP
pipeline_parallel_size = PP
```

這裡的 `EP=2` 是 CPY planner 看到的 capability metadata。vLLM backend config
實際開 expert parallel 的欄位是：

```json
{
  "enable_expert_parallel": true
}
```

也就是：

```text
enable_expert_parallel = false -> expert_parallel_size = 1
enable_expert_parallel = true  -> expert_parallel_size = 2
```

MoE 判斷第一版可以先用 config 裡的 model name / model path 判斷，例如：

```text
model contains "moe"
or backend_config.pretrained_model_name_or_path contains "moe"
```

目前 MoE capability 可以宣告：

```text
supports_tp = True
supports_dp = True
supports_ep = True
supports_state_export = False
supports_state_restore = False
max_num_gpus = 4
```

`supports_ep=True` 只表示 allowlist 裡有已驗證的 EP config，不表示所有 TP / DP
組合都可以任意打開 EP。

## Ownership

大鼻主要碰這些檔案：

```text
sllm/backends/capability.py
sllm/backends/backend_utils.py
sllm/backends/vllm_backend.py
sllm/backends/vllm_capability.py
sllm/backends/vllm_context_metadata.py
sllm/backends/vllm_state_metadata.py
sllm/backends/vllm_runtime_metadata.py
tests/spotserve_test/test_backend_capability.py
tests/spotserve_test/test_vllm_context_metadata.py
tests/spotserve_test/test_vllm_state_metadata.py
tests/spotserve_test/test_vllm_runtime_metadata.py
```

大鼻原則上不要改：

```text
sllm/controller.py
sllm/routers/
sllm/schedulers/
sllm/spot/reparallelization.py
```

如果 capability 需要 CPY planner 支援新的欄位，先在文件或 PR comment
寫清楚 contract，讓 CPY 這邊接。

## Shared Interface

CPY 已經定義了 `ParallelPlan`：

```python
@dataclass(frozen=True)
class ParallelPlan:
    model_name: str
    backend: str
    tensor_parallel_size: int
    data_parallel_size: int
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    num_replicas: int = 1
    num_gpus: int = 1
    target_nodes: list[str] = field(default_factory=list)
    reason: str = "replan"
```

大鼻要補 `BackendCapability`：

```python
@dataclass(frozen=True)
class BackendCapability:
    backend: str
    model_name: str
    supports_tp: bool
    supports_dp: bool
    supports_ep: bool
    supports_state_export: bool
    supports_state_restore: bool
    max_num_gpus: int
    supported_configs: list[ParallelPlan]
```

建議放在：

```text
sllm/backends/capability.py
```

或如果想少新增檔案，也可以放：

```text
sllm/backends/backend_utils.py
```

但比較推薦獨立檔案，避免 `backend_utils.py` 變成雜物區。

## V6 Implementation

第一版請先保守，不要假裝 vLLM 已經支援所有 dynamic TP / DP / EP。
dense vLLM 只宣告目前設定中的 TP shape；MoE vLLM 只宣告上面列出的
verified allowlist。

建議新增：

```text
sllm/backends/capability.py
sllm/backends/vllm_capability.py
```

### `sllm/backends/capability.py`

```python
from dataclasses import dataclass
from typing import Any, Mapping

from sllm.spot.reparallelization import ParallelPlan


@dataclass(frozen=True)
class BackendCapability:
    backend: str
    model_name: str
    supports_tp: bool
    supports_dp: bool
    supports_ep: bool
    supports_state_export: bool
    supports_state_restore: bool
    max_num_gpus: int
    supported_configs: list[ParallelPlan]


def get_backend_capability(
    model_config: Mapping[str, Any],
) -> BackendCapability | None:
    backend = model_config.get("backend")
    if backend == "vllm":
        from sllm.backends.vllm_capability import get_vllm_capability

        return get_vllm_capability(model_config)
    return None
```

### `sllm/backends/vllm_capability.py`

最小 dense vLLM path 可以長這樣：

```python
from typing import Any, Mapping

from sllm.backends.capability import BackendCapability
from sllm.spot.reparallelization import ParallelPlan


def get_vllm_capability(
    model_config: Mapping[str, Any],
) -> BackendCapability:
    model_name = str(model_config["model"])
    backend_config = model_config.get("backend_config", {})
    configured_num_gpus = int(model_config.get("num_gpus", 1) or 1)
    tensor_parallel_size = int(
        backend_config.get("tensor_parallel_size", configured_num_gpus) or 1
    )

    num_gpus = max(configured_num_gpus, tensor_parallel_size)

    supported_configs = [
        ParallelPlan(
            model_name=model_name,
            backend="vllm",
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            num_replicas=1,
            num_gpus=num_gpus,
            reason="current_vllm_config",
        )
    ]

    return BackendCapability(
        backend="vllm",
        model_name=model_name,
        supports_tp=True,
        supports_dp=True,
        supports_ep=False,
        supports_state_export=False,
        supports_state_restore=False,
        max_num_gpus=num_gpus,
        supported_configs=supported_configs,
    )
```

這版的意思是：

```text
dense vLLM supports exactly the currently configured TP shape.
SLLM control plane can represent DP as replicas.
EP/state restore are not claimed for dense vLLM yet.
```

這樣是安全的，因為 planner 不會選到 backend 沒承諾能跑的 config。

MoE vLLM path 則需要另外產生 verified allowlist：

```python
VLLM_MOE_SUPPORTED_SHAPES = (
    {"tensor_parallel_size": 4, "data_parallel_size": 1, "pipeline_parallel_size": 1, "expert_parallel_size": 1},
    {"tensor_parallel_size": 4, "data_parallel_size": 1, "pipeline_parallel_size": 1, "expert_parallel_size": 2},
    {"tensor_parallel_size": 2, "data_parallel_size": 1, "pipeline_parallel_size": 1, "expert_parallel_size": 1},
    {"tensor_parallel_size": 2, "data_parallel_size": 2, "pipeline_parallel_size": 1, "expert_parallel_size": 1},
    {"tensor_parallel_size": 2, "data_parallel_size": 1, "pipeline_parallel_size": 1, "expert_parallel_size": 2},
    {"tensor_parallel_size": 2, "data_parallel_size": 2, "pipeline_parallel_size": 1, "expert_parallel_size": 2},
    {"tensor_parallel_size": 1, "data_parallel_size": 4, "pipeline_parallel_size": 1, "expert_parallel_size": 1},
    {"tensor_parallel_size": 1, "data_parallel_size": 4, "pipeline_parallel_size": 1, "expert_parallel_size": 2},
    {"tensor_parallel_size": 2, "data_parallel_size": 1, "pipeline_parallel_size": 2, "expert_parallel_size": 1},
    {"tensor_parallel_size": 2, "data_parallel_size": 1, "pipeline_parallel_size": 2, "expert_parallel_size": 2},
)
```

如果未來實測更多 MoE config，例如 `TP=4, DP=2` 或更大的
expert parallel setting，再把那些 config 加進 allowlist。CPY planner 只應該
選 `supported_configs` 裡出現的 plan。

## What CPY Will Consume

CPY planner 之後會這樣用：

```text
available GPUs
-> generate planner candidates
-> convert candidate to ParallelPlan
-> filter by BackendCapability.supported_configs
-> select best legal ParallelPlan
```

所以大鼻提供的 `supported_configs` 必須是「backend 能執行」的集合。

不要在 capability 裡做 policy decision，例如：

```text
preempt 後應該選 TP=2 還是 DP=4
```

這是 CPY planner 的責任。

大鼻只回答：

```text
TP=2, DP=1, EP=1 可不可以跑？
TP=1, DP=4, EP=1 可不可以跑？
TP=2, DP=1, EP=2 可不可以跑？
```

## Tests

建議新增：

```text
tests/spotserve_test/test_backend_capability.py
```

第一版測這些即可：

```python
def test_dense_vllm_capability_reports_current_tp_config():
    capability = get_backend_capability(
        {
            "model": "vllm-dense",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {
                "tensor_parallel_size": 2,
            },
        }
    )

    assert capability is not None
    assert capability.backend == "vllm"
    assert capability.supports_tp is True
    assert capability.supports_dp is True
    assert capability.supports_ep is False
    assert capability.supports_state_export is False
    assert capability.supports_state_restore is False
    assert capability.supported_configs[0].tensor_parallel_size == 2
    assert capability.supported_configs[0].num_gpus == 4
```

MoE path 要測 verified allowlist：

```python
def test_vllm_moe_capability_reports_verified_allowlist():
    capability = get_backend_capability(
        {
            "model": "vllm-moe",
            "backend": "vllm",
            "num_gpus": 1,
            "backend_config": {
                "pretrained_model_name_or_path": "Qwen/Qwen1.5-MoE-A2.7B",
                "tensor_parallel_size": 1,
                "trust_remote_code": True,
            },
        }
    )

    configs = {
        (
            plan.tensor_parallel_size,
            plan.data_parallel_size,
            plan.pipeline_parallel_size,
            plan.expert_parallel_size,
            plan.num_gpus,
        )
        for plan in capability.supported_configs
    }

    assert capability.supports_ep is True
    assert capability.max_num_gpus == 4
    assert configs == {
        (4, 1, 1, 1, 4),
        (4, 1, 1, 2, 4),
        (2, 1, 1, 1, 2),
        (2, 2, 1, 1, 4),
        (2, 1, 1, 2, 2),
        (2, 2, 1, 2, 4),
        (1, 4, 1, 1, 4),
        (1, 4, 1, 2, 4),
        (2, 1, 2, 1, 4),
        (2, 1, 2, 2, 4),
    }
```

也要測 dense default：

```text
backend_config 沒有 tensor_parallel_size 時，預設 TP = model_config.num_gpus。
```

## Config Contract

大鼻可以先支援這些 config 欄位：

```text
model
backend
num_gpus
backend_config.pretrained_model_name_or_path
backend_config.tensor_parallel_size
backend_config.pipeline_parallel_size
backend_config.enable_expert_parallel
backend_config.supports_state_export
backend_config.supports_state_restore
```

如果某個欄位不存在，請保守 fallback：

```text
tensor_parallel_size = 1
pipeline_parallel_size = 1
enable_expert_parallel = false
expert_parallel_size = 1 in ParallelPlan metadata
state export/restore = false
```

MoE V6 第一版不從 arbitrary config 自動推導所有合法 shape，而是使用 verified
allowlist。也就是 config 看起來像 MoE 時，`supported_configs` 應該回傳上面
列出的 TP/DP/EP 組合；如果 planner 產生其他組合，CPY 要 filter 掉。

## Definition Of Done

大鼻這部分完成時，應該滿足：

- `BackendCapability` dataclass exists.
- `get_backend_capability(model_config)` exists.
- vLLM model config can return a conservative capability.
- `supported_configs` contains at least the current configured vLLM shape.
- MoE vLLM returns the verified TP/DP/EP allowlist.
- EP is true only for MoE allowlist configs that were verified.
- state export / restore are false unless大鼻已確認 backend 真的支援。
- unit tests cover dense vLLM TP config, dense default TP behavior, and MoE allowlist.
- no CPY router/scheduler/controller main-flow changes are required.

## Current Backend Answers For 大鼻

第一版 capability 已用保守方式回答：

- dense vLLM 只宣告目前設定中的 `tensor_parallel_size`，不宣告任意 DP / PP / EP。
- MoE vLLM 只宣告已實測過的 allowlist；EP=2 只代表 allowlist 中的 config 可用。
- state export / restore 不宣告 true KV restore；`supports_state_export = false`，
  `supports_state_restore = false`。
- dense `num_gpus` 取 `max(model_config.num_gpus, tensor_parallel_size)`，避免
  top-level config 比 TP 小時產生不可執行 plan。
- dynamic replan 不做線上改 TP；CPY 若選到新 plan，backend 應重建對應 actor。

Future work:

- 若要新增 `TP=4, DP=2` 或更大的 EP，必須先用 vLLM/MoE 實測通過再加入 allowlist。
- 若 vLLM 未來能 expose 更細的 expert / KV metadata，再擴充 capability metadata。

---

# Version 7 Backend Metadata Contract

CPY 已經可以先做 low-cost context migration planner：

```text
context metadata
-> cost matrix
-> fixed-warmup-cost assignment
-> MigrationPlan
-> migration metrics
```

大鼻在 V7 不需要做 matching。大鼻要提供的是：

```text
每個 request / instance 目前有哪些 context 可以 reuse
reuse 的 token / block 數量是多少
backend 能不能 export / restore state
```

CPY planner 會用這些 metadata 做：

```text
old instance -> new instance
```

的 minimum-cost mapping。

`MigrationTarget.warmup_cost` 是 target-level fixed cost。也就是同一個
target 如果接兩個 request，只算一次 warmup，不是每個 assignment 都算一次。
因此 CPY V7 在 `warmup_cost > 0` 時不是純 Hungarian/KM；它會解帶有 target
fixed opening cost 的 assignment。
CPY 輸出的 `cost_matrix` 是 marginal source-to-target cost；fixed warmup 會
反映在每個 `MigrationPlan.estimated_cost` 和 `total_estimated_cost`。

## CPY V7 Interface

CPY 會使用這些資料結構：

```python
@dataclass(frozen=True)
class ContextMetadata:
    request_id: str | None
    instance_id: str
    node_id: str
    num_tokens: int = 0
    context_blocks: int = 0
    reusable_tokens_by_target: Mapping[str, int] = field(default_factory=dict)
    reusable_blocks_by_target: Mapping[str, int] = field(default_factory=dict)
```

```python
@dataclass(frozen=True)
class MigrationTarget:
    instance_id: str
    node_id: str
    capacity: int = 1
    warmup_cost: float = 0.0
```

```python
@dataclass(frozen=True)
class MigrationPlan:
    request_id: str | None
    old_instance_id: str
    new_instance_id: str
    old_node_id: str
    new_node_id: str
    estimated_cost: float
    reusable_tokens: int = 0
    reusable_context_blocks: int = 0
    reason: str = "low_cost_mapping"
```

CPY owns:

```text
sllm/spot/context_migration.py
```

大鼻只需要讓 backend 能提供 `ContextMetadata` 的真實資料。

## What 大鼻 Should Provide For V7

新增 backend metadata helper：

```text
sllm/backends/vllm_context_metadata.py
```

第一版 helper 負責把 vLLM runtime metadata 正規化成 CPY 可讀的
`ContextMetadata` payload：

```python
from typing import Any, Mapping


def get_vllm_context_metadata(
    model_name: str,
    instance_id: str,
    node_id: str,
    runtime_metadata: Mapping[str, Any],
) -> dict:
    return {
        "request_id": runtime_metadata.get("request_id"),
        "instance_id": instance_id,
        "node_id": node_id,
        "model_name": model_name,
        "backend": "vllm",
        "num_tokens": _token_count(runtime_metadata),
        "context_blocks": context_block_count_from_runtime(runtime_metadata),
        "reusable_tokens_by_target": (
            runtime_metadata.get("reusable_tokens_by_target", {})
        ),
        "reusable_blocks_by_target": (
            runtime_metadata.get("reusable_blocks_by_target", {})
        ),
        "supports_state_export": bool(
            runtime_metadata.get("supports_state_export", False)
        ),
        "supports_state_restore": bool(
            runtime_metadata.get("supports_state_restore", False)
        ),
        "metadata": dict(runtime_metadata.get("metadata", {}) or {}),
    }
```

`token_count(runtime_metadata)` 的優先順序：

```text
1. runtime_metadata["num_tokens"]
2. len(runtime_metadata["tokens"])
3. len(runtime_metadata["prompt_tokens"]) + len(runtime_metadata["output_tokens"])
4. 0
```

`context_blocks` 的優先順序：

```text
1. runtime_metadata["context_blocks"]
2. runtime_metadata["kv_block_count"] / ["num_kv_blocks"]
3. len(runtime_metadata["block_ids"] / ["kv_block_ids"])
4. block count from runtime_metadata["block_table"]
5. the same explicit fields under runtime_metadata["kv_transfer_params"]
6. 0
```

也就是說，如果 vLLM runtime / scheduler / block manager 能透過
`kv_transfer_params` 或其他明確欄位 expose block ids / block table，backend
helper 會把它轉成 `context_blocks`。如果沒有明確 block metadata，仍然回 0，
不使用 token count 假裝 KV block count。

`ContextMetadata.from_dict()` 目前只消費 migration planner 需要的核心欄位：

```text
request_id
instance_id
node_id
num_tokens
context_blocks
reusable_tokens_by_target
reusable_blocks_by_target
```

`model_name`、`backend`、`supports_state_export`、`supports_state_restore`、
`metadata` 是 backend payload 的補充欄位。這些欄位讓 CPY 或 debug/reporting
可以知道資料來源和 state capability；即使 CPY V7 planner 暫時忽略，也應該由
backend 保守填上。

同時可以在 backend base class 補保守 hook：

```python
async def get_context_metadata(
    self,
    instance_id: str = "",
    node_id: str = "",
) -> list[dict]:
    return []
```

vLLM backend 則可以用目前 request trace 裡的 `RequestOutput` 產生 token-level
metadata：

```text
request_id = RequestOutput.request_id
model_name = self.model_name
backend = vllm
num_tokens = len(prompt_token_ids + output_token_ids)
metadata.prompt_token_count = len(prompt_token_ids)
metadata.generated_token_count = len(output_token_ids)
metadata.num_cached_tokens = RequestOutput.num_cached_tokens
metadata.kv_transfer_params_available = RequestOutput.kv_transfer_params is not None
context_blocks = explicit KV block count if safely exposed, else 0
reusable_tokens_by_target = {}
reusable_blocks_by_target = {}
supports_state_export = false
supports_state_restore = false
```

第一版可以很保守：

```text
num_tokens = known generated/prompt token count if available
context_blocks = 0 if backend cannot expose KV blocks yet
reusable_* = empty dict if backend cannot prove reuse
```

如果 vLLM 只能提供 token ids，不能提供 KV blocks，請明確表示：

```text
context_blocks = 0
supports_state_export = false
supports_state_restore = false
```

這樣 CPY planner 仍可做 token-level estimated mapping，但不會假裝已經有
KV migration。

### Current V7 Status And Remaining Backend Work

目前大鼻提供的 `get_vllm_context_metadata()` schema 已經可以取代
`benchmarks/spotserve/context_migration_synthetic.json` 裡的 `sources` 欄位：

```text
VllmBackend.get_context_metadata()
-> ContextMetadata.from_dict()
-> plan_low_cost_migration()
```

CPY router 也已經能在 spot preemption / dead-node live path 中呼叫
`get_context_metadata()`，並用 READY instances 自動建立 migration targets。

但這只代表 V7 的 live planning path 已接上。若要讓 live metadata 真正取代
synthetic benchmark 裡的 rich context data，大鼻需要提供或保守回答下列資料：

```text
request_id: active request id, stable across export / restore / replay
num_tokens: prompt + generated token count for the active request
context_blocks: active KV cache block count for that request
reusable_tokens_by_target: tokens reusable on each target instance/node
reusable_blocks_by_target: KV blocks reusable on each target instance/node
supports_state_export: whether backend can export real request/KV state
supports_state_restore: whether backend can restore real request/KV state
```

目前大鼻可以提供 / 保守回答：

```text
request_id: yes, from RequestOutput.request_id for active requests
num_tokens: yes, from prompt_token_ids + generated token_ids
context_blocks: yes if explicit block metadata or kv_transfer_params is exposed,
                otherwise conservative 0
supports_state_export: false for true request/KV state
supports_state_restore: false for true request/KV state
```

目前不能安全提供非空值的是 target-specific reuse：

```text
reusable_blocks_by_target
reusable_tokens_by_target
```

原因是現在 live path 只把 source instance 資訊傳給 backend：

```text
CPY -> VllmBackend.get_context_metadata(instance_id, node_id)
```

backend 可以知道 source request 的 token / cache metadata，但不知道每個
target instance/node 目前有哪些 prefix cache 或 KV blocks。因此現在合理輸出是：

```text
reusable_tokens_by_target = {}
reusable_blocks_by_target = {}
```

若要填非空值，需要先有 target-specific 證據，例如 CPY 把 target list 傳給
backend，或 backend/runtime 自己維護 prefix-cache / KV-transfer registry。
在沒有這些資料前，不能把 `num_cached_tokens`、`context_blocks` 或 prefix
length 直接填給所有 target。

MoE 目前不需要 expert metadata 才能做 token-level planning；若未來要宣稱
MoE KV/expert-level reuse，才需要補 expert routing / expert-parallel state。

所以現在 CPY planner 可以跑，但 cost / reuse ratio 是保守估計；不能宣稱已經
完成真正 low-cost KV block context migration，也不能拿來公平比較「V7 比
naive 快多少」。

大鼻不需要實作 matching algorithm，也不需要決定 target。大鼻只需要把上述
metadata 從 vLLM runtime / scheduler / block manager 裡 expose 出來。CPY 會
負責：

```text
target selection
capacity handling
warmup cost
minimum-cost assignment
context_migration metrics
```

這裡要注意：V7 backend hook 只提供 metadata，不做 matching，也不決定要把
request 搬去哪個 target。matching 仍然由 CPY 的
`sllm/spot/context_migration.py` 負責。

## CPY KV Cache Warmup Path

CPY 已在 router live path 補上 opt-in KV cache warmup：

```text
router_config.enable_context_migration = true
router_config.enable_kv_cache_migration = true

spot preemption/dead event
-> source backend get_context_metadata()
-> CPY plan_low_cost_migration()
-> source backend get_current_tokens()
-> target backend resume_kv_cache(request_datas=tokens)
-> context_migration.kv_cache_migration metric
```

這條路徑會用既有 `resume_kv_cache()` 在 target 做 token replay / prefix-cache
warmup。它可以驗證 control plane 是否真的把 source tokens 送到 target
backend，也會在 benchmark summary 產生：

```text
kv_cache_migration_attempts
kv_cache_migration_successes
kv_cache_migration_tokens
kv_cache_migration_latest
```

但這仍然不是 true vLLM KV block serialization / transfer / request binding。
目前 checkout 裡 `VllmBackend.supports_state_restore()` 仍然是 `False`，
`restore_inference_state()` 仍回 `vllm_kv_restore_not_available`。等大鼻補上
真正 restore hook 後，CPY 才能把 V8 stateful recovery 從 token replay
fallback 升級成真正 KV restore。

## Reuse Metadata Meaning

`reusable_tokens_by_target` 和 `reusable_blocks_by_target` 的 key 可以先用：

```text
target instance_id
```

或：

```text
target node_id
```

例如：

```json
{
  "request_id": "req-1",
  "instance_id": "old-vllm-0",
  "node_id": "node-0",
  "num_tokens": 512,
  "context_blocks": 32,
  "reusable_tokens_by_target": {
    "new-vllm-0": 512,
    "node-0": 512
  },
  "reusable_blocks_by_target": {
    "new-vllm-0": 32,
    "node-0": 32
  }
}
```

意思是：

```text
如果搬到 new-vllm-0 或 node-0，可以 reuse 512 tokens / 32 blocks。
```

如果大鼻不知道 target-specific reuse，先留空：

```json
"reusable_tokens_by_target": {},
"reusable_blocks_by_target": {}
```

CPY planner 會用 conservative default 估算。

## V7 Current Backend Answers

第一版 vLLM backend metadata 已採保守答案：

| Question | Current answer |
| --- | --- |
| vLLM runtime 是否能列出目前 active request ids？ | 可以。`VllmBackend.request_trace` 保留 active request id，並可透過 `return_all_request_ids()` / `return_all_results()` 讀取。注意：`trace_debug=false` 時 completed request 會被刪除，所以這裡只代表 ongoing request。 |
| 每個 request 是否能拿到 prompt tokens + generated tokens？ | 可以拿到 token ids。`RequestOutput.prompt_token_ids` 提供 prompt tokens，`RequestOutput.outputs[0].token_ids` 提供目前 generated tokens。V7 metadata 會回報 `num_tokens`，並在 `metadata.prompt_token_count` / `metadata.generated_token_count` 保留數量。 |
| 每個 request 是否能知道 KV cache block count？ | 如果 vLLM `RequestOutput.kv_transfer_params` 或 runtime metadata 明確提供 `context_blocks` / `kv_block_count` / `block_ids` / `block_table`，helper 會回報 block count；否則保守回 `context_blocks = 0`。 |
| block id / block table 是否能安全 expose？ | backend helper 可以讀取已提供的 block ids / block table 來計算 count，但 V7 context metadata 第一版不把 raw block table 交給 CPY planner。 |
| 同 node restore 是否能 reuse GPU KV blocks？ | 目前不能宣告。沒有 true KV restore hook 前，same-node reuse 不標成可用。 |
| cross node restore 是否只能 token replay？ | 是。第一版 cross-node 只支援 fallback token replay / retry，不做 KV transfer。 |
| vLLM 是否有 state export hook？ | 有 backend-level token snapshot export：`VllmBackend.export_inference_state()`。但這不是 vLLM true KV export，所以 V7 context metadata 中 `supports_state_export = false`。 |
| vLLM 是否有 state restore hook？ | 沒有 true KV restore。`VllmBackend.restore_inference_state()` 目前回 `restored = false`、`reason = vllm_kv_restore_not_available`。 |
| MoE request 是否需要 expert metadata 才能正確估 reuse？ | 對目前 token-level estimated migration 不需要；若未來要估 KV/expert-level reuse，才需要 expert/routing metadata。 |

目前保守輸出：

```text
supports_state_export = false
supports_state_restore = false
context_blocks = explicit KV block count if safely exposed, else 0
reusable_tokens_by_target = {}
reusable_blocks_by_target = {}
```

Future work 是確認 vLLM 是否能安全 expose target-specific same-node reuse、
cross-node transfer，以及 MoE expert/routing metadata。

## V7 Definition Of Done For 大鼻

大鼻 V7 metadata 部分完成時，應該滿足：

- backend helper can return context metadata for vLLM instances.
- metadata includes request id, instance id, node id, token count if available.
- KV block fields are present, even if conservative zero.
- complete V7 requires non-zero real `context_blocks` when KV blocks exist.
- complete V7 requires real target-specific `reusable_tokens_by_target` and
  `reusable_blocks_by_target`, or an explicit explanation that reuse is not
  available for that target/backend.
- state export / restore capability is explicit.
- `VllmBackend.get_context_metadata()` can return conservative token-level
  metadata from active request traces.
- live metadata can replace synthetic `sources` input without losing token,
  block, or reuse information needed by the planner.
- no matching algorithm is implemented in backend code.
- no CPY router/scheduler/controller main-flow changes are required.

建議測試：

```text
tests/spotserve_test/test_vllm_context_metadata.py
```

至少覆蓋：

```text
tokens -> num_tokens
prompt_tokens + output_tokens -> num_tokens
explicit reusable_*_by_target maps are preserved
payload can be parsed by ContextMetadata.from_dict()
```

CPY 會把這些 metadata 接進：

```text
sllm/spot/context_migration.py
```

並產生：

```text
MigrationPlan
context_migration metrics
estimated reuse ratio
```

---

# Version 8 Backend State Restore Contract

CPY 已經可以先做 stateful recovery control flow：

```text
request failure / preemption
-> export InferenceState
-> decide restore_state / fallback_token_replay / retry
-> restore state on target backend when supported
-> state_recovery metrics
```

大鼻在 V8 不需要決定 recovery policy。大鼻要提供的是：

```text
backend 能不能 export inference state
backend 能不能 restore inference state
exported state 到底包含哪些 vLLM/KV metadata
```

CPY policy 會根據 backend 回答做：

```text
supports_state_restore=true  -> try restore_state
supports_state_restore=false -> fallback generated-token replay / retry
```

## CPY V8 Interface

CPY 會使用：

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

CPY owns:

```text
sllm/spot/stateful_recovery.py
sllm/routers/roundrobin_router.py
```

## Backend Hooks 大鼻要接

Base backend 已經有 conservative default：

```python
async def supports_state_restore(self) -> bool:
    return False

async def export_inference_state(
    self,
    request_data: dict | None = None,
    current_output: list[list[int]] | None = None,
    completed_tokens: int | None = None,
) -> dict:
    ...

async def restore_inference_state(
    self,
    state: dict,
    request_data: dict | None = None,
) -> dict:
    ...
```

大鼻要在 vLLM backend 補真實版本。第一版如果做不到真 KV restore，請明確保持：

```text
supports_state_restore = false
state_kind = "token_snapshot"
```

這樣 CPY 會 fallback，不會假裝有 true stateful recovery。

V8 backend-side 第一版新增：

```text
sllm/backends/vllm_state_metadata.py
```

helper 會產生 CPY `InferenceState.from_dict()` 可讀的 payload：

```text
request_id
instance_id
node_id
backend = vllm
model_name
tokens
completed_tokens
state_kind = token_snapshot
supports_restore = false
metadata.cache_engine = vllm
metadata.can_restore_same_node = false
metadata.can_restore_cross_node = false
metadata.reason = vllm_kv_restore_not_available
```

`VllmBackend.export_inference_state()` 可以先從 `current_output` 或
`get_current_tokens()` 取得 token snapshot。`VllmBackend.restore_inference_state()`
第一版必須保守回：

```json
{
  "restored": false,
  "reason": "vllm_kv_restore_not_available"
}
```

也就是說，V8 第一版只讓 CPY 有明確 token snapshot 可以做 fallback token
replay；不宣告真 KV restore。

## vLLM State Metadata 建議

如果 vLLM 能 expose KV/cache metadata，`InferenceState.metadata` 可以包含：

```json
{
  "prompt_token_count": 128,
  "generated_token_count": 32,
  "kv_block_count": 10,
  "block_ids": [],
  "block_table": {},
  "cache_engine": "vllm",
  "can_restore_same_node": false,
  "can_restore_cross_node": false
}
```

第一版可以保守：

```json
{
  "tokens": [1, 2, 3],
  "completed_tokens": 3,
  "state_kind": "token_snapshot",
  "supports_restore": false,
  "metadata": {
    "reason": "vllm_kv_restore_not_available"
  }
}
```

## V8 Current Backend Answers

第一版 vLLM state hook 提供 token snapshot，不宣稱 true KV restore：

- active request token snapshot 由 `get_current_tokens()` / `request_trace` 取得。
- request token snapshot 可包含 prompt token ids + generated token ids。
- KV block table / block ids 預設為空，metadata 標註
  `vllm_kv_restore_not_available`。
- same-node / cross-node restore 都回報不可用。
- `restore_inference_state()` 回傳 `restored = false`，讓 CPY fallback。
- MoE expert/routing state 尚未保存。

Future work 是確認 vLLM 是否能提供穩定 KV block ownership、serialize / transfer
KV blocks，以及 restore 後 request id binding 語意。

## V8 Definition Of Done For 大鼻

大鼻 V8 metadata / runtime 部分完成時，應該滿足：

- `VllmBackend.supports_state_restore()` 回答真實 capability。
- `export_inference_state()` 至少能回傳 request id、tokens、completed tokens。
- 如果不能真 KV restore，`supports_restore=false`。
- 如果能真 KV restore，metadata 需說清楚 block ids / block table / node
  限制。
- `restore_inference_state()` 回傳 `{ "restored": true }` 之前，必須真的讓
  後續 generation 從 state 接續。
- backend 不實作 CPY policy decision，不實作 matching，不改 scheduler。
- unit tests cover vLLM token snapshot payload and compatibility with
  `InferenceState.from_dict()`.

建議測試：

```text
tests/spotserve_test/test_vllm_state_metadata.py
```

CPY 會根據這些 hooks 產生：

```text
StateRecoveryPlan
state_recovery metrics
fallback generated-token replay / retry
```

---

# Version 9 Backend Runtime Metadata Contract

CPY 已經可以做 spot-risk-aware scheduling：

```text
worker node metadata
-> risk score
-> ranked ready nodes
-> allocation decision
```

大鼻在 V9 不需要寫 scheduler。大鼻要提供的是：

```text
model loading cost
model resource profile
GPU usage / free capacity
optional spot risk / remaining lifetime metadata
```

CPY scheduler 會根據這些資料排序 node。

## CPY V9 Interface

CPY 使用：

```python
@dataclass(frozen=True)
class NodeRiskScore:
    node_id: str
    state: str
    free_gpu: int
    total_gpu: int
    spot_risk: float
    remaining_lifetime_s: float
    loading_cost: float
    score: float
    reason: str = "risk_aware_ranking"
```

```python
@dataclass(frozen=True)
class SchedulingDecision:
    action: str
    model_name: str
    requested_gpus: int
    selected_node_id: str | None
    candidates: list[NodeRiskScore]
    reason: str = "risk_aware_scheduling"
```

CPY owns:

```text
sllm/spot/risk_aware_scheduling.py
sllm/schedulers/fcfs_scheduler.py
```

## Metadata 大鼻可以提供

Node-level metadata:

```json
{
  "node_id": "node-0",
  "free_gpu": 2,
  "total_gpu": 4,
  "spot_risk": 0.25,
  "remaining_lifetime_s": 1800,
  "loading_cost": 12.5
}
```

Model-level metadata:

```json
{
  "model_name": "vllm-moe",
  "num_gpus": 2,
  "estimated_load_time_s": 20.0,
  "gpu_memory_required_gb": 40.0
}
```

第一版可以保守：

```text
spot_risk = 0.0 if unknown
remaining_lifetime_s = default_remaining_lifetime_s
loading_cost = measured/estimated model load time if available, else 0
```

如果大鼻不能提供 spot provider risk，CPY 仍可用 synthetic risk metadata 做
benchmark，不會阻塞 Version 9 CPY side。

V9 backend-side 第一版新增：

```text
sllm/backends/vllm_runtime_metadata.py
```

它提供兩種 helper：

```text
get_vllm_model_resource_profile(...)
get_vllm_runtime_metadata(...)
```

model resource profile 會回傳：

```text
model_name
backend = vllm
num_gpus = TP * PP * DP
tensor_parallel_size
pipeline_parallel_size
data_parallel_size
expert_parallel_enabled
estimated_load_time_s
gpu_memory_required_gb
gpu_memory_utilization
max_model_len
max_num_seqs
```

runtime metadata 會回傳 CPY risk-aware scheduler 可讀的欄位：

```text
node_id
free_gpu
total_gpu
spot_risk
remaining_lifetime_s
loading_cost
```

`VllmBackend.init_backend()` 可以記錄 engine load time，並讓
`VllmBackend.get_runtime_metadata()` 回傳 conservative runtime metadata。
如果 backend 不知道 GPU free capacity 或 spot risk，先回 `0` 或空 metadata，
不要自行做 ranking。

CPY scheduler 現在可以 opt in live backend metadata refresh：

```json
{
  "scheduler_config": {
    "enable_spot_risk_aware": true,
    "enable_backend_runtime_metadata": true,
    "runtime_metadata_timeout_s": 1.0
  }
}
```

live path 會做：

```text
model_instance[model_name][instance_id] -> node_id
ray.get_actor(instance_id)
-> VllmBackend.get_runtime_metadata(instance_id, node_id)
-> merge into worker node metadata
-> node_risk_score()
```

大鼻仍然不需要實作 scheduler ranking。大鼻只要讓 backend actor 的
`get_runtime_metadata()` 回傳準確或明確保守的欄位即可。多個 instance 在同一
node 時，CPY 會用保守聚合：

```text
spot_risk = max(values)
remaining_lifetime_s = min(values)
loading_cost = max(values)
```

如果沒有 cloud provider，`spot_risk` / `remaining_lifetime_s` 可以繼續由
`scheduler_config.node_risk` 或 synthetic benchmark 提供；backend metadata
主要先提供 loading cost 和 model resource profile。

## V9 Current Backend Answers

第一版 vLLM runtime metadata 已提供：

- `VllmBackend.init_backend()` 會記錄 model load time，作為 loading cost。
- resource profile 會回報 TP / PP / DP、EP enabled flag、`max_model_len`、
  `max_num_seqs`、`gpu_memory_utilization`。
- GPU usage / free capacity、spot risk、remaining lifetime 若 runtime 沒提供，
  就回 conservative defaults。
- cloud provider risk predictor 不在 backend 實作；CPY benchmark 可用 synthetic
  metadata。
- MoE loading cost 目前不做獨立預估；若 runtime 提供實測 load time，profile 會
  使用該值。

## V9 Definition Of Done For 大鼻

大鼻 V9 metadata 部分完成時，應該滿足：

- model resource profile 可以被 CPY scheduler 讀取。
- loading cost 有 conservative estimate。
- GPU usage / free capacity metadata 是最新或明確標註 stale。
- spot risk / remaining lifetime 如果不可用，要明確留空或用 default。
- backend 不做 scheduler ranking，不改 CPY policy decision。
- unit tests cover vLLM resource profile and risk-aware scheduler compatible
  runtime metadata.

建議測試：

```text
tests/spotserve_test/test_vllm_runtime_metadata.py
```

CPY 會根據這些 metadata 產生：

```text
SchedulingDecision
risk_aware_scheduling metrics
health-only vs risk-aware benchmark comparison
```
