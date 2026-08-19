# MoE-aware SpotServe Planner

分析日期：2026-08-19

## 研究定位

不要只把 SpotServe 原本針對 dense Transformer 的方法直接套到 MoE。MoE
inference 多了 expert placement、expert routing skew、expert dispatch
communication，所以 SpotServe 的三個核心應該改成 MoE-aware 版本。

建議對外說法：

```text
We extend SpotServe with MoE-aware re-parallelization,
expert-locality-aware context migration, and expert-compatible
stateful recovery.
```

## 核心想法

原本 SpotServe 主要處理：

```text
GPU availability
-> TP / PP / DP parallel plan
-> context / KV migration
-> stateful recovery
```

MoE 版本應該變成：

```text
GPU availability
+ expert placement
+ global expert hotness
+ per-request routed-token history
+ recent-window expert hotness
+ all-to-all communication cost
+ KV cache compatibility
-> MoE-aware ParallelPlan
-> expert-locality-aware migration
-> expert-compatible recovery
```

## 新增 Planner 架構

建議新增：

```text
sllm/spot/moe_placement.py
```

它不取代現有三個 planner，而是成為 MoE metadata 與三大核心之間的共同層。

```text
vLLM runtime metadata
  -> MoEExpertMetadata
  -> ExpertPlacementState
  -> ExpertPlacementPlan
  -> Reparallelization / Context Migration / Stateful Recovery
```

## Runtime Metadata

MoE-aware planner 至少需要以下 metadata：

```text
model_name
model_revision
num_layers
num_experts
top_k
tensor_parallel_size
pipeline_parallel_size
expert_parallel_size
expert_parallel_enabled
expert_ids_by_rank
rank_id
node_id
gpu_id
expert_weight_size_bytes
routed_tokens_by_expert
routed_tokens_by_layer
per_request_routed_tokens_by_expert
recent_window_routed_tokens_by_expert
expert_load
expert_cache_warm
```

其中 routed-token statistics 應分成三層，不要全部混成同一個 histogram：

| Metadata | 用途 |
|---|---|
| `global_expert_hotness` | placement / load balance / hot expert replication |
| `per_request_expert_route_histogram` | migration target locality hint |
| `recent_window_expert_hotness` | 短期 dispatch cost prediction |

`per_request_expert_route_histogram` 很重要，但它只是 historical locality
hint，不是 correctness requirement。Autoregressive generation 的後續 token
routing 可能改變，所以歷史 histogram 只能影響 cost model，不能直接決定一個
target 能不能 restore。

## Suggested Data Structures

```python
@dataclass(frozen=True)
class ExpertShard:
    layer_id: int
    expert_id: int
    rank_id: str
    node_id: str
    gpu_id: str
    weight_size_bytes: int = 0
    routed_tokens: int = 0
    load_score: float = 0.0


@dataclass(frozen=True)
class ExpertPlacementState:
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    shards: list[ExpertShard]


@dataclass(frozen=True)
class ExpertPlacementPlan:
    model_name: str
    target_parallel_plan: ParallelPlan
    expert_to_target_rank: dict[str, str]
    moved_expert_count: int
    moved_weight_bytes: int
    estimated_dispatch_cost: float
    estimated_load_balance_penalty: float
    reason: str
```

`expert_to_target_rank` 的 key 可以用：

```text
"layer:{layer_id}/expert:{expert_id}"
```

## Core 1: MoE-aware Re-parallelization

### 原本做法

目前 `ParallelPlan` 主要描述：

```text
TP / PP / DP / EP size
target nodes
num replicas
```

這對 dense model 夠用，但對 MoE 不夠，因為 EP size 只說「切幾份」，沒有說：

- 哪些 expert 在哪張 GPU
- hot experts 是否集中在同一張 GPU
- preempted GPU 上有哪些 experts
- 搬 expert weights 的成本是多少
- 新 placement 是否增加跨 node expert dispatch

### MoE-aware 做法

MoE-aware re-parallelization 應該同時選：

```text
TP / PP / DP / EP
+ expert placement
+ target node / rank mapping
```

建議 scoring：

```text
score =
  gpu_utilization_bonus
+ throughput_estimate
- model_load_cost
- expert_weight_movement_cost
- hot_expert_imbalance_penalty
- cross_node_dispatch_penalty
- unavailable_expert_penalty
```

候選 plan 必須滿足：

```text
ready GPUs >= required GPUs
all required experts are covered by the placement,
  subject to the runtime's replication/partition semantics
hot experts are not overloaded on one GPU
TP/PP/EP shape is supported by runtime
KV/cache layout remains restorable if in-flight requests will migrate
```

不要把 `all experts are placed exactly once` 寫成硬限制。實際 MoE serving
可能有 expert replication、shared experts、hot expert duplication、hybrid
TP+EP，或 runtime-specific partition semantics。planner 需要尊重 runtime
宣告的 placement semantics，而不是假設每個 expert 只能出現一次。

### Planner Flow

```text
spot event
-> update ready/preempting/dead nodes
-> collect current expert placement and routed-token histogram
-> generate TP/PP/DP/EP candidates
-> build expert placement for each candidate
-> estimate movement + dispatch + imbalance cost
-> select best MoE-aware plan
-> apply through vLLM deployment adapter
```

## Core 2: Expert-locality-aware Context Migration

### 關鍵觀念

MoE expert weights 通常不是 per-request state。真正的 request state 主要還是：

```text
attention KV cache
generated tokens
sampling state
request metadata
```

但 MoE migration target selection 不應該只看 KV cache。它也要看 request
接下來可能會用到哪些 experts。

### 新增 Request Routing Profile

每個 active request 可以維護：

```text
request_id
tokens
kv_blocks
expert_route_histogram
last_n_tokens_expert_histogram
top_experts
```

`expert_route_histogram` 形式：

```text
layer_id -> expert_id -> routed_token_count
```

### Target Scoring

context migration target 的 cost 應改成：

```text
cost =
  kv_transfer_or_recompute_cost
+ expert_dispatch_cost
+ queue_penalty
```

先保持 cost function 簡單，避免一開始就變成很難解釋的 heuristic soup。
更細的 warmup、cross-node、local-hot-expert bonus 可以先作為
`expert_dispatch_cost` 或 `kv_transfer_or_recompute_cost` 的內部估計，不要先把
每一項都暴露成獨立 weight。

其中 expert locality 的語意是：

```text
local_hot_expert_bonus
= request 的 hot experts 已在 target local GPU/rank 上的比例
```

這個值只代表「歷史 routing 暗示這個 target 可能比較便宜」。它不是 restore
correctness 條件，也不能保證 request 後續 token 一定繼續走同一批 experts。

### Planner Flow

```text
preempt/dead event
-> collect source request KV metadata
-> collect source request expert_route_histogram
-> collect target expert placement
-> compute KV compatibility
-> compute expert locality score
-> choose target with lowest total migration cost
-> restore KV if possible, otherwise token replay
```

## Core 3: Expert-compatible Stateful Recovery

### 原本做法

目前 stateful recovery 主要看：

```text
state_kind
tokens
completed_tokens
KV runtime_state
TP/PP/cache compatibility
same-node/cross-node restore support
```

### MoE-aware 做法

MoE recovery 應在 `InferenceState.metadata` 中加入：

```text
expert_parallel_size
expert_parallel_enabled
expert_placement_fingerprint
expert_route_histogram
gate_model_revision
moe_backend
top_k
```

target selection 應先檢查 restore correctness，再估計 expert locality：

```text
same model revision
same gate behavior / tokenizer / sampling config
compatible TP/PP/cache layout
expert locality score
remote expert dispatch cost
```

重要：KV restore correctness 和 expert locality 必須分開判斷。這是
MoE-aware V8 的核心原則。

```text
KV compatible != expert-locality optimal
expert placement changed != KV restore impossible
expert_route_histogram = historical locality hint, not a restore requirement
```

很多情況下 EP placement 改了，attention KV 仍然可以 restore。MoE expert
placement 主要影響後續 FFN dispatch cost，而不是已經產生的 attention KV
cache。因此 EP mismatch 不應該被自動當成 fallback 條件。

只有在 runtime/state encoding 對特定 EP layout 有硬相依時，EP mismatch 才是
restore incompatibility。否則：

```text
KV restore 可以成功
後續 expert dispatch cost 可能變高
metric 必須把兩者分開記錄
```

## Integration Points

### Existing Files To Extend

```text
sllm/backends/vllm_runtime_metadata.py
sllm/backends/vllm_context_metadata.py
sllm/backends/vllm_state_metadata.py
sllm/backends/vllm_backend.py
sllm/spot/reparallelization.py
sllm/spot/context_migration.py
sllm/spot/stateful_recovery.py
sllm/routers/roundrobin_router.py
```

### New File

```text
sllm/spot/moe_placement.py
```

### Config Proposal

```json
{
  "router_config": {
    "enable_reparallelization": true,
    "enable_context_migration": true,
    "recovery_policy": "stateful_recovery",
    "enable_moe_aware_planning": true,
    "moe_planner_config": {
      "expert_dispatch_weight": 1.0,
      "kv_cost_weight": 1.0,
      "queue_penalty_weight": 1.0,
      "hot_expert_window_tokens": 256,
      "allow_remote_expert_dispatch": true,
      "require_ep_compatible_restore": false
    }
  }
}
```

## Metrics

新增 metrics：

```text
moe_planning_events
moe_global_hot_experts
moe_request_hot_experts
moe_recent_window_hot_experts
moe_selected_expert_parallel_size
moe_moved_expert_count
moe_moved_weight_bytes
moe_hot_expert_locality_ratio
moe_estimated_dispatch_cost
moe_remote_expert_dispatch_tokens
moe_expert_rebalance_events
moe_kv_restore_compatible
moe_recovery_ep_compatible
moe_recovery_correctness_fallback
moe_recovery_locality_penalty
```

報告中應該分開呈現：

```text
reparallelization result
context/KV migration result
state restore result
expert placement result
```

避免只用 `success_rate` 代表整個系統成功。

## Milestones

### Milestone A: Metadata-only MoE Awareness

目標：先不改 placement，只收集與報告 MoE metadata。

完成條件：

- vLLM backend 回傳 `expert_parallel_enabled`、`expert_parallel_size`。
- 若 runtime 可取得，回傳 `expert_ids_by_rank`。
- 分開回報 global hotness、per-request route histogram、
  recent-window hotness。
- benchmark summary 顯示 MoE metadata。

### Milestone B: Expert-locality Target Selection

目標：先讓 context migration target selection MoE-aware，但不搬 expert
weights。

完成條件：

- active request 有 `expert_route_histogram`。
- target 有 expert placement metadata。
- planner 選 target 時考慮 hot expert locality。
- metric 顯示 `moe_hot_expert_locality_ratio`。

### Milestone C: MoE-aware Stateful Recovery

目標：recovery target selection 分離 KV restore correctness 與 expert locality。

完成條件：

- `InferenceState.metadata` 帶 MoE routing/placement fingerprint。
- target planner 先判斷 KV/cache restore compatibility。
- expert locality 只影響 target ranking，不直接否決 restore。
- 若 runtime/state encoding 對 EP layout 有硬相依，才用 EP mismatch 觸發
  correctness fallback。
- restore 後量測 remote expert dispatch / locality penalty。

### Milestone D: Expert-aware Re-parallelization

目標：re-parallelization planner 真的能決定 expert placement。

完成條件：

- `ExpertPlacementPlan` 可序列化到 metrics。
- preempted GPU 上的 experts 可被重新配置到 ready GPUs。
- planner cost 同時考慮 GPU capacity、expert movement、dispatch cost。

### Milestone E: Physical Cross-node Validation

目標：把 same-host simulation 擴展到真正多機 GPU。

完成條件：

- source/target 在不同 physical nodes。
- NIXL 或等價 transport 有正向 restore 結果。
- `can_restore_cross_node=true` 只在真實跨機驗證通過後開啟。

## Validation Matrix

| Test | Dense baseline | MoE black-box | KV-only | Expert-only | KV + expert locality |
|---|---:|---:|---:|---:|---:|
| Preempt idle worker | no new traffic | same | same | same | same |
| Preempt active worker | retry / replay / restore | same | prefer KV-compatible target | prefer expert-local target | optimize combined cost |
| Hot expert concentrated on lost GPU | not applicable | no special behavior | no special behavior | prefer local hot experts | balance KV reuse and dispatch |
| Target has KV but poor expert locality | KV target preferred | KV target preferred | KV target preferred | may choose expert-local target | cost model decides |
| EP layout changed | usually unsupported restore | unclear unless guarded | restore if KV compatible | locality may change | correctness and locality reported separately |
| Cross-node restore | only if supported | only if supported | KV transport cost | expert dispatch cost | separate KV transport and expert dispatch costs |

MoE-aware experiments 至少要拆成三個 baseline：

```text
A. KV-only target selection
B. Expert-locality-only target selection
C. KV + expert locality combined
```

否則如果 latency 變好，很難知道改善來自 KV reuse 還是 expert locality。

## 最小可行版本

最小可行的 MoE-aware extension 不需要一開始就搬 expert weights。可以先做：

```text
1. 收集 MoE metadata
2. 建立 global / per-request / recent-window routed-token statistics
3. 在 context migration target selection 加 expert locality score
4. 在 stateful recovery 中分離 KV restore correctness 與 expert locality
5. 報告 hot expert locality、remote dispatch cost、fallback 原因
```

這樣就能把研究重點從：

```text
SpotServe 可以跑在 MoE model 上
```

推進到：

```text
SpotServe 的 migration/recovery 決策會利用 MoE expert routing 特性。
```

建議先停在這個最小版本做完整實驗。它能回答一個乾淨的研究問題：

```text
在 spot preemption recovery 中，除了 KV locality 之外，
加入歷史 expert routing locality，能不能降低 recovery 後的
expert dispatch cost 或 tail latency？
```

真正動態搬 expert weights 可以留到後續階段，避免一開始就把工程範圍拉太大。

## 收斂後的 Phase Plan

```text
Phase 1
MoE metadata collection
-> expert placement
-> global/request-level/recent-window routed-token statistics

Phase 2
MoE-aware target selection
-> KV compatibility
-> expert locality
-> queue cost

Phase 3
MoE-aware stateful recovery
-> separate KV restore correctness from expert locality
-> measure remote expert dispatch after restore

Phase 4
True expert-aware re-parallelization
-> EP shape
-> expert remapping / replication
-> weight movement

Phase 5
physical cross-node validation
```

## 最安全的階段性 Claim

```text
We first implement a SpotServe-style control plane for vLLM MoE serving.
Then, we extend its planning decisions with MoE-specific runtime signals,
including expert placement and routed-token hotness, so that
re-parallelization, context migration, and stateful recovery can prefer
targets with better expert locality while preserving KV/cache compatibility.
```
