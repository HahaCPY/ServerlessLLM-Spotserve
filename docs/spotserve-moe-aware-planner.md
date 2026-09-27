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

## 目前實作邊界

目前 `docs/spotserve-version5-vllm-moe.md` 已經明確寫出 MoE path 是
vLLM MoE black-box integration，且 expert routing、expert migration、
MoE-specific recovery optimization 都是 out of scope。這不應該被放在 gap
analysis 當成單純錯誤，而應該視為 MoE-aware planner 要往前推進的研究邊界。

程式上 `sllm/backends/vllm_capability.py` 有 MoE supported shapes，也能表示
`enable_expert_parallel=True` 後的 derived effective EP；但目前還沒有做到：

- expert-level placement decision
- expert hotness / routed-token load balancing
- expert KV / expert weight migration cost model
- preemption 後針對 expert shard 的 recovery policy

因此現階段最安全的定位是：

```text
可以宣稱：SpotServe-style control plane 可以服務 vLLM MoE model。
不能宣稱：已經把 SpotServe 完整應用到 MoE expert-level serving。
```

MoE-aware planner 要補上的不是單一 bug，而是研究 extension：

- 在 runtime metadata 加入 `num_experts`, `effective_expert_parallel_size`,
  `expert_placement_snapshot`, `expert_load`,
  `per_request_expert_route_histogram`。
- 讓 re-parallelization planner 真的把 expert placement 放進候選與 cost model。
- 加 MoE-specific benchmark：preempt 某個 expert-heavy rank，看新 plan 是否降低
  expert movement / recomputation。

## 與 vLLM EPLB 的責任邊界

MoE-aware SpotServe planner 不應該被描述成重新實作 vLLM 的 Expert Parallel
Load Balancing。比較安全的責任切分是：

```text
vLLM EPLB:
  steady-state intra-deployment expert load balancing
  logical/physical/redundant expert placement within one deployment

SpotServe MoE planner:
  preemption/resource-change-aware topology planning
  recovery/migration target selection across available workers
  deciding when to recreate or move a deployment after GPU availability changes
```

也就是說，SpotServe planner 關心的是 spot/preemption event 後「新的服務拓撲與
target 選擇」；vLLM EPLB 則可在新 deployment 內繼續處理局部 expert load
balance。後續若做到真正 expert-aware re-parallelization，文件與實驗都要明確說明
哪些決策由 SpotServe planner 做，哪些交給 vLLM EPLB。

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
vllm_data_parallel_size
sllm_replica_count
expert_physical_replication_factor
expert_parallel_enabled
effective_expert_parallel_size
expert_parallel_size_source
expert_placement_snapshot
placement_epoch
placement_source
rank_id
node_id
gpu_id
expert_weight_size_bytes
global_expert_hotness
per_request_expert_route_histogram
recent_window_expert_hotness
expert_load
expert_weight_resident
recent_expert_execution_count
expert_weight_loading_required
```

`effective_expert_parallel_size` 應優先從 runtime metadata 取得。若 runtime 沒有
直接回報，才使用目前 vLLM EP enabled 的預設語意：

```text
effective_expert_parallel_size = tensor_parallel_size * vllm_data_parallel_size
```

這裡的 `sllm_replica_count` 只代表 ServerlessLLM/Ray actor replica 數，不代表
vLLM runtime DP，也不代表 expert 的完整 replica 數。若要描述 expert replication，
應使用獨立欄位，例如 `expert_physical_replication_factor`。

`expert_placement_snapshot` 代表某一時間點的 placement view，而不是 model 的永久
靜態屬性。若 runtime 有 EPLB、redundant experts、expert relocation，planner
必須同時記錄 `placement_epoch` 或 `placement_version`，避免用過期 placement 做
target selection。

Default ordering 應該是：

```text
spot / preemption event
-> re-parallelization
   decide deployment topology / expert placement view
-> context migration
   choose request targets using the selected placement epoch
-> stateful recovery
   verify KV/state compatibility and restore request state
```

因此 Phase 2 / Phase 3 的 planner decision 需要帶上 target-side
`placement_epoch` / `expert_placement_fingerprint`。在真正執行 prefix warmup 或
state restore 前，router 需重新讀取 target runtime metadata；若 current placement
與 planner 使用的 placement marker 不一致，就把這次 decision 標記為 stale，並避免
用舊 placement 的 expert locality 做成功 claim。

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

此外，per-request / per-layer / per-expert routed-token statistics 應標成
required instrumentation 或 optional runtime capability。vLLM serving interface
不應被假設一定會直接提供這些欄位；實作上可能需要 patch MoE router、fused MoE
path、或額外 tracing/aggregation。

## Suggested Data Structures

```python
@dataclass(frozen=True)
class ExpertShard:
    layer_id: int
    expert_id: int
    physical_expert_id: int | None
    rank_id: str
    node_id: str
    gpu_id: str
    weight_size_bytes: int = 0
    weight_resident: bool = True
    routed_tokens: int = 0
    recent_execution_count: int = 0
    load_score: float = 0.0


@dataclass(frozen=True)
class ExpertPlacementState:
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    vllm_data_parallel_size: int
    sllm_replica_count: int
    expert_parallel_enabled: bool
    effective_expert_parallel_size: int
    expert_parallel_size_source: str
    expert_physical_replication_factor: int
    placement_epoch: int
    placement_source: str
    shards: list[ExpertShard]


@dataclass(frozen=True)
class ExpertPlacementPlan:
    model_name: str
    target_parallel_plan: ParallelPlan
    expert_to_target_rank: dict[str, str]
    placement_epoch: int
    movement_observation_available: bool
    movement_source: str
    moved_expert_count: int
    stationary_expert_count: int
    unknown_movement_expert_count: int
    moved_weight_bytes: int | None
    estimated_expert_weight_movement_cost_ms: float | None
    estimated_dispatch_cost: float
    estimated_load_balance_penalty: float
    reason: str
```

`expert_to_target_rank` 的 key 可以用：

```text
"layer:{layer_id}/expert:{expert_id}"
```

`expert_id` 表示 logical expert；`physical_expert_id` 表示 runtime 在 EPLB /
redundant expert / relocation 後實際放置的 physical expert。沒有 redundant
experts 時兩者可以相同或省略 `physical_expert_id`，但 planner 不應假設
logical expert 永遠只有一個固定 physical placement。

## Core 1: MoE-aware Re-parallelization

### 原本做法

目前 `ParallelPlan` 主要描述：

```text
TP / PP / vLLM DP / effective EP size
target nodes
ServerlessLLM replica count
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
TP / PP / vLLM DP / effective EP
+ ServerlessLLM replica count
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
-> collect current expert placement snapshot and routed-token histogram
-> generate TP/PP/vLLM-DP/effective-EP candidates
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
per_request_expert_route_histogram
last_n_tokens_expert_histogram
top_experts
```

Phase 2 planner 的 canonical input key 固定為
`per_request_expert_route_histogram`。planner 不再把
`per_request_routed_tokens_by_expert`、`expert_route_histogram` 或
`routed_tokens_by_expert` 當成 request-locality input，避免 runtime contract
模糊。

`per_request_expert_route_histogram` 形式：

```text
request_id -> "layer:{layer_id}/expert:{expert_id}" -> routed_token_count
```

這個 profile 是 optional runtime capability。Phase 2 已新增 vLLM patch path：
當 backend config 開啟 `enable_moe_route_instrumentation=true` 時，
ServerlessLLM 會設定 `VLLM_SPOTSERVE_MOE_TRACE=1`，patched vLLM runtime 會在
fused MoE layer 取得實際 selected top-k expert ids，並輸出 canonical
`per_request_expert_route_histogram`。若 runtime 沒有這個能力，benchmark fixture
仍可注入 `_spotserve_per_request_expert_route_histogram` 做 ablation；若完全沒有
routing history，planner 應退化成 KV-only / queue-only target selection。

runtime-provided histogram 必須用 source 標清楚：

```text
moe_route_histogram_source = vllm_runtime_topk
moe_route_histogram_kind = runtime_observed_topk
```

benchmark/request fixture 只能標成 `request_instrumentation`，不能拿來宣稱已經從
vLLM router kernel 真實量到每個 token 的 expert routing。

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

其中 expert locality 的核心語意是 routing-weighted locality：

```text
routing_weighted_expert_locality
= local_routed_token_weight / total_routed_token_weight

estimated_remote_routing_ratio
= 1 - routing_weighted_expert_locality

estimated_remote_routed_tokens
= total_routed_token_weight * estimated_remote_routing_ratio

expert_dispatch_cost
= estimated_remote_routing_ratio * expert_dispatch_weight
```

這個值只代表「歷史 routing 暗示這個 target 可能比較便宜」。它不是 restore
correctness 條件，也不能保證 request 後續 token 一定繼續走同一批 experts。

### Planner Flow

```text
preempt/dead event
-> collect source request KV metadata
-> collect source request per_request_expert_route_histogram
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
expert_parallel_enabled
effective_expert_parallel_size
sllm_replica_count
vllm_data_parallel_size
expert_placement_fingerprint
expert_placement_epoch
per_request_expert_route_histogram
gate_model_revision
moe_backend
top_k
```

target selection 應先檢查 restore correctness，再估計 expert locality：

```text
model semantic compatibility:
  same model revision
  compatible gate behavior
  compatible tokenizer/token ids

state serialization compatibility:
  same state_kind
  compatible sampling state
  compatible request metadata encoding

KV physical/layout compatibility:
  compatible TP/PP/cache layout
  compatible block size / dtype / attention backend
  supported same-node or cross-node KV transport

locality/cost ranking:
  expert placement topology change
  expert locality score
  remote expert dispatch cost
```

重要：KV restore correctness 和 expert locality 必須分開判斷。這是
MoE-aware V8 的核心原則。

```text
KV compatible != expert-locality optimal
expert placement changed != KV restore impossible
per_request_expert_route_histogram = historical locality hint,
  not a restore requirement
expert_placement_fingerprint = locality/topology change detector,
  not a restore rejection rule by itself
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

因此 `expert_placement_fingerprint` mismatch 應主要觸發重新估算
`expert_dispatch_cost` / locality penalty，而不是直接拒絕 KV restore。

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
      "use_runtime_effective_expert_parallel_size": true,
      "route_histogram_source": "runtime_or_instrumentation",
      "hot_expert_window_tokens": 256,
      "allow_remote_expert_dispatch": true,
      "require_ep_compatible_restore": false,
      "delegate_steady_state_balancing_to_vllm_eplb": true
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
moe_route_histogram_available
moe_route_histogram_source
moe_runtime_effective_expert_parallel_size
moe_expert_parallel_size_source
moe_selected_effective_expert_parallel_size
moe_selected_sllm_replica_count
moe_selected_vllm_data_parallel_size
moe_expert_physical_replication_factor
moe_placement_epoch
moe_placement_source
moe_moved_expert_count
moe_stationary_expert_count
moe_unknown_movement_expert_count
moe_moved_weight_bytes
moe_estimated_expert_weight_movement_cost_ms
moe_expert_weight_resident_count
moe_expert_weight_loading_required_count
moe_hot_expert_locality_ratio
moe_estimated_remote_routing_ratio
moe_estimated_remote_routed_tokens
moe_estimated_dispatch_cost
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

目標：先不改 placement，只收集與報告 MoE metadata。（已開始實作）

完成條件：

- vLLM backend 回傳 `expert_parallel_enabled`、
  `effective_expert_parallel_size`、`expert_parallel_size_source`。
- 分開回傳 `vllm_data_parallel_size` 與 `sllm_replica_count`。
- 若 runtime 可取得，回傳 `expert_placement_snapshot`、`placement_epoch`、
  `placement_source`。
- 分開回報 global hotness、per-request route histogram、
  recent-window hotness。
- 若 per-request route histogram 需要 patch/tracing，benchmark 必須標示
  `moe_route_histogram_source`。
- benchmark summary 顯示 MoE metadata。

目前實作進度：

- 新增 `sllm/spot/moe_placement.py`，提供 `ExpertShard`、
  `ExpertPlacementState`、`ExpertPlacementPlan` 的 metadata-only schema。
- `sllm/backends/vllm_runtime_metadata.py` 已新增 aware aliases：
  `vllm_data_parallel_size`、`sllm_replica_count`、
  `expert_physical_replication_factor`、`expert_placement_available`、
  `placement_epoch`、`placement_source`、`moe_route_histogram_available`、
  `moe_route_histogram_source`。
- `effective_expert_parallel_size` 現在採 runtime-first 語意：runtime 有明確回報
  時使用 runtime value，否則才依 vLLM EP enabled 的預設語意由 `TP * DP`
  推導。
- `sllm/backends/vllm_context_metadata.py` 與
  `sllm/backends/vllm_state_metadata.py` 已允許這些 aware 欄位通過 metadata
  export。
- vLLM backend 已補上 Phase 2 runtime instrumentation path：若 patched runtime
  hook 回傳 `per_request_expert_route_histogram`，會優先保留為 runtime-provided
  metadata；若 benchmark request 帶
  `_spotserve_per_request_expert_route_histogram`，backend 會在建立
  `SamplingParams` 前移除該私有欄位，並轉成 canonical
  `per_request_expert_route_histogram`。若兩者都沒有，會明確標示 route
  histogram unavailable。
- target placement 目前支援 configured `expert_placement_snapshot`，或在本地
  MoE model `config.json` 可讀時由 `num_hidden_layers` 與 expert count 推導一份
  instance-level coverage snapshot，並標示 `placement_source` 為
  `derived_from_model_config`。這是 observability / planner input，不宣稱已完成
  physical expert migration。
- stateful recovery planner 已改成不把 EP mismatch 預設視為 KV restore
  incompatibility；只有 `state_restore_requires_ep_layout=true` 時才把 EP layout
  mismatch 當 hard reject。

### Milestone B: Expert-locality Target Selection

目標：先讓 context migration target selection MoE-aware，但不搬 expert
weights。（已開始實作）

完成條件：

- active request 有 `per_request_expert_route_histogram`。
- target 有 expert placement metadata。
- planner 選 target 時考慮 hot expert locality。
- metric 顯示 `moe_hot_expert_locality_ratio`、
  `moe_estimated_remote_routing_ratio`、
  `moe_estimated_remote_routed_tokens`、`moe_estimated_dispatch_cost`。
- metric 顯示 placement-derived dispatch breakdown：
  `moe_routed_tokens`、`moe_local_routed_tokens`、
  `moe_remote_routed_tokens`、`moe_remote_routing_ratio`，以及
  by-layer / by-expert routed-token breakdown。
- metric 顯示 `context_migration_queue_penalty_cost`、
  `context_migration_avg_queue_pressure`、`context_migration_max_queue_depth`。
- 若沒有 route histogram，planner 可退化成 KV-only target selection，並在
  metrics 中標示 route histogram unavailable。

目前實作進度：

- `sllm/spot/context_migration.py` 已在 `MigrationTarget` 加入 target metadata，
  並在 `MigrationPlan` / `MigrationDecision` 中輸出
  `expert_locality_available`、`hot_expert_locality_ratio`、
  `estimated_remote_routing_ratio`、`estimated_remote_routed_tokens`、
  `expert_dispatch_cost`、`queue_depth`、`queue_pressure`、
  `queue_penalty_cost`。
- 新增 `estimate_expert_dispatch_cost()`，MVP 只保留一套
  routing-weighted expert locality cost。
- 新增 `estimate_queue_penalty_cost()`，讓 target selection 不只看 target
  是否還有 capacity，也把 target 現有 `concurrency` / `queue_depth` 以及同一輪
  migration 已排在前面的 planned requests 納入 soft penalty。queue cost 預設不改變
  舊 planner 行為；需要設定 `queue_penalty_weight`、`queue_pressure_weight`，或 target
  明確帶入 `queue_penalty` 才會影響 target assignment。
- `plan_low_cost_migration()` 會在 planner config 啟用
  `enable_moe_expert_locality` 或設定 expert dispatch cost 參數時，把
  expert dispatch cost 加入 target assignment；若 histogram 或 placement snapshot
  不可用，會退化為原本 KV / queue / warmup cost。
- `RoundRobinRouter` 的 context migration target collection 會嘗試讀取 target
  runtime metadata，並把 `expert_placement_snapshot`、`placement_epoch`、
  `moe_route_histogram_available` 等欄位放入 target metadata；同時會把 target
  的 `concurrency`、`max_queue_length`、`queue_depth` 傳給 planner。
- `make_context_migration_event()` 與 benchmark analyzer 已新增
  `moe_hot_expert_locality_ratio`、`moe_estimated_remote_routing_ratio`、
  `moe_estimated_remote_routed_tokens`、`moe_estimated_dispatch_cost`、
  `moe_routed_tokens`、`moe_local_routed_tokens`、
  `moe_remote_routed_tokens`、`moe_remote_routing_ratio`、
  `moe_local_routed_tokens_by_layer`、
  `moe_remote_routed_tokens_by_layer`、
  `moe_local_routed_tokens_by_expert`、
  `moe_remote_routed_tokens_by_expert`、
  `context_migration_selected_target_ids`、
  `context_migration_selected_plan_kv_migration_cost`、
  `context_migration_selected_plan_expert_dispatch_cost`、
  `context_migration_selected_plan_queue_penalty_cost`、
  `context_migration_queue_penalty_cost`、
  `context_migration_avg_queue_pressure`、`context_migration_max_queue_depth`
  等欄位。
- context migration / core applied benchmark config 已設定
  `queue_penalty_weight=1.0`，因此 standard benchmark 會啟用這個 queue cost
  component；若 target 當下沒有 queue pressure，對應 metric 仍會自然為 0。
- context migration / core applied benchmark config 已啟用
  `enable_moe_expert_locality=true` 與 `expert_dispatch_weight=10.0`；
  `context_migration_vllm_performance.jsonl` 與
  `spotserve_core_vllm_performance.jsonl` 的 warm-prefix request 會注入
  `_spotserve_per_request_expert_route_histogram`，用來驗證 runtime
  instrumentation 到 planner/metrics 的資料流。
- V7 runtime observability 已補上 selected-plan cost breakdown：
  `selected_target_ids`、`selected_plan_total_estimated_cost`、
  `selected_plan_kv_migration_cost`、`selected_plan_expert_dispatch_cost`、
  `selected_plan_queue_penalty_cost`、`context_source_count`、
  `context_target_count`。applied performance configs 會額外開啟
  `emit_candidate_component_costs=true`，因此 router metrics 會保留每個 source
  request 對每個 candidate target 的 KV / expert / queue / total cost。
- runtime observability 欄位已補上 `moe_route_histogram_kind`。因此 benchmark
  summary 會同時顯示 `route_source` 與 `route_kind`：
  `vllm_runtime_topk/runtime_observed_topk` 才能用來主張已從 patched vLLM
  fused MoE top-k path 取得 routing；`request_instrumentation/request_fixture`
  只能作為 deterministic benchmark fixture；`synthetic/synthetic_ablation`
  只能作為 planner cost ablation。
- 新增 real-GPU smoke：
  `python -m sllm.spot.moe_route_instrumentation_smoke`。它不注入
  `_spotserve_per_request_expert_route_histogram`，只在 live context metadata
  回報 `moe_route_histogram_source=vllm_runtime_topk` 時通過。

### Phase 2 可驗證實驗

Phase 2 的 MVP 先用 synthetic ablation 驗證 planner，再用 end-to-end vLLM
benchmark 驗證 runtime instrumentation 是否真的接到 planner。synthetic workload
可以隔離 cost model 本身，確認 KV / expert / queue component 會正確影響 target
selection；end-to-end run 則用來確認 patched vLLM 的 runtime top-k routing
metadata 真的被 V7 flow 消費到。

執行：

```bash
python scripts/run_context_migration_phase2_ablation.py \
  --input benchmarks/spotserve/context_migration_phase2_ablation.json \
  --output-dir results/spotserve_context_migration_phase2_ablation
```

這個實驗不需要啟動 container、Ray 或 vLLM。它會產生：

- `report.json`
- `latest_summary.json`
- `latest_comparisons.json`
- 每個 run 的 `migration_plan.json`
- 每個 run 的 `migration_metrics.jsonl`

四組 ablation 的預期 target selection：

| Run | Active cost | Expected target | 驗證重點 |
|---|---|---|---|
| `phase2-kv-only` | KV cost | `target-kv-busy-remote-expert` | 同 node KV/context reuse 會贏 |
| `phase2-kv-plus-expert-locality` | KV + expert dispatch cost | `target-expert-busy` | routing-weighted expert locality 能抵銷 KV locality |
| `phase2-kv-plus-queue` | KV + queue cost | `target-idle-remote-expert` | busy target 的 queue penalty 會讓 planner 選 idle target |
| `phase2-kv-plus-expert-plus-queue` | KV + expert dispatch + queue cost | `target-expert-idle` | combined cost 會選 expert-local 且 idle 的 target |

此實驗的 pass/fail 條件是：

- `report.json` 的 `passed=true`。
- 四個 run 的 `selected_targets` 符合上表。
- `candidate_component_costs` 中可以看到每個 candidate 的
  `kv_migration_cost`、`expert_dispatch_cost`、`queue_penalty_cost` 和
  `total_estimated_cost`，因此能解釋 target 為什麼改變。
- `phase2-kv-plus-queue` 中 busy same-node target 的 queue cost 必須高於 idle
  target，證明 queue cost 不是只出現在 metrics，而是真的進入 target ranking。

### Runtime Top-k Observability Smoke

若要驗證 patched vLLM runtime 真的量到 MoE selected top-k，而不是 request
fixture，先重新 build 含 `runtime_moe_metadata.patch` 的 image。因為
`--skip-build` 只會同步 ServerlessLLM Python package，不會重新 patch 已安裝的
vLLM，所以第一次驗證 runtime top-k hook 時不要加 `--skip-build`：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
scripts/prepare_spotserve.sh --deploy-set context-migration-performance
```

再在 worker venv 執行：

```bash
podman exec sllm_worker_0 bash -lc '
SPOTSERVE_MOE_ROUTE_INSTRUMENTATION_MODEL=/models/Qwen2-MoE-Tiny \
  /opt/venvs/worker/bin/python -m sllm.spot.moe_route_instrumentation_smoke
'
```

通過條件：

- `status=passed`
- `moe_route_histogram_source=vllm_runtime_topk`
- `moe_route_histogram_kind=runtime_observed_topk`
- `per_request_expert_route_histogram` 非空

如果這支 smoke 還沒通過，V7 end-to-end benchmark 中的 expert locality
結果仍應標成 planner/fixture validation，不能宣稱已完成 true runtime routing
instrumentation。

### V7 End-to-end Runtime Top-k Check

2026-08-30 的 V7 context migration performance run 已確認 end-to-end path 可以吃到
patched vLLM runtime routing metadata：

```text
Benchmark:
  benchmark_matrix_context_migration_performance.yaml
Model:
  /models/Qwen2-MoE-Tiny
Disabled:
  successes=8/8
  p95=48238.92ms
Applied:
  successes=8/8
  p95=4010.09ms
  context_migrations=1
  kv_successes=1
  route_source=vllm_runtime_topk
  route_kind=runtime_observed_topk
  kv_cost=293.00
  expert_cost=0.00
  queue_cost=0.00
```

這代表：

- vLLM fused MoE top-k path 的 runtime instrumentation 已產生
  `per_request_expert_route_histogram`。
- V7 context migration planner / metrics 已消費到 runtime-provided histogram，
  不是只靠 request fixture。
- selected-plan cost breakdown 已能分開回報 KV cost、expert dispatch cost、queue
  cost。

仍不能宣稱：

- 已完成 physical expert migration。
- 已量到真實 remote expert dispatch traffic。
- V7 已完成 true KV block transfer。此 run 的 reusable blocks 仍為 `0`，所以它
  驗證的是 low-cost target selection 與 prefix warmup/context planning。

### Milestone C: MoE-aware Stateful Recovery

目標：recovery target selection 分離 KV restore correctness 與 expert locality。
（已開始實作）

完成條件：

- `InferenceState.metadata` 帶 MoE routing profile、placement fingerprint、
  placement epoch。
- target planner 先分別判斷 model semantic compatibility、state serialization
  compatibility、KV physical/layout compatibility。
- expert locality 只影響 target ranking，不直接否決 restore。
- 若 runtime/state encoding 對 EP layout 有硬相依，才用 EP mismatch 觸發
  correctness fallback。
- `expert_placement_fingerprint` mismatch 只作為 topology/locality signal，除非
  runtime 明確宣告 state encoding 對 placement 有硬相依。
- restore 後量測或估計 remote expert dispatch / locality penalty。

目前實作進度：

- `sllm/spot/stateful_recovery.py` 已把 restore target compatibility 拆成三層：
  model semantic compatibility、state serialization compatibility、KV layout
  compatibility。
- EP layout compatibility 已從 KV restore correctness 中分離。預設情況下，
  `effective_expert_parallel_size`、`expert_parallel_enabled` 或
  `expert_placement_fingerprint` mismatch 不會直接 reject restore；只有 source
  state 或 target metadata 明確宣告 `state_restore_requires_ep_layout=true` 時，
  EP mismatch 才會變成 hard incompatibility。
- 通過 correctness gate 的候選 target 會再用 routing-weighted expert locality
  排序，並輸出 recovery-side `hot_expert_locality_ratio`、
  `estimated_remote_routing_ratio`、`estimated_remote_routed_tokens`、
  `expert_dispatch_cost`。
- `VllmBackend.restore_inference_state()` 已移除
  `expert_parallel_enabled` 的預設 hard cache-config check，改成只在
  `state_restore_requires_ep_layout=true` 時檢查 EP layout。這避免
  `KV compatible != expert-locality optimal` 的情況被錯誤 fallback。
- `InferenceState.metadata` / vLLM metadata pass-through 已補上
  `expert_placement_fingerprint`、`expert_placement_epoch`、
  `state_restore_requires_ep_layout`、`gate_model_revision`、`moe_backend`、
  `top_k`、`sampling_state_encoding`、`request_metadata_encoding`。
- `make_state_recovery_event()`、benchmark analyzer、V8/core comparison fields
  已新增 recovery compatibility 與 locality metrics。
- 目前 recovery-side expert dispatch 是由 request route histogram + target
  placement metadata 估計，不是真實 vLLM all-to-all / remote dispatch traffic
  counter。真實 traffic counter 仍屬後續工作。

2026-09-16 的 V8 stateful recovery performance run 已驗證 Phase 3 的主要
runtime restore path：

```text
Benchmark:
  benchmark_matrix_stateful_recovery_performance.yaml
Model:
  /models/Qwen2-MoE-Tiny
Token replay:
  successes=3/3
  p95=48739.13ms
Stateful recovery:
  successes=3/3
  p95=5157.64ms
  state_restores=1/1
  state_tokens=16
  state_blocks=6
  response_blocks=6
  state_fallbacks=0
  true_kv_restores=1
  true_kv_rate=100.00%
  true_kv_blocks=6
  supports_state_restore=1
  recovery_kv_compatible=1
  recovery_ep_required=0
  recovery_ep_mismatch=1
  recovery_locality=1.00
  recovery_remote_tokens=0
  recovery_expert_cost=0.00
```

這個結果的重點不是 `EP mismatch=1` 有問題，而是相反：planner/runtime
正確把它視為 topology/locality signal。因為 `recovery_ep_required=0`，
EP mismatch 不會硬擋 KV restore；同一個 run 同時有
`recovery_kv_compatible=1`、`state_blocks=6`、`response_blocks=6`、
`true_kv_restores=1`，代表 KV restore correctness 與 expert locality 已經
被分開報告。

仍不能宣稱已量測真實 remote expert dispatch traffic；這次
`recovery_remote_tokens=0` / `recovery_expert_cost=0.00` 是根據 route histogram
與 target placement metadata 的估計結果。

2026-09-16 的 V7-V9 core combined run 也確認三個核心可以在同一個 applied
benchmark 中一起運作，但這個 matrix 不應取代 standalone V8 true-KV-restore
證據：

```text
Benchmark:
  benchmark_matrix_spotserve_core_performance.yaml
Baseline:
  successes=8/8
Applied:
  successes=8/8
  context_migrations=1
  route_source=vllm_runtime_topk
  route_kind=runtime_observed_topk
  kv_successes=1
  state_events=1
  state_fallbacks=1
  true_kv_restores=0
  supports_state_restore=0
  recovery_kv_compatible=1
  recovery_ep_required=0
  recovery_ep_mismatch=1
  recovery_locality=1.00
  risk_scheduling_events=3
```

這組結果可以用來 claim「V7 context planning、V8 stateful recovery、V9
risk-aware scheduling 的 code paths 可以合在同一個 live benchmark 內執行」。
但它仍不應被寫成 physical expert migration、真實 remote expert dispatch traffic
已完成，或 true KV restore 已在 core matrix 中完成。true KV restore 的主要
證據應使用 standalone `benchmark_matrix_stateful_recovery_performance.yaml`。

### Phase 4 前置小步：Expert Dispatch Observability

在真正做 expert-aware re-parallelization 以前，先補一個可驗證的 observability
baseline：用 runtime-observed MoE route histogram 加上 target expert placement，
推導 restore/migration 後有多少 routed-token weight 會命中 target-local experts，
以及多少會落到 target-local placement 之外。

目前 local expert 的定義是 planner-level target placement coverage：

```text
expert key in target expert_placement_snapshot
=> counted as local / placement-covered

expert key missing from target expert_placement_snapshot
=> counted as remote / outside target placement
```

因此 `moe_local_routed_tokens` / `moe_remote_routed_tokens` 的 granularity 是
`target_instance_or_deployment`，不是 per-rank GPU locality。若 target snapshot
本身覆蓋所有 experts，remote routed tokens 可以是 0；這只代表 planner
metadata 看起來都 target-covered，不代表 vLLM runtime 沒有 EP all-to-all 或
remote NCCL traffic。

這一步已補的欄位：

```text
moe_dispatch_observation_available_count
moe_routed_tokens
moe_local_routed_tokens
moe_remote_routed_tokens
moe_remote_routing_ratio
moe_local_routed_tokens_by_layer
moe_remote_routed_tokens_by_layer
moe_local_routed_tokens_by_expert
moe_remote_routed_tokens_by_expert
moe_locality_definition = target_placement_coverage
moe_locality_granularity = target_instance_or_deployment
moe_remote_routing_definition = missing_from_target_placement_snapshot
moe_rank_locality_available = false
moe_physical_dispatch_traffic_available = false
```

這些欄位會出現在：

- context migration event / analyzer summary：
  `context_migration_moe_*`
- stateful recovery event / analyzer summary：
  `state_recovery_moe_*`
- benchmark summary：
  `remote_dispatch_tokens`、`remote_dispatch_ratio`、
  `recovery_remote_dispatch_tokens`、`recovery_remote_dispatch_ratio`

Phase 4 前置也補了 placement epoch handshake：

```text
target_placement_epoch
target_expert_placement_fingerprint
placement_handshake_attempts
placement_handshake_successes
placement_handshake_failures
placement_handshake_stale
```

context migration prefix warmup 與 stateful recovery restore 前都會重新讀 target
runtime metadata。如果 target placement marker 已經改變，migration 會 skip 該
prefix warmup，recovery 會退回 token replay fallback。這讓 V7/V8 的 locality
結果可以明確綁定 planner 當下看到的 placement epoch。

語意邊界：

- `moe_routed_tokens` 來自 runtime top-k route histogram，是 observed routing
  signal。
- `moe_local_routed_tokens` / `moe_remote_routed_tokens` 是根據 target
  expert placement metadata 推導出的 placement-locality breakdown。
- 這仍不是 NIC / NCCL / all-to-all byte counter，也不是 physical expert
  migration。它的用途是建立 Phase 4 的 baseline：之後如果改 expert placement
  或做 hot expert replication，應該能讓 `moe_remote_routed_tokens` /
  `moe_remote_routing_ratio` 下降。

### Milestone D: Expert-aware Re-parallelization

目標：re-parallelization planner 真的能決定 expert placement。

完成條件：

- `ExpertPlacementPlan` 可序列化到 metrics。（已完成 logical plan）
- preempted GPU 上的 experts 可被重新配置到 ready GPUs。（尚未完成
  physical weight movement）
- planner cost 同時考慮 GPU capacity、expert movement、dispatch cost。
- 明確定義 SpotServe planner 與 vLLM EPLB 的責任邊界：SpotServe 負責
  resource-change / preemption-aware topology planning，vLLM EPLB 負責
  deployment 內部 steady-state expert balancing。

目前 Phase 4 的第一個前置小步已完成：

- V6 re-parallelization planner 會從 selected `ParallelPlan` 與 MoE model
  topology 產生 deterministic logical `ExpertPlacementPlan`。
- vLLM 的 EP 預設使用 `linear` placement：以連續的 expert ID 分配到各
  EP rank；若 backend config 設定 `expert_placement_strategy=round_robin`，
  planner 才使用交錯分配。logical plan 必須和 vLLM 實際策略一致，否則
  request 仍可能成功，但 `runtime_placement_verified` 應該失敗。
- 若 router/head container 看不到模型的 `config.json`，planner 會嘗試使用
  active vLLM worker runtime metadata；只要 runtime 回報
  `expert_placement_snapshot`，logical planner 可以從 snapshot 反推出
  layer/expert topology。
- plan 會輸出 `placement_epoch`、`placement_fingerprint`、
  `required_expert_count`、`covered_expert_count`、`planned_shard_count`、
  `target_rank_count` 與 `physical_weight_migration=false`。
- 若 replan 真正建立新的 vLLM actor，adapter 會把 logical placement plan、
  epoch、fingerprint、placement snapshot 放進新 actor backend config，讓後續
  context migration/recovery target metadata 可以綁定同一個 placement marker。

這一步仍然不是 physical expert migration。它只是讓 controller 先能說清楚
「新 topology 下 experts 應該在哪裡」，下一步才是把這個 plan 接到 vLLM EP
runtime 的 rank mapping / weight loading / expert movement。

目前 Phase 4 的第二個前置小步也已完成：

- vLLM backend/runtime metadata 會把 placement contract 與 placement snapshot
  拆開回報。`expert_placement_plan_fingerprint` 代表 planner 產生的
  `ExpertPlacementPlan` contract；`expert_placement_snapshot_fingerprint`
  代表 target runtime 當下提供的 expert placement snapshot。
- 舊欄位 `expert_placement_fingerprint` 保留為相容 marker；若有 planner
  contract，會優先代表 plan fingerprint，否則才退回 snapshot fingerprint。
- 新增 `expert_placement_contract_available`、
  `expert_placement_contract_bound`、`expert_placement_contract_snapshot_match`、
  `expert_placement_plan_applied`、`expert_placement_plan_verified` 與
  `expert_placement_contract_reason`。
- `expert_placement_plan_applied=false` / `verified=false` 是目前預期狀態：
  表示 controller 已把 logical placement contract 傳到 backend config，但
  vLLM runtime 還沒有真的回報它已套用新的 EP rank mapping 或完成 expert
  weight movement。
- context migration / stateful recovery 事件與 analyzer summary 會輸出 target
  contract 狀態，例如
  `context_migration_selected_target_expert_placement_contracts`、
  `context_migration_selected_target_expert_placement_plan_applied`、
  `context_migration_selected_target_expert_placement_plan_verified`、
  `state_recovery_target_expert_placement_contracts`、
  `state_recovery_target_expert_placement_plan_applied` 和
  `state_recovery_target_expert_placement_plan_verified`。
- V7 context migration target metadata allowlist 會保留上述 contract / hook
  欄位，因此若 target 是由 V6 logical `ExpertPlacementPlan` 建出來的，
  migration planner 的 selected plan 可以看見 plan fingerprint、snapshot
  fingerprint、contract reason，以及 apply/verify hook 狀態。
- V6 deployment plan matching 會把 expert placement marker 納入比較。也就是
  TP/DP/PP/replica/target nodes 相同時，若 `ExpertPlacementPlan` fingerprint
  不同，仍會被視為新的 plan，而不是誤判成 `unchanged`。

因此 Phase 4 現在可以驗證：

```text
logical placement plan generated
-> movement diff computed against current runtime placement when available
-> placement contract delivered to vLLM backend metadata
-> target migration/recovery sees the same contract marker
-> runtime-applied / verified remain false until vLLM EP runtime integration
```

Phase 4 的第三個前置小步定義了 runtime apply/verify hook 邊界：

```text
apply_expert_placement_plan(expert_placement_plan=...)
verify_expert_placement_plan(expert_placement_plan=...)
```

backend 初始化 vLLM engine 後，若 backend config 帶有
`expert_placement_plan`，會嘗試呼叫上述 hook。為了相容現有 vLLM，hook
不存在時不會讓部署失敗，而是輸出：

```text
expert_placement_apply_hook_available=false
expert_placement_apply_attempted=false
expert_placement_apply_success=false
expert_placement_apply_reason=runtime_apply_hook_unavailable
expert_placement_verify_hook_available=false
expert_placement_verify_attempted=false
expert_placement_verify_success=false
expert_placement_verify_reason=runtime_verify_hook_unavailable
expert_placement_plan_applied=false
expert_placement_plan_verified=false
```

Phase 4D 補上 patched vLLM observe-only hook plumbing。也就是
`runtime_moe_metadata.patch` 會在 `AsyncLLM -> EngineCore -> WorkerBase`
暴露 `apply_expert_placement_plan()` 與 `verify_expert_placement_plan()`。
目前這兩個 hook 只確認 runtime 看到了 SpotServe 的 logical placement
contract，不會改 vLLM EP rank mapping，也不會搬 expert weights。因此重建 image
後，新的預期輸出會變成：

```text
expert_placement_apply_hook_available=true
expert_placement_apply_attempted=true
expert_placement_apply_success=false
expert_placement_apply_reason=physical_expert_placement_migration_not_supported
expert_placement_verify_hook_available=true
expert_placement_verify_attempted=true
expert_placement_verify_success=false
expert_placement_verify_reason=physical_expert_placement_verification_not_supported
expert_placement_plan_applied=false
expert_placement_plan_verified=false
```

Phase 4E 補上 logical expert movement diff / cost estimate。planner 在產生
新的 `ExpertPlacementPlan` 時，優先把 target placement 和舊 actor 實測的
`runtime_expert_placement_shards` 做 normalized diff。由 model config 推導的
`expert_placement_snapshot` 不是實測 placement，不能用來宣稱觀測到 movement：

```text
current expert placement snapshot
+ selected target ExpertPlacementPlan
-> movement_observation_available
-> moved_expert_count
-> stationary_expert_count
-> unknown_movement_expert_count
-> moved_weight_bytes
-> estimated_expert_weight_movement_cost_ms
```

這一步解決的是「planner 是否知道新地圖和舊地圖差在哪裡」。它仍然不是
physical expert migration：`physical_weight_migration=false`、runtime
`apply_success=false` 仍然是正確狀態；`verify_success=true` 只表示新 actor
的 placement 符合 plan。若 runtime 沒有提供 current
placement snapshot，movement observation 會是 unavailable，planner 不會憑空
宣稱 expert 被搬動。若 runtime 未提供 expert weight size，`moved_weight_bytes`
會是 `null`（未知），不是 0；`estimated_expert_weight_movement_cost_ms`
也會是 `null`，除非另有不依賴 bytes 的固定成本設定。即使
`moved_weight_bytes` 有值，它也只是「換了 logical EP rank 的權重大小總和」，
不是 actor recreate 期間實際傳輸的 bytes。沒有設定成本係數時，非零 movement
的 cost estimate 同樣是 `null`，不是零成本。

movement cost 目前只作為 cost model 的可觀測 component；若設定
`expert_weight_movement_cost_ms_per_gib`、
`expert_weight_movement_cost_ms_per_expert` 或
`expert_weight_movement_bandwidth_bytes_per_s`，它會被加進 selected
`replan_window_cost_ms`，並受
`expert_weight_movement_penalty_weight` 控制。沒有設定 expert weight size 或
movement cost 時，不會假設 moved bytes 或 movement cost 為 0；scoring
仍只使用可估計的成本，實驗報表另外標出未知 weight bytes 的事件數。

benchmark/analyzer 會新增檢查：

```text
replanning_expert_placement_plan_movement_observation_events
replanning_max_expert_placement_plan_moved_experts
replanning_total_expert_placement_plan_moved_weight_bytes
replanning_expert_placement_plan_unknown_weight_bytes_events
replanning_expert_placement_plan_unknown_cost_events
replanning_avg_expert_placement_plan_weight_movement_cost_ms
replanning_avg_selected_expert_weight_movement_cost_estimate_ms
```

Phase 4F 補上 controlled movement ablation。這不是 live vLLM benchmark，
而是一個不用 GPU/Ray 的 synthetic planner test：它刻意讓 current placement
把 4 個 experts 都放在 `node-a/ep-rank-0`，再提供兩個 candidate plan：

```text
stationary_single_rank
-> all experts remain on node-a/ep-rank-0
-> moved_experts = 0

split_across_two_ep_ranks
-> expert 1/3 move to node-b/ep-rank-1
-> moved_experts = 2
-> moved_weight_bytes = 2 MiB
-> estimated movement cost = 20 ms
```

跑法：

```bash
python scripts/run_reparallelization_phase4_movement_ablation.py \
  --input benchmarks/spotserve/reparallelization_phase4_movement_ablation.json \
  --output-dir results/spotserve_reparallelization_phase4_movement_ablation
```

通過條件：

```text
report.json: passed = true

phase4-movement-unpenalized:
  selected_reason = split_across_two_ep_ranks
  selected_expert_placement_moved_expert_count = 2
  selected_expert_placement_moved_weight_bytes = 2097152
  selected_expert_weight_movement_cost_estimate_ms = 20

phase4-movement-penalized:
  selected_reason = stationary_single_rank
  selected_expert_placement_moved_expert_count = 0
  selected_expert_weight_movement_cost_estimate_ms = 0
```

這個 ablation 的重點是驗證兩件事：

- planner 的 movement diff 真的能產生非零 moved expert / bytes / cost。
- `expert_weight_movement_penalty_weight` 真的會影響 selected plan。

它仍然不是 physical expert migration；`physical_weight_migration=false` 仍是
正確狀態。

若未來 patched vLLM EP runtime 實作 hook，`apply` 應該只在 runtime 已接受並
套用 planned rank mapping / expert layout 時回 `applied=true`；
`verify` 應該重新讀 runtime 實際 placement，確認和 contract 相符後才回
`verified=true`。只有 `expert_placement_plan_verified=true` 時，才能把 Phase
4 claim 從「logical/control-plane placement」推進到「runtime-applied
placement」。

2026-09-15 的 V6 re-parallelization performance benchmark 已驗證目前
logical/control-plane placement、movement diff 與 observe-only runtime hook
plumbing path：

```text
benchmark_matrix_reparallelization_performance.yaml
model = /models/Qwen2-MoE-Tiny

Disabled:
  successes=3/8
  success_rate=37.50%
  p95=180102.20ms
  trace_success=1

Applied:
  successes=8/8
  success_rate=100.00%
  p95=14069.87ms
  trace_success=1
  replans=1
  applied=1
  failed=0
  exec_model=expert_aware_actor_recreate
  actor_recreate=1
  live_migration=0
  runtime_workers=1
  exec_ms=16340.36
  cost_model=1
  expert_plan=1
  expert_plan_shards=8
  expert_plan_coverage=1.00
  expert_movement_observation=1
  expert_plan_moved=0
  expert_plan_moved_bytes=0
  expert_move_ms=0.00
  runtime_apply_hooks=1
  runtime_apply_attempted=1
  runtime_apply_success=0
  runtime_verify_hooks=1
  runtime_verify_attempted=1
  runtime_verify_success=0
  runtime_plan_applied=0
  runtime_plan_verified=0
  physical_expert_migration=0
  runtime_verification_level=contract_seen_only
  runtime_verified_placement=0
  runtime_remap_ep=0
  runtime_a2a_counters=0
```

這代表：

- preemption trace replay 本身成功，不是 benchmark 忽略 `/spot/event` timeout。
- V6 planner 有進入 replan，並成功 apply 新的 actor/recreate plan。
- workload-aware cost model 有輸出 selected replan-window / load / migration
  estimate。
- logical `ExpertPlacementPlan` 有產生並被 metrics/analyzer 看到，8 個 logical
  expert shards 都被 placement 覆蓋。
- Phase 4E movement diff 有被 metrics/analyzer 看到；這次
  `moved_expert_count=0`、`moved_weight_bytes=0` 是合理結果，因為此 matrix 是
  single-worker same-node recreate，selected logical placement 和 current runtime
  snapshot 沒有 expert location 差異。
- patched vLLM observe-only placement hook 已被 runtime 暴露且被呼叫，所以
  metrics 從 `runtime_apply_hook_unavailable` 推進到 hook available/attempted。
- `runtime_apply_success=0`、`runtime_verify_success=0`、
  `runtime_plan_applied=0`、`runtime_plan_verified=0` 是目前正確結果，因為
  hook 仍是 observe-only，沒有真的改 vLLM EP rank mapping 或搬 expert weights。
- `runtime_verification_level=contract_seen_only` 代表 runtime 已看過 placement
  contract；它不是 `physical_migration_verified`。

這組 run 使用 single-worker same-node recreate mode，因此
`runtime_workers=1` 是預期結果：它表示 same-node recreate capacity entry 代表
一個真實 runtime worker node。這仍只驗證 controller、logical plan、metric path，
不是 multi-worker relocation。`actor_recreate=1`、`live_migration=0` 和
`physical_expert_migration=0` 也仍然是預期結果；真正的 expert weight movement
要等 vLLM EP runtime hook 可以回報 `applied=true` 且 `verified=true` 之後才能
宣稱。

### Phase 5A: vLLM EP Runtime Capability Audit

Phase 5 的第一步不是直接搬 expert weights，而是先檢查 vLLM runtime 是否具備
可安全套用 `ExpertPlacementPlan` 的能力。這一步新增
`sllm.spot.vllm_ep_runtime_audit`，用來回答：

```text
1. patched vLLM 是否有 MoE top-k routing instrumentation
2. gpu_model_runner forward path 是否包住 moe_request_context(...)
3. fused MoE method 是否真的呼叫 record_moe_routing(...)
4. WorkerBase / EngineCore / AsyncLLM 是否有 apply/verify placement hook
5. apply/verify hook 是 observe-only，還是真的回報 physical migration applied
```

在 worker container 裡可以這樣跑：

```bash
podman exec sllm_worker_0 bash -lc '
/opt/venvs/worker/bin/python -m sllm.spot.vllm_ep_runtime_audit
'
```

如果 running container 回 `No module named sllm.spot.vllm_ep_runtime_audit`，
代表目前 container 還沒同步這個新 module；先用 `SPOTSERVE_SYNC_SOURCE=1`
重新 prepare 一次同一個 deploy set，或 rebuild image 後再跑 audit。

若只想檢查 source tree、不 import runtime：

```bash
/opt/venvs/worker/bin/python -m sllm.spot.vllm_ep_runtime_audit \
  --source-root /opt/venvs/worker/lib/python3.11/site-packages/vllm \
  --no-runtime-probe
```

目前預期分類是：

```text
phase5_gate.classification = observe_only_expert_placement_contract
phase5_gate.can_claim_physical_expert_migration = false
phase5_gate.recommended_execution_model = expert_aware_actor_recreate
```

這代表 Phase 4 產生的 `ExpertPlacementPlan` 已經能被 runtime hook 看到，但
vLLM 仍沒有真的更新 live EP rank mapping 或搬 expert weight tensor。若未來
runtime hook 回報：

```text
apply_expert_placement_plan -> applied=true, physical_weight_migration=true
verify_expert_placement_plan -> verified=true, physical_weight_migration=true
```

audit 才能把 gate 推進到：

```text
phase5_gate.classification = physical_expert_migration_supported
phase5_gate.can_claim_physical_expert_migration = true
```

所以 Phase 5A 的階段性結論會是：

```text
Current implementation supports expert-aware actor recreate planning.
It does not yet support live physical expert weight migration.
```

2026-09-06 在 `sllm_worker_0` 實際執行 Phase 5A audit 的結果符合這個 gate：

```text
vLLM version = 0.11.2
spotserve_moe_importable = true
forward_path_has_moe_request_context = true
route_recording_hooks = 2
apply_verify_boundary_present = true
observe_only_markers_present = true

apply_expert_placement_plan:
  callable = true
  applied = false
  physical_weight_migration = false
  reason = physical_expert_placement_migration_not_supported
  hook_kind = spotserve_observation_only

verify_expert_placement_plan:
  callable = true
  verified = false
  physical_weight_migration = false
  contract_seen_by_runtime = true
  reason = physical_expert_placement_verification_not_supported
  hook_kind = spotserve_observation_only

phase5_gate:
  classification = observe_only_expert_placement_contract
  can_claim_physical_expert_migration = false
  recommended_execution_model = expert_aware_actor_recreate
```

這代表 Phase 5A 已經確認 runtime boundary 是存在且可被呼叫的，但目前仍只
能作為 placement contract / observability boundary，不能把它寫成 live
physical expert migration。

Phase 5A 後續補強的 runtime observability 會把這個邊界拆得更清楚：

```text
expert_placement_contract_seen_by_runtime = true
expert_placement_contract_seen_by_all_workers = true
expert_placement_plan_applied = false
expert_placement_plan_verified = false
expert_placement_physical_weight_migration = false
```

這代表每個 vLLM worker hook 已經看過同一份 placement contract，但 runtime
仍沒有證明 expert tensor 被 live remap 或搬移。因此它只能證明 control-plane
到 runtime hook 的 handshake 成功，不能宣稱 physical expert migration。

### Phase 5B: Expert-aware Actor Recreate Execution Model

Phase 5B 先把目前能安全宣稱的 execution model 固定下來：

```text
ExpertPlacementPlan is consumed by recreated vLLM actors.
It is not applied by live in-place expert weight migration.
```

也就是說，Phase 4 的 `ExpertPlacementPlan` 會進入
`VllmDeploymentAdapter._plan_backend_config(...)`，再隨著新的 vLLM actor
建立流程一起交給 runtime metadata / observe-only hook。這讓系統可以用新的
logical placement 來做後續 migration / recovery planning，但目前不宣稱：

```text
live EP rank remapping = true
expert weight tensor moved in-place = true
remote expert dispatch traffic directly measured = true
```

新的 benchmark / metrics 欄位會明確呈現這個邊界：

```text
reparallelization_execution_model = actor_recreate
expert_placement_execution_model = expert_aware_actor_recreate
expert_placement_runtime_contract_mode = observe_only_contract
expert_placement_live_migration_enabled = false
expert_placement_physical_migration_required = false

replanning_expert_placement_actor_recreate_events > 0
replanning_expert_placement_live_migration_events = 0
replanning_expert_placement_physical_migration_required_events = 0
replanning_expert_placement_runtime_contract_seen > 0
replanning_expert_placement_runtime_physical_weight_migration = 0
replanning_expert_placement_runtime_verification_levels = contract_seen_only
replanning_expert_placement_runtime_verified_placement = 0
replanning_expert_placement_runtime_can_verify_physical_placement = 0
replanning_expert_placement_runtime_can_remap_live_ep_rank = 0
replanning_expert_placement_runtime_can_measure_all_to_all = 0
```

如果未來真的完成 physical expert migration，這些欄位才應該轉成：

```text
expert_placement_execution_model = live_expert_weight_migration
expert_placement_live_migration_enabled = true
expert_placement_physical_migration_required = true
replanning_expert_placement_runtime_verification_levels = physical_migration_verified
replanning_expert_placement_runtime_verified_placement > 0
replanning_expert_placement_runtime_physical_weight_migration > 0
replanning_expert_placement_runtime_can_verify_physical_placement > 0
```

所以目前 Phase 5B 的 claim 是：

```text
Current implementation executes expert-aware re-parallelization by recreating
vLLM actors with a logical expert placement contract.
It does not execute live physical expert weight migration.
```

也就是：

```text
contract_seen_only
-> runtime 已收到 / 看過 ExpertPlacementPlan contract
-> 不代表 expert tensor 已搬動
-> 不代表 live EP rank mapping 已更新
-> 不代表 all-to-all dispatch traffic 已下降

physical_migration_verified
-> runtime apply 成功
-> runtime verify 成功
-> physical_weight_migration=true
-> 才能宣稱真正 physical expert weight migration
```

### Phase 5C: Runtime-verified Expert Placement

Phase 5C 把 runtime boundary 往前推一步：不再只看
`ExpertPlacementPlan` contract 是否被 runtime hook 看過，而是讓 patched vLLM
嘗試回報「目前這個 worker 實際 resident 的 local experts」。

vLLM MoE expert layout 目前可從 `FusedMoE` layer 讀到：

```text
global_num_experts / logical_num_experts
local_num_experts
expert_map
moe_parallel_config.ep_rank
moe_parallel_config.ep_size
```

vLLM 的 EP 語意是：

```text
EP enabled:
  each device owns a set of experts fully
  expert_map maps global expert id -> local expert index

EP disabled / ep_size = 1:
  all experts are local to the single EP rank
```

因此 Phase 5C 新增的 runtime introspection 會：

```text
1. WorkerBase 從 self.model_runner.get_model() 取得 loaded model。
2. 掃描 named_modules() 中疑似 FusedMoE 的 layer。
3. 透過 expert_map 找出本 worker local 的 global expert ids。
4. 掃描 local expert weight tensors，估算 resident weight bytes。
5. 回報 runtime_expert_placement_shards / worker_snapshots。
```

新的驗證語意：

```text
contract_seen_only
-> runtime hook 看過 ExpertPlacementPlan
-> 不代表 actual local experts 與 plan 相符

runtime_placement_verified
-> runtime 已掃到 resident expert weights
-> actual local experts 與 ExpertPlacementPlan 的 rank subset 相符
-> 仍不代表 live migration，因為 physical_weight_migration=false

physical_migration_verified
-> runtime apply/verify 都成功
-> physical_weight_migration=true
-> 才能宣稱 live physical expert migration
```

這次實作更新：

- `vllm.spotserve_moe.inspect_runtime_expert_placement()`：read-only 掃描
  loaded model 的 local experts。
- `WorkerBase._spotserve_runtime_expert_placement_snapshot()`：把 loaded model
  actual placement snapshot 傳給 apply/verify hook。
- `get_moe_runtime_metadata()`：回報
  `runtime_expert_placement_available`、
  `runtime_expert_placement_worker_count`、
  `runtime_expert_placement_shard_count`、
  `runtime_expert_placement_shards`。
- `verify_expert_placement_plan()`：只驗證屬於本 worker / EP rank 的 subset；
  multi-rank plan 不會要求每個 worker 擁有全部 experts。
- `estimate_expert_dispatch_cost()`：target 同時有 logical plan 與 runtime
  actual placement 時，優先使用 `runtime_expert_placement_shards`。

這一步回答了三個前置問題：

```text
experts 權重在哪？
-> vLLM FusedMoE layer 的 local expert tensors，例如 w13 / w2 類 weights。

EP rank 怎麼決定 local experts？
-> FusedMoE.expert_map 與 moe_parallel_config.ep_rank / ep_size。

worker init/load model 後能不能重新掛載 expert weights？
-> 目前沒有安全 live remap API。現階段只能 read-only 驗證 resident experts；
   真正 relocation 仍應走 controlled actor recreate with changed placement。
```

所以 Phase 5C 後，合理 claim 是：

```text
The runtime can report and verify actual resident expert placement when the
loaded vLLM model exposes FusedMoE expert maps.
```

仍不能 claim：

```text
physical expert weight migration
live EP rank remapping
real all-to-all traffic reduction
```

### Phase 5D: Quiescent Fixed-EP Expert Remap (Experimental)

現在新增一條明確 opt-in 的 runtime 路徑：`sllm/vllm_expert_remap.py`
使用 vLLM EPLB 的 `rearrange_expert_weights_inplace()` 在既有 EP group
內搬運 unquantized FusedMoE expert tensor，並原地更新 `expert_map`。
搬前、搬後以每個 expert 的權重 SHA-256 比對；所有 EP ranks 的預檢必須
通過，且 engine 沒有 unfinished requests，才會開始 collective transfer。

限制與語意要保持嚴格：

- 預設關閉。worker 環境需 `VLLM_SPOTSERVE_EXPERT_REMAP=1`，傳入的
  `ExpertPlacementPlan` 也需 `live_expert_remap=true`、非空
  `placement_fingerprint` 與完整 `expert_to_target_rank`。
- 僅支援 **相同 TP/EP size 且 DP=1**、單 replica、每個 expert 恰好一份、
  各 rank expert 數量不變、unquantized FusedMoE。EPLB、quantized weights、
  redundant experts、shared fused experts 和動態 EP 縮放都會在預檢拒絕。
- 這裡的「live」預設只表示**不重建既有 engine 的固定 EP remap**。
  2026-09-22 後新增一個更嚴格的 opt-in active-request 模式：
  plan 必須帶 `allow_active_requests=true`，worker 也必須設定
  `VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1`。runtime 只允許在 synchronous
  `EngineCore.step()` 邊界執行，且 `batch_queue is None`、`async_scheduling`
  為 false。distributed collective 或 verify 失敗時，runtime/controller 會將
  execution 標為 failed，不能把該次 replan 計為成功；目前不宣稱具有
  transaction rollback 或失敗後原 engine 可安全繼續服務。
- `physical_weight_migration=true` 只在至少一個 expert 跨 EP rank 進入
  本 rank 且搬後權重摘要一致時回報；同 rank slot 重排不算跨 rank 傳輸。
- `require_cross_node=true` 是 hard gate：runtime 必須看到 distinct
  `SPOTSERVE_PHYSICAL_HOST_ID`，且至少一個 expert shard 從不同 physical host
  進入本 rank，才會回報 `cross_node_weight_migration=true`。same-host mp
  驗證會正確停在 `cross_node_requires_distinct_physical_host_ids`。
- 這一步尚未量測或降低真實 inference all-to-all traffic，也不支援
  live EP group size 變更。舊 benchmark 的 `expert_plan=1` 仍不能當成
  physical migration 成功的證據。

已驗證三層：worker 環境的 5 個預檢單元測試、兩張 GPU 上的 vLLM
EPLB primitive + SpotServe 搬運器合成 tensor 實測，以及真實
Qwen2-MoE-Tiny TP2/EP2 runtime apply/verify。真實模型從 linear placement
`rank0={0,1}, rank1={2,3}` remap 為 round-robin placement
`rank0={0,2}, rank1={1,3}`；兩層合計搬移 4 個 expert shards、786432 bytes，
兩個 worker 都回報 `physical_weight_migration=true` 與
`weights_and_expert_map_verified`，remap 前後相同 prompt 的 token IDs 一致。
可重跑：

```bash
podman cp sllm/vllm_expert_remap.py sllm_worker_0:/opt/venvs/worker/lib/python3.11/site-packages/sllm/vllm_expert_remap.py
podman cp tests/spotserve_test/test_vllm_expert_remap.py sllm_worker_0:/tmp/test_vllm_expert_remap.py
podman cp scripts/verify_spotserve_expert_weight_transfer.py sllm_worker_0:/tmp/verify_spotserve_expert_weight_transfer.py
podman exec sllm_worker_0 /opt/venvs/worker/bin/python /tmp/test_vllm_expert_remap.py
podman exec sllm_worker_0 /opt/venvs/worker/bin/python /tmp/verify_spotserve_expert_weight_transfer.py
podman cp scripts/verify_spotserve_qwen_ep_remap.py sllm_worker_0:/tmp/verify_spotserve_qwen_ep_remap.py
podman exec sllm_worker_0 bash -lc '
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
/opt/venvs/worker/bin/python /tmp/verify_spotserve_qwen_ep_remap.py \
  --model /models/Qwen2-MoE-Tiny \
  --output /tmp/qwen_quiescent_remap_report.json
'
podman exec sllm_worker_0 bash -lc '
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
/opt/venvs/worker/bin/python /tmp/verify_spotserve_qwen_ep_remap.py \
  --model /models/Qwen2-MoE-Tiny \
  --active-request \
  --gpu-memory-utilization 0.45 \
  --output /tmp/qwen_active_remap_report.json
'
```

要驗證 **新的 vLLM apply hook**，必須重新 build image，因為
`--skip-build` 不會更新已安裝的 vLLM patch；上述 `podman cp` 只驗證
搬運器本身，並不更新 engine/core/worker hooks。正式 controller 路徑可用
`benchmark_matrix_expert_remap_performance.yaml` 驗證，成功時至少必須看到：

```text
replanning_expert_placement_runtime_apply_success > 0
replanning_expert_placement_runtime_verify_success > 0
replanning_expert_placement_runtime_physical_weight_migration > 0
replanning_expert_placement_runtime_verified_placement > 0
replanning_total_expert_placement_runtime_moved_expert_shards > 0
replanning_total_expert_placement_runtime_moved_weight_bytes > 0
```

active-request validation additionally requires:

```text
replanning_expert_placement_runtime_active_request_remap > 0
replanning_expert_placement_runtime_step_boundary_barrier > 0
```

2026-09-21 的 TP2/EP2 Qwen2-MoE-Tiny 單次端到端驗證：

```text
run: 2026-09-21_11-50-59_vllm-expert-remap-applied
requests: 8/8; trace replay: 1/1; replan applied: 1/1
runtime apply/verify: 1/1; verification: physical_migration_verified
runtime moved: 4 expert shards, 786432 bytes; remap duration: 190.90 ms
runtime actual placement: 2 workers, 8 shards
live EP remap: 0; all-to-all counters: 0
```

`runtime moved` 是 worker 回報的搬動量，和 planner 預估的
`expert_plan_moved=4`、`expert_plan_moved_mb=0.75` 分開記錄。
這是一個 correctness / observability run，沒有 latency 對照組，
因此 `8/8` 與 p95 不代表 remap 降低了服務延遲。

2026-09-22 後，active-request 的 standalone verifier 會在同一個 TP2/EP2
engine 內先跑一個 deterministic baseline request，再於第二個 streaming
request 已經產生第一個 token 後呼叫 `apply_expert_placement_plan()`。通過
條件是 `active_requests_at_barrier=true`、`step_boundary_barrier=true`，
且 active request 的 token ids 與 baseline 完全一致。這只能支持
「synchronous step-boundary active-request fixed-EP remap」，不是任意
continuous batching / async scheduling 下的無停頓 remap。

2026-09-22 的 controller active-request benchmark (`02-33-05`) 是一個
**未通過 active remap gate** 的紀錄：請求 `3/3`、trace `1/1`、runtime
apply/verify `1/1`，且 runtime 回報搬動 4 個 expert shards / 0.75 MiB；
但 `exec_model=quiescent_fixed_ep_remap`、`actor_recreate=1`、
`runtime_active_remap=0`、`runtime_step_barrier=0`。因此它只驗證了
quiescent physical remap，不能作為 active-request remap 的實驗結果。

後續修正了兩個 controller/fixture 問題。首次 replan 先建立 vLLM
deployment adapter，再取得 current actor snapshot，避免因 adapter 尚未
建立而誤走 actor recreate；缺少 runtime MoE topology 時，benchmark
config 提供兩層、每層四個 experts 的模型拓樸。active-request trace 改用
`add` 容量事件，使原 actor 保持 READY；`preempt` 會把 actor 標為
PREEMPTING，不能在相同即將退役的 actor 上把原地 remap 當成服務恢復。
router 也禁止對非 READY actor 選用原地 remap。

2026-09-23 已完成 controller active-request benchmark 重跑：

```text
run: 2026-09-23_01-08-25_vllm-expert-remap-active-request-applied
requests: 3/3; trace replay: 1/1; replan applied: 1/1
execution: active_fixed_ep_remap; actor recreate: 0
runtime apply/verify: 1/1; verification: physical_migration_verified
runtime active remap: 1; synchronous step-boundary barrier: 1
runtime moved: 4 expert shards, 786432 bytes; remap duration: 168.12 ms
runtime actual placement: 2 workers, 8 shards
live EP-size remap: 0; all-to-all counters: 0
```

這次 trace 的 `add` event 延後到 35 秒，確保 warmup 結束且長 request 已進入
runtime；workload 使用 test-only token pacing 擴大 active window。這個結果
通過 active-request fixed-EP gate，但仍不是 active EP resize、跨節點搬動或
all-to-all traffic reduction 的證據。

2026-09-24 再次重跑時發現上述 warmup + 固定 35 秒事件仍會受 DELTA chunk
數量影響：`_spotserve_token_delay_s` 是每個 runtime output chunk 的延遲，
不是每 token 延遲。若 request 提早完成，physical remap 仍會成功，但
`runtime_active_remap=0`。fixture 因此收斂為長 request 在 `t=0` 送出、每
chunk 延遲 1 秒、`add` event 在 `t=1` 觸發，並移除會阻塞 request dispatch
的 warmup。修正後結果為：

```text
run: 2026-09-24_12-27-04_vllm-expert-remap-active-request-applied
requests: 2/2; trace replay: 1/1; replan applied: 1/1
execution: active_fixed_ep_remap; actor recreate: 0
runtime apply/verify: 1/1; verification: physical_migration_verified
runtime active remap: 1; synchronous step-boundary barrier: 1
runtime moved: 4 expert shards, 786432 bytes; remap duration: 117.99 ms
runtime actual placement: 2 workers, 8 shards
```

同日第一次啟動曾在 `EngineCore` 初始化階段失敗，尚未進入 trace replay；
立即重跑可正常啟動。當時 Ray 另回報 session filesystem 超過 95% 使用率，
因此該次只記為 environment/startup failure，不納入 remap correctness 結果。

2026-09-25 再次遇到第一個 actor 啟動失敗、controller 已建立 replacement
actor 的情況。舊 benchmark runner 只要在 model status 看見任一歷史 failed
actor 就立即中止，即使 replacement actor 已處於 `starting`，因此會把正常的
startup retry 誤判為整個 setup 失敗。`wait_for_ready_instances()` 現在會在仍有
非 failed/dead startup candidate 時繼續等待；只有連續三次輪詢都只剩 failed
actor 才提早失敗。修正後重跑結果為：

```text
run: 2026-09-25_08-55-34_vllm-expert-remap-active-request-applied
requests: 2/2; trace replay: 1/1; replan applied: 1/1
execution: active_fixed_ep_remap; actor recreate: 0
runtime apply/verify: 1/1; verification: physical_migration_verified
runtime active remap: 1; synchronous step-boundary barrier: 1
runtime moved: 4 expert shards, 0.75 MiB; remap duration: 133.18 ms
runtime actual placement: 2 workers, 8 shards
request latency p95: 3192.94 ms
```

這筆結果再次通過 active-request fixed-EP gate；它不擴張既有 claim，仍不代表
EP-size resize、跨節點 expert 搬動或 all-to-all traffic reduction。

### 2026-09-25：EP resize、cross-node 與 all-to-all 驗證補強

目前三項缺口已拆成獨立、fail-closed 的 runtime contract：

1. **動態 EP size**：vLLM process group 無法在既有 engine 內安全改大小，因此
   不宣稱 live in-place resize。當 source 與 target 的 effective EP size 不同時，
   controller 會走 controlled actor recreate，並回報
   `actor_recreate_ep_resize`、source EP、target EP 與
   `dynamic_ep_resize=true`。`scripts/verify_spotserve_ep_transition.py` 會比對
   TP2/EP2 與 TP4/EP4 的 runtime-observed ownership、coverage 與 execution model。
2. **跨節點 expert weight movement**：planner 新增
   `require_cross_node_expert_migration`。啟用後會把 `require_cross_node=true`
   傳到 runtime。runtime 必須觀察到至少兩個不同
   `SPOTSERVE_PHYSICAL_HOST_ID`，且至少一個 expert shard 的來源與目的 host
   不同，否則 apply 直接失敗。單機多 container 不得通過這個 gate。
3. **真實 all-to-all instrumentation**：vLLM patch 現在掛在
   `CudaCommunicator.dispatch()` / `combine()`，記錄實際 all-to-all manager
   invocation 的 calls、input/output tensor payload bytes、backend 與 internode
   calls。這不是由 routing histogram 推估。這些數值是 collective API boundary
   的 tensor payload，不等同 NIC wire bytes。

`scripts/verify_spotserve_all_to_all_traffic.py` 可在固定 workload 前後取 runtime
counter delta；帶入 `--baseline-report` 時會計算 payload reduction ratio，若
candidate 未下降會 fail closed。只有 baseline/candidate 使用相同 workload 且
counter delta 均大於零時，才可宣稱 all-to-all payload reduction。

這批 runtime hook 需要 rebuild image；僅 `SPOTSERVE_SYNC_SOURCE=1` 不會修改
image 內的 vLLM communicator：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
VLLM_SPOTSERVE_A2A_TRACE=1 \
SPOTSERVE_REQUIRE_ALL_TO_ALL_INSTRUMENTATION=1 \
scripts/prepare_spotserve.sh --deploy-set expert-remap-active-performance
```

目前 coding 狀態：controlled EP resize 與 A2A counter 路徑已完成；兩者仍需在
rebuild image 上跑 runtime verifier。cross-node hard gate 已完成，但實際
cross-node success 必須使用至少兩台 physical worker hosts。

已安裝 active-remap vLLM patch 的 image 可用下列命令同步 controller
程式並重跑；若 image 尚未包含 patched runtime，先不要使用 `--skip-build`：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
scripts/prepare_spotserve.sh --skip-build --deploy-set expert-remap-active-performance

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_expert_remap_active_request_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 --trace-event-timeout 600 \
  --ray-address auto --ray-namespace sllm
'
```

通過條件包含 `trace_success=1`、`exec_model=active_fixed_ep_remap`、
`actor_recreate=0`、`runtime_active_remap=1`、`runtime_step_barrier=1`、
`runtime_apply_success=1`、`runtime_verify_success=1`。單看成功率或
`runtime_physical_migration=1` 不足以判定 active-request 路徑成功。

改變 EP size 目前不走 in-place live remap。`target_rank_count != current EP
size` 會被 runtime preflight 拒絕；要從 TP2/EP2 變成 TP4/EP4，現階段
定義為 controlled actor recreate：先建立新 actor、由新 runtime 回報 actual
expert placement，再用 before/after placement snapshot 驗證哪些 experts
改變 EP owner。可用 `scripts/verify_spotserve_ep_transition.py` 在 4 張 GPU
環境驗證這條 actor-recreate transition。

這些結果支持「fixed-EP physical expert migration」與「controlled EP-size
transition via actor recreate」。因為 `can_remap_live_ep_rank=false` 且
`can_measure_all_to_all=false`，仍不能宣稱 arbitrary live EP resize 或
real all-to-all traffic reduction。

### 2026-09-26：DP2 all-to-all runtime gate

2026-09-25 的 active-request 結果中，`runtime_a2a_counters=1` 但
`runtime_a2a_calls=0`。這不是 physical remap 失敗，而是測試使用
`TP=2, DP=1`。vLLM 0.11.2 的 `FusedMoEParallelConfig.use_all2all_kernels`
只有在 `data_parallel_size > 1 and use_ep` 時才成立；因此 EP size 雖由
`TP * DP = 2` 推導，該配置仍不會走 all-to-all prepare/finalize path。

本階段補強如下：

- `can_measure_all_to_all` 改為 fail closed：只有 runtime 實際觀察到至少一個
  dispatch/combine collective 才為 true；僅開啟 trace 環境變數不算成功。
- DP2 可用來量測 A2A traffic，但 physical remap 仍 fail closed。vLLM 每個 DP
  rank 有獨立 EngineCore；目前 hook 無法保證所有 DP EngineCores 同時進入
  expert-transfer collective，因此允許 DP2 remap 會 deadlock。
- vLLM MoE capability 新增 `TP1 x DP2` 與 `TP2 x DP2` shape。
- planner 的 `min_data_parallel_size` 與 ServerlessLLM `min_replica_count`
  已拆開，不再把 vLLM DP 誤當成獨立 serving replicas。
- 新增 `benchmark_matrix_expert_remap_dp2_a2a_performance.yaml`，使用
  `TP=1, DP=2, EP=2` 與 `allgather_reducescatter`，只做 inference，不觸發
  remap。benchmark 保留 model，後續由
  `verify_spotserve_all_to_all_traffic.py` 驗證 counter delta。

DP2 gate 的通過條件是：

```text
benchmark success_rate = 1.0
a2a-report.delta.collective_calls > 0
a2a-report.observed_payload_bytes > 0
```

2026-09-25 的 DP2 runtime gate 已通過：benchmark requests `2/2`，p95
`1149.20 ms`；後續 verifier 對四個 requests 觀察到：

```text
collective call delta: 512
input payload delta: 486080 bytes
output payload delta: 488128 bytes
total observed payload: 974208 bytes
internode call delta: 0
```

這證明 `TP1 x DP2 x EP2` inference 確實走過 patched vLLM 的 real
dispatch/combine collective boundary。`internode=0` 符合本次 single-host
環境。這筆結果不是 traffic reduction 證據，因為尚未有相同 workload 的
baseline/candidate payload 比較。

這只能證明真實 collective traffic 可觀測。對
`allgather_reducescatter` 而言，總 payload 可能不隨 expert placement 改變；
因此「traffic reduction」仍必須以相同 workload 的 baseline/candidate counter
delta 實測，且 candidate bytes 嚴格小於 baseline 才能宣稱。不能用 routing
histogram 或 estimated remote tokens 代替這個結果。

### 2026-09-26：All-DP-Engine Remap Coordinator

前一版 DP2 remap 只透過單一 EngineCore 進入 expert-transfer collective，另一個
DP rank 沒有同時進入，因而 deadlock。第一版 all-DP broadcast 仍有另一個 race：
若一個 EngineCore 有 active request、另一個已 idle，前者會拒絕、後者卻會進入
transfer collective。2026-09-26 的 `03-34-13` run 正好觸發此情況，event 在
600 秒後 timeout，requests 為 `0/2`；因此該 run 是失敗結果，不能當作 DP2
physical remap 證據。

runtime patch 現改為 all-DP-engine 兩階段協議：

```text
AsyncLLM
-> enumerate core_engines
-> concurrently call side-effect-free preflight on every DP EngineCore
-> if any EngineCore rejects, no EngineCore enters transfer
-> only if every preflight succeeds, concurrently commit apply
-> verify on every DP EngineCore
-> every EngineCore enters its local worker collective
-> aggregate every DP engine and worker result
```

placement plan 只有在 `dp_engine_coordinated=true` 且
`dp_engine_count == runtime data_parallel_size` 時才能通過 DP remap preflight。
runtime metadata 也改為聚合所有 DP EngineCore，避免 planner 只看到 DP rank 0
而把其餘 experts 標成 `missing_current_expert`。

新增 `verify_spotserve_dp2_coordinated_remap.py` 作為主要 correctness gate。它不以
固定 trace timestamp 猜測 request 是否完成，而是依序等待 warmup request 完成、
router concurrency 歸零，再送 placement event，最後驗證 apply/verify、runtime
placement 與 post-remap inference。原本的 timed benchmark 仍可作 concurrency
stress test，但不能單獨證明 quiescent DP2 remap。

2026-09-26 sequential gate 已通過：pre-remap request 與 post-remap request 都
完成，event 前 router concurrency 為零；runtime 對 2 個 DP EngineCore 完成
apply/verify，實際搬動 4 個 expert shards、786432 bytes，remap duration 為
123.13 ms，且 runtime placement verification 成功。因此目前可宣稱同一個
physical host 上的 `TP1 x DP2 x EP2` coordinated physical expert remap 已驗證。
這筆結果不代表 cross-host migration、EP resize 或 A2A traffic reduction。

這項 runtime patch 需要 rebuild image，`SPOTSERVE_SYNC_SOURCE=1` 無法更新
image 內的 `vllm/v1/engine/async_llm.py`。

vLLM 0.11.2 本身有 elastic-EP resize API，但只支援 Ray DP backend。目前
ServerlessLLM backend actor 已先向 Ray 持有整組 GPU，若直接啟用 vLLM nested
Ray DP actors，會形成兩層 GPU resource ownership。完成 allocation ownership
整合以前，live in-place EP resize 仍不可安全啟用；EP2 -> EP4 繼續使用
controlled actor recreate，不把它誤稱為 live resize。

三項剩餘驗證的精確狀態：

```text
dynamic EP size:
  coding complete as controlled actor recreate; TP2/EP2 -> TP4/EP4 verifier pending

cross-node movement:
  runtime hard gate complete; success requires two distinct physical host IDs

real all-to-all traffic:
  DP2 inference/measurement and coordinated physical-remap gates passed;
  same-host sparse dispatch payload reduction passed;
  internode traffic reduction remains unverified
```

### A2A Reduction Experiment

`benchmark_matrix_expert_remap_a2a_reduction_performance.yaml` 定義一個有順序
的 baseline/remap experiment：

```text
warmup
-> baseline counter snapshot
-> replay workload
-> baseline counter delta
-> coordinated physical expert remap
-> candidate counter snapshot
-> replay the identical workload
-> candidate counter delta
-> compare observed payload bytes
```

為避免把 cache reuse 當成 communication reduction，這個 config 強制關閉
prefix caching。workload 使用偶數 request 數量，讓 DP round-robin 的起始位置在
兩個 measurement windows 一致；runner 也要求前後 inference outputs 完全相同、
runtime apply/verify 成功且兩邊都觀測到真實 collective calls。

`traffic_reduced=true` 且 candidate payload 嚴格小於 baseline 時，才支援 A2A
reduction claim。runner 另外要求 measurement kind 必須是
`runtime_sparse_transfer_payload`，避免把固定大小 collective 的估計值誤當成
destination-aware sparse transfer。若 `allgather_reducescatter` 前後 bytes 相同，
實驗仍可成功完成，
但會回報 `payload_invariant_for_allgather_reducescatter` 與
`claim_supported=false`。這代表 backend 的 collective volume 不受 expert
ownership 改變，而不是把零改善包裝成成功。

2026-09-26 先以 `allgather_reducescatter` 執行三次重複 median，得到負向
baseline：

```text
physical expert shards moved: 4
baseline collective calls: 1000
candidate collective calls: 1000
baseline observed payload: 1801872 bytes
candidate observed payload: 1801872 bytes
baseline bytes / collective: 1801.872
candidate bytes / collective: 1801.872
payload reduction ratio: 0.0
traffic_reduced: false
claim_supported: false
interpretation: payload_invariant_for_allgather_reducescatter
```

接著加入 `spotserve_sparse` backend。它從 runtime `topk_ids` 與目前
`expert_map` 建立每個 token 的 destination ranks，本地 expert 路由不進網路，遠端
token 則使用 variable-size `all_to_all_single` dispatch/combine。相同 workload、
相同輸出與三次重複 median 的結果為：

```text
all-to-all backend: spotserve_sparse
measurement kind: runtime_sparse_transfer_payload
physical expert shards moved: 4
outputs match: true
baseline collective calls: 992
candidate collective calls: 1000
baseline observed payload: 1148928 bytes
candidate observed payload: 931392 bytes
payload reduction: 217536 bytes
payload reduction ratio: 0.189338 (18.93%)
traffic_reduced: true
claim_supported: true
interpretation: measured_collective_payload_reduction
```

因此目前可以宣稱：在這個 same-host、DP2/EP2、未量化 Qwen2-MoE-Tiny workload
中，physical expert remap 配合 destination-aware sparse dispatch，使實際送收的
GPU collective payload median 降低 18.93%。這仍不是跨實體主機或 NIC traffic
reduction 證據；該次結果的 `internode_calls=0`，跨節點 claim 必須另外驗證。

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
| EP layout changed | not applicable | runtime-dependent | restore if KV compatible | locality may change | correctness and locality reported separately |
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
2. 標示 runtime-provided vs instrumentation-derived metadata
3. 建立 global / per-request / recent-window routed-token statistics
   若 per-request routing 無法取得，先退化為 KV-only target selection
4. 在 context migration target selection 加 expert locality score
5. 在 stateful recovery 中分離 KV restore correctness 與 expert locality
6. 報告 hot expert locality、remote dispatch cost、fallback 原因
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
-> runtime effective EP / placement snapshot
-> runtime-provided vs instrumentation-derived metadata
-> global/request-level/recent-window routed-token statistics

Phase 2
MoE-aware target selection
-> KV compatibility
-> expert locality
-> queue cost
-> explicit expert_dispatch_cost definition

Phase 3
MoE-aware stateful recovery
-> separate model semantic / state serialization / KV layout compatibility
-> separate KV restore correctness from expert locality
-> measure remote expert dispatch after restore

Phase 4
Expert dispatch observability baseline
-> runtime routing + placement-derived local/remote routed-token breakdown
-> per-layer / per-expert remote routed-token reports
-> placement_epoch / expert_placement_fingerprint handshake
-> no physical expert movement yet

Phase 5
True expert-aware re-parallelization
-> EP shape
-> expert remapping / replication
-> weight movement
-> clear responsibility boundary with vLLM EPLB

Phase 6
physical cross-node validation
```

## 最安全的階段性 Claim

```text
We first implement a SpotServe-style control plane for vLLM MoE serving.
Then, we extend its planning decisions with MoE-specific runtime signals,
including runtime placement snapshots and routed-token hotness when available,
so that re-parallelization, context migration, and stateful recovery can prefer
targets with better expert locality while preserving KV/cache compatibility and
leaving steady-state intra-deployment expert balancing to the vLLM runtime.
```
