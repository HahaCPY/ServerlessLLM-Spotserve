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

這個 profile 是 optional runtime capability。若 runtime 不能直接提供，MVP 可以先
用 patched tracing 或 offline instrumentation 收集；若完全沒有 routing history，
planner 應退化成 KV-only / queue-only target selection。

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
moe_moved_weight_bytes
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

### Phase 2 可驗證實驗

Phase 2 的 MVP 應先用 synthetic ablation 驗證 planner，而不是直接依賴
end-to-end vLLM latency。原因是 vLLM runtime 目前還不一定能穩定輸出真實
per-request expert routing instrumentation；synthetic workload 可以先確認 cost
model 本身是否正確影響 target selection。

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

### Milestone C: MoE-aware Stateful Recovery

目標：recovery target selection 分離 KV restore correctness 與 expert locality。

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
- restore 後量測 remote expert dispatch / locality penalty。

### Milestone D: Expert-aware Re-parallelization

目標：re-parallelization planner 真的能決定 expert placement。

完成條件：

- `ExpertPlacementPlan` 可序列化到 metrics。
- preempted GPU 上的 experts 可被重新配置到 ready GPUs。
- planner cost 同時考慮 GPU capacity、expert movement、dispatch cost。
- 明確定義 SpotServe planner 與 vLLM EPLB 的責任邊界：SpotServe 負責
  resource-change / preemption-aware topology planning，vLLM EPLB 負責
  deployment 內部 steady-state expert balancing。

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
True expert-aware re-parallelization
-> EP shape
-> expert remapping / replication
-> weight movement
-> clear responsibility boundary with vLLM EPLB

Phase 5
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
