# SpotServe on MoE 問題盤點

分析日期：2026-08-19

## 一句話結論

目前這份專案的方向沒有錯：它已經把 SpotServe 的三個核心概念接到
ServerlessLLM/vLLM 控制平面上，並且讓 vLLM MoE model 可以跑在這套
preemption-aware flow 裡。

但目前不能直接宣稱「完整實作 SpotServe on MoE」。比較準確的說法是：

```text
已完成：SpotServe-style control-plane prototype + vLLM MoE compatibility
部分完成：vLLM patched runtime/NIXL 的 stateful KV restore smoke
尚未完成：MoE-aware expert placement / expert migration / expert recovery
```

## 三大核心對應架構

| SpotServe 核心 | 專案主要位置 | 目前狀態 |
|---|---|---|
| Dynamic re-parallelization | `sllm/spot/reparallelization.py`, `sllm/spot/reparallelization_executor.py`, `sllm/spot/vllm_deployment_adapter.py`, `sllm/routers/roundrobin_router.py` | 控制平面與 vLLM actor 重建/切流有做，但不是論文完整 optimizer，也不是 in-place MoE expert repartition |
| Low-cost context migration | `sllm/spot/context_migration.py`, `sllm/backends/vllm_context_metadata.py`, `sllm/routers/roundrobin_router.py` | 有 context metadata 與低成本 mapping planner；但 V7 本身多數是 planning / prefix warmup，不等於 true KV block migration |
| Stateful recovery | `sllm/spot/stateful_recovery.py`, `sllm/backends/vllm_state_metadata.py`, `sllm/backends/vllm_backend.py`, `sllm/routers/roundrobin_router.py` | recovery decision、fallback、patched vLLM/NIXL hooks 有接；真 KV restore 依賴 patched runtime，未 patch 時是 token replay fallback |
| Preemptible simulation | `sllm/spot/trace_reader.py`, `sllm/spot/preemption_simulator.py`, `sllm/app_lib.py`, `sllm/controller.py` | JSONL trace replay 模擬 `add/remove/preempt/recover/dead`，不是真 cloud spot provider integration |

## 主要問題

### 1. MoE 目前多數仍是 black-box integration

目前 `docs/spotserve-version5-vllm-moe.md` 已經明確寫出 MoE path 是
vLLM MoE black-box integration，且 expert routing、expert migration、
MoE-specific recovery optimization 都是 out of scope。

程式上 `sllm/backends/vllm_capability.py` 有 MoE supported shapes，也能表示
`enable_expert_parallel=True` 後的 derived effective EP；但目前沒有做到：

- expert-level placement decision
- expert hotness / routed-token load balancing
- expert KV / expert weight migration cost model
- preemption 後針對 expert shard 的 recovery policy

影響：

```text
可以宣稱：SpotServe-style control plane 可以服務 vLLM MoE model。
不能宣稱：已經把 SpotServe 完整應用到 MoE expert-level serving。
```

建議補強：

- 在 runtime metadata 加入 `num_experts`, `effective_expert_parallel_size`,
  `expert_ids_by_rank`, `expert_load`, `routed_tokens_by_expert`。
- 讓 re-parallelization planner 真的把 expert placement 放進候選與 cost model。
- 加 MoE-specific benchmark：preempt 某個 expert-heavy rank，看新 plan 是否降低
  expert movement / recomputation。

### 2. `expert_parallel_size` 的規劃與實際 vLLM config 可能不一致 ✅

vLLM 的 MoE EP 不是獨立設定 `expert_parallel_size=N`。正確語意是：

```text
enable_expert_parallel = True / False
effective_expert_parallel_size = tensor_parallel_size * data_parallel_size
```

修正前，`sllm/backends/vllm_capability.py` 會產生
`expert_parallel_size=2` 等候選，而 `sllm/spot/vllm_deployment_adapter.py`
只把它轉成：

```python
config["enable_expert_parallel"] = plan.expert_parallel_size > 1
```

更大的問題是：`data_parallel_size` 在 planner 中其實曾代表
ServerlessLLM 要建立幾個 Ray Actor，但在 vLLM 中它代表同一個 distributed
vLLM deployment 內部的 DP degree。也就是：

```text
planner data_parallel_size = replica count
vLLM data_parallel_size = runtime DP
```

這兩個混在一起時，planner 以為 `DP=4`，但 executor 實際建立的是四個
`data_parallel_size=1` 的獨立 vLLM actors。

影響：

```text
planner DP / EP plan != vLLM runtime DP / effective EP
```

建議補強：

- `ParallelPlan.data_parallel_size` 只代表 vLLM runtime DP。
- 新增 `replica_count` 代表 ServerlessLLM/Ray Actor 數量。
- planner 不再自由設定 `expert_parallel_size`。
- `effective_expert_parallel_size` 改成 derived value：
  `TP * DP` when `enable_expert_parallel=True`，否則為 1。
- `get_runtime_metadata()` 回報 actual TP/DP/EP-enabled/effective EP，並標示
  `parallel_plan_mismatch`。

已修正：

- `ParallelPlan` 已拆成 `data_parallel_size` 與 `replica_count`。
- `ParallelPlan.effective_expert_parallel_size` 由 `enable_expert_parallel`、
  `tensor_parallel_size`、`data_parallel_size` 推導。
- `VllmDeploymentAdapter` 會把 `data_parallel_size` 傳給 vLLM，並用
  `replica_count` 建立 ServerlessLLM Ray actors。
- 第一階段 planner/capability 保守限制 vLLM runtime DP 為 1；原本多 DP 的語意
  轉成 `replica_count`。
- re-parallelization decision / metrics 增加 `selected_replica_count`、
  `selected_enable_expert_parallel`、
  `selected_effective_expert_parallel_size`。
- runtime metadata 增加 `planned_effective_expert_parallel_size`、
  `effective_expert_parallel_size`、`parallel_plan_mismatch`。

### 3. V7 context migration 不等於 true KV migration ✅

`sllm/spot/context_migration.py` 做的是低成本 assignment planning。router 的
live path 會收集 source/target context metadata，推估 reuse，然後可選擇呼叫
`resume_kv_cache()`。

但 `sllm/backends/vllm_backend.py` 的 `resume_kv_cache()` 是用 token batch 再跑
一次短 generate/prefill 來 warm target cache。這是 prefix/cache warmup，不是把
source 的 KV block serialization 後傳到 target，也不是把原 request binding 到
target。

影響：

```text
context_migration_plan_count > 0 只能證明 planner 有產生 mapping。
kv_cache_migration_successes > 0 不一定代表 true KV block transfer 成功。
```

實驗解讀：

```text
V7 本身不一定需要 true KV cache transfer。
V7 的核心目標是 low-cost target selection，而不是 runtime-level KV block
serialization / attach。
```

因此，這個修正主要是語意、event 與 metric claim 的修正。若只比較
「語意修正前的 V7」和「語意修正後的 V7」，理論上 latency / throughput
不應該有明顯差異，因為 target selection 與 prefix warmup 的執行路徑沒有被改成
true KV transfer。差異應該主要出現在 log / metrics 的解讀：

```text
以前：
kv_cache_migration_successes > 0
容易被誤解成 true KV migration。

現在：
prefix_warmup_successes > 0
true_kv_block_transfer = false
```

如果 V7 語意修正前後的性能差異很大，應優先檢查 benchmark config、metric
parser、runtime 狀態或 workload 是否一致，而不是把差異歸因於 true KV
migration。

建議補強：

- metric 命名區分：
  - `context_migration_plan_*`
  - `prefix_warmup_*`
  - `kv_restore_*`
- 只有當 `state_kind=vllm_kv_snapshot`、`supports_restore=true`、
  `restored_blocks>0` 時，才宣稱 true KV migration / restore。

已修正：

- `RoundRobinRouter` 的執行路徑改成回報 `prefix_warmup`，而不是把
  `resume_kv_cache()` 直接稱為 true KV migration。
- 舊欄位 `kv_cache_migration` 保留為相容 alias，但會標示
  `deprecated_alias=true`、`operation_kind=prefix_warmup`、
  `true_kv_block_transfer=false`。
- `make_context_migration_event()` 新增：
  - `context_migration_plan_count`
  - `prefix_warmup_attempts`
  - `prefix_warmup_successes`
  - `prefix_warmup_tokens`
  - `kv_restore_attempts`
  - `kv_restore_successes`
  - `kv_restore_restored_blocks`
  - `true_kv_block_transfer`
- `VllmBackend.resume_kv_cache()` 回傳 structured prefix-warmup result，並明確標示
  `true_kv_block_transfer=false`。

### 4. V8 stateful recovery 依賴 patched vLLM runtime ✅

`sllm/backends/vllm_backend.py` 透過 runtime hooks 尋找：

```text
get_request_kv_metadata
get_all_request_kv_metadata
export_inference_state
restore_inference_state
supports_state_restore
```

這些 hooks 來自 `sllm_store/vllm_patch/runtime_kv_metadata.patch` 和
`sllm_store/vllm_patch/runtime_kv_restore.patch`。如果執行環境沒有打 patch，
`VllmBackend.supports_state_restore()` 會保守回 false，最後走 token snapshot /
generated-token replay fallback。

影響：

```text
stateful_recovery policy 成功完成 request，不一定代表 KV restore 成功。
需要看 state_restore_successes_total、state_restore_fallback_count、
restored_blocks 等欄位。
```

建議補強：

- benchmark 報告中強制列出：
  - `supports_state_restore`
  - `state_kind`
  - `state_restore_successes_total`
  - `state_restore_fallback_count`
  - `restored_blocks`
- CI 或 smoke test 在跑 NIXL 實驗前先執行 patch capability check。

已修正：

- `VllmBackend.supports_state_restore()` 只在 runtime hook 同時具備
  export / restore 能力，且 `supports_state_restore()` probe 沒有回 false 時才回
  true；未 patch 的 vLLM 會保守走 token snapshot / generated-token replay。
- `VllmBackend.restore_inference_state()` 不把 `restored_blocks <= 0` 當成成功；
  只有真的 attach 到 KV blocks 的結果才會回報 restored success。
- request metrics 新增 restore evidence：
  - `supports_state_restore`
  - `state_kind`
  - `state_restore_reason`
  - `state_restored_blocks`
  - `state_restore_staged`
- benchmark analyzer 會從 router metrics 彙總：
  - `state_restored_blocks_total`
  - `supports_state_restore_requests`
  - `state_restore_staged_count`
  - `true_kv_restore_successes_total`
  - `state_kinds`
  - `state_restore_reasons`
- benchmark analyzer 也會從 raw response 的 `_spotserve_kv_restore` 彙總：
  - `response_kv_restore_events`
  - `response_kv_restore_successes`
  - `response_kv_restore_restored_blocks`
  - `response_kv_restore_cached_tokens`
  - `response_kv_restore_reasons`
- benchmark analyzer 會把 router metrics 與 raw response evidence 合併成 unified
  true KV restore 欄位：
  - `true_kv_restore_successes_total`
  - `true_kv_restored_blocks_total`
  - `true_kv_restore_evidence_sources`
- benchmark CLI summary 現在會直接印出 state fallback、state blocks、
  response blocks、true KV blocks、supports-state-restore request count 和 state
  kinds；不用再只靠 request success rate 判斷 V8 是否真的 restore。
- `scripts/prepare_spotserve.sh` 在 stateful recovery / NIXL 相關 deploy set 前會檢查
  patched vLLM hooks 是否存在，並確認 stateful-recovery model 在 worker container
  內可見。

修正後的解讀：

```text
stateful_recovery request 成功
!= true KV restore 成功

true KV restore evidence 至少要看到：
state_restore_successes_total > 0
true_kv_restore_successes_total > 0
true_kv_restored_blocks_total > 0

若 router request metrics 是舊版或沒有帶 state-restored blocks，
vLLM/NIXL response evidence 仍可證明 attach 成功：
response_kv_restore_successes > 0
response_kv_restore_restored_blocks > 0
```

### 5. Re-parallelization 是 heuristic + actor recreate，不是論文完整 controller（部分修正）

`sllm/spot/reparallelization.py` 的 candidate selection 主要依照：

```text
GPU utilization
data parallel replica count
target replica GPU shape distance
unused GPU count
```

這是合理的 control-plane prototype，但 SpotServe 論文中的 parallelization
controller 會考慮 throughput、latency、monetary cost、batch size `B`、workload
變化等 trade-off。現有版本也主要透過 `VllmDeploymentAdapter` 建新 vLLM actors、
ready 後切流、drain 舊 actors，不是 runtime 內部原地重分片。

影響：

```text
可以說：capacity event 後會重新選 parallel plan 並套用到 vLLM actors。
不能說：已完整重現 SpotServe parallelization controller。
```

建議補強：

- 在 planner 中加入 workload arrival rate、batch size、latency estimate、
  model load time、migration cost。
- benchmark 至少要有多 worker node，才能顯示 preemption 後 replan 的服務能力改善。
- 分開呈現 replan-window startup cost 與 post-replan steady-state latency。

已開始修正：

- `sllm/spot/reparallelization.py` 加入 optional workload/cost-aware score。
  預設關閉；只有 `enable_workload_cost_model=true` 時，candidate 才會把
  arrival rate、batch size、latency estimate、model load time、migration cost、
  queue penalty 放進 score。這避免舊 V6 heuristic 在未設定時被偷偷改掉。
- `sllm/routers/roundrobin_router.py` 會收集最近 request arrival / latency，
  replan 時把 `runtime_workload` 傳給 planner，也會記錄 actor recreate/apply 的
  `execution_duration_ms`。
- `sllm/spot/metrics.py` 與 `scripts/analyze_spotserve_benchmark.py` 已輸出
  `replanning_avg_execution_duration_ms`,
  `replanning_avg_selected_replan_window_cost_ms`,
  `replanning_avg_selected_load_time_estimate_ms`,
  `replanning_avg_selected_migration_cost_estimate_ms`,
  `replanning_cross_node_targets`,
  `replanning_multi_worker_targets` 等欄位。
- `benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml`
  的 comparison 會自動帶出 replan-window / post-replan phase latency，以及上述
  replan cost/control-plane 指標。
- 新增 `benchmarks/spotserve/benchmark_matrix_reparallelization_multi_worker_performance.yaml`，
  搭配不含 `synthetic_worker_nodes` 的 V6 configs，讓 planner 從 scheduler/Ray
  的 runtime worker snapshot 做 target selection。summary 會用
  `replanning_max_runtime_worker_node_count` 確認 planner 是否真的看到多個 runtime
  workers。
- `scripts/prepare_spotserve.sh --deploy-set reparallelization-multi-worker-performance`
  會啟動 `sllm_worker_0` / `sllm_worker_1`，並要求至少 2 個 Ray `worker_node`
  resources 後才進入 benchmark。
- V6 performance traces 改用 `instance_selector=ready`，避免 hardcode synthetic
  node id 導致 spot event 沒有打到真正 live instance。
- V6 vLLM configs 加上 `count_preempting_toward_capacity=true`，避免一般
  autoscaler 在 preempting actor 尚未釋放 Ray/GPU resource 時，額外建立
  pending replacement actor；actor recreate 應由 replan controller 控制。
- router bulk shutdown 現在會對 inference instances 呼叫 scheduler
  `deallocate_resource`，避免 benchmark baseline 的 preempting actor 在
  `delete_after_run` 後仍佔用 Ray/GPU capacity，導致下一個 applied run 卡在
  pending actor。

仍未完成：

- 這不是完整 SpotServe optimizer，也還沒有 monetary cost model。
- 新增的 multi-worker matrix 可以驗證多個 runtime worker container，但如果兩個
  worker 都在同一台 host 上，仍不能 claim physical cross-node validation；真正強
  claim 需要不同 host / 不同 failure domain。
- 目前是 actor recreate / ready 後切流，不是 runtime 內部 in-place repartition。

這三件事的意思如下。

第一，planner 不應只看「GPU 塞不塞得下」，也要看「跑起來值不值得」。
目前的 V6 planner 比較像 capacity-aware heuristic：GPU 夠不夠、shape 差多少、
replica 數怎麼變。但完整 SpotServe controller 需要把 workload 和 serving cost
也納入：

```text
arrival rate
→ 請求進來的速度。流量高時，重建 actor 的空窗成本會被放大。

batch size
→ 實際 batching 能力。某個 parallel plan 可能 GPU utilization 高，
  但 batch shape 不適合目前 workload。

latency estimate
→ 預估不同 plan 下的 request latency / tail latency。
  不能只選 GPU shape 最接近的 plan。

load time
→ 新 actor 載入模型、建立 vLLM engine、warmup 到 ready 的時間。
  V6 最大成本通常就在這段。

migration cost
→ 切流、drain 舊 actor、request replay / state restore 的成本。
  如果搬過去的成本比留在原 plan 還高，就不一定值得 replan。
```

換句話說，未來 planner 的 objective 應該從：

```text
Can this plan fit available GPUs?
```

提升成：

```text
Is this plan worth switching to under the current workload?
```

第二，benchmark 需要真正的多 worker node 才能展示 preemption 下的
re-parallelization 效果。目前 single-host / same-host 實驗可以驗證控制流程，
但研究 claim 比較弱。更有說服力的情境是：

```text
Node A 發生 preemption
↓
原本在 Node A 上的 capacity 消失
↓
controller 重新選 parallel plan
↓
new vLLM actors 被建立到 Node B / Node C
↓
ready 後切流並恢復服務能力
```

這樣才能展示 V6 的價值是「capacity event 後重新部署到剩餘 worker nodes」，
而不只是同一台 host 裡重建 actor。

第三，latency 要拆成兩段呈現：

```text
replan-window latency
→ preemption 剛發生、舊 actor drain、planner 選新 plan、
  新 actor 建立 / load model / ready 的期間。

post-replan steady-state latency
→ 新部署穩定後，正常 serving 的 latency。
```

這個拆分很重要，因為 V6 的主要成本通常不是新 plan 穩定後的 serving latency，
而是 replan window 內的 startup / load / 切流成本。如果只看整體 p95，會把
「重建期間的巨大延遲」和「穩定後的服務品質」混在一起，導致很難判斷
re-parallelization 到底改善了哪一段、傷害了哪一段。

目前較安全的 claim 是：

```text
V6 已具備 capacity event 後重新選 parallel plan、
重建 vLLM actors、ready 後切流 / drain 舊 actors 的 control-plane prototype。

V6 planner 已開始納入 workload/cost-aware score hook，
可以量測 replan-window startup cost、post-replan steady-state latency、
以及 synthetic/runtime worker placement signal。

但它仍不是完整 SpotServe optimizer，monetary cost、physical cross-node
validation 與 runtime 內部 in-place repartition 仍未完成，因此不能宣稱已完整重現
SpotServe parallelization controller。
```

### 6. Preemptible instance 目前是 trace replay，不是 cloud provider integration

`sllm/spot/trace_reader.py` 支援 `add/remove/preempt/recover/dead`，而
`sllm/spot/preemption_simulator.py` 依照 JSONL 時間播放事件。這適合研究與
benchmark，但不是 AWS/GCP/Azure 的真實 spot/preemptible notice integration。

另外，`preempt` 只會把 instance/node 標成 preempting，停止接新 request；
真正移除或死亡仍需 trace 裡有 `dead` / `remove` 事件，或 backend request 自己回
`preempted`。

影響：

```text
目前沒有強制模擬 cloud grace period deadline。
如果 trace 沒有 dead/remove，preempting instance 可能只是停止接新流量。
```

建議補強：

- trace event 加上 `grace_period_s`。
- simulator 在 `preempt` 後自動排程 `dead`，除非中間收到 `recover`。
- metrics 記錄 notice time、deadline、state export time、restore finish time。

### 7. `MigrationRouter` 是舊路徑，可能已經不適合 SpotServe flow

`sllm/controller.py` 如果 `enable_migration=true`，會使用
`sllm/routers/migration_router.py`。但這個 class 看起來仍沿用舊欄位，例如
`ready_instances`，而目前 `RoundRobinRouter` 使用的是
`ready_inference_instances`。

SpotServe 新實作主要集中在 `RoundRobinRouter`：

```text
enable_reparallelization
enable_context_migration
recovery_policy=stateful_recovery
```

影響：

```text
若啟動時打開 enable_migration，可能繞過 SpotServe 新路徑，甚至踩到舊欄位錯誤。
```

建議補強：

- 明確標記 `MigrationRouter` deprecated，或移除 controller 對它的自動切換。
- 若仍要保留，需把它更新到 `ready_inference_instances` 與新 recovery/stateful path。
- SpotServe 範例文件應提醒不要用 `enable_migration=true` 啟用舊 router。

### 8. 現有實驗多為 single-host / same-host simulation

文件中已有記錄：Tiny/Qwen MoE 的 NIXL restore smoke、cross-container 模擬、
four-container fleet churn 都有價值，但多數仍在同一台主機上跑。這可以證明控制流程、
container packaging、NIXL same-host 或 network namespace path，但還不是真正
physical cross-node GPU cluster validation。

影響：

```text
不能宣稱已完成 physical cross-node KV restore。
can_restore_cross_node 應維持 false，除非真的有跨機器正向結果。
```

建議補強：

- 用兩台以上 GPU nodes 跑 source/target NIXL restore。
- 報告中分開寫：
  - same-process / same-node
  - same-host multi-container
  - physical cross-node

### 9. 成功率指標可能掩蓋 fallback

目前 recovery path 設計有 fallback，這是好的；但如果只看 request success rate，
很容易把「重試成功」誤讀成「KV restore 成功」。

影響：

```text
success_rate=100% 可能只是 naive retry 或 token replay 成功。
```

建議補強：

每份 SpotServe/MoE 報告至少列：

```text
failed_attempts_total
retry_count_total
recovered_tokens_total
recovery_fallback_count
state_restore_attempts_total
state_restore_successes_total
state_restore_fallback_count
state_restored_tokens_total
restored_blocks
context_migration_reusable_context_blocks
```

### 10. Risk-aware scheduling 是額外功能，不屬於原本三核心

`sllm/spot/risk_aware_scheduling.py` 和 `sllm/schedulers/fcfs_scheduler.py`
提供 spot risk / remaining lifetime / loading cost ranking。這是合理延伸，但不是
使用者一開始說的 SpotServe 三大核心。

影響：

```text
不要把 V9 risk-aware scheduling 混成 SpotServe 三核心完成度證據。
```

建議補強：

- 在報告中獨立成「additional scheduling extension」。
- 若要宣稱 production risk-aware scheduling，需要真 cloud provider 或 predictor
  的 live data validation。

## 建議的最小修正清單

1. 文件措辭修正：把「SpotServe on MoE」改成
   「SpotServe-style control plane for vLLM MoE」。
2. metric 修正：把 planning、prefix warmup、true KV restore 三種結果分開。
3. EP 修正：把 ServerlessLLM `replica_count` 和 vLLM
   `data_parallel_size` 分開；只宣稱 derived `effective_expert_parallel_size`。
4. router 修正：避免 SpotServe flow 使用舊 `MigrationRouter`。
5. simulator 修正：`preempt` 加 grace-period deadline 與自動 `dead`。
6. 實驗補強：至少補一組 physical cross-node NIXL restore，或明確標示目前是 same-host。

## 最安全的對外說法

```text
本專案將 SpotServe 的 re-parallelization、context migration planning、
stateful recovery 三個核心流程實作在 ServerlessLLM/vLLM 控制平面上，並驗證
vLLM MoE backend 可以接入這些流程。現階段 MoE 仍主要以 vLLM black-box 方式
服務；expert-aware placement/migration/recovery 與 physical cross-node KV
restore 尚未完整完成。
```
