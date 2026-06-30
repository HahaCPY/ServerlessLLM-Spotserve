# Revised CPY Plan: ServerlessLLM → SpotServe-style MoE Recovery System

## Overall Goal

最終目標是設計一個：

> Spot-aware recovery mechanism for MoE serving on Spot GPU clusters.

也就是結合：

* ServerlessLLM 的 control plane、fast loading、multi-worker serving 能力
* SpotServe-style scheduling / recovery 思想
* vLLM MoE backend

來降低 Spot GPU 被回收時對 MoE serving 的影響。

但要注意：

第一階段不是直接做完整 SpotServe 論文的三大核心，而是先把 ServerlessLLM 的 control plane 改造成可以支援 preemption、retry、replay、metrics 和 benchmark 的基礎系統。

---

# Version 1: Control-plane Prototype — Done

> 不修改。此版本已完成。

Version 1 已完成 SpotServe-style control plane 的基本骨架：

* worker state
* preemption-aware routing
* synthetic trace reader / simulator
* `none` / `naive_retry` / `generated_token_replay`
* JSONL metrics
* dummy backend benchmark
* benchmark report workflow

Version 1 的重點是確認整條流程可以跑通：

```text
Trace
→ Controller
→ Router
→ Worker State
→ Retry / Replay Policy
→ Metrics
→ Report
```

限制：

* `recover` 當時只被 parsed，沒有真正 dispatch。
* generated-token replay 是 best-effort，不是真正 KV cache recovery。
* 沒有 dynamic reparallelization。
* 沒有 low-cost context migration。
* 沒有 true stateful inference recovery。

---

# Version 2: Recover Dispatch + Node Health

## Goal

Version 2 補上 Version 1 的 control-plane 缺口，讓 synthetic trace 裡的 `recover` 事件真的影響系統狀態，並讓 scheduler 避開不健康 node。

這一版仍然不做真正 SpotServe 三大核心，只是讓 preempt / recover / dead 的生命週期完整。

## In Scope

* `recover` trace event dispatch
* controller-level `handle_recover`
* router-level `handle_recover`
* `PREEMPTING` instance recover 回 `READY`
* `DEAD` instance 不可被 recover 復活
* scheduler 新增 node health state
* FCFS / storage-aware scheduler 避開 `PREEMPTING` / `DEAD` node
* `sllm replay-trace` CLI
* repeatable setup script

## Out Of Scope

* true cloud spot provider
* spot risk ranking
* dynamic reparallelization
* KV cache migration
* expert-aware scheduling

## Definition of Done

* trace 中的 `recover` 不再只是 parsed，而是 dispatch 到 controller。
* Scheduler 知道 node recovered / preempting / dead。
* Router 可以把 matching `PREEMPTING` instance 恢復成 `READY`。
* `DEAD` instance 不會被 recover 復活。
* `sllm replay-trace` 可以從 CLI 執行。

---

# Version 3: Recovery Correctness Validation

## Goal

Version 3 的目標是驗證 retry / generated-token replay 真的有被觸發，而不是只看到所有 request 都成功。

這一版要回答：

> 當 request 真的在 generation 中途失敗時，naive retry 和 generated-token replay 是否真的被觸發？

## Why Needed

如果 benchmark 結果顯示：

```text
none: 46/46
naive_retry: 46/46
token_replay: 46/46
```

這只能證明 control-plane 沒壞，不能證明 recovery policy 有效果。

Version 3 要讓 benchmark 明確產生：

```text
failed_attempts > 0
retry_count > 0
recovered_tokens > 0
```

## In Scope

* forced backend failure
* forced mid-generation preemption
* dummy backend replay correctness
* transformers backend replay correctness
* request-level recovery metrics validation
* recovered token count validation
* fallback path validation

## Out Of Scope

* vLLM MoE
* true KV cache migration
* dynamic reparallelization
* low-cost context migration
* real cloud spot provider

## Tasks

### 1. Add forced failure workload

新增 workload，讓 request 一定會在指定 token 或指定時間失敗：

```text
request starts
→ backend generates partial tokens
→ worker preempted / dead
→ router retries or replays
```

### 2. Strengthen dummy backend

Dummy backend 需要支援：

* partial generation
* current token reporting
* forced failure after N tokens
* resume from replayed tokens

### 3. Validate transformers backend

Dummy backend 通過後，用 transformers small model 驗證 generated-token replay。

Transformers backend 比 vLLM 更容易觀察 token 狀態，適合做 correctness validation。

### 4. Add recovery correctness report

Report 要明確區分：

* policy enabled
* policy actually triggered
* replay succeeded
* replay fallback to retry
* replay not needed

## Deliverables

* `benchmark_matrix_recovery_correctness.yaml`
* forced-failure workload
* dummy replay correctness report
* transformers replay smoke report

## Definition of Done

* naive retry benchmark 中能看到 `retry_count > 0`
* generated-token replay benchmark 中能看到 `recovered_tokens > 0`
* fallback case 可以被記錄
* dummy backend correctness test 通過
* transformers backend replay smoke test 通過

---

# Version 4: vLLM Dense Black-box Integration

## Goal

Version 4 把 recovery pipeline 從 dummy / transformers 推進到 vLLM dense model。

這一版仍然不碰 MoE，也不碰 vLLM internals。

Version 4 要回答：

> 把 backend 換成 vLLM dense model 後，SpotServe-style router / retry / replay pipeline 是否仍能跑通？

## In Scope

* vLLM dense model
* single model
* multiple replicas
* synthetic trace
* retry policy
* best-effort generated-token replay
* vLLM capability detection
* vLLM dense benchmark report

## Out Of Scope

* MoE model
* expert parallel
* vLLM scheduler internals
* PagedAttention modification
* true KV cache migration

## Tasks

### 1. Add vLLM dense benchmark config

新增：

```text
examples/spotserve/config-vllm-dense-none.json
examples/spotserve/config-vllm-dense-naive-retry.json
examples/spotserve/config-vllm-dense-token-replay.json
```

建議先用小模型，例如：

```text
Qwen/Qwen3-0.6B
```

### 2. Add vLLM dense benchmark matrix

新增：

```text
benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml
```

至少包含：

* no preemption
* preemption + no retry
* preemption + naive retry
* preemption + generated-token replay

### 3. Validate vLLM replay behavior

vLLM path 要保守解讀：

* 如果能拿到 current tokens，記錄 `recovered_tokens`
* 如果不能 resume，fallback naive retry
* 不宣稱 true KV recovery

## Deliverables

* vLLM dense config
* vLLM dense benchmark matrix
* vLLM dense report
* vLLM replay limitation notes

## Definition of Done

* vLLM dense model 可以透過 SLLM deploy
* synthetic trace 可以 replay
* preempting worker 不接新 request
* retry policy 可在 vLLM dense backend 下運作
* token replay policy 可運作或清楚 fallback
* report 清楚區分 triggered / not triggered / fallback

---

# Version 5: vLLM MoE Black-box Integration

## Goal

Version 5 接大鼻提供的 vLLM + MoE config。

這一版仍然維持 black-box boundary：

> CPY 不改 MoE expert dispatch，不改 vLLM MoE router，不改 CUDA kernel。

Version 5 要回答：

> ServerlessLLM + SpotServe control plane 能不能包住 vLLM MoE worker，並在 preemption trace 下完成 retry / recovery benchmark？

## Important Clarification

Version 5 會「使用 MoE」，但不會「修改 MoE」。

也就是：

```text
CPY control plane
→ Router / Retry / Replay / Metrics
→ vLLM MoE worker
```

但不碰：

```text
MoE expert dispatch
expert placement
expert migration
MoE CUDA kernel
vLLM MoE internal router
```

## In Scope

* vLLM MoE worker as black box
* 大鼻提供的 MoE config
* single MoE model
* multiple replicas
* synthetic trace
* retry / replay benchmark
* latency / throughput baseline

## Out Of Scope

* expert-aware scheduling
* expert placement
* expert migration
* MoE kernel optimization
* KV cache migration

## Tasks

### 1. Integrate 大鼻 config

大鼻提供：

* model name
* vLLM launch config
* TP / DP / EP setting
* `enable_expert_parallel` feasibility
* API smoke test
* baseline latency / throughput

CPY 接到：

```text
examples/spotserve/config-vllm-moe-*.json
```

### 2. Run smoke validation

先只確認：

```text
deploy
→ one request
→ response OK
```

### 3. Run trace benchmark

使用固定 workload / trace：

* steady-low
* steady-high
* burst

比較：

* none
* naive_retry
* generated_token_replay

### 4. Compare dense vs MoE

報告中新增：

```text
vLLM dense
vs
vLLM MoE
```

觀察：

* latency
* p95
* throughput
* retry overhead
* replay fallback rate

## Deliverables

* MoE black-box configs
* MoE smoke validation
* MoE trace benchmark report
* dense vs MoE comparison table

## Definition of Done

* vLLM MoE worker 可 deploy
* single MoE model + multiple replicas 可跑
* synthetic trace replay 可完成
* retry / replay benchmark 有 report
* 結果中不宣稱 expert-aware behavior，只宣稱 black-box worker recovery behavior

---

# Version 6: Dynamic Reparallelization Planner

## Goal

Version 6 開始對齊 SpotServe 第一個核心：

> Dynamic reparallelization

目標是在 GPU availability 改變時，根據目前可用 GPU 數量與 workload，選擇新的 parallel configuration。

這一版先做 planner，不一定馬上真的重建 vLLM worker。

## SpotServe Core Mapping

對應 SpotServe：

```text
Dynamic reparallelization
```

## In Scope

* cluster GPU availability tracking
* model parallel config representation
* candidate parallel config generation
* cost model / heuristic ranking
* synthetic replanning benchmark
* support dense / MoE config metadata

## Out Of Scope

* actual vLLM internal repartition
* live expert migration
* KV cache migration
* CUDA kernel modification

## Tasks

### 1. Define parallel configuration schema

例如：

```json
{
  "tp": 2,
  "dp": 1,
  "pp": 1,
  "ep": 2,
  "num_gpus": 4
}
```

Dense model 可以只用：

```text
TP / DP / PP
```

MoE model 可以加入：

```text
EP
```

### 2. Track available GPU count

當 trace 發生：

```text
preempt node
dead node
recover node
```

系統要知道目前還剩多少可用 GPU。

### 3. Generate candidate configs

例如原本：

```text
8 GPUs
TP=4, EP=2
```

Spot event 後只剩：

```text
4 GPUs
```

planner 產生：

```text
TP=2, EP=2
TP=4, EP=1
TP=2, DP=2
```

### 4. Rank candidate configs

先用簡單 heuristic：

```text
score = estimated_latency + reload_cost + risk_penalty
```

Version 6 只需要能選出一個 best config。

### 5. Emit replanning decision

不要急著真的重啟 worker。

先記錄：

```json
{
  "event": "reparallelization_decision",
  "old_config": {"tp": 4, "ep": 2},
  "new_config": {"tp": 2, "ep": 2},
  "reason": "node_preempted"
}
```

## Deliverables

* parallel config schema
* dynamic repartition planner
* replanning metrics
* synthetic replanning benchmark

## Definition of Done

* GPU availability 改變時，planner 能產生新的 parallel config
* report 能顯示 old config → new config
* dense / MoE 都能以 metadata 形式被 planner 處理
* 不需要真的修改 vLLM internals

---

# Version 7: Spot-risk-aware Scheduling

## Goal

Version 7 在 Version 6 的基礎上加上 risk-aware scheduling。

Version 2 只做 node health filter：

```text
避開 PREEMPTING / DEAD node
```

Version 7 要做：

```text
根據 spot risk / remaining lifetime / loading cost 做 placement ranking
```

## In Scope

* synthetic spot risk score
* scheduler ranking
* storage-aware + spot-aware combined score
* on-demand fallback placeholder
* risk-aware benchmark

## Out Of Scope

* real cloud provider integration
* real spot market prediction
* production autoscaling
* expert-aware scheduling

## Tasks

### 1. Add spot risk model

從 trace 或 config 產生 node risk：

```json
{
  "node_id": "worker0",
  "spot_risk": 0.8,
  "expected_lifetime_s": 120
}
```

### 2. Extend scheduler ranking

FCFS scheduler：

```text
READY node first
lower risk first
then FCFS
```

Storage-aware scheduler：

```text
score = loading_cost + alpha * spot_risk
```

### 3. Add scheduler policy config

例如：

```json
{
  "scheduler_config": {
    "spot_aware": true,
    "risk_weight": 0.5,
    "on_demand_fallback": false
  }
}
```

### 4. Benchmark scheduler impact

比較：

* node-health-only scheduler
* spot-risk-aware scheduler
* storage-aware-only scheduler
* storage-aware + spot-aware scheduler

## Deliverables

* scheduler risk config
* FCFS spot-risk ranking
* storage-aware spot-risk ranking
* scheduler benchmark report

## Definition of Done

* scheduler 不只避開 dead/preempting node，也能根據 risk 排序
* benchmark 能比較 health-only vs risk-aware
* report 能顯示 preemption count、retry count、p95 latency、success rate 是否改善

---

# Version 8: Low-cost Context / Expert Mapping

## Goal

Version 8 對齊 SpotServe 第二個核心：

> Low-cost context migration

目標是當 parallel configuration 改變時，選擇一個能最大化 context / expert / worker reuse 的 mapping。

這一版先做 mapping algorithm，不一定真正搬 KV cache。

## SpotServe Core Mapping

對應 SpotServe：

```text
Low-cost context migration
```

## In Scope

* old placement representation
* new placement representation
* reuse cost matrix
* bipartite graph matching
* KM / Hungarian algorithm
* mapping decision metrics
* expert-level mapping metadata

## Out Of Scope

* actual KV cache transfer
* actual expert weight migration
* CUDA kernel modification
* vLLM internal expert dispatch modification

## Tasks

### 1. Define old/new mapping

例如：

```text
Old:
expert_0 → gpu_0
expert_1 → gpu_1
expert_2 → gpu_2

New candidate:
expert_0 → gpu_1
expert_1 → gpu_0
expert_2 → gpu_3
```

### 2. Build cost matrix

Cost 可以包含：

* context reuse benefit
* expert weight already loaded
* checkpoint loading cost
* node health
* node spot risk

例如：

```text
cost(expert_i, gpu_j)
```

越低代表越適合。

### 3. Run matching algorithm

使用 Hungarian / KM algorithm 找最小 cost mapping。

### 4. Emit mapping decision

記錄：

```json
{
  "event": "context_mapping_decision",
  "old_mapping": "...",
  "new_mapping": "...",
  "reused_context": 12,
  "migration_cost": 3.4
}
```

## Deliverables

* mapping schema
* cost matrix builder
* KM / Hungarian matching implementation
* mapping decision metrics
* synthetic mapping benchmark

## Definition of Done

* 給定 old mapping + new candidate workers，可以輸出 minimum-cost mapping
* report 能顯示 reuse ratio / migration cost
* 不需要真的搬 KV cache
* 不需要真的改 vLLM expert dispatch

---

# Version 9: True Stateful Inference Recovery

## Goal

Version 9 對齊 SpotServe 第三個核心：

> Stateful inference recovery

目標是從 generated-token replay 進一步變成真正保存推論狀態，讓被中斷的 request 可以從 KV cache / request state 繼續，而不是從 prompt 重新計算。

## SpotServe Core Mapping

對應 SpotServe：

```text
Stateful inference recovery
```

## In Scope

* request state checkpoint
* token-level progress tracking
* KV cache handle / metadata tracking
* backend capability interface
* vLLM KV cache export / import investigation
* fallback to generated-token replay
* correctness benchmark

## Out Of Scope

* full production KV migration across arbitrary vLLM versions
* CUDA kernel modification unless absolutely necessary
* expert-aware KV placement optimization

## Tasks

### 1. Define inference state interface

新增 backend-neutral interface：

```python
get_inference_state(request_id)
restore_inference_state(request_id, state)
```

State 可以包含：

* generated tokens
* current position
* sampling state
* KV cache handle / block metadata
* model config
* backend-specific metadata

### 2. Implement dummy true-state recovery

先在 dummy backend 做真正 state restore，確保 router / controller 邏輯正確。

### 3. Investigate vLLM KV state

對 vLLM 先做 feasibility study：

* KV cache block metadata 在哪裡
* request state 在哪裡
* PagedAttention block table 如何表示
* 是否能安全 export / import
* 是否需要 patch vLLM

### 4. Add vLLM experimental backend path

如果可行，做 experimental path：

```text
worker A exports KV metadata
→ worker B imports KV metadata
→ continue generation
```

如果不可行，明確 fallback：

```text
generated-token replay
```

### 5. Benchmark against replay

比較：

* no recovery
* naive retry
* generated-token replay
* true stateful recovery

指標：

* recovered tokens
* recomputation cost
* p95 latency
* correctness
* fallback count

## Deliverables

* inference state interface
* dummy stateful recovery
* vLLM KV feasibility notes
* experimental vLLM stateful recovery path
* replay vs stateful recovery benchmark

## Definition of Done

* dummy backend 可以做到真正 state restore
* vLLM path 至少完成 feasibility report
* 若 vLLM path 可行，能展示 single request stateful recovery
* 若不可行，明確說明限制並 fallback generated-token replay

---

# Version 10: Expert-aware Recovery for MoE

## Goal

Version 10 才真正進入 MoE expert-level recovery。

這一版不再只是把 MoE 當黑盒，而是開始考慮 expert placement、expert hotness、expert migration 和 expert-aware scheduling。

## In Scope

* expert-level metadata
* expert hotness tracking
* expert placement decision
* expert-aware recovery policy
* MoE-specific recovery benchmark

## Out Of Scope

* CUDA kernel optimization
* full production expert migration
* multi-region cloud scheduling

## Tasks

### 1. Track expert usage

記錄：

* request → expert route
* expert load
* hot experts
* expert-to-worker placement

### 2. Expert-aware placement

當 spot preemption 發生時，不只看 worker：

```text
which worker is alive?
```

也看：

```text
which experts are affected?
which experts are hot?
which experts should be recovered first?
```

### 3. Expert-aware recovery policy

例如：

* hot expert 優先 restore
* cold expert 延後 restore
* on-demand worker 優先放 hot expert
* spot worker 放 cold expert

### 4. Compare with black-box MoE recovery

比較：

* MoE black-box retry / replay
* expert-aware recovery

指標：

* p95 latency
* throughput
* expert load imbalance
* recovery time
* request failure rate

## Deliverables

* expert metadata collection
* expert-aware scheduling prototype
* expert recovery benchmark
* final MoE recovery comparison

## Definition of Done

* 系統能觀察 expert-level load
* preemption 後能做 expert-aware recovery decision
* benchmark 能顯示 expert-aware policy 相比 black-box policy 是否改善

---

# Version 11: Real Cloud Spot Integration

## Goal

Version 11 才接真實 cloud spot provider。

這一版把 synthetic trace event 換成真實 provider event。

## In Scope

* cloud spot metadata watcher
* provider adapter
* preemption notice to SpotEvent
* optional replacement node registration
* real spot smoke test

## Out Of Scope

* production-grade cloud autoscaler
* complex price optimizer
* multi-region scheduling

## Tasks

### 1. Add provider abstraction

新增：

```text
sllm/spot/providers/
    base.py
    aws.py
    gcp.py
```

先可以只做一個 provider。

### 2. Convert provider event to SpotEvent

例如：

```text
AWS spot interruption notice
→ SpotEvent(event="preempt", node_id=...)
```

### 3. Reuse existing control plane

不要重寫 recovery。

真實 event 仍然走：

```text
provider watcher
→ controller.handle_preemption()
→ router.handle_preemption()
```

## Deliverables

* provider watcher
* real spot smoke validation
* cloud integration notes

## Definition of Done

* 真實 spot notice 可以觸發 controller
* preempting worker 不接新 request
* request 可 retry / replay 到其他 replica
* synthetic trace pipeline 和 real provider pipeline 共用同一套 handler

---

# Version Mapping to SpotServe Core Ideas

| SpotServe Core Idea         | Corresponding Version | Notes                                           |
| --------------------------- | --------------------: | ----------------------------------------------- |
| Dynamic reparallelization   |                    V6 | 先做 planner，再考慮實際 worker rebuild                 |
| Low-cost context migration  |                    V8 | 先做 mapping / KM algorithm，不急著搬 KV               |
| Stateful inference recovery |                    V9 | 從 generated-token replay 升級到 true state restore |
| Expert-aware MoE recovery   |                   V10 | 這是你們自己的 MoE extension，不是原始 control-plane v1     |

---

# Updated Milestone Table

| Version | Name                              | Main Goal                                              | Status         |
| ------- | --------------------------------- | ------------------------------------------------------ | -------------- |
| V1      | Control-plane prototype           | worker state, trace replay, retry, replay, metrics     | Done           |
| V2      | Recover + node health             | complete trace lifecycle and scheduler health filter   | Done / Current |
| V3      | Recovery correctness              | prove retry/replay is actually triggered               | Next           |
| V4      | vLLM dense black-box              | validate vLLM dense backend under trace                | Planned        |
| V5      | vLLM MoE black-box                | integrate 大鼻 MoE worker without touching MoE internals | Planned        |
| V6      | Dynamic reparallelization planner | choose new parallel config under GPU loss              | Planned        |
| V7      | Spot-risk scheduler               | rank placement by risk and loading cost                | Planned        |
| V8      | Low-cost context mapping          | KM/Hungarian mapping for low migration cost            | Planned        |
| V9      | True stateful recovery            | investigate and prototype KV/state recovery            | Research       |
| V10     | Expert-aware MoE recovery         | expert-level recovery policy                           | Research       |
| V11     | Real cloud spot                   | replace synthetic trace with real provider events      | Future         |

---

# Updated Research Milestone

## First Research Milestone

完成：

* V1 control-plane prototype
* V2 recover + node health
* V3 recovery correctness
* V4 vLLM dense black-box
* V5 vLLM MoE black-box

這一階段可以宣稱：

> We built a SpotServe-style control-plane prototype on ServerlessLLM and evaluated retry / generated-token replay policies under synthetic preemption traces for dummy, transformers, vLLM dense, and vLLM MoE black-box backends.

不能宣稱：

* dynamic reparallelization
* low-cost context migration
* true KV cache recovery
* expert-aware recovery

---

## Second Research Milestone

完成：

* V6 dynamic reparallelization planner
* V7 spot-risk scheduler
* V8 low-cost context mapping

這一階段可以宣稱：

> We extend the prototype with SpotServe-style repartitioning and low-cost mapping decisions to reduce recovery cost under changing GPU availability.

---

## Third Research Milestone

完成：

* V9 true stateful recovery
* V10 expert-aware MoE recovery

這一階段才可以宣稱：

> We implement expert-aware recovery and investigate true stateful inference recovery for MoE serving on spot GPU clusters.

---

# Core Principle

不要把三個層級混在一起：

## Current control-plane layer

```text
preempt
→ drain
→ retry
→ generated-token replay
→ metrics
```

## SpotServe algorithm layer

```text
dynamic reparallelization
→ low-cost context mapping
→ stateful inference recovery
```

## MoE expert layer

```text
expert placement
→ expert hotness
→ expert-aware recovery
```

Version 1 已經完成第一層的雛形。

後續版本要逐步往第二層和第三層推進，避免一開始就把 generated-token replay 誤寫成 true KV cache recovery，或把 MoE black-box integration 誤寫成 expert-aware scheduling。
