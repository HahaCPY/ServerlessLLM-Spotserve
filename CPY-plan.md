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

Version 5 integrates the vLLM MoE backend provided by the vLLM side while keeping it as a black-box service.

This version answers:

> Can the existing SpotServe-style control plane operate correctly when the backend is changed from a dense model to a MoE model?

The objective is integration rather than MoE optimization.

## In Scope

- vLLM MoE backend
- single MoE model
- multiple replicas
- synthetic SpotServe trace
- retry policy
- generated-token replay
- benchmark comparison

## Out Of Scope

- expert routing
- expert dispatch
- CUDA kernels
- expert-aware scheduling
- expert migration

## Tasks

### 1. Integrate vLLM MoE backend

Deploy a MoE model through ServerlessLLM using the configuration provided by the vLLM team.

### 2. Validate SpotServe pipeline

Verify:

```text
Spot Trace
→ Controller
→ Router
→ vLLM MoE Worker
→ Retry / Replay
→ Metrics
→ Report
```

### 3. Benchmark

Compare:

- dense model
- MoE model

under:

- none
- naive retry
- generated-token replay

## Deliverables

- vLLM MoE deployment
- MoE benchmark matrix
- Dense vs MoE report
- Smoke validation

## Definition of Done

- MoE model deploys successfully.
- Spot trace replays successfully.
- Retry / Replay work with MoE backend.
- Benchmark completes successfully.

---

# Version 6: Dynamic Reparallelization Planner

## Goal

Implement the first core idea of SpotServe:

> Dynamic Reparallelization

Generate a new deployment configuration after GPU resources change because of spot preemption.

## In Scope

- GPU availability tracking
- Parallel configuration schema
- Candidate configuration generation
- Replanning heuristic
- Replanning metrics
- Synthetic replanning benchmark

## Out Of Scope

- MoE optimization
- CUDA kernels
- KV cache migration

## Definition of Done

- Planner generates a new deployment configuration.
- Benchmark records replanning decisions.

---

# Version 7: Low-cost Context Migration Planner

## Goal

Implement the second core idea of SpotServe:

> Low-cost Context Migration

Compute the minimum-cost mapping between old workers and new workers without implementing true KV cache migration.

## In Scope

- Context metadata
- Cost matrix
- Worker mapping
- Hungarian / KM algorithm
- Migration metrics

## Out Of Scope

- True KV cache migration
- Expert migration
- vLLM internals

## Definition of Done

- Produce minimum-cost mapping.
- Report migration cost and reuse ratio.

---

# Version 8: Stateful Inference Recovery

## Goal

Implement the third core idea of SpotServe:

> Stateful Inference Recovery

Extend generated-token replay toward true inference-state recovery.

## In Scope

- Backend-independent state interface
- Dummy state recovery
- vLLM feasibility study
- Recovery benchmark

## Out Of Scope

- Production KV migration
- CUDA optimization

## Definition of Done

- Dummy backend restores inference state.
- vLLM feasibility documented.

---

# Version 9: Spot-risk-aware Scheduling

## Goal

Extend scheduling with spot awareness by considering:

- Spot risk
- Remaining lifetime
- Loading cost

instead of only filtering unhealthy nodes.

## In Scope

- Spot-risk model
- Risk-aware scheduler
- Scheduler benchmark

## Out Of Scope

- Real cloud provider
- Autoscaling
- MoE scheduling

## Definition of Done

- Compare health-only and risk-aware scheduling.

---

# Version 10: Expert-aware Recovery

## Goal

Extend SpotServe with MoE-specific recovery mechanisms.

## In Scope

- Expert metadata
- Expert hotness
- Expert-aware placement
- Expert-aware recovery benchmark

## Out Of Scope

- Production expert migration

## Definition of Done

- Compare black-box recovery and expert-aware recovery.

---

# Updated Research Milestones

## Milestone 1 (Completed)

- Version 1
- Version 2
- Version 3
- Version 4

## Milestone 2 (MoE Integration)

- Version 5

## Milestone 3 (SpotServe Core)

- Version 6
- Version 7
- Version 8
- Version 9

## Milestone 4 (Research Extension)

- Version 10
