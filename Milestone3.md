# New Milestone 3 Plan: Parallel Work Split for CPY and 大鼻

## Goal

Milestone 3 的目標是共同完成 SpotServe core：

1. Dynamic Reparallelization
2. Low-cost Context Migration
3. Stateful Inference Recovery
4. Spot-risk-aware Scheduling

但為了避免 merge conflict，兩人的分工不直接用「功能」切，而是用「layer」切：

```text
CPY:
ServerlessLLM control plane / planner / scheduler / router / metrics

大鼻:
vLLM backend capability / MoE runtime metadata / config validation / execution contract
```

核心原則是：

> CPY 負責「決策與策略」，大鼻負責「提供 backend 能力與資訊」。
> CPY 不改 vLLM internals；大鼻不改 ServerlessLLM router / scheduler 主流程。

👉 換句話說：

> **大鼻不是完全不寫 code，而是「主要寫的是描述能力與提供資訊的 code」，而不是改核心 serving 邏輯。**

---

# Shared Interface First

Milestone 3 一開始先共同定義三個 interface，之後兩邊各自實作。

## 1. ParallelPlan

用來描述重新平行化後的 deployment plan。

```python
@dataclass
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

CPY 負責產生這個 plan。
大鼻負責確認這個 plan 能不能被 vLLM / MoE backend 執行。

---

## 2. BackendCapability

用來描述 backend 支援什麼。

```python
@dataclass
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

CPY 不需要知道 vLLM 內部怎麼做 TP / DP / EP。
CPY 只需要知道這個 backend 支援哪些 config。

👉 這裡就是大鼻的核心工作之一：
**把 vLLM 能做什麼「整理成資料」給 CPY 用。**

---

## 3. MigrationPlan

用來描述舊 worker 到新 worker 的 mapping。

```python
@dataclass
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

CPY 負責做 mapping decision。
大鼻負責提供哪些 context / state 是可重用的 metadata。

---

# Work Split

## Track A: CPY — Control-plane Planner

CPY 主要負責 ServerlessLLM 這層。

### CPY owns files

```text
sllm/controller.py
sllm/routers/roundrobin_router.py
sllm/schedulers/
sllm/spot/
benchmarks/spotserve/
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

### CPY responsibilities

1. Dynamic Reparallelization Planner

   * 根據 GPU availability 產生 ParallelPlan
   * 根據 spot event 決定是否 replan
   * 記錄 old config → new config

2. Low-cost Context Migration Planner

   * 建立 cost matrix
   * 做 Hungarian / KM matching
   * 輸出 MigrationPlan

3. Stateful Recovery Control Logic

   * 定義 request state machine
   * 決定 request 要 retry、replay，還是 stateful recover
   * 如果 backend 不支援 state restore，fallback generated-token replay

4. Spot-risk-aware Scheduling

   * node risk score
   * remaining lifetime
   * loading cost
   * scheduler ranking

5. Metrics and Benchmark

   * replanning metrics
   * migration cost metrics
   * recovery metrics
   * benchmark matrix
   * report

---

## Track B: 大鼻 — Backend Capability and Runtime Metadata

大鼻主要負責 **提供 backend 能力與資訊（但仍會寫輔助 code）**，而不是改 ServerlessLLM 主流程。

### 大鼻 owns files

```text
examples/spotserve/
sllm/backends/vllm_backend.py
sllm/backends/backend_utils.py
docs/vllm_moe_config.md
benchmarks/spotserve/configs/
```

如果需要新增檔案，建議新增在：

```text
sllm/backends/vllm_capability.py
sllm/backends/vllm_moe_capability.py
examples/spotserve/config-vllm-moe-*.json
```

---

### 大鼻 responsibilities（重點解釋）

👉 可以理解成三種工作：

### 1. 提供「能力描述」（最核心）

* 哪些 TP / DP / EP 可以用
* 最大 GPU 數
* 是否支援 state export / restore

👉 這些會變成 `BackendCapability`

---

### 2. 提供「runtime metadata」

例如：

* instance 現在用多少 GPU
* memory usage
* TP / DP / EP config
* node / instance id

👉 CPY 用這些資訊做 scheduler / planner

---

### 3. 做「輕量 code 支援」（不是改核心）

例如：

* capability provider（回傳 BackendCapability）
* config validation script
* metadata export function
* MoE config example

👉 這些是「讓 CPY 能用 backend」，不是改 vLLM 核心邏輯

---

### 4. Feasibility / 調查型工作

例如：

* vLLM 能不能做 KV restore
* MoE expert metadata 能不能拿

👉 這些會影響 CPY 的策略（fallback 或進階功能）

---

# Version-by-Version Parallel Plan

## V6: Dynamic Reparallelization Planner

### CPY

CPY 做 decision layer：

```text
spot event
    ↓
available GPU changes
    ↓
replanning planner
    ↓
ParallelPlan
```

### 大鼻

大鼻提供：

```text
哪些 TP / DP / EP config 可以用
```

👉 不負責「怎麼選」，只負責「哪些能選」

---

## V7: Low-cost Context Migration Planner

### CPY

做 mapping：

```text
cost matrix → matching → MigrationPlan
```

### 大鼻

提供：

```text
哪些 context 可以 reuse
有哪些 metadata 可以用
```

👉 不做 matching，只提供資訊

---

## V8: Stateful Inference Recovery

### CPY

決定：

```text
recover / replay / retry
```

### 大鼻

回答：

```text
能不能 restore state？
```

👉 如果不能：

```text
CPY fallback → token replay
```

---

## V9: Spot-risk-aware Scheduling

### CPY

做 scheduler：

* risk score
* ranking

### 大鼻

提供：

* model resource profile
* loading cost
* GPU usage

👉 CPY 用這些資訊做決策

---

# Conflict Avoidance Rules

## Rule 1: File ownership

CPY 不改：

```text
vLLM internals
MoE dispatch
CUDA kernel
```

大鼻不改：

```text
controller.py
router
scheduler policy
```

---

## Rule 2: Interface-first PR

先定義資料結構，再各自實作。

---

## Rule 3: Separate modules

CPY 寫 planner
大鼻寫 capability / metadata

---

## Rule 4: Benchmark 最後整合

避免一開始就互相卡住

---

# Final Division Summary

| Area                       | CPY                        | 大鼻                                     |
| -------------------------- | -------------------------- | -------------------------------------- |
| Dynamic Reparallelization  | Planner / decision         | supported config / backend capability  |
| Low-cost Context Migration | cost matrix / matching     | runtime metadata / reuse info          |
| Stateful Recovery          | recovery policy / fallback | vLLM KV feasibility / state capability |
| Spot-risk Scheduling       | scheduler ranking          | model resource profile                 |
| Benchmarks                 | benchmark runner / metrics | backend config / baseline              |

---

# Main Principle

```text
CPY:
決定系統要做什麼

大鼻:
告訴系統 backend 能做什麼
```