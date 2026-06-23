# CPY Plan: ServerlessLLM -> SpotServe-style Controller (v1)

## Goal

CPY 負責回答：

> ServerlessLLM 要怎麼變成 SpotServe-style controller？

第一版目標是把 ServerlessLLM 的 control plane 改造成可以模擬 SpotServe-style 行為的系統：

- 不改 vLLM internals
- 不改 MoE router / expert dispatch
- 不碰 CUDA kernel
- 將 vLLM worker / deployment 視為 black-box inference service

重點放在：

- request routing
- worker lifecycle
- worker state
- spot preemption simulation
- preempting worker drain
- retry policy
- generated-token replay / recovery
- metrics and logging
- reproducible benchmark workflow
- visual benchmark report

第一版先追求整個系統跑通，而不是追求最完美的 spot-aware scheduling。

---

## First-version Scope

### In Scope

第一版要做：

- single model
  - 不需要 multi-model，會很複雜
- multiple replicas / instances
  - 當一個 worker 快被搶走或已經 dead，request 要能轉移到另一個還活著的 replica
- vLLM worker as black box
  - 因為是大鼻負責
- synthetic SpotServe trace
  - 用 synthetic trace 來模擬 spot instance
- preempting worker 不再接新 request
- prompt + generated tokens replay
  - 有點像低階版的 recovery，但不是真正的 KV cache 搬移
- Benchmark
  - naive retry baseline
  - generated-token replay best effort
  - baseline vs retry vs replay metrics
  - repeatable benchmark runner
  - visual report for later presentations and comparisons

### Out Of Scope

第一版先不要做：

- true cloud spot instance integration
- full multi-model scheduling optimization
- production Grafana / Prometheus dashboard
- expert-aware scheduling
- vLLM MoE internal routing
- expert dispatch
- CUDA kernel
- PagedAttention / KV cache internals
- true KV cache migration

重要原則：

> 先完成 SpotServe-style control plane，再考慮 MoE-aware 或 expert-aware scheduling。

---

## System Boundary

目前 ServerlessLLM request path：

```text
Client
  |
  v
FastAPI
  |
  v
Controller
  |
  v
Per-model Router
  |
  v
Backend Worker Actor
  |
  v
vLLM / Transformers / Dummy
  |
  v
GPU
```

CPY 第一版主要修改：

```text
Controller
Router
Scheduler
Worker state
Trace replay
Recovery policy
Metrics
```

CPY 第一版不修改：

```text
vLLM internals
MoE expert dispatch
CUDA kernel
```

---

## Split With 大鼻

### 大鼻: vLLM + MoE Backend

大鼻負責回答：

> vLLM 到底能不能在我們環境跑 MoE？要怎麼設定？

大鼻提供：

- 可執行 vLLM + MoE config
- MoE model candidate list
- worker 啟動方式
- API 測試方法
- baseline latency / throughput
- TP / DP / PP / EP / `enable_expert_parallel` 可行性分析

CPY 不需要知道：

- expert 怎麼切
- expert dispatch 怎麼做
- vLLM MoE CUDA kernel 怎麼跑

### CPY: SpotServe Control Plane

CPY 負責回答：

> ServerlessLLM 要怎麼變成 SpotServe-style controller？

CPY 提供：

- controller 如何呼叫 worker
- request routing
- worker lifecycle
- worker state machine
- spot trace replay
- preemption handling
- drain policy
- retry policy
- generated-token recovery policy
- metrics and comparison

大鼻不需要知道：

- spot trace 格式
- preemption state machine
- retry implementation
- recovery implementation

### Shared Contract

雙方只約定：

> vLLM worker / deployment 是一個可以接 request 的黑盒服務。

建議共同 request contract：

```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "request_id": "optional-stable-id"
}
```

建議共同 response contract：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

CPY 內部可以額外記錄：

```json
{
  "request_id": "...",
  "attempt": 1,
  "assigned_model": "...",
  "assigned_instance": "...",
  "assigned_node": "...",
  "worker_state": "ready",
  "preempted": false,
  "retry_count": 0,
  "recovered_tokens": 0,
  "latency_ms": 0
}
```

---

## Current Code Map

### Request Entry

- `sllm/cli/clic.py`
  - CLI entrypoint。
  - 現有 commands:
    - `sllm start`
    - `sllm deploy`
    - `sllm delete`
    - `sllm status`
  - 第一版可以先不加 CLI，trace simulator 可以先用 `python -m ...` 跑。
  - 第二版再加 `sllm replay-trace` 或 `sllm simulate-preemption`。

- `sllm/cli/_cli_utils.py`
  - `start_server()` 啟動 Ray、FastAPI、controller actor。
  - `deploy_model()` 送 `/register` 到 head server。
  - `delete_model()` 送 `/delete`。
  - `show_status()` 查 `/v1/models`。

- `sllm/app_lib.py`
  - FastAPI API layer。
  - `/register` 呼叫 controller 的 `register()`。
  - `/v1/chat/completions` 呼叫 per-model router 的 `inference(..., "generate")`。
  - `/v1/embeddings` 呼叫 per-model router 的 `inference(..., "encode")`。
  - 第一版通常不用大改。
  - 可選：加 debug endpoint，例如 `/admin/preempt`，方便手動觸發 preemption。

### Controller / Scheduler / Router

- `sllm/controller.py`
  - 系統大腦。
  - `start()` 建立 `StoreManager`、scheduler、router class。
  - `register()` 下載/register model，建立 per-model router actor。
  - CPY 要在這裡加 preemption event handling:
    - node-level preemption
    - instance-level preemption
    - call router hook

- `sllm/store_manager.py`
  - 管理每個 worker node 上的 `sllm-store` server。
  - 知道 model 在哪些 node、有沒有 loaded to host memory。
  - 第一版可以先不動。
  - 第二版如果要考慮 checkpoint/cache locality，再把 spot risk 和 storage-aware scheduling 合併。

- `sllm/schedulers/fcfs_scheduler.py`
  - 最簡單的 node allocation。
  - 第一版先不急著改 scheduler；先讓 router 不把 request 丟給 `PREEMPTING` / `DEAD` instance。
  - 第二版再從這裡開始做 spot-aware allocation：
    - 不要 allocate 到 dead node
    - 不要 allocate 到 preempting node
    - optional: spot/on-demand node preference

- `sllm/schedulers/storage_aware_scheduler.py`
  - 已經會看 model location、store info、hardware info 估 loading latency。
  - 第二版可加：
    - spot risk
    - preemption probability
    - on-demand fallback
    - checkpoint loading time + preemption risk combined score

- `sllm/routers/roundrobin_router.py`
  - 第一版最重要的檔案。
  - 每個 model 一個 router actor。
  - 維護:
    - `starting_inference_instances`
    - `ready_inference_instances`
    - `deleting_inference_instances`
    - FT instance pools
  - `_auto_scaler_loop()` 決定 instance 數量。
  - `_load_balancer_loop()` 把 request 分配給 ready instance。
  - `_start_instance()` 透過 scheduler allocate node，再啟動 backend actor。
  - CPY 的 worker state、preempting drain、naive retry、generated-token replay 主要落在這裡。

- `sllm/routers/migration_router.py`
  - 目前像 live migration prototype。
  - 注意：它使用 `ready_instances`，但 `RoundRobinRouter` 現在是 `ready_inference_instances`。
  - 第一版只參考 concept，不建議直接基於它改。

### Backend Worker

- `sllm/inference_instance.py`
  - `start_instance()` 依 backend name 建立 Ray backend actor。
  - 支援:
    - `vllm`
    - `transformers`
    - `dummy`

- `sllm/backends/vllm_backend.py`
  - vLLM backend 黑盒 worker。
  - `generate()` 包 `AsyncLLMEngine.generate()`。
  - 目前已有 `get_current_tokens()`。
  - 目前已有 `resume_kv_cache()`，但這不等於完整 true KV recovery。
  - 第一版優先使用既有 `get_current_tokens()`；必要時才加 best-effort token replay helper。

- `sllm/backends/transformers_backend.py`
  - Transformers backend。
  - 有比較明確的 `InferenceStatus`。
  - 有 `get_current_tokens()` 和 `resume_generate()`。
  - generated-token replay 可以先用這個 backend 驗證，再搬到 vLLM。

- `sllm/backends/dummy_backend.py`
  - 適合做 controller/router unit test，不依賴 GPU。
  - 第一版 preemption/retry 先用 dummy backend 測通最穩。

---

## Request Flow To Understand

### Deploy Path

```text
sllm deploy
  |
  v
sllm/cli/clic.py::deploy
  |
  v
sllm/cli/_cli_utils.py::deploy_model
  |
  v
POST /register
  |
  v
sllm/app_lib.py::register_handler
  |
  v
SllmController.register()
  |
  v
StoreManager.register()
  |
  v
create per-model router
```

### Inference Path

```text
POST /v1/chat/completions
  |
  v
sllm/app_lib.py::generate_handler
  |
  v
sllm/app_lib.py::inference_handler
  |
  v
ray.get_actor(model_name, namespace="models")
  |
  v
RoundRobinRouter.inference(request_data, "generate")
  |
  v
router request_queue
  |
  v
_load_balancer_loop()
  |
  v
backend_instance.generate.remote(...)
  |
  v
VllmBackend.generate()
```

### Worker Create Path

```text
RoundRobinRouter._auto_scaler_loop()
  |
  v
_create_instance()
  |
  v
_start_instance()
  |
  v
model_loading_scheduler.allocate_resource()
  |
  v
start_instance.options(...).remote(...)
  |
  v
sllm/inference_instance.py::start_instance
  |
  v
Ray backend actor
  |
  v
backend_instance.init_backend()
```

### Worker Stop Path

```text
RoundRobinRouter._stop_instance()
  |
  v
_finish_instance()
  |
  v
backend_instance.stop.remote()
  |
  v
ray.kill(backend_instance)
  |
  v
model_loading_scheduler.deallocate_resource()
```

---

## Worker State Machine

新增 worker / instance state：

```python
class InstanceState(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    PREEMPTING = "preempting"
    DEAD = "dead"
```

建議位置：

```text
sllm/utils.py
```

### State Meaning

- `STARTING`
  - Ray actor 建立中。
  - Backend 還沒 ready。
  - 不接 request。

- `READY`
  - 可以接新 request。
  - 若 concurrency 未滿，router 可以分配 request。

- `BUSY`
  - 正在處理 request。
  - 注意：`BUSY` 不代表不能接 request。
  - 如果 backend / actor 支援 concurrency，而且 instance 的 `concurrency < max_queue_length`，仍然可以接新 request。
  - 第一版不一定需要顯式使用 `BUSY`，因為目前已有 `concurrency`。
  - 可以保留作 metrics 狀態。

- `DRAINING`
  - 不接新 request。
  - 已經在跑的 request 可以完成。
  - 用於 graceful stop 或 preemption warning。

- `PREEMPTING`
  - trace 宣告 worker 即將被回收。
  - 不接新 request。
  - 嘗試取得 generated/current tokens。
  - 視 recovery policy 做 retry 或 token replay。

- `DEAD`
  - worker 不可用。
  - router 必須移除此 instance。
  - scheduler 必須 deallocate resource。

### Minimal State Transition

```text
STARTING
  |
  v
READY
  |
  +--> DRAINING --> DEAD
  |
  +--> PREEMPTING --> DEAD
```

### Router Allocation Rule

不要直接用 `state == READY` 判斷能不能接 request。第一版應該保留一個統一入口：

```python
await instance.can_accept_request()
```

原因是 backend 可能支援 concurrency，所以 `BUSY` 不一定等於不能接 request。

一定不能接新 request：

```text
STARTING
DRAINING
PREEMPTING
DEAD
```

可以接新 request：

```text
READY and concurrency < max_queue_length
BUSY and concurrency < max_queue_length
```

如果第一版沒有顯式使用 `BUSY`，也沒關係；重點是所有 request allocation 都走 `can_accept_request()`，不要散落成多個 state 判斷。

第一版可以保留原本 `InstanceHandle.ready` bool，避免一次改太多：

```python
ready: bool = False
state: InstanceState = InstanceState.STARTING
```

新增 helper：

```python
async def can_accept_request(self) -> bool:
    return (
        self.ready
        and self.state not in {
            InstanceState.STARTING,
            InstanceState.DRAINING,
            InstanceState.PREEMPTING,
            InstanceState.DEAD,
        }
        and self.concurrency < self.max_queue_length
    )
```

---

## Phase 0: Code Reading

目的：先完整看懂現有系統，不急著改。

要看懂：

- request path
- worker create path
- worker stop path
- scheduler allocation path
- router request queue
- migration router prototype

Deliverable:

- request flow notes
- worker lifecycle notes
- preemption hook points

建議輸出：

```text
docs/spotserve-flow.md
```

或直接放在 `examples/spotserve/README.md`。

---

## Phase 1: Trace Reader

新增：

```text
sllm/spot/
    __init__.py
    trace_reader.py
```

### Trace Format

第一版使用 JSONL：

```jsonl
{"time": 5.0, "event": "preempt", "node_id": "0"}
{"time": 12.0, "event": "recover", "node_id": "0"}
{"time": 20.0, "event": "preempt", "instance_id": "Qwen_xxx"}
{"time": 30.0, "event": "dead", "node_id": "0"}
```

### SpotEvent Dataclass

```python
@dataclass
class SpotEvent:
    time: float
    event: str
    node_id: Optional[str] = None
    model_name: Optional[str] = None
    instance_id: Optional[str] = None
```

### Validation

Trace reader 要檢查：

- `time >= 0`
- `event in {"preempt", "recover", "dead"}`
- event 至少要有 `node_id` 或 `instance_id`
- events 按 time 排序，或 parser 讀完後排序

### Deliverable

- `examples/spotserve/spot_trace_sample.jsonl`
- `tests/spotserve_test/test_trace_reader.py`
- trace reader unit test 不需要 GPU

---

## Phase 2: Worker State + Drain

修改：

```text
sllm/utils.py
sllm/routers/roundrobin_router.py
```

### `sllm/utils.py`

新增：

```python
class InstanceState(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    PREEMPTING = "preempting"
    DEAD = "dead"
```

Extend `InstanceHandle`：

```python
state: InstanceState = InstanceState.STARTING

async def can_accept_request(self):
    ...

async def mark_ready(self):
    ...

async def mark_draining(self):
    ...

async def mark_preempting(self):
    ...

async def mark_dead(self):
    ...
```

### `roundrobin_router.py`

修改 `_start_instance()`：

```python
instance.ready = True
instance.state = InstanceState.READY
```

修改 `_load_balancer_loop()`：

```python
if await instance.can_accept_request():
    allocated = await instance.add_requests(1)
```

注意：不要在 `_load_balancer_loop()` 裡手寫 `instance.state == InstanceState.READY` 這種判斷。所有能不能接 request 的邏輯都放進 `can_accept_request()`，因為 `BUSY` instance 在 backend 支援 concurrency 時仍可能可以接 request。

修改 `_stop_instance()` / `_finish_instance()`：

```python
await instance.mark_draining()
...
await instance.mark_dead()
```

新增：

```python
async def mark_instance_preempting(self, instance_id: str):
    ...

async def mark_instance_dead(self, instance_id: str):
    ...

async def drain_instance(self, instance_id: str):
    ...
```

### Deliverable

- `PREEMPTING` instance 不接新 request。
- `DEAD` instance 不接新 request。
- 現有非 spot 行為維持。

---

## Phase 3: Preemption Simulation Hooks

新增：

```text
sllm/spot/preemption_simulator.py
```

### Event Path

```text
trace time
  |
  v
preempt event
  |
  v
controller
  |
  v
router
  |
  v
instance PREEMPTING
```

### Controller Hook

在 `sllm/controller.py` 加：

```python
async def handle_preemption(
    self,
    node_id: Optional[str] = None,
    instance_id: Optional[str] = None,
    model_name: Optional[str] = None,
):
    ...
```

行為：

- 如果有 `model_name`，只找該 model router。
- 如果只有 `node_id`，broadcast 給所有 registered model routers。
- router 自己判斷哪些 instance 在該 node。

### Router Hook

在 `sllm/routers/roundrobin_router.py` 加：

```python
async def handle_preemption(
    self,
    node_id: Optional[str] = None,
    instance_id: Optional[str] = None,
):
    ...
```

行為：

- 找 matching instance。
- mark `PREEMPTING`。
- 不再接新 request。
- optional: start recovery / drain task。

### Simulator CLI

第一版可以先用 module 跑：

```bash
python -m sllm.spot.preemption_simulator \
  --trace examples/spotserve/spot_trace_sample.jsonl \
  --speedup 10
```

第二版再加：

```bash
sllm replay-trace --trace examples/spotserve/spot_trace_sample.jsonl
```

### Deliverable

- synthetic trace 可以 replay。
- 可以手動 mark instance preempting。
- preempting worker 不接新 request。

---

## Phase 4: Naive Retry Baseline

目的：先有最簡單 recovery baseline。

### Policy

如果 worker dead / preempted / backend exception：

```text
request fail
  |
  v
mark instance dead or preempting
  |
  v
requeue same request
  |
  v
assign another instance
  |
  v
retry
```

限制：

```python
max_retries = 2
```

### Router Config

在 deploy config 裡支援：

```json
{
  "router_config": {
    "recovery_policy": "naive_retry",
    "max_retries": 2
  }
}
```

### Implementation Note

目前 `RoundRobinRouter.inference()` 做了：

- increment `request_count`
- queue request
- wait allocation
- call backend
- decrement concurrency
- decrement `request_count`

加 retry 時要避免：

- retry 被算成新的 external request
- concurrency 沒有扣回來
- failed instance 還留在 ready pool

建議拆出 helper：

```python
async def _allocate_instance_for_request(self):
    ...

async def _call_backend(self, instance, request_data, action):
    ...

async def _release_instance_request(self, instance):
    ...
```

Naive retry pseudo-code：

```python
async def inference(self, request_data, action):
    request_id = request_data.get("request_id", ...)
    max_retries = self.router_config.get("max_retries", 0)

    async with self.request_count_lock:
        self.request_count += 1

    try:
        for attempt in range(max_retries + 1):
            instance = await self._allocate_instance_for_request()
            try:
                result = await self._call_backend(instance, request_data, action)
                if "error" not in result:
                    return result
            except Exception:
                await instance.mark_dead()
            finally:
                await instance.add_requests(-1)

        return {"error": "request failed after retries"}
    finally:
        async with self.request_count_lock:
            self.request_count -= 1
```

### Metrics

記錄：

- retry_count
- latency
- success rate
- failed_attempts
- final_status

### Deliverable

比較：

- no retry
- naive retry

---

## Phase 5: Metrics

新增：

```text
sllm/spot/metrics.py
```

輸出格式：JSONL。

### Request-level Metrics

Example：

```json
{
  "type": "request",
  "request_id": "xxx",
  "model": "Qwen/Qwen3-0.6B",
  "policy": "naive_retry",
  "success": true,
  "latency_ms": 1200,
  "retry_count": 1,
  "failed_attempts": 1,
  "assigned_instances": ["inst-a", "inst-b"],
  "preempted": true,
  "recovered_tokens": 0
}
```

### Instance-level Metrics

Example：

```json
{
  "type": "instance_state",
  "timestamp": 123.4,
  "model": "Qwen/Qwen3-0.6B",
  "instance_id": "inst-a",
  "node_id": "0",
  "from": "ready",
  "to": "preempting",
  "reason": "trace_event"
}
```

### System-level Metrics

最後分析：

- throughput
- p50 latency
- p95 latency
- p99 latency
- success rate
- retry overhead
- preemption count
- recovered token count

### Deliverable

能產生：

```text
results/spotserve/*.jsonl
```

能比較：

- latency
- p95
- success rate
- retry count

---

## Phase 6: Generated-token Recovery (Best-effort, Not True KV Migration)

第一版不做 true KV cache recovery，也不做 true KV migration。

第一版只做：

```text
generated tokens
  |
  v
token replay
  |
  v
continue generation best effort
```

如果 backend 不支援：

```text
fallback to naive retry
```

### Why Best Effort

`VllmBackend` 目前有：

- `get_current_tokens()`
- `resume_kv_cache()`

但這不代表我們已經能做到完整 in-flight KV cache recovery。

所以 v1 說法要保守：

> generated-token replay 是 best-effort，不保證 true KV recovery，也不保證 true KV migration。

### Router Config

```json
{
  "router_config": {
    "recovery_policy": "generated_token_replay",
    "max_retries": 2
  }
}
```

### Recovery Flow

```text
worker PREEMPTING
  |
  v
router asks backend.get_current_tokens()
  |
  v
store current/generated tokens
  |
  v
assign another worker
  |
  v
continue with token replay if possible
  |
  v
fallback to naive retry if not possible
```

### Backend Capability Detection

Router 不要假設所有 backend 都支援 recovery。

Pseudo-code：

```python
supports_tokens = hasattr(instance.backend_instance, "get_current_tokens")
supports_resume = hasattr(instance.backend_instance, "resume_generate")
```

實際 Ray actor 上 `hasattr` 不一定直接可靠，所以可先用 try/except：

```python
try:
    tokens = await instance.backend_instance.get_current_tokens.remote()
except Exception:
    tokens = None
```

### vLLM First-version Strategy

對 vLLM：

- 優先使用既有 `get_current_tokens()`。
- 如果拿得到 generated tokens：
  - 用 prompt + generated tokens replay。
  - 或用 `input_tokens` path 重新送進 `VllmBackend.generate()`。
  - 減少 remaining `max_tokens`。
- 如果不穩：
  - fallback naive retry。
  - metrics 記 `recovery_fallback=true`。

### Deliverable

比較：

```text
No recovery
Naive retry
Generated-token replay
```

指標：

- recovered tokens
- latency overhead
- success rate
- fallback count

---

## Phase 7: Integration With vLLM + MoE

接大鼻提供的：

```text
vLLM + MoE worker
```

CPY 視為：

```text
black-box backend
```

第一個整合測試：

```text
single model
multiple replicas
synthetic spot trace
```

比較：

| Policy | Success Rate | P50 Latency | P95 Latency | Retry Count | Recovered Tokens |
|---|---:|---:|---:|---:|---:|
| none | | | | | |
| naive retry | | | | | |
| token replay | | | | | |

### Experiment Order

建議順序：

1. dummy backend
   - 不需要 GPU。
   - 驗證 router state / retry / metrics。

2. transformers backend small model
   - 驗證 generated-token replay concept。

3. vLLM dense small model
   - 驗證 vLLM black-box integration。

4. vLLM MoE model
   - 使用大鼻提供的 config。
   - 做最終 comparison。

---

## Phase 8: Benchmark Harness + Visualization

目的：把 benchmark 做成之後可以一直重複使用的工具，而不是一次性的實驗 script。

第一版 benchmark 要回答：

- preemption 發生時，request 失敗率是多少？
- naive retry 可以救回多少 request？
- generated-token replay 是否比 naive retry 少重算？
- recovery 帶來多少 latency overhead？
- 不同 trace intensity 下，policy 差異是否穩定？
- vLLM dense / vLLM MoE black-box worker 的結果是否可比較？

### Benchmark Principles

Benchmark 要固定四件事：

```text
workload
trace
policy
backend/model config
```

每次跑完都要保留：

```text
raw metrics
summary table
plots
run metadata
```

這樣之後換 policy、換 model、換 trace 時，結果才可以直接並排比較。

### Benchmark Runner

新增：

```text
benchmarks/spotserve/run_benchmark.py
benchmarks/spotserve/benchmark_matrix.yaml
```

建議 command：

```bash
python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix.yaml \
  --output results/spotserve
```

Runner 負責：

- 啟動固定 workload。
- 啟動 synthetic preemption trace replay。
- 跑不同 recovery policy。
- 收集 JSONL metrics。
- 產出 summary JSON / CSV。
- 呼叫 visualization script 產生 report。

### Benchmark Matrix

第一版 matrix：

| Scenario | Backend | Model | Policy | Trace | Goal |
|---|---|---|---|---|---|
| no-preemption | dummy | fake model | none | none | sanity check |
| preemption-no-retry | dummy | fake model | none | synthetic-light | failure baseline |
| preemption-naive-retry | dummy | fake model | naive_retry | synthetic-light | retry baseline |
| preemption-token-replay | transformers | small model | generated_token_replay | synthetic-light | replay validation |
| vllm-dense | vllm | small dense model | naive/replay | synthetic-light | vLLM black-box |
| vllm-moe | vllm | 大鼻提供 MoE | naive/replay | synthetic-light/heavy | final comparison |

### Workload Design

至少準備三種 workload：

```text
steady-low
steady-high
burst
```

要記錄：

- request arrival time
- request_id
- prompt length
- max_tokens
- temperature
- expected model

建議格式：

```jsonl
{"time": 0.0, "request_id": "req-0001", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}
{"time": 0.2, "request_id": "req-0002", "messages": [{"role": "user", "content": "explain serverless inference"}], "max_tokens": 128}
```

### Result Directory Layout

每次 benchmark 建議產生一個 run id：

```text
results/spotserve/
  2026-xx-xx_hh-mm_policy-comparison/
    run_metadata.json
    raw_requests.jsonl
    raw_instance_states.jsonl
    raw_trace_events.jsonl
    summary.json
    summary.csv
    plots/
      latency_cdf.html
      latency_box.html
      throughput_timeseries.html
      success_rate_bar.html
      retry_count_bar.html
      recovered_tokens_bar.html
      preemption_timeline.html
    report.html
    report.md
```

`run_metadata.json` 至少包含：

```json
{
  "git_commit": "...",
  "backend": "vllm",
  "model": "Qwen/Qwen3-0.6B",
  "policy": "generated_token_replay",
  "trace": "synthetic-light",
  "workload": "steady-low",
  "num_replicas": 2,
  "max_retries": 2,
  "gpu": "...",
  "started_at": "..."
}
```

### Visualization

第一版建議用 offline Plotly HTML：

- 不需要開 dashboard server。
- 圖可以互動。
- demo 時可以直接打開 `report.html`。
- 也可以輸出 PNG 給投影片。

新增：

```text
scripts/analyze_spotserve_benchmark.py
scripts/plot_spotserve_benchmark.py
```

必備圖表：

1. Latency CDF
   - 比較 `none`、`naive_retry`、`generated_token_replay`。
   - 看 tail latency 是否被 retry 拉高。

2. Latency box plot
   - 看 p50 / p95 / p99 分布。
   - 適合放簡報。

3. Throughput time series
   - x-axis 是 time。
   - y-axis 是 completed requests / second。
   - 可以看 preemption 發生後吞吐掉多少。

4. Success rate bar chart
   - 每個 policy 一根 bar。
   - 最直觀展示 retry/replay 有沒有救回 request。

5. Retry / fallback stacked bar
   - 顯示 retry count、fallback count。
   - 區分「成功是因為 replay」還是「最後 fallback naive retry」。

6. Recovered tokens bar chart
   - 顯示每個 policy 或每個 run 的 recovered token 數。
   - 用來支持 generated-token replay 的價值。

7. Preemption timeline
   - x-axis 是時間。
   - worker state transition 用不同顏色表示。
   - 標出 `READY`、`DRAINING`、`PREEMPTING`、`DEAD`。
   - 這張圖很適合解釋系統行為。

### Report Format

`report.html` 內容建議：

- Experiment title
- Git commit / config / model / trace / workload
- Summary table
- Key takeaways
- Plots
- Raw artifact links

`report.md` 內容建議：

```markdown
# SpotServe Benchmark Report

## Run Metadata

## Summary

| Policy | Success Rate | P50 | P95 | P99 | Throughput | Retry Count | Recovered Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|

## Plots

## Notes
```

### Visual Style

圖表要保持固定風格，避免每次看起來都不一樣：

- policy color 固定：
  - `none`: gray
  - `naive_retry`: blue
  - `generated_token_replay`: green
  - `fallback`: orange
  - `failure`: red
- y-axis 單位清楚標示。
- latency 一律用 ms。
- throughput 一律用 req/s。
- 所有圖都要有 title、legend、axis label。
- summary table 數字固定小數位。

### Deliverable

完成後應該能做到：

```bash
python benchmarks/spotserve/run_benchmark.py --config benchmarks/spotserve/benchmark_matrix.yaml
python scripts/analyze_spotserve_benchmark.py results/spotserve/<run_id>
python scripts/plot_spotserve_benchmark.py results/spotserve/<run_id>
```

產出：

- `summary.csv`
- `summary.json`
- `report.html`
- `report.md`
- reusable plots

---

## Files To Modify

### Must Modify

#### `sllm/routers/roundrobin_router.py`

加入：

- worker state checks
- drain policy
- preemption hook
- retry wrapper
- generated-token replay hook
- metrics emission

最重要修改點：

- `_load_balancer_loop()`
- `inference()`
- `_start_instance()`
- `_stop_instance()`
- `_finish_instance()`
- `_shutdown_instance()`

#### `sllm/controller.py`

加入：

```python
handle_preemption()
```

用途：

- trace simulator 發 event 給 controller。
- controller 找到對應 router。
- router mark instance/node preempting。

#### `sllm/utils.py`

新增：

```python
InstanceState
```

擴充：

```python
InstanceHandle
```

### Optional Later

#### `sllm/schedulers/fcfs_scheduler.py`

第一版先不要太早改 scheduler。先完成 router-level 行為：

- `PREEMPTING` instance 不接新 request。
- `DEAD` instance 不接新 request。
- retry / recovery 可以重新分配到其他 ready instance。

第二版才加最小 spot-aware allocation：

- 避免配置到 dead node。
- 避免配置到 preempting node。
- 保留原本 FCFS 行為。
- optional: spot/on-demand node preference。

#### `sllm/schedulers/storage_aware_scheduler.py`

第二版才加：

- storage-aware + spot-aware ranking
- risk-aware placement
- on-demand fallback

#### `sllm/app_lib.py`

可選 debug endpoint：

```text
/admin/preempt
```

#### `sllm/cli/clic.py`

可選 CLI：

```text
sllm replay-trace
sllm simulate-preemption
```

---

## New Files

```text
sllm/spot/
    __init__.py
    trace_reader.py
    preemption_simulator.py
    recovery_policy.py
    metrics.py

benchmarks/spotserve/
    README.md
    run_benchmark.py
    benchmark_matrix.yaml
    workloads/
        steady_low.jsonl
        steady_high.jsonl
        burst.jsonl

scripts/
    analyze_spotserve_benchmark.py
    plot_spotserve_benchmark.py

examples/spotserve/
    README.md
    spot_trace_sample.jsonl
    config-vllm-spot.json

reports/spotserve/
    README.md

tests/spotserve_test/
    test_trace_reader.py
    test_router_state.py
    test_naive_retry.py
```

---

## Do Not Touch In v1

第一版不要碰：

```text
vLLM scheduler internals
MoE internals
expert dispatch
CUDA kernel
PagedAttention internals
true KV cache migration
```

`vllm_backend.py` 第一版盡量少改。

可以做：

- 使用既有 `get_current_tokens()`。
- 加很小的 best-effort token replay helper，如果真的必要。

不要做：

- 改 vLLM engine internals。
- 改 expert parallel。
- 改 MoE routing。

---

## Risk

### MigrationRouter Risk

`MigrationRouter` 目前看起來可能跟 `RoundRobinRouter` 的 state pool 不同步。

原因：

- `MigrationRouter` 使用 `ready_instances`。
- `RoundRobinRouter` 使用 `ready_inference_instances`。

處理：

- v1 不依賴 `MigrationRouter`。
- 只參考它的 migration/recovery idea。

### vLLM Recovery Risk

vLLM generated-token recovery 不保證 true KV recovery。

處理：

- v1 寫成 best-effort token replay。
- fallback naive retry。
- metrics 明確記錄 fallback。

### Ray Actor Failure Risk

Ray actor 被 kill / preempt 時，exception timing 可能不穩。

處理：

- router catch broad Ray exceptions。
- mark instance dead。
- retry or fail gracefully。

### Multi-model Risk

ServerlessLLM 原本是 multi-model，但 v1 不要一開始做 full multi-model spot scheduling。

處理：

- v1 只保證 single model + multiple replicas。
- data structure 保留 `model_name`，不要封死 multi-model extension。

### GPU Environment Risk

vLLM + MoE 可能受環境限制。

處理：

- 先 dummy backend。
- 再 transformers backend。
- 再 vLLM dense。
- 最後接 vLLM MoE。

### Benchmark Comparability Risk

如果每次 benchmark 的 workload、trace、model config、replica count 不固定，結果會很難比較。

處理：

- 每次 run 都保存 `run_metadata.json`。
- benchmark matrix 固定 workload / trace / policy。
- latency 圖表固定顯示 p50 / p95 / p99。
- 至少跑 warmup，再跑正式 measurement。
- 圖表配色固定，避免同一個 policy 每次顏色不同。

---

## Milestones

### M1: Code Reading

完成條件：

- request flow 看懂。
- worker create/stop flow 看懂。
- scheduler allocation 看懂。
- migration router limitation 記錄下來。

產出：

- request flow notes
- worker lifecycle notes

### M2: Trace Reader

完成條件：

- JSONL trace 可 parse。
- validation 可抓錯。
- unit test 不需要 GPU。

產出：

- `sllm/spot/trace_reader.py`
- `examples/spotserve/spot_trace_sample.jsonl`
- `tests/spotserve_test/test_trace_reader.py`

### M3: Worker State + Drain

完成條件：

- instance 有 `STARTING/READY/DRAINING/PREEMPTING/DEAD`。
- `PREEMPTING` 不接新 request。
- `DEAD` 不接新 request。
- 原本 round-robin 行為不壞。

產出：

- `InstanceState`
- router state transition
- router state tests

### M4: Naive Retry

完成條件：

- worker dead 後 request 可 retry。
- retry 次數有限制。
- request 不會消失。
- metrics 記 retry count。

產出：

- retry baseline
- no retry vs retry comparison

### M5: Metrics

完成條件：

- request-level JSONL metrics。
- instance state transition JSONL metrics。
- 可算 success rate / p95 latency。
- metrics schema 足夠支援後續 benchmark visualization。

產出：

- `sllm/spot/metrics.py`
- `results/spotserve/*.jsonl`

### M6: Generated-token Replay

完成條件：

- preempting worker 可嘗試取 current/generated tokens。
- 另一個 worker 可 best-effort replay。
- 不支援時 fallback naive retry。
- metrics 記 recovered tokens / fallback。

產出：

- token replay policy
- retry vs replay comparison

### M7: vLLM + MoE Black-box Integration

完成條件：

- 接大鼻提供的 vLLM + MoE config。
- single model multiple replicas 可跑。
- synthetic spot trace 可 replay。
- policy comparison table 有數據。

產出：

- vLLM + MoE black-box experiment
- baseline vs retry vs token replay result

### M8: Benchmark Harness + Visualization

完成條件：

- benchmark runner 可重跑固定 workload / trace / policy matrix。
- 每次 run 都有 metadata、raw JSONL、summary CSV/JSON。
- 可產生 `report.html` 和 `report.md`。
- 圖表至少包含 latency CDF、latency box plot、throughput time series、success rate、retry/fallback、preemption timeline。
- 同一組圖表可用於 dummy、transformers、vLLM dense、vLLM MoE。

產出：

- `benchmarks/spotserve/run_benchmark.py`
- `benchmarks/spotserve/benchmark_matrix.yaml`
- `scripts/analyze_spotserve_benchmark.py`
- `scripts/plot_spotserve_benchmark.py`
- `results/spotserve/<run_id>/report.html`
- `results/spotserve/<run_id>/report.md`

---

## Definition Of Done

第一版完成條件：

- [ ] synthetic spot trace 可 replay
- [ ] preempting worker 不接新 request
- [ ] request 不會消失
- [ ] naive retry 可運作
- [ ] token replay 可運作，或 fallback 清楚記錄
- [ ] baseline vs retry vs replay 有數據
- [ ] success rate 與 p95 latency 可比較
- [ ] benchmark 可以用固定 config 重跑
- [ ] benchmark report 有 summary table 和互動圖表
- [ ] plots 可以直接放進報告或簡報
- [ ] 可以接大鼻提供的 vLLM + MoE worker as black box

---

## Recommended PR Order

### PR 1: Trace Reader Only

不要碰 router。

內容：

- `sllm/spot/trace_reader.py`
- sample trace
- unit test

### PR 2: Worker State

內容：

- `InstanceState`
- `InstanceHandle` helper
- router allocation check
- preempting/draining state tests

### PR 3: Preemption Simulator

內容：

- controller hook
- router hook
- preemption simulator
- manual synthetic trace replay

### PR 4: Naive Retry + Metrics

內容：

- retry wrapper
- metrics JSONL
- no retry vs retry experiment

### PR 5: Benchmark Harness + Visualization

內容：

- benchmark runner
- benchmark matrix config
- analyzer script
- Plotly report / Markdown report
- no retry vs naive retry visual comparison

### PR 6: Generated-token Replay

內容：

- generated-token replay policy
- backend capability detection
- fallback naive retry
- retry vs replay experiment

### PR 7: vLLM + MoE Integration

內容：

- 大鼻提供 config
- black-box worker experiment
- final comparison table

---

## Core Principle

不要從：

```text
MoE internals
expert dispatch
CUDA kernel
```

開始。

先讓：

```text
ServerlessLLM
  +
vLLM worker
  +
spot trace
  +
preemption recovery
```

完整跑通。

之後才考慮：

```text
expert-aware scheduling
MoE migration
KV cache migration
```
