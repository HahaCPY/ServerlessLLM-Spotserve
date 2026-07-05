# Milestone 3 Backend Capability Handoff

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
```

重點是：大鼻不需要決定「要選哪個 plan」。大鼻只需要告訴 CPY：

```text
這個 backend 對這個 model，到底支援哪些 TP / DP / PP / EP config。
```

CPY 仍然負責根據 spot event 和 GPU availability 選 plan。

## Ownership

大鼻主要碰這些檔案：

```text
sllm/backends/backend_utils.py
sllm/backends/vllm_backend.py
sllm/backends/vllm_capability.py
sllm/backends/vllm_moe_capability.py
examples/spotserve/config-vllm-moe-*.json
docs/vllm_moe_config.md
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

## Minimal Implementation

第一版請先保守，不要假裝 vLLM 已經支援所有 dynamic TP / DP / EP。

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
vLLM supports exactly the currently configured TP shape.
SLLM control plane can represent DP as replicas.
EP/state restore are not claimed yet.
```

這樣是安全的，因為 planner 不會選到 backend 沒承諾能跑的 config。

## MoE Follow-up

如果要支援 MoE capability，可以再新增：

```text
sllm/backends/vllm_moe_capability.py
```

它可以先從 config 判斷：

```text
backend = vllm
model path / model id looks like MoE
trust_remote_code = true
tensor_parallel_size = N
```

但第一版仍然建議保守：

```text
supports_ep = False
expert_parallel_size = 1
```

等大鼻確認 vLLM runtime 真的能提供 expert parallel 或 expert metadata，再打開：

```text
supports_ep = True
supported_configs includes expert_parallel_size > 1
```

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
def test_vllm_capability_reports_current_tp_config():
    capability = get_backend_capability(
        {
            "model": "vllm-moe-none",
            "backend": "vllm",
            "num_gpus": 2,
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
    assert capability.supported_configs[0].num_gpus == 2
```

也要測 default：

```text
backend_config 沒有 tensor_parallel_size 時，預設 TP=1。
```

## Config Contract

大鼻可以先支援這些 config 欄位：

```text
model
backend
num_gpus
backend_config.tensor_parallel_size
backend_config.pipeline_parallel_size
backend_config.expert_parallel_size
backend_config.supports_state_export
backend_config.supports_state_restore
```

如果某個欄位不存在，請保守 fallback：

```text
tensor_parallel_size = 1
pipeline_parallel_size = 1
expert_parallel_size = 1
state export/restore = false
```

## Definition Of Done

大鼻這部分完成時，應該滿足：

- `BackendCapability` dataclass exists.
- `get_backend_capability(model_config)` exists.
- vLLM model config can return a conservative capability.
- `supported_configs` contains at least the current configured vLLM shape.
- EP and state restore are false unless大鼻已確認 backend 真的支援。
- unit tests cover vLLM TP config and default TP=1.
- no CPY router/scheduler/controller main-flow changes are required.

## Open Questions For 大鼻

請大鼻確認以下問題，確認後再更新 capability：

- vLLM dense 是否只支援 `tensor_parallel_size`，還是能明確表示 DP？
- vLLM MoE 是否能 expose expert metadata？
- vLLM MoE 是否有可用的 expert parallel config？
- state export / state restore 是否有任何可用 hook？
- `num_gpus` 應該以 ServerlessLLM top-level `num_gpus` 為準，還是以
  `backend_config.tensor_parallel_size` 為準？
- dynamic replan 時，backend 是否允許「線上改 TP」，還是必須重建 actor？

第一版可以全部保守回答。重點是先把 contract 接起來，讓 CPY planner 可以
filter legal plans。
