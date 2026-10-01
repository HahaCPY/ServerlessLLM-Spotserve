# SpotServe MoE Performance Experiment Guide

這份文件提供 performance 實驗的標準跑法。目標是在同一台 4-GPU 主機、相同模型、
相同 workload 與相同 preemption trace 下，比較 SpotServe-style baseline 與本專案的
MoE-aware 實作。

## Claim Boundary

目前實驗環境是 single-host、multi-GPU：

```text
1 physical host
└── 4 x NVIDIA GeForce RTX 5070 Ti
```

因此可以做的公平比較是：

```text
SpotServe-style baseline
vs.
MoE-aware context migration / recovery / re-parallelization
```

兩者必須使用相同硬體、模型、GPU 數量、workload、trace 與 benchmark 設定。

可以使用的結論：

> On a single-host multi-GPU testbed, the MoE-aware extension is compared
> against a SpotServe-style baseline under identical trace-driven preemption
> workloads.

不可直接將本專案的 latency、throughput 或加速比與 SpotServe 論文在多台 AWS
instances 上公布的絕對數字比較。本實驗不包含 cross-host network、internode A2A
或真實 cloud provider termination notice。

## Experiment Groups

正式結果至少包含以下四組：

| Group | Matrix | 目的 |
| --- | --- | --- |
| E1 | `benchmark_matrix_reparallelization_performance.yaml` | V6 re-parallelization baseline vs applied |
| E2 | `benchmark_matrix_context_migration_performance.yaml` | V7 migration baseline vs MoE-aware target selection |
| E3 | `benchmark_matrix_stateful_recovery_performance.yaml` | V8 token replay vs true state/KV restore |
| E4 | `benchmark_matrix_spotserve_core_performance.yaml` | baseline vs combined migration/recovery flow |

另外執行下列機制實驗：

| Group | Matrix | 目的 |
| --- | --- | --- |
| M1 | `benchmark_matrix_spotserve_core_trace_sweep.yaml` | 多種 preemption/recovery trace 的穩定性 |
| M2 | `benchmark_matrix_expert_remap_a2a_reduction_performance.yaml` | 相同 workload remap 前後的 A2A payload reduction |
| M3 | `benchmark_matrix_elastic_ep_resize_performance.yaml` | 同 actor、active-request drain 的 EP2 -> EP4 resize |

## Before Running

記錄程式版本與硬體環境：

```bash
git rev-parse HEAD
nvidia-smi
podman --version
podman images docker.io/serverlessllm/sllm
```

確認模型存在：

```bash
test -f /work/spotserve-models/Qwen2-MoE-Tiny/config.json && echo MODEL_OK
```

確認 GPU 沒有其他使用者的 workload。M3 必須有四張 GPU，每張至少 8192 MiB
free memory：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv
```

不要終止其他使用者的程序。如果 GPU 不足，等待資源釋放後再跑。

## Image And Source Setup

如果目前 image 已包含最新 vLLM patches，後續只需使用 `--skip-build`。
在全新機器或 vLLM patch 有變更時，第一次 prepare 必須省略 `--skip-build`，完成
一次 image rebuild。一般 Python/controller 修改可使用 `SPOTSERVE_SYNC_SOURCE=1`
同步到 container。

每次切換 deploy set 都要重新執行對應的 prepare command。prepare 可能重建
container，所以切換前先保存上一組結果。

## E1: Re-parallelization

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set reparallelization-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效 run 必須符合：

```text
trace_replay_success = 1
replanning_events >= 1
replanning_execution_applied >= 1
replanning_execution_failed = 0
```

主要報告 `success_rate`、`latency_p95_ms`、`throughput_req_s`、
`phase_replan_window_latency_p95_ms`、`phase_post_replan_latency_p95_ms` 與
`replanning_avg_execution_duration_ms`。

## E2: Context Migration

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set context-migration-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效的 MoE runtime run 應包含：

```text
context_migration_events >= 1
context_migration_plan_count >= 1
route_source = vllm_runtime_topk
route_kind = runtime_observed_topk
```

同時保存 planner 的 cost breakdown：

```text
KV migration cost
expert dispatch cost
queue penalty cost
hot expert locality ratio
estimated remote routing ratio/tokens
```

`route_source=request_instrumentation` 或 fixture 只能算 planner sanity check，不能算
真實 vLLM routing instrumentation 實驗。

## E3: Stateful Recovery

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set stateful-recovery-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效的 true KV restore run 必須符合：

```text
state_restore_attempts_total >= 1
state_restore_successes_total >= 1
true_kv_restore_successes_total >= 1
true_kv_restored_blocks_total > 0
supports_state_restore_requests >= 1
state_restore_fallback_count = 0
```

只看到 `recovered_tokens_total > 0` 不足以證明 true KV restore，因為 token replay
也可能產生 recovered token。

## E4 And M1: Combined Core

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CORE_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CORE_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set spotserve-core-performance
```

先跑主要比較：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

再跑 multi-trace sweep：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_trace_sweep.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

每個 baseline/candidate pair 都必須確認 `trace_replay_success=1`。如果 trace replay
失敗，即使 request success rate 是 100%，該 run 仍然無效。

## M2: Physical Remap And A2A Reduction

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
VLLM_SPOTSERVE_A2A_TRACE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set expert-remap-active-performance
```

執行相同 workload 的 remap 前後比較：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_expert_remap_a2a_reduction_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

只有同時符合下列條件才能宣稱 A2A traffic reduction：

```text
traffic_reduced = true
claim_supported = true
candidate observed_payload_bytes < baseline observed_payload_bytes
payload_reduction_ratio > 0
```

這裡量到的是 same-host sparse collective payload，不是 internode NIC traffic。

## M3: Elastic EP2 To EP4

這組測試獨占四張 GPU。先確認每張 GPU 都有至少 8192 MiB free memory。

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set reparallelization-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_elastic_ep_resize_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效 run 必須符合：

```text
before_worker_count = 2
after_worker_count = 4
runtime_verified_placement = true
actor_recreated = false
active_request_observed = true
active_request_completed = true
elastic_ep_admission_drained = true
changed_experts 非空
```

這證明 same-host live in-place EP2 -> EP4，不代表 cross-host EP resize。

## Repetitions

先跑一次 smoke，確認所有 success gates。正式實驗每個主要 matrix 至少跑 5 次；若時間
允許，跑 10 次。baseline 與 candidate 必須保留在同一個 matrix 中執行，不要手動拆開，
以免環境狀態不同。

每次執行前記錄：

```text
git commit
日期與 run index
GPU 型號與空閒記憶體
matrix/config 名稱
model path
任何非預設環境變數
```

不要只平均各次輸出的 P95。正式彙整應優先使用所有 raw request latency 合併後重新計算
P50/P95/P99，並另外報告各 run 的 median 與 variability。

## Preserve Results

benchmark 結果位於 `sllm_head` 的 `/tmp/spotserve-work/results`。切換 deploy set 或
重建 container 前，先複製到 host：

```bash
mkdir -p results-export
podman cp sllm_head:/tmp/spotserve-work/results \
  results-export/results-$(date +%Y%m%d-%H%M%S)
```

每個 run 至少保留：

```text
summary.json
raw_requests.jsonl
router metrics JSONL
trace_replayer.log
report.html
特殊 verifier 的 report.json
```

## Invalid Run Rules

以下任一情況發生時，不得把該 run 放入效能統計：

- `trace_replay_success=0` 或 `trace_replay_failed>0`。
- benchmark setup/cleanup 出現 error，且模型狀態無法確認乾淨。
- baseline 與 candidate 使用不同模型、GPU 數、workload 或 trace。
- true KV restore claim 只有 token replay counter，沒有 restored blocks。
- runtime routing claim 使用 fixture，而不是 `vllm_runtime_topk`。
- physical placement claim 沒有 `runtime_verified_placement=true`。
- elastic EP run 發生 actor recreate，或 active request 未完成。
- GPU 同時被其他 workload 佔用，造成 OOM、scheduler pending 或異常 throttling。

cleanup warning 不一定代表 run 無效；需進一步確認後續 model deployment、trace replay
與 success gates 都成功。若無法確認，將該次標成 invalid 並重跑。

## Final Tables

主表按 E1-E4 分別列出：

```text
success rate
clean success rate
fallback rate
P50 / P95 / P99 latency
throughput
replan/migration/failure window latency
post-replan/post-migration/post-recovery latency
```

MoE mechanism table 另外列出：

```text
expert locality ratio
estimated and observed remote routing ratio
expert remap duration and moved bytes
A2A baseline/candidate payload and reduction ratio
EP transition, changed experts, and actor identity preservation
```

最後在圖表與報告標題中明確註明：

```text
single-host, 4 x RTX 5070 Ti, trace-driven preemption simulation
```
