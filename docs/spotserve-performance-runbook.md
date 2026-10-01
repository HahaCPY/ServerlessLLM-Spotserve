# SpotServe MoE Performance Runbook

這份文件給負責正式 performance 實驗的組員使用。請固定使用同一個 commit、模型、
硬體、workload 與 trace。不要把 smoke test、失敗重跑或不同環境的結果混入正式統計。

詳細指標定義見 `docs/spotserve-performance-experiment-guide.md`；本文件只保留交接後
實際需要執行的順序、命令、成功條件與結果保存方式。

## 1. 實驗範圍

主實驗環境：

```text
1 physical host
4 x NVIDIA GeForce RTX 5070 Ti, 16 GiB
Qwen2-MoE-Tiny
trace-driven preemption simulation
```

正式比較：

| ID | Matrix | 比較內容 |
| --- | --- | --- |
| E1 | `benchmark_matrix_reparallelization_performance.yaml` | re-parallelization disabled vs applied |
| E2 | `benchmark_matrix_context_migration_performance.yaml` | baseline vs MoE-aware migration planning |
| E3 | `benchmark_matrix_stateful_recovery_performance.yaml` | token replay vs true KV/state restore |
| E4 | `benchmark_matrix_spotserve_core_performance.yaml` | SpotServe-style baseline vs combined MoE-aware flow |
| M1 | `benchmark_matrix_spotserve_core_trace_sweep.yaml` | combined flow 在多種 trace 下的穩定性 |
| M2 | `benchmark_matrix_expert_remap_a2a_reduction_performance.yaml` | physical remap 前後 A2A payload reduction |
| M3 | `benchmark_matrix_elastic_ep_resize_performance.yaml` | active-request EP2 -> EP4 resize |
| M4 | `benchmark_matrix_simulated_cross_host_expert_remap_performance.yaml` | 單機 2+2 GPU failure-domain simulation |

E1-E4 是主要效能表。M1-M4 是機制驗證或補充實驗，不要把它們的 latency 直接混入
E1-E4 平均。

## 2. 固定版本與環境

在 repo 根目錄執行：

```bash
git status --short
git rev-parse HEAD
date --iso-8601=seconds
nvidia-smi
podman --version
podman image inspect docker.io/serverlessllm/sllm:latest \
  --format '{{.Id}} {{.Created}}'
test -f /work/spotserve-models/Qwen2-MoE-Tiny/config.json && echo MODEL_OK
```

將輸出存進實驗紀錄。正式實驗前必須滿足：

```text
working tree 狀態已記錄
四張 GPU 沒有其他 workload
模型 config.json 存在
所有 run 使用同一個 git commit 與 image
```

若 `sllm_store/vllm_patch/`、Dockerfile 或依賴在目前 image 建立後曾改動，第一組
prepare 必須拿掉 `--skip-build`，重建一次 image。之後的重跑都使用
`--skip-build`。Python/controller 變更透過 `SPOTSERVE_SYNC_SOURCE=1` 同步。

## 3. 執行規則

1. 每個 matrix 先跑一次 smoke，確認下方 success gate。
2. 正式結果每個主要 matrix 至少跑 5 次，時間足夠時跑 10 次。
3. baseline 與 candidate 必須由同一份 matrix 連續執行，不要拆開跑。
4. 同一時間只執行一個 benchmark。
5. 每次切換 deploy set 前先把 `/tmp/spotserve-work/results` 複製回 host。
6. 特殊 verifier 的固定 `report.json` 每次會覆寫，完成後要立即備份。

建立 host 端結果目錄：

```bash
mkdir -p performance-results
```

每次 matrix 跑完後執行：

```bash
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mkdir -p "performance-results/$RUN_TAG"
podman cp sllm_head:/tmp/spotserve-work/results \
  "performance-results/$RUN_TAG/results"
git rev-parse HEAD > "performance-results/$RUN_TAG/git-commit.txt"
nvidia-smi > "performance-results/$RUN_TAG/nvidia-smi.txt"
```

## 4. E1 Re-parallelization

準備：

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

有效條件：

```text
trace_replay_success = 1
replanning_events >= 1
replanning_execution_applied >= 1
replanning_execution_failed = 0
```

主要記錄：success rate、P95、throughput、replan-window latency、post-replan
latency 與 replan execution duration。

## 5. E2 Context Migration

準備：

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

有效條件：

```text
context_migration_events >= 1
context_migration_plan_count >= 1
route_source = vllm_runtime_topk
route_kind = runtime_observed_topk
```

另存 KV cost、expert dispatch cost、queue cost、expert locality 與 estimated remote
routing。若 route source 是 fixture，只能標成 planner sanity check。

## 6. E3 Stateful Recovery

準備：

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

有效條件：

```text
state_restore_attempts_total >= 1
state_restore_successes_total >= 1
true_kv_restore_successes_total >= 1
true_kv_restored_blocks_total > 0
supports_state_restore_requests >= 1
state_restore_fallback_count = 0
```

`recovered_tokens_total > 0` 本身不能證明 true KV restore。

## 7. E4 Combined Core And M1 Trace Sweep

兩組共用一次 prepare：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CORE_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CORE_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set spotserve-core-performance
```

E4：

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

M1：

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

所有 baseline/candidate pair 都必須有：

```text
trace_replay_success = 1
trace_replay_failed = 0
```

即使 request success rate 是 100%，trace 失敗的 run 仍然無效。

## 8. M2 Expert Remap And A2A Reduction

準備：

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

執行：

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

有效條件：

```text
traffic_reduced = true
claim_supported = true
candidate observed_payload_bytes < baseline observed_payload_bytes
payload_reduction_ratio > 0
```

這裡量到的是 runtime collective tensor payload，不是實體 NIC bytes。

## 9. M3 Elastic EP2 To EP4

這組獨占四張 GPU。確認每張至少有 8192 MiB free memory：

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv
```

準備：

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

有效條件：

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

## 10. M4 Two-failure-domain Simulation

這組使用獨立容器，把同一台主機拆成 GPU 0-1 與 GPU 2-3。執行前先保存一般
benchmark 結果，並停止一般 SpotServe stack，確保四張 GPU 都空閒：

```bash
podman stop sllm_head sllm_worker_0 sllm_worker_1 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.free --format=csv
```

啟動：

```bash
SPOTSERVE_REPO_ROOT="$PWD" \
MODEL_FOLDER=/work/spotserve-models \
podman-compose \
  -f "$PWD/examples/spotserve/docker-compose.simulated-cross-host.yml" \
  up -d
```

執行：

```bash
podman exec sllm_sim_head bash -lc '
cd /workspace/spotserve &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_simulated_cross_host_expert_remap_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效條件：

```text
host_mode = simulated_failure_domain
physical_cross_host_verified = false
simulated_cross_host_verified = true
physical_host_count = 1
failure_domain_count = 2
ray_node_count >= 2
verified_placement = true
physical_weight_migration = true
cross_failure_domain_migration = true
cross_failure_domain_moved_shards > 0
cross_failure_domain_moved_bytes > 0
inter_container_a2a_observed = true
```

保存結果：

```bash
RUN_TAG="$(date +%Y%m%d-%H%M%S)-simulated-domains"
mkdir -p "performance-results/$RUN_TAG"
podman cp sllm_sim_head:/workspace/spotserve/results/spotserve_simulated_cross_host_expert_remap_performance \
  "performance-results/$RUN_TAG/"
```

停止並釋放 GPU：

```bash
SPOTSERVE_REPO_ROOT="$PWD" \
MODEL_FOLDER=/work/spotserve-models \
podman-compose \
  -f "$PWD/examples/spotserve/docker-compose.simulated-cross-host.yml" \
  down
```

報告中必須稱為：

```text
single-physical-host, two-failure-domain simulation
```

不能稱為 physical multi-host deployment 或實體 NIC performance。

## 11. Invalid Run

下列任一情況發生，該 run 不得納入統計：

- `trace_replay_success=0` 或 `trace_replay_failed>0`。
- setup error、所有 request failed、模型一直停在 `starting`。
- baseline/candidate 使用不同模型、GPU 數、workload 或 trace。
- KV restore 沒有 positive restored-block counters。
- runtime routing 使用 fixture 而不是 `vllm_runtime_topk`。
- physical placement 沒有 `runtime_verified_placement=true`。
- elastic EP 發生 actor recreate，或 active request 未完成。
- GPU 被其他 workload 佔用、OOM、Ray resource pending 或明顯 throttling。

單獨的 cleanup warning 不一定使結果無效；若後續部署、trace 與 success gate 全部成功，
可保留。無法確認時標成 invalid 並重跑。

## 12. 最終交付

每個有效 run 至少保留：

```text
summary.json
raw_requests.jsonl
router metrics JSONL
trace_replayer.log
report.html
特殊 verifier 的 report.json
git commit 與 nvidia-smi 紀錄
```

E1-E4 主表報告：

```text
success rate / clean success rate / fallback rate
P50 / P95 / P99 latency
throughput
failure/replan/migration window latency
post-replan/post-migration/post-recovery latency
```

MoE 機制表另外報告：

```text
expert locality 與 remote-routing ratio
expert moved shards / bytes / remap duration
A2A baseline/candidate payload 與 reduction ratio
EP transition、changed experts、actor identity preservation
```

不要直接拿本機 RTX 5070 Ti 的絕對 latency 與 SpotServe 論文的 AWS T4 多機數字做
加速比。公平結論應限定為同一台測試機、相同 workload 與 trace 下的 baseline vs
MoE-aware implementation。
