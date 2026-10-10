# CSCC Ray 正式 F1/F2（單次）

## 固定設計

- Model：`/home/ps2026004/models/Qwen1.5-MoE-A2.7B-Chat`
- GPU pool：8 個 Ray GPU workers；每個 worker 1 張 RTX 5090
- 初始服務：2 個 DP2/EP2 instances（共 4 張 active GPU，保留 migration headroom）
- Workload：24 個不同 prompt；每個 8192 input tokens、最多 2048 output tokens
- Preemption target：request 約生成 512 tokens；grace period 30 秒
- Logical capacity：`8 → 7 → 5 → 6 → 7 → 8`
- 正式 configuration：每組一次

F1：

1. `rerouting`
2. `reparallelization`
3. `original_spotserve`（MoE-aware reconfiguration 與 migration 都關閉）
4. `moe_spotserve`（兩者都開啟）

F2 是同一 protocol 下的 2×2 ablation：

| Configuration | MoE-aware Reparallelization | MoE-aware Migration |
| --- | --- | --- |
| Original SpotServe | off | off |
| Reparallelization only | on | off |
| Migration only | off | on |
| Full MoE-SpotServe | on | on |

F2 的 off/off 與 on/on 和 F1 的第 3、4 組完全相同，因此直接引用同一列結果；總計是 6
次唯一 formal GPU runs。報告以 p95 latency 為主要 F2 指標，同時輸出平均 latency、
throughput、KV restore、planner 與 EP readback 證據。

## 從 WSL 提交

先建立 8-GPU Ray session headers（會要求輸入 API token）：

```bash
cscc-ray-login
cscc-ray-use 8
```

確認 `cscc-ray-use 8` 顯示的 image 是：

```text
192.168.110.1:30003/p26000110/spotserve-moe-ray:2.58.0-vllm0.11.2-roce
```

提交正式 Job：

```bash
ray job submit \
  --no-wait \
  -- \
  bash -lc 'cd /home/ps2026004/ServerlessLLM-Spotserve && python -u deploy/cscc-ray/run_formal_f1_f2.py --model /home/ps2026004/models/Qwen1.5-MoE-A2.7B-Chat --output-dir results/cscc_ray_20261011/formal_f1_f2_single_01'
```

請保存 submit 後顯示的 `<job-id>`，再查看：

```bash
ray job status <job-id>
ray job logs <job-id>
```

這是一個長時間工作。若 WSL log WebSocket 中斷，先執行 `ray job status <job-id>`；不要因
WebSocket close 1006 就重送工作。只有狀態明確為 `FAILED` 才檢查 log 與 artifacts。

## 結果

平台 shared storage 上的主要檔案：

```text
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/formal_f1_f2_single_01/results.md
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/formal_f1_f2_single_01/state.json
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/formal_f1_f2_single_01/run-ledger.json
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/formal_f1_f2_single_01/formal-driver.log
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/formal_f1_f2_single_01/serverlessllm.log
```

成功條件是 `state.json` 與 `cscc-wrapper-state.json` 都為 `passed`，F1 四列與 F2 四列的
`Valid n` 都是 1。F2 兩列會標記為由 F1 reuse，這不是重複樣本。

## 解讀限制

- 每組只有一次，因此百分比是 point estimate，沒有信賴區間。
- 這是 logical trace：Ray workers 在 Job 期間仍被保留，只是 scheduler 在指定時間把 GPU
  標成 preempting/dead/ready；不能宣稱 K8s 實際刪除了 worker Pod。
- Cross-worker KV restore 使用已通過 canary 的 experimental NIXL direct API opt-in；正式
  stateful 組仍要求 restored blocks、attach receipt 與 acknowledgment，失敗就中止而不降級。
- distinct Ray Node ID 證明是不同 worker pods，不等於已證明在不同 physical hosts。
- 若某組 fail-closed，先用該組 log 找出第一個 evidence gate；不要直接重跑整個矩陣。
