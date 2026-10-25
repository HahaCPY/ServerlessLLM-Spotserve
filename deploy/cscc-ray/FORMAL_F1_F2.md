# CSCC Ray 正式 F1/F2（單次）

## 固定設計

- Model：`/home/ps2026004/models/Qwen1.5-MoE-A2.7B-Chat`
- GPU pool：8 個 Ray GPU workers；每個 worker 1 張 RTX 5090
- 初始服務：2 個 DP2/EP2 instances（共 4 張 active GPU，保留 migration headroom）
- Workload：24 個不同 prompt；每個 8192 input tokens、最多 2048 output tokens
- Preemption target：request 約生成 512 tokens；grace period 30 秒
- Logical capacity：`8 → 7 → 8 → 6 → 7 → 8`
- Trace timing：570 秒單 GPU notice、690 秒恢復；1170 秒雙 GPU notice、1260/1320 秒逐步恢復
- 正式 configuration：每組一次
- Single-pass 實際執行 6 個 unique cells；F1/F2 共用的 off/off 與 on/on 不重跑，六組以
  固定 seed 一起隨機排序，避免 F2-only configurations 永遠最後執行

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

此 protocol 使用 CSCC 的 gang session。`X-CSCC-Gang` 只在建立 Ray session 時讀取；若
Ray 分頁仍有舊 session，請先按「結束工作階段」，等它完全消失後再提交。不可搶佔的
`rtx5090-queue`、8-GPU quota 與 image 必須都已核准。

先建立 8-GPU Ray session headers（會要求輸入 API token）：

```bash
cscc-ray-login
cscc-ray-use 8
```

保留既有 token、queue 與 image，加入 gang header：

```bash
export RAY_JOB_HEADERS="$(
python3 - <<'PY'
import json
import os

headers = json.loads(os.environ["RAY_JOB_HEADERS"])
headers["X-CSCC-GPUs"] = "8"
headers["X-CSCC-Gang"] = "true"
print(json.dumps(headers, separators=(",", ":")))
PY
)"
```

不要印出 Authorization token；只檢查非敏感欄位：

```bash
python3 - <<'PY'
import json
import os

headers = json.loads(os.environ["RAY_JOB_HEADERS"])
for key in ("X-CSCC-GPUs", "X-CSCC-Queue", "X-CSCC-Image", "X-CSCC-Gang"):
    print(f"{key}: {headers.get(key)}")
PY
```

確認 `cscc-ray-use 8` 顯示的 image 是：

```text
192.168.110.1:30003/p26000110/spotserve-moe-ray:2.58.0-vllm0.11.2-roce
```

提交正式 Job：

```bash
ray job submit \
  --no-wait \
  --runtime-env-json '{"env_vars":{"N_GPUS":"8","WAIT_SECONDS":"0"}}' \
  -- \
  bash -lc 'cd /home/ps2026004/ServerlessLLM-Spotserve && python -u deploy/cscc-ray/run_formal_f1_f2.py --model /home/ps2026004/models/Qwen1.5-MoE-A2.7B-Chat --output-dir results/cscc_ray_20261024/formal_f1_f2_single_01'
```

`X-CSCC-Gang: true` 讓平台在 8 張 GPU 尚未同時可用時一張都不啟動；8 張到齊後，程式內
的 8-bundle placement group 才會 ready 並開始 preflight/profile/formal runs。
`WAIT_SECONDS=0` 表示 placement group 不設使用者端 timeout；它會等待到 ready，或直到你
執行 `ray job stop <job-id>`／在 Ray 分頁結束 session。平台本身的 project、queue 與最長
24 小時 session 上限仍然有效，程式無法把平台期限改成真正無限。第一次建立
session 若在兩分鐘內未完成，CLI 可能回 `503`；只有在**沒有拿到 job ID**時才依
`Retry-After` 重送相同 submit。拿到 job ID 後不可重送。

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
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261024/formal_f1_f2_single_01/results.md
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261024/formal_f1_f2_single_01/state.json
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261024/formal_f1_f2_single_01/run-ledger.json
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261024/formal_f1_f2_single_01/formal-driver.log
/home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261024/formal_f1_f2_single_01/serverlessllm.log
```

成功條件是 `state.json` 與 `cscc-wrapper-state.json` 都為 `passed`，F1 四列與 F2 四列的
`Valid n` 都是 1。F2 兩列會標記為由 F1 reuse，這不是重複樣本。

## 解讀限制

- 每組只有一次，因此百分比是 point estimate，沒有信賴區間。
- 這是 logical trace：Ray workers 在 Job 期間仍被保留，只是 scheduler 在指定時間把 GPU
  標成 preempting/dead/ready；不能宣稱 K8s 實際刪除了 worker Pod。
- 第一輪 recovery 與第二輪雙 GPU churn 已分開，避免第二個 notice 在第一輪 grace deadline
  後僅 30 秒便發生。
- Cross-worker KV restore 使用已通過 canary 的 experimental NIXL direct API opt-in；正式
  stateful 組仍要求 raw-response restored blocks、native completion receipt、零 fallback，
  並確認 source 只在 receipt 後且 grace deadline 前釋放；失敗就中止而不降級。此 receipt
  是完成 request 的證據，不是 early connector transfer ACK。
- distinct Ray Node ID 證明是不同 worker pods，不等於已證明在不同 physical hosts。
- 若某組 fail-closed，先用該組 log 找出第一個 evidence gate；不要直接重跑整個矩陣。
