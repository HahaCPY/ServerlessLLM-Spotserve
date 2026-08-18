# Four-version recovery comparison

這份報告記錄本次重新製作 trace 後的實際實驗。四個版本的定義如下：

- **No Recovery**：source 被 preempt 後 request 直接失敗。
- **Rerouting**：送到事先 READY 的完整 replica，不改 parallel config、不建立新 engine。
- **Reparallelization**：依剩餘 GPU 建立新的合法 vLLM parallel config，不搬 KV。
- **Modified/NIXL**：既有 target 透過 NIXL restore KV state，再繼續 request，不重建 target engine。

## 環境與 trace

- 主機：4 張 RTX 5070 Ti（每張約 16 GiB），GPU0–GPU3 全部列為可 add/remove 資源。
- 每個 vLLM worker 是獨立 container，但仍是同主機模擬，不是實體跨節點。
- Tiny：`/work/spotserve-models/Qwen2-MoE-Tiny`，source/target 為 TP1。
- Qwen：`/work/spotserve-models/Qwen1.5-MoE-A2.7B`（下文簡稱 Qwen2.7B/MoE），source/target 為 TP2。
- trace：`examples/spotserve/spot_trace_tiny_capacity_formula.jsonl` 與 `examples/spotserve/spot_trace_qwen15_moe_a27b_capacity_formula.jsonl`。
- trace 只使用 `add`、`remove`、`DONE`；事件由容量公式產生，維持模型最低可運作 GPU 數，且不會 add 超過 node-0 到 node-3。
- planner 在每次 add/remove 後重新計算，不只在初始化決定；Tiny 的實際 smoke 曾選出 TP1、TP2、TP2×DP2 等配置，Qwen 曾選出 TP2 與 TP2×DP2。

### Trace 公式

容量先由週期變化和 burst 組合產生：

```text
capacity(t) = clip(round(
    2 + 1.3*sin(2*pi*t/180s)
      + 0.7*sin(2*pi*t/70s)
      + burst(t)), minimum_capacity, 4)
```

`burst(t)` 在指定區間增加或減少容量，讓 trace 有連續 add、多 GPU remove、低容量區間與四 GPU 高峰，而不是固定一個 remove 接一個 add。Tiny 的 `minimum_capacity=1`，Qwen 的 `minimum_capacity=2`，因為 Qwen checkpoint 在這台機器上至少需要 TP2。每個時間點把公式要求的容量和目前 active set 做差，差集轉成同一筆 add/remove event；active set 會輪替 GPU0–GPU3，確保 source 可能被移除但仍保留合法 target，而且不會超過四張 GPU。

每個模型都執行 4 個版本 × context 64/240/480 × 每格 3 次，共 36 個 cells。下表是三次平均；No Recovery 的「通過」代表測試流程正常觀察到預期 failure，不代表 request 成功。

## Tiny 完整結果（36/36 cells passed）

Recovery Data 格式：`重算 tokens / 還原 blocks / target 產生 tokens；新 engine；placement 改變；target 事前 READY`。

| Context | Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---:|---|---:|---:|---:|---:|---|
| 64 | No Recovery | 21.351 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |
| 64 | Rerouting | 21.729 | 0.526 | 8.684 | 100% | 66.3 / 0 / 5；否；否；是 |
| 64 | Reparallelization | 162.098 | 0.504 | 9.719 | 100% | 66 / 0 / 5；是；是；否 |
| 64 | Modified/NIXL | 21.405 | 0.305 | 6.324 | 100% | 0 / 7 / 2；否；否；是 |
| 240 | No Recovery | 21.457 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |
| 240 | Rerouting | 21.670 | 0.493 | 9.914 | 100% | 243 / 0 / 5；否；否；是 |
| 240 | Reparallelization | 162.015 | 0.466 | 10.488 | 100% | 243 / 0 / 5；是；是；否 |
| 240 | Modified/NIXL | 21.415 | 0.301 | 6.401 | 100% | 0 / 18 / 2；否；否；是 |
| 480 | No Recovery | 21.414 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |
| 480 | Rerouting | 21.644 | 0.493 | 9.913 | 100% | 483 / 0 / 5；否；否；是 |
| 480 | Reparallelization | 162.550 | 0.472 | 10.323 | 100% | 483 / 0 / 5；是；是；否 |
| 480 | Modified/NIXL | 21.863 | 0.301 | 6.403 | 100% | 0 / 32 / 2；否；否；是 |

### Tiny 分析

1. No Recovery 在三個 context 都 0% 成功率，符合 failure baseline。
2. Rerouting 和 Reparallelization 都把完整 context 重新算一次，重算量約等於輸入長度；Reparallelization 的 recovery time 約 162 秒，主要是新 TP1 engine 啟動。
3. Modified/NIXL 的三個 context 都是 0 重算，分別還原約 7、18、32 blocks，且不建立新 engine；這是低 context migration cost 的主要證據。
4. P99 是本 harness 兩個觀察輸出時間的單請求 p99 proxy，不是 production workload 的統計 P99；throughput 也只用一次 continued output 計算，應解讀為保守比較值。

### Tiny 平均值總表

以下格式對應早期報告的摘要表，但數值改成這次每個 cell 三次重複的平均。Tiny 的 Reparallelization 是 TP1→TP1；舊表中的 TP1→TP2 是不同配置的早期試跑結果。

| Context | No Recovery | Rerouting | Reparallelization | Modified/NIXL |
|---:|---|---|---|---|
| 64 | request failed | continued；21.729 s；重算 66 | continued；162.098 s；TP1→TP1；重算 66 | continued；21.405 s；還原 7 blocks；重算 0 |
| 240 | request failed | continued；21.670 s；重算 243 | continued；162.015 s；TP1→TP1；重算 243 | continued；21.415 s；還原 18 blocks；重算 0 |
| 480 | request failed | continued；21.644 s；重算 483 | continued；162.550 s；TP1→TP1；重算 483 | continued；21.863 s；還原 32 blocks；重算 0 |

### Target 階段細節

這張表和早期報告中的 target-stage 表是同一個概念；早期數字是單次試跑，下面是本次三次平均。它只量 target 開始 generate 到第一個 output 的時間，Reparallelization 不包含新 engine 的啟動時間。

| Context | Rerouting target time | Reparallelization target time | Modified/NIXL target time |
|---:|---:|---:|---:|
| 64 | 0.526 s | 0.504 s（不含新 engine 啟動） | 0.305 s |
| 240 | 0.493 s | 0.466 s（不含新 engine 啟動） | 0.301 s |
| 480 | 0.493 s | 0.472 s（不含新 engine 啟動） | 0.301 s |

Rerouting 和 Modified/NIXL 的完整 `recovery_time` 看起來接近，原因是兩者都使用事先 READY 的 target；從 preemption 到結果的時間主要被 source pause、container/control socket、trace 協調等固定成本主導。在同主機的 Tiny 測試裡，NIXL 傳輸本身很快，沒有大到足以拉開完整 wall-clock time。真正的差異在 recovery data：Rerouting 要重算 66/243/483 tokens，而 Modified/NIXL 是 0 重算並還原 7/18/32 blocks；target-stage 時間也顯示 NIXL 約 0.30 s，比 Rerouting 約 0.49–0.53 s 快。

## Qwen2.7B/MoE 已完成部分

Qwen 使用 `/work/spotserve-models/Qwen1.5-MoE-A2.7B`，在本報告中簡稱 Qwen2.7B/MoE；source 是 GPU0+GPU1 的 TP2，target 是 GPU2+GPU3 的 TP2。

目前已完成 27/36 個預定 cell：context 64 和 240 的四版本各 3 次，以及 context 480 的 No Recovery 三次。下表只列已完成且 status=passed 的 cell 平均；No Recovery 的 0% 是它本來就應該呈現的 failure baseline。

Recovery Data 格式：`重算 tokens / 還原 blocks / target 產生 tokens；新 engine；placement 改變；target 事前 READY`。

| Context | Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---:|---|---:|---:|---:|---:|---|
| 64 | No Recovery | 22.734 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |
| 64 | Rerouting | 39.552 | 0.537 | 3.216 | 100% | 66 / 0 / 2；否；否；是 |
| 64 | Reparallelization | 197.212 | 0.570 | 3.095 | 100% | 66 / 0 / 2；是；是；否 |
| 64 | Modified/NIXL | 22.780 | 0.306 | 6.056 | 100% | 0 / 5 / 2；否；否；是 |
| 240 | No Recovery | 22.901 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |
| 240 | Rerouting | 28.721 | 0.512 | 3.384 | 100% | 242 / 0 / 2；否；否；是 |
| 240 | Reparallelization | 193.763 | 0.527 | 3.314 | 100% | 242 / 0 / 2；是；是；否 |
| 240 | Modified/NIXL | 30.551 | 0.359 | 5.219 | 100% | 0 / 16 / 2；否；否；是 |
| 480 | No Recovery | 132.301 | — | 0.000 | 0% | 0 / 0 / 0；否；否；否 |

### Qwen2.7B/MoE 平均值摘要

以下格式和 Tiny 摘要表一致。64、240 是各 cell 三次平均；480 目前只有 No Recovery 三次完成，因此其他三個版本保留為尚未完成。

| Context | No Recovery | Rerouting | Reparallelization | Modified/NIXL |
|---:|---|---|---|---|
| 64 | request failed | continued；39.552 s；重算 66 | continued；197.212 s；TP2→TP2；重算 66 | continued；22.780 s；還原 5 blocks；重算 0 |
| 240 | request failed | continued；28.721 s；重算 242 | continued；193.763 s；TP2→TP2；重算 242 | continued；30.551 s；還原 16 blocks；重算 0 |
| 480 | request failed | 尚未完成（GPU2 無法使用） | 尚未完成（GPU2 無法使用） | 尚未完成（GPU2 無法使用） |

已完成 Qwen context 的 target 階段平均：

| Context | Rerouting target time | Reparallelization target time | Modified/NIXL target time |
|---:|---:|---:|---:|
| 64 | 0.537 s | 0.570 s（不含新 engine 啟動） | 0.306 s |
| 240 | 0.512 s | 0.527 s（不含新 engine 啟動） | 0.359 s |
| 480 | 尚未完成 | 尚未完成 | 尚未完成 |

### Qwen 目前遇到的問題

Qwen 480 的 Rerouting 第 1、2 次在 target startup 等待 ready 時逾時，後續 480 的 Reparallelization、Modified/NIXL 尚未執行。診斷時確認 GPU2 被其他使用者 PID `2633059`（`/home/tmpp/Cangjie-llm/.venv/bin/python3`）佔用約 15.5 GiB，GPU3 則是空的；因此 TP2 target 無法使用 GPU2+GPU3。這個程序沒有被終止，本次自己的 source/target container 已清理。

Qwen 的 480 No Recovery 三次雖然完成，但第三次 engine startup 因共享 GPU/記憶體壓力延長到約 513 秒；這是 failure baseline 的觀察成本，不是 recovery 成功。待 GPU2 釋放後，應用同一份 Qwen trace 重跑缺失的 480 cell，才可以宣稱 Qwen 的完整 36-cell 結果。

## 限制

- 這是同一台主機上的多 container/GPU 模擬，不能宣稱真正跨節點 NIXL。
- planner/deployment smoke 會實際啟動與停止 vLLM，但它本身不等同於 in-flight request migration；完整 recovery matrix 的 Modified/NIXL 才執行實際 KV export/restore。
- 實驗結束後只清理由本次測試建立的 container/process，不碰其他使用者資源。

原始逐次結果保存在 `/tmp/four-version-tiny-capacity-full*.json` 與 `/tmp/four-version-qwen-capacity-full*.json`。
