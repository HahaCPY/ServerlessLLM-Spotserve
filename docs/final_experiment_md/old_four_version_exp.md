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

這張 target-stage 表之所以在 64、240、480 看起來接近，是因為它只從 target 已 READY（Modified/NIXL 則已完成 restore）開始量到第一個 output。Tiny 模型在 GPU 上處理 64–480 token 的 prefill 本身不到主要時間，container/control socket、scheduler、第一次 output 等固定成本和量測抖動反而佔主導；所以 Rerouting 只從 0.526 降到 0.493 秒，Modified/NIXL 只從 0.305 降到 0.301 秒，不能解讀成三種 context 的計算量完全一樣。Reparallelization 欄位還刻意排除了新 engine 啟動，因此它只代表 target ready 後的階段，不代表完整 recovery。

這也解釋了為什麼少數 Modified/NIXL 的總時間反而比較長：`recovery_time` 還包含 NIXL export、abort、restore 和 container 協調；Rerouting 則直接把完整 token list 交給既有 target。兩者都在同一台主機、同一組 container 上執行，NIXL 傳輸距離很短，所以省下的重算時間被控制成本和量測抖動抵消。這次 harness 只等待 2–5 個 continued output token，也不足以讓長時間生成成本充分放大。

### 為什麼 240/480 context 時總時間仍然接近

這不是單純代表 NIXL 沒有作用，而是目前的總時間可以近似寫成：

```text
recovery time
  = source/container 控制成本
  + NIXL transfer 或 context 重算
  + target 第一個 output
```

Rerouting 和 Modified/NIXL 都使用事先 READY 的 target，因此兩者都要付出 source pause、container stop、control socket、request state 切換等固定成本；這些成本約 20–30 秒，會蓋過 240 或 480 token prefill 的差距。對目前 GPU 而言，480 token 仍然是偏小的 context，target-stage 只有約 0.49 秒（Rerouting）和 0.30 秒（Modified/NIXL），相較於整體控制成本很難拉開 wall-clock 差距。

此外，目前 Rerouting 是在同一台主機上透過 control socket 交給 target 的 token list，沒有完整計入跨節點網路、頻寬限制或遠端 context 傳輸成本，因此 Rerouting 的成本可能被低估。harness 也只等待 2–5 個 continued output token；雖然 worker 可以生成較長輸出，但這次主要量測的是 request 能否接回與第一個 output，而不是長時間生成吞吐量。

因此 Qwen 240 出現 Modified/NIXL 30.551 秒、略高於 Rerouting 28.721 秒並不矛盾：Modified 額外包含 export、abort、NIXL transfer、restore，短 context 和同主機環境下，這些協調成本可能暫時大於省下的重算時間；但 target-stage 仍是 Modified/NIXL 0.359 秒、Rerouting 0.512 秒，且重算量是 `0` 對 `242` tokens。

目前結果應解讀為：Modified/NIXL 已證明能成功還原 KV、把重算 token 從 `66/242/483` 降到 `0`，但不能僅用這個短 context、短輸出的總 wall-clock 宣稱一定比 Rerouting 快。要把差距放大，下一版應使用 4K/8K context、128–256 個 continued output token、多個同時執行的 long requests，並分段記錄 preemption、source stop、export、NIXL transfer、restore、prefill 和 first-token；最好再加入真正跨節點或限速網路。

### 為什麼 Modified/NIXL 實驗有固定 target

這是 recovery harness 的刻意設計，不是 planner 不能動態重規劃。四版本實驗要單獨比較「KV migration」和「重新部署」的差異，因此 Modified/Rerouting 都先建立 READY target；preemption 時只改變 source 的可用性。若同時讓 planner 重新選 plan，就會把 planner、engine startup、KV restore 三種成本混在一起。

另外有獨立的 dynamic planner/deployment smoke：每次 trace 的 add/remove 後都重新呼叫 planner，並由 vLLM deployment adapter 實際套用新 ParallelPlan；Tiny 曾套用 TP1、TP2、TP2×DP2，Qwen 曾套用 TP2、TP2×DP2。因此 planner 的動態重規劃路徑本身有驗證。

目前尚未完成的是「planner 在 in-flight Modified/NIXL request 中重新選 target、建立相容 engine，再把同一份 KV state restore 過去」的整合測試。現有四版本 Modified 只驗證既有相容 target 的 NIXL restore；若 target 本身也被移除，應由 router 先重新規劃，再判斷新 target 是否支援相容的 state restore，否則只能 fallback 或重新計算。

### 這個實驗的目的與有效範圍

固定 target 是一個控制變因（controlled ablation）：它回答的是「在 source 被 preempt、且有相容 READY target 的前提下，KV restore 是否能避免完整 context 重算」。因此它可以有效比較 Rerouting、Reparallelization 和 Modified/NIXL 的 recovery data、target latency 與 engine startup 成本，但不能單獨代表 target 也被 preempt、planner 重新選 plan、再接上 NIXL 的完整 production 流程。

如果把這張表解讀成整個 SpotServe 系統的 end-to-end resilience，確實會失真；正確做法是把它標成 controlled recovery benchmark，並另外做 dynamic planner + target loss + compatible NIXL restore 的端到端實驗。兩者合在一起，才能同時回答「NIXL migration 本身是否有效」和「系統遇到實際容量變動時能否自動重規劃並完成 migration」。

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
- Tiny 4K smoke 曾嘗試啟動，但 `/work/spotserve-models/Qwen2-MoE-Tiny/config.json` 的 `max_position_embeddings=512`，模型架構不支援 4K/8K context；兩次 source initialization 都在 recovery 前退出，因此沒有產生 4K 結果。若要做真正 4K/8K，需換用支援該 context 的 checkpoint（例如 Qwen2.7B 類模型）並重新跑矩陣。
- 實驗結束後只清理由本次測試建立的 container/process，不碰其他使用者資源。

原始逐次結果保存在 `/tmp/four-version-tiny-capacity-full*.json` 與 `/tmp/four-version-qwen-capacity-full*.json`。

## Qwen2.7B/MoE 長 context 追加實驗：4K 與 8K budget

為了觀察 context 變長後三種 recovery strategy 的差異，另外使用同一個 Qwen checkpoint 和同一份 trace 執行長 context 實驗。這一組依照使用者指定，只比較 Rerouting、Reparallelization 和 Modified/NIXL；每個 context、每個 mode 各執行 2 次，表格數值是兩次的平均。

### 實驗設定

- Model：`/work/spotserve-models/Qwen1.5-MoE-A2.7B`（本報告簡稱 Qwen2.7B/MoE）。
- Trace：`examples/spotserve/spot_trace_qwen15_moe_a27b_capacity_formula.jsonl`，共 16 個 trace events；兩個 context 使用同一份 trace。
- Source：GPU0+GPU1、TP2；target：GPU2+GPU3、TP2。
- 4K：prompt 4096 tokens，固定產生 256 tokens。
- 8K budget：prompt 7936 tokens，固定產生 256 tokens，總序列長度 8192；因模型 `max_position_embeddings=8192`，不能再使用完整 8192-token prompt 後額外產生 output。
- 每個 mode 重複 2 次；四個 GPU 都納入可用資源範圍。
- 這仍是同一主機上的 multi-container 模擬，不是實體跨節點 NIXL。

### 兩次平均結果

Recovery Data 格式為：`重算 tokens / 還原 blocks / target 產生 tokens；新 engine；placement 改變；target 事前 READY`。表中的 Success Rate 是兩次 run 的平均成功率。

| Context | Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---:|---|---:|---:|---:|---:|---|
| 4K | Rerouting | 21.485 | 0.252 | 6.384 | 100% | 4098 / 0 / 2；否；否；是 |
| 4K | Reparallelization | 191.288 | 0.273 | 6.028 | 100% | 4098 / 0 / 2；是；是；否 |
| 4K | Modified/NIXL | 24.582 | 1.877 | 1.054 | 100% | 0 / 257 / 2；否；否；是 |
| 8K budget | Rerouting | 22.288 | 0.964 | 2.026 | 100% | 7938 / 0 / 2；否；否；是 |
| 8K budget | Reparallelization | 191.712 | 0.994 | 1.977 | 100% | 7938 / 0 / 2；是；是；否 |
| 8K budget | Modified/NIXL | 26.563 | 3.165 | 0.632 | 100% | 0 / 497 / 2；否；否；是 |

### 每次 run 的原始數值

下表保留兩次執行的主要量測，方便核對平均值不是單次結果。兩次 run 的 Success Rate 都是 100%。

| Context | Version | Run 1：Recovery / P99 / Throughput | Run 2：Recovery / P99 / Throughput |
|---:|---|---|---|
| 4K | Rerouting | 21.509 s / 0.247 s / 6.443 | 21.460 s / 0.257 s / 6.325 |
| 4K | Reparallelization | 191.250 s / 0.283 s / 5.882 | 191.326 s / 0.263 s / 6.175 |
| 4K | Modified/NIXL | 24.791 s / 1.845 s / 1.070 | 24.373 s / 1.909 s / 1.037 |
| 8K budget | Rerouting | 22.258 s / 0.964 s / 2.027 | 22.317 s / 0.965 s / 2.025 |
| 8K budget | Reparallelization | 191.455 s / 0.981 s / 1.990 | 191.969 s / 1.007 s / 1.963 |
| 8K budget | Modified/NIXL | 26.996 s / 2.921 s / 0.680 | 26.131 s / 3.409 s / 0.584 |

### Target 階段平均

Target Recovery 是 target 開始接手 request 到第一個 output 的時間；Reparallelization 的完整 engine startup 已包含在 Recovery Time，但 target 階段欄位不重複計入它。

| Context | Rerouting target time (s) | Reparallelization target time (s) | Modified/NIXL target time (s) |
|---:|---:|---:|---:|
| 4K | 0.252 | 0.273（不含新 engine 啟動） | 1.877 |
| 8K budget | 0.964 | 0.994（不含新 engine 啟動） | 3.165 |

### 追加實驗分析

1. **KV migration 的差異在長 context 下更清楚。** 4K 時 Rerouting/Reparallelization 都必須重算 4098 tokens；8K budget 時重算量增加到 7938 tokens。Modified/NIXL 兩組都是重算 0，還原 blocks 則從 257 增加到 497，表示 context 變長時確實搬移了更多 KV state，而不是改用 token replay。
2. **Reparallelization 的主要成本是新 engine 啟動。** 它兩個 context 的完整 recovery time 都約 191 秒，遠高於另外兩種；這是因為它會建立新的 GPU2+GPU3 TP2 engine，且 `engine_created=true`、`placement_changed=true`。
3. **Rerouting 的完整時間仍低於 Modified/NIXL，但不能因此判定它更有效率。** 本機 controlled harness 中，Rerouting 直接把 token list 交給事前 READY 的 target；Modified/NIXL 需要額外執行 export、abort、NIXL transfer、restore。這些固定協調成本和本機量測抖動，會讓 4K/8K 的 wall-clock 差距不一定反映重算量差距。真正可見的優勢是 Modified/NIXL 避免了 4098/7938 tokens 的重算。
4. **8K 的 target-stage latency 上升。** Rerouting 從 0.252 s 上升到 0.964 s，Modified/NIXL 從 1.877 s 上升到 3.165 s；這符合較長 prompt 的 prefill/restore 壓力。不過本次每次只觀察 2 個 continued output tokens，throughput 與 P99 仍是 recovery harness 的單 request proxy，不是 production workload 的長時間 P99。
5. **planner 的範圍要分開解讀。** 這個長 context 四版本實驗是 controlled recovery benchmark，Rerouting/Modified 使用事先 READY 的相容 target，沒有把 planner 動態重規劃混進 recovery time。獨立的 planner/deployment smoke 會在 trace add/remove 後重新選 `ParallelPlan` 並套用配置；但「in-flight request 中 planner 改 plan，再對新 target 做相容 NIXL restore」仍是下一個整合測試，不應由本表宣稱已完成。

### 結論與限制

這次 4K 與 8K budget 實驗的 6/6 cells、共 12 次 run 全部 `status=passed`，三種模式的 Success Rate 都是 100%。結果證明 Modified/NIXL 在長 context 下能成功還原更多 KV blocks 並保持 0 token recomputation；但因為目前是同機、事前 READY target、短 continued output 的 controlled harness，不能直接把 24–27 秒的 Modified/NIXL recovery time 當成跨節點 production latency，也不能把它解讀成 planner 已在 request 中途動態切換部署。

原始結果檔：`/tmp/qwen-four-version-4k.json`、`/tmp/qwen-four-version-8k.json`，以及各 mode 的 `.p4096.r*.json` / `.p7936.r*.json`。

## Trace-driven live preemption 追加驗證

前面的 4K/8K controlled matrix 是先讓 source 在第一次 output 後 pause，再套用 trace；因此它不是由容量事件自然觸發的 live preemption，而且 Rerouting 當時有事前 READY 的 target。為避免這兩個偏差，新增了 trace-driven 模式：source 先持續生成，直到 trace 的 source `remove` 事件才抓 live metadata 並執行 recovery。

### 新 trace

新增兩份不超過四張 GPU 的容量 trace：

- Tiny：`examples/spotserve/spot_trace_tiny_realtime_capacity.jsonl`
- Qwen2.7B/MoE：`examples/spotserve/spot_trace_qwen15_moe_a27b_realtime_capacity.jsonl`

兩份 trace 都使用「最低容量起步 → 分段 add 到滿容量 → 壓力下降時 remove source → 交錯 add/remove 恢復」的公式。Qwen 的最低容量是 2 GPU，Tiny 的最低容量是 1 GPU；事件永遠只使用 `node-0` 到 `node-3`，不會超過本機四張 GPU。以 Qwen 為例，容量狀態是：

```text
2 GPU (source TP2)
  -> add 到 3 GPU
  -> add 到 4 GPU
  -> remove node-0,node-1，剩下 GPU2+GPU3
  -> 後續交錯 add/remove，模擬容量恢復與再次縮減
```

### Rerouting baseline 的修正

新的 Rerouting 不再預先啟動 backup engine。source 被 remove 後，才在剩餘 GPU 上建立同樣 TP 的 target，再把完整 context 交給 target 重算。這使 Rerouting 和 Reparallelization 都包含 target engine startup，不會因預熱 replica 而得到不合理的優勢；差異只剩是否改變 parallel configuration，以及 Modified 是否保留 KV state。

### Modified 的完整流程

`--dynamic-planner` 開啟時，Modified 在 source remove 事件後會：

1. 以 trace 的 active GPU 建立 worker-node snapshot。
2. 呼叫 `plan_dynamic_reparallelization()` 選出新的 `ParallelPlan`。
3. 依 selected TP/DP/target GPU 建立實際 target replicas。
4. 在 source request 仍 live 時 export KV state。
5. target READY 後 abort source、restore KV，並繼續原 request。

因此這條路徑同時驗證 planner configuration application 與 KV migration；如果 planner 選出 DP2，harness 會真的建立兩個 target replicas，而不是只把 DP2 寫進 log。

### Tiny trace-driven smoke 結果

以下三次使用同一份 `spot_trace_tiny_realtime_capacity.jsonl`、`--preempt-after-new-tokens=0`（由 trace remove 觸發）、256-token generation、四張 GPU。三個 request 都在 trace remove 時仍屬於 live request；這一輪 source 在 prefill/live window 被移除，並非固定生成一半後才移除。

| Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---|---:|---:|---:|---:|---|
| Rerouting（無 backup） | 163.464 | 0.499 | 32.732 | 100% | 重算 64 / 還原 0 / 產生 18；新 engine；placement 改變；target 非事前 READY |
| Reparallelization | 163.055 | 0.498 | 32.732 | 100% | 重算 64 / 還原 0 / 產生 18；新 engine；placement 改變；target 非事前 READY |
| Modified/NIXL + dynamic planner | 170.947 | 0.320 | 32.309 | 100% | 重算 0 / 還原 5 / 產生 12；新 engine；placement 改變；planner 選 TP1×DP2、建立 2 replicas |

這一輪最重要的不是宣稱 Modified wall-clock 一定最短，而是確認：Rerouting 已移除 backup 優化；Modified 的 planner 選擇有實際套用；Modified 保持 0 token recomputation 並成功 restore KV。Modified 比兩個 baseline 多約 8 秒，是因為它在 preempt 後要建立 planner 選出的兩個 replicas，還要支付 export/abort/restore；這是把完整功能納入後的真實成本，不應再拿事前 READY target 的舊結果比較。

### 解讀與下一步

- trace-driven 模式已解決「先 pause、再跑 trace」的時序問題；preemption 現在由容量 remove event 觸發。
- 這次 Tiny 的 remove 時刻落在 source prefill/live window，`source_generated_before_preemption=0`；因此不能把它稱為「生成一半 token」實驗。
- 若要放大三者的效能差距，下一輪應使用 Qwen2.7B/MoE 的 4K/8K context、較長 output，並把 trace speedup 校準到 request 已有數十到數百 decode tokens 但尚未完成的區間。
- 這仍是同主機 multi-container；尚未代表真正跨節點 NIXL 網路成本。

原始 trace-driven 結果：`/tmp/modified-dynamic-tiny-trace-driven-dp2.json`、`/tmp/rerouting-no-backup-tiny-trace-driven.json`、`/tmp/reparallelization-tiny-trace-driven.json`。

## 正式四版本長 request batch 實驗設計

前面的結果已確認單一 request 的 NIXL restore、planner 套用與 no-backup Rerouting 路徑，但單一短 request 會讓固定的 container/planner 成本蓋過 KV migration 的優勢。下一輪正式實驗改成同一個 preemption event 同時遷移多個長 request，讓四個版本在相同容量變化下比較整批 recovery 效能。

### 實驗目的

回答以下問題：

1. source 被移除時，四種 recovery policy 能否讓整批 request 完成或明確失敗？
2. context 與 request 數量增加時，Modified/NIXL 是否能用 KV restore 避免大量重算？
3. planner 動態選出的 target configuration 是否真的被建立並接上 KV migration？
4. 在不預熱 backup engine 的公平條件下，Modified 的 batch recovery time、P99 與 throughput 是否優於另外兩個可恢復 baseline？

### 固定硬體與 workload

- Model：`/work/spotserve-models/Qwen1.5-MoE-A2.7B`（Qwen2.7B/MoE）。
- Source：GPU0+GPU1、TP2；preemption 後 target 資源為 GPU2+GPU3、TP2。
- GPU 範圍固定為 `0 1 2 3`，最多四張卡；不使用其他 GPU。
- Trace：`examples/spotserve/spot_trace_qwen15_moe_a27b_batch_recovery.jsonl`。
- Trace 先由 2 GPU 擴張到 4 GPU，再 remove `node-0,node-1`；之後保留 add/remove 事件模擬容量恢復與再次縮減。
- Request batch：4、8 個同時執行的 request，各 request 使用相同 context 長度和 output budget，但 prompt 內容用 deterministic index 區分。
- Context 組合：4096 tokens、7168 tokens；模型的 8192 上限下，7168 組合固定最多 512 output tokens。
- Output 組合：512、1024 tokens；總序列長度必須小於 `max_model_len=8192`。
- 每個 cell 至少重複 5 次，報告平均、標準差與 P95/P99。

### 四個版本的嚴格定義

| Version | Preemption 後動作 | Target startup | Context/KV 行為 |
|---|---|---|---|
| No Recovery | source 停止，request failure | 無 | 不 retry、不 reroute、不 replay |
| Rerouting（no backup） | preempt 後建立同 TP 的完整 target | 包含 | 重新計算每個 request 的完整 context |
| Reparallelization | preempt 後建立 generic planner/legal TP/DP configuration | 包含 | 重新計算每個 request 的完整 context |
| Modified/NIXL | planner 重新選相容 configuration，建立 target replicas，再 restore | 包含 | export/abort/NIXL restore，重算 tokens 應為 0 |

Rerouting 不預先建立 READY backup engine；否則會把 target startup 從它的 recovery time 移除，造成不公平優勢。Modified 也不使用預熱 target：target 必須由 preemption 後的 planner/deployment path 建立，這樣三個可恢復版本都承擔相同的 engine startup 類型成本。

### 執行流程

```text
1. 啟動 source TP2
2. 送入 4/8 個長 request，確認都已在 source 建立 live KV metadata
3. 依 batch trace 推進 add/remove capacity event
4. remove node-0,node-1，觸發 source preemption
5. No Recovery：停止 source，記錄 batch failure
6. Rerouting：建立同 TP target，完整重算所有 context
7. Reparallelization：建立 generic selected plan，完整重算所有 context
8. Modified：呼叫 planner，套用 selected ParallelPlan，建立 target replicas
9. source export；target READY 後 abort source 並逐 request NIXL restore
10. 等待整批 request 完成，核對每個完整 token sequence
11. 清理本次 container/network，進行下一次 repeat
```

Trace-driven 模式使用 `preempt-after-new-tokens=0`，不把 preemption 綁死在某個固定 output token。實驗前先用短校準 run 選擇 `trace-speedup`，要求 remove 事件發生時 request 已有 live KV、但仍未完成；校準值必須記錄在結果中，不能每個版本分別調整。

### 必須記錄的指標

| 指標 | 定義 |
|---|---|
| Recovery Time | 從 source remove 到整批 request 都完成 recovery 的 wall-clock time，包含 target startup、planner、NIXL 與重算 |
| Recovery P99 | 每個 request 從 preemption 到第一個 continued output 的 P99 |
| Time-to-all-complete | 最後一個 request 完成的時間 |
| Effective Throughput | 整批 continued output tokens / recovery wall-clock time |
| Success Rate | 成功完成 request 數 / batch request 數 |
| Recovery Data | 每個 request 的 recomputed tokens、restored blocks、target generated tokens、engine_created、placement_changed、target_preexisting |
| Sequence Equality | target 完整 sequence 與未 preempt deterministic reference 是否一致 |
| Planner/transfer breakdown | planner decision、target startup、export、abort、NIXL transfer、restore、first output 各階段時間 |

### 結果表格範本

每個 context/request-count cell 產生一張平均表，表中的五個核心欄位固定如下：

| Context | Batch | No Recovery | Rerouting（no backup） | Reparallelization | Modified/NIXL |
|---:|---:|---|---|---|---|
| 4096 | 4 | failure；Recovery/P99/Throughput/Success/Recovery Data | 待執行 | 待執行 | 待執行 |
| 4096 | 8 | failure；Recovery/P99/Throughput/Success/Recovery Data | 待執行 | 待執行 | 待執行 |
| 7168 | 4 | failure；Recovery/P99/Throughput/Success/Recovery Data | 待執行 | 待執行 | 待執行 |
| 7168 | 8 | failure；Recovery/P99/Throughput/Success/Recovery Data | 待執行 | 待執行 | 待執行 |

詳細表格欄位為：`Recovery Time`、`Recovery P99`、`Time-to-all-complete`、`Effective Throughput`、`Success Rate`、`Recovery Data`、`Sequence Equality`。每一格都必須由相同 repeat 數的實際 run 平均得到，不能以單次成功結果代替。

### 預期可觀察的差異與公平性

這個 workload 讓固定 startup/planner 成本由 4/8 個 request 分攤，而 Rerouting/Reparallelization 的重算成本會隨 `batch_size × context_length` 增長。若 Modified/NIXL 實作正確，應觀察到：

- Modified 的 `recomputed_tokens=0`，且 `restore_fallbacks=0`。
- context 或 batch 增加時，Rerouting/Reparallelization 的 recovery tail 與重算量明顯增加。
- Modified 的 batch `Time-to-all-complete` 和 `Effective Throughput` 優於兩個可恢復 baseline；這個優勢來自實際省下的重算，不是預先啟動 target 或縮短量測區間。
- No Recovery 維持 0% request success，作為 failure baseline。

若 Modified 在這個長 batch workload 仍未勝出，應記錄為真實結果，並拆解是 target startup、NIXL bandwidth、planner decision 或 GPU memory pressure 造成；不能藉由刪除 startup 或只報 target-stage 數值來製造優勢。

### 尚未執行的部分

本節是正式實驗 protocol 與結果表格範本；上表的 `待執行` 不代表已完成。需要先把現有單 request harness 擴充為 batch driver（`max_num_seqs=4/8`、每個 request 的 state export/restore 與 sequence comparison），再依同一份 batch trace 執行四個版本和所有 repeats。完成後才能將本節的待執行欄位換成實測平均值。

## 實際執行一次：Tiny 四版本 controlled pilot

為了先確認四版本流程在目前共享 GPU 環境可完整執行，先跑一個單 request、四版本各一次的 controlled pilot。這不是上面正式的 4/8-request Qwen batch 終版，而是先驗證 recovery 流程和指標是否能從真實 container/vLLM/NIXL 執行得到。

### 執行設定

- Model：`/work/spotserve-models/Qwen2-MoE-Tiny`。
- Prompt：240 tokens；`max_model_len=512`；最多產生 128 tokens。
- GPU：source 使用 GPU0；trace target 資源使用 GPU1、GPU3；GPU2 刻意不使用，因為當時被其他使用者佔用。
- Trace：`examples/spotserve/spot_trace_tiny_pilot_gpu013.jsonl`。
- 四個版本都在第一個 generated token 後 pause/preempt（`--preempt-after-new-tokens=1`），確保 source request 在 recovery 前仍是 live。
- 每一個 cell 都建立自己的 source/target container；Rerouting 不預熱 backup，target startup 計入 Recovery Time。
- Modified/NIXL 開啟 dynamic planner，讓 planner 在 remove event 後重新選 plan 並實際建立 target replicas。

### 實測結果

| Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---|---:|---:|---:|---:|---|
| No Recovery | 21.520 | — | 0.000 | 0% | 重算 0 / 還原 0 / 產生 0；無新 engine；placement 未改變；無 target |
| Rerouting（no backup） | 162.096 | 0.465 | 34.806 | 100% | 重算 263 / 還原 0 / 產生 18；新 engine；placement 改變；target 非事前 READY |
| Reparallelization | 162.897 | 0.478 | 34.063 | 100% | 重算 263 / 還原 0 / 產生 18；新 engine；placement 改變；target 非事前 READY |
| Modified/NIXL | 170.659 | 0.323 | 32.052 | 100% | 重算 0 / 還原 17 / 產生 12；新 engine；placement 改變；target 非事前 READY |

Modified 的 planner/部署細節如下：

```text
active target GPUs: [1, 3]
selected plan: TP1 × DP2
target groups: [[1], [3]]
configuration_applied: true
restore_success: true
recomputed_tokens: 0
restored_blocks: 17
```

### 這次結果怎麼解讀

1. **四個版本都真的走過 container recovery path。** No Recovery 按定義 failure；另外三個版本都成功接回 request，Success Rate 是 100%。
2. **Modified/NIXL 的核心優勢是資料量，不是這次的總 wall-clock。** Rerouting/Reparallelization 都重算 263 tokens；Modified 重算 0、還原 17 blocks。這證明 KV state 有實際從 source 接到 target。
3. **Modified 的 P99 較低，但完整 Recovery Time 約多 8–9 秒。** 這一輪 Modified 額外支付 planner、兩個 DP target startup、export/abort/restore；Tiny context 太短，省下的重算時間不足以抵銷固定控制成本。
4. **Rerouting 和 Reparallelization 接近是合理的。** Tiny 的 planner/legal configuration 在剩餘資源上最後都是單 GPU TP1，兩者都建立一個同形狀 target 並重算，所以差異主要只剩少量控制開銷。
5. **這不是 production claim。** 它是單 request、短 context、同主機 multi-container pilot；尚未證明 4/8 長 request batch 下 Modified 的總時間一定勝出，也不是跨節點網路測試。

### Qwen2.7B pilot 的阻塞紀錄

同一輪也嘗試以 Qwen2.7B/MoE、4096-token prompt 跑四版本，但 source container 在 `ready` handshake 前 exit code 1，尚未產生有效結果。診斷確認當時 GPU2 被其他使用者的 `/home/test/Cangjie-llm/.venv/bin/python3` 佔用約 14.5 GiB；Qwen 的 TP2 target 需要 GPU2+GPU3，因此不能在不影響他人的情況下啟動公平的 target。這次沒有填入 Qwen 偽數據，待 GPU2 釋放後再用正式 batch protocol 執行。

另外，Tiny 的第一次 trace-driven 嘗試在 source prefill/live 邊界觸發 remove，metadata 讀到 prompt 但 export 時 request 已離開 EngineCore live list，結果為 `request_not_active`。因此本 pilot 使用相同 preemption token 的明確 live barrier；該 race 會保留為下一輪 trace-driven harness 修正項目。

原始結果：`/tmp/tiny-four-version-controlled-pilot.json`，逐 cell log/report 為 `/tmp/tiny-four-version-controlled-pilot.*.p240.r1.{json,log}`。

## 為了明顯呈現 Modified 優勢而重新設計的正式比較

Tiny pilot 的結論只能是「功能成功」，不能用來展示 Modified 的效能優勢。原因是 Tiny 只有 512-token context，而且 Modified 這次建立了兩個 DP target；固定的 engine startup 成本大於省下的重算成本。下一次正式比較不再使用 Tiny 或單 request。

### 必須改變的 workload

- 使用 Qwen1.5-MoE-A2.7B，source/target 都是 TP2。
- 同一個 preemption event 同時影響 8 個長 request。
- Context 使用 4096 和 7168 tokens，output budget 使用 512 tokens。
- 每個 request 在 preempt 前先產生 32 個 decode tokens，確保是真正的 in-flight request，而不是 prefill race。
- Rerouting、Reparallelization、Modified 都在 preempt 後才建立 target；三者都把 target startup 算入 Recovery Time。
- Rerouting 不預熱 backup；Modified 在只有 GPU2+GPU3 的 target 容量下只能選 TP2×DP1，避免 Modified 因額外 DP replica 被不公平加重 startup。
- 每一組至少重複 5 次，四個版本使用完全相同的 trace、prompt、preemption 時間與 GPU 配置。

### 預期可直接看到的差異

以 8 requests、4096 context 為例，Rerouting/Reparallelization 至少要重算約 `8 × 4096 = 32,768` 個 prompt tokens；7168 context 則約 `57,344` 個。Modified 應該是：

```text
recomputed_tokens = 0
restored_blocks > 0
restore_fallbacks = 0
```

因為三個可恢復版本都承擔相同的 target engine startup，差異主要會落在「整批 context prefill 重算」與「NIXL KV restore」；這比 Tiny 單 request 更能放大 Modified 的優勢。報告必須同時列出 Recovery Time、P99、Effective Throughput、Success Rate、Recovery Data、Time-to-all-complete 及 stage breakdown，不能只報 target-stage latency。

### 目前阻塞

這組 Qwen 正式比較需要同時使用 GPU0–3，且 preempt 後 target 需要 GPU2+GPU3。現在 GPU2 仍被其他使用者程序佔用約 14.5 GiB；在不終止他人程序的前提下，無法執行公平的 Qwen TP2 target。因此 Tiny pilot 的結果不會被拿來宣稱 Modified 已經比較快；待 GPU2 釋放後，才執行上述 8-request 長 context matrix。

## Qwen 無法使用時的 Tiny 備案實驗

若 GPU2 持續被其他使用者佔用，可以使用 Tiny 做一組可重現的備案。這組的目標不是宣稱 Tiny 等同 Qwen2.7B，而是把 request 數量提高、固定 target startup，清楚觀察 Modified 少做 context recomputation 的效果。

### 備案 trace

使用：`examples/spotserve/spot_trace_tiny_backup_batch_gpu013.jsonl`。

```text
0s       add node-0                 → source TP1
30s      add node-1
60s      add node-3                 → 目前可用 GPU0、GPU1、GPU3
120s     remove node-0,node-3       → source 被 preempt，只留下 GPU1 target
180s     add node-0
240s     add node-3
300s     remove node-1
360s     add node-1
420s     DONE
```

這份 trace 刻意不使用 node-2，因此 GPU2 被占用時仍可執行；preemption 後只留下 GPU1，四個版本都只能建立一個 TP1 target，Modified 不會因為選到 DP2 而多付一倍 engine startup。

### 建議 workload

- Model：`/work/spotserve-models/Qwen2-MoE-Tiny`。
- 同時送出 8 個 deterministic request。
- 每個 request 使用 480-token prompt、最多產生 32 tokens；總長度不超過 Tiny 的 512-token 上限。
- 在產生 16 個 decode tokens 後 preempt，確保 request 已經有 live KV，而不是停在 prefill race。
- 四個版本都使用相同 trace、相同 prompt、相同 preemption token 和相同的單一 target startup。
- 每個 cell 重複 5 次。

### 預期可觀察數值

source 在 preempt 時每個 request 約有 `480 + 16 = 496` 個 prefix tokens；Rerouting/Reparallelization 整批約需重算：

```text
8 × 496 = 3,968 tokens
```

Modified/NIXL 應記錄：

```text
recomputed_tokens = 0
restored_blocks > 0
restore_fallbacks = 0
planner selected plan = TP1 × DP1
```

結果表要列出 `Recovery Time`、`P99 Latency`、`Effective Throughput`、`Success Rate`、`Recovery Data`、`Sequence Equality`，另外分開列出 target startup、context prefill/recompute、NIXL transfer、restore 和 first output。因為四版本只建立一個 target，若 Modified 仍比 baseline 慢，差異就能直接追到 NIXL/控制成本，而不是多建立 target 的不公平因素。

## Tiny 四版本單次執行結果（本次要求）

為了完成「Tiny、每個版本重複 1 次」，採用同一組已通過的 Tiny pilot 設定，避免把不同 context 或不同 trace 混在一起比較：

- trace：`examples/spotserve/spot_trace_tiny_pilot_gpu013.jsonl`（只用 GPU0、GPU1、GPU3，避開被占用的 GPU2）。
- prompt：240 tokens；`max_model_len=512`；最多生成 128 tokens。
- source 在生成 1 個 token 後 pause，trace 移除 GPU0；Modified 由 live planner 重新選擇 target。
- Rerouting 不預熱 backup engine；四個版本都把 target 建立時間計入 Recovery Time。
- 每個版本各執行 1 次，結果來自 `/tmp/tiny-four-version-controlled-pilot.*.p240.r1.json`。

| Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---|---:|---:|---:|---:|---|
| No Recovery | 21.520 | — | 0.000 | 0% | 重算 0 / 還原 0 / 產生 0；無新 engine；無 target |
| Rerouting（no backup） | 162.096 | 0.465 | 34.806 | 100% | 重算 263 / 還原 0 / 產生 18；新 engine；placement 改變 |
| Reparallelization | 162.897 | 0.478 | 34.063 | 100% | 重算 263 / 還原 0 / 產生 18；新 engine；placement 改變 |
| Modified/NIXL | 170.659 | 0.323 | 32.052 | 100% | 重算 0 / 還原 17 / 產生 12；新 engine；planner 套用 TP1×DP2 |

### 單次結果解讀

1. No Recovery 按定義失敗；其餘三個版本都能把 request 接回，成功率 100%。
2. Modified/NIXL 確實做到 `recomputed_tokens=0`、還原 17 個 KV blocks，且 planner 在 remove 後選出 `TP1×DP2`、target groups `[[1],[3]]` 並建立新配置。
3. 這個 Tiny/240 單 request 組合中，Modified 的 P99 最低，但總 Recovery Time 比兩個 baseline 多約 8 秒；原因是 context 太短，planner、兩個 target engine startup 和 NIXL 控制成本大於省下的 263-token 重算。
4. 因此這一輪是功能與動態 planner smoke，不足以宣稱 Modified 的 wall-clock 一定較快；要凸顯效能差距，仍需執行前述 8-request、4K/7K 長 context batch。

補充：曾嘗試直接以 backup batch trace 啟動 8 個 Tiny request，EngineCore 在目前硬體上超過 8 分鐘仍未 ready，已停止並確認容器/GPU 資源釋放；沒有把該次 startup failure 當成 recovery 數據。

## Qwen2.7B/MoE 4096 長 context 實測紀錄（硬體可行版）

本節記錄本次實際執行的 Qwen2.7B/MoE batch 實驗，不把尚未完成的 Modified/NIXL 填成成功數據。

### 實驗設定

- Model：`/work/spotserve-models/Qwen1.5-MoE-A2.7B`（Qwen2.7B/MoE）。
- Hardware：4 張 RTX 5070 Ti 16GB；Qwen TP2 每張 GPU 約使用 13.4GiB。
- Workload：2 個同時執行的 4096-token request；每個 request 最多產生 64 tokens，preempt 前先產生 32 tokens。
- vLLM：`tensor_parallel_size=2`、`max_model_len=4352`、`max_num_batched_tokens=2048`、`gpu_memory_utilization=0.94`、`cpu_offload_gb=6`。
- Trace：baseline 使用 `examples/spotserve/spot_trace_qwen15_moe_a27b_batch_recovery.jsonl`，source 在第一個 remove event 失效；Modified 因 GPU2 被外部程序佔用，使用 `examples/spotserve/spot_trace_qwen15_moe_a27b_gpu013_batch_recovery.jsonl`，source=GPU0+GPU1、target=GPU0+GPU3。
- Target 沒有使用前一輪的 backup engine；每輪都重新建立 container。Rerouting/Reparallelization 的 target startup 由 preempt 後開始計時。

### 已完成的三個 baseline（各一次）

`Recovery Data` 格式為：`重算 tokens / 還原 blocks / target 產生 tokens`；`Recovery Time` 是從 preempt 到所有 request 完成（No Recovery 則是 failure observation）。

| Context | Version | Recovery Time (s) | P99 Latency (s) | Effective Throughput (tokens/s) | Success Rate | Recovery Data |
|---:|---|---:|---:|---:|---:|---|
| 4096 × 2 requests | No Recovery | 4.120 | — | 0.000 | 0% | 0 / 0 / 0；無 target |
| 4096 × 2 requests | Rerouting | 175.587 | 0.258 | 0.364 | 100% | 8192 / 0 / 64；新 TP2 engine |
| 4096 × 2 requests | Reparallelization | 177.006 | 0.273 | 0.362 | 100% | 8192 / 0 / 64；新 TP2 engine |
| 4096 × 2 requests | Modified/NIXL | **未完成** | — | — | — | export 在 EngineCore timeout |

Baseline 的原始結果檔：

- `/tmp/qwen-batch-4096-b2.no_recovery.json`
- `/tmp/qwen-batch-4096-b2.rerouting.json`
- `/tmp/qwen-batch-4096-b2.reparallelization.json`

Rerouting 的 target-stage time 是 1.516 秒，Reparallelization 是 1.562 秒；兩者約 175 秒的主要差異來自全新 TP2 engine 的啟動與 8192-token context 重算，而不是第一個 target output 本身。

### Modified/NIXL 的診斷結果

單一 Qwen TP2 + NIXL engine 的獨立 probe 已成功，記錄到：

```text
NIXL is available
GPU KV cache size: 10,736 tokens
NIXL handshake metadata collected: workers=2 nonempty=2
NIXL listener metadata received: entries=2
NIXL_READY
```

這證明 NIXL library、實際 CUDA KV buffer registration、TP0/TP1 handshake 都能初始化。可是放回 in-flight recovery harness 後，source request 在 preempt barrier 可以成功停住，接著 `export_inference_state()` 的 EngineCore utility 沒有在 30 秒內回覆；即使改成直接使用 EngineCore request ID 並加 bounded timeout，仍無法取得 `runtime_state.kv_transfer_params`，因此沒有啟動 target，也沒有把這次標成 restore success。

目前 GPU2 的外部佔用者是 `wuling` 的 `/home/test/Cangjie-llm/.venv/bin/python`，約 9GiB；另有既有 Ray/vLLM EngineCore。這些程序未被終止，所以 Modified 本次改用 GPU0+GPU3 target。這是資源保護措施，不是公平的四 GPU 對照結果。

### 本次可下的結論

1. Qwen TP2 的兩個 baseline 已完成真實長 context、真實 container startup 和完整 request completion；兩者都重算 8192 tokens，Rerouting 略快約 1.4 秒。
2. Qwen 單 engine 的 NIXL 初始化已通過，但「in-flight request export → target restore」仍未通過；因此目前不能宣稱 Qwen Modified/NIXL 的 recovery time、P99 或 throughput 優於 baseline。
3. 4096 的 8-request 與 7168 context 尚未執行；在 GPU2 被外部佔用、且 export utility 尚未完成前，不應產生或填入推測數值。
4. 下一個修正點是讓 vLLM EngineCore 在 request 被 paused 時仍能處理 SpotServe 的 export utility（目前 harness 的 simulated pause 會讓 async output queue 停在 barrier）。完成後再用同一份 trace 重跑四個版本，才是有效的 Modified/NIXL 對照。
