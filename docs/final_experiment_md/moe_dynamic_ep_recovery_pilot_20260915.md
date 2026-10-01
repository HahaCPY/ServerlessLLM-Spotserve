# Granite dynamic EP：單次恢復候選量測（2026-09-15）

這是 **EP enabled、配置可變的實際 GPU 恢復 pilot**，已量測 2A 卡組與 2B 形狀的候選成本；**尚不是 Original SpotServe 對 MoE-aware SpotServe 的正式 A/B**。每格只有一次 fresh target，沒有獨立校準、三次重複或可信區間。報告時應稱為「可行性與單次成本」，不能稱為 MoE 最佳化的加速比。

## 設置與 source 正確性

- 模型：原始 BF16 Granite 3.1 3B A800M Instruct，`GraniteMoeForCausalLM`，32 個 MoE layer、每層 40 個 expert、top-k 8；checkpoint revision `a02780686e08a03fe0d2679a293b5c74a90efa89`。
- GPU：RTX 5070 Ti ×4；source GPU0/1 `TP2/DP1/EP2`；target 候選 `{1,2}`、`{1,3}`、`{2,3}` × `TP2/DP1/EP2`、`TP1/DP2/EP2`。vLLM `0.23.1rc1.dev562+g3483240b7.d20260716`、Torch `2.11.0+cu130`、Triton MoE、BF16，prefix caching off，C1。
- Workload：同一個普通文字 seed 的 tokenizer token 延展至 input 512 tokens；source 生成 128 tokens，在 64 tokens 精確凍結；target 重放凍結的 576-token host 前綴並完成餘下 64 tokens。恢復是 **host-token handoff + target GPU prefill/replay**，沒有 NIXL KV 直接恢復或實體 expert 權重移動。
- Source：凍結數 `64`、overshoot `0`、allocated KV blocks `36`；擷取 `575` 個 token 的即時 top-k expert routing，experts 0–19 / 20–39 的全層 assignment 為 `76,357` / `70,843`（比例 `1.078`）。凍結輸出前綴與不中斷 source 完全一致。[source raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/source.json)。

## 實驗結果

表中的「target launch→final」從**各候選 container 建立時**開始，到該候選完成剩餘 64 tokens；含 engine startup、一次暖機生成、host handoff 與餘下生成，**不含 source 關閉與候選之間的等待**。六個 target 是依序啟動、共用一個已驗證 source prefix，故這不是六次同時發生的 notice→client final 事件。首輪 raw JSON 的 `notice_to_final_s` 欄名不精確，這裡依實際計時起點更名；補測 raw JSON 已改為 `candidate_launch_to_final_s`。

| GPU 卡組 | 實際 EP 形狀 | Engine ready (s) | 暖機後 ready (s) | Host handoff (ms) | 剩餘生成 (s) | Target launch→final (s) | 完整輸出檢查 | 原始資料 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `{1,2}` | `TP2/DP1/EP2` | 92.34 | 94.60 | 0.72 | 1.90 | 96.50 | passed | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/target-12-tp2dp1.json) |
| `{1,2}` | `TP1/DP2/EP2` | 97.96 | 100.11 | 1.56 | 1.99 | 102.10 | passed | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt3/target-12-tp1dp2.json) |
| `{1,3}` | `TP2/DP1/EP2` | 92.66 | 94.92 | 0.70 | 1.86 | 96.79 | passed | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/target-13-tp2dp1.json) |
| `{1,3}` | `TP1/DP2/EP2` | — | — | — | — | — | 未完成：GPU 佔用 | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt3/target-13-tp1dp2.json) |
| `{2,3}` | `TP2/DP1/EP2` | 94.14 | 96.38 | 0.99 | 1.86 | 98.24 | passed | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/target-23-tp2dp1.json) |
| `{2,3}` | `TP1/DP2/EP2` | 98.25 | 100.44 | 1.55 | 2.03 | 102.47 | passed | [raw](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/target-23-tp1dp2.json) |

### 2A：固定 `TP2/DP1/EP2`，變更 target 卡組

由上表比較三組 target launch→final。三張可用卡上 `{1,2}`、`{1,3}`、`{2,3}` 都能恢復相同 source prefix；最快與最慢的單次差距為 1.74 秒。這是實體 placement 敏感度 pilot，沒有 Original/MoE-aware 選組決策的 paired A/B。

### 2B：固定卡組，變更 `TP/DP`（EP2 保持 enabled）

| GPU 卡組 | TP2/DP1/EP2 launch→final (s) | TP1/DP2/EP2 launch→final (s) | DP2 − TP2 (s) |
| --- | ---: | ---: | ---: |
| `{1,2}` | 96.50 | 102.10 | 5.60 |
| `{1,3}` | 96.79 | — | — |
| `{2,3}` | 98.24 | 102.47 | 4.23 |

這是同卡組的 parallel shape 敏感度 pilot。由於新 target 建立時間主導總成本，應同時看表中的暖機後 ready 與真正剩餘生成，不把單次差值解讀為 MoE 效益。

## 原始 SpotServe／MoE-aware 對照的狀態與限制

| 項目 | 目前證據 | 可報告結論 |
| --- | --- | --- |
| EP2 source freeze、route capture、target output | source 與各成功格的 raw JSON | 普通 512/128 工作負載下的恢復可行性 |
| 2A 實體卡組候選 | 同形狀三個卡組，每格 n=1 | 單次候選成本；沒有 planner A/B |
| 2B EP parallel shape 候選 | 同卡組兩個 EP2 形狀，每格 n=1 | 單次配置敏感度；沒有 planner A/B |
| Original SpotServe vs MoE-aware SpotServe | 未跑 paired 且沒有獨立量測的 rank dispatch / route-conditioned cost model | **不能報告加速比** |

首輪 DP2 兩格曾因 driver 的 DP rank inspection 只讀到第一個 rank、且本地 rank 皆為 0 而被誤判失敗。修正後逐一核對兩個實體 GPU UUID、MoE 權重與輸出，`{1,2}` 已補測成功；`{1,3}` 在最後一次補測前，GPU1 出現非本實驗 VLLM/Ray 佔用，安全預檢拒絕啟動，因此該格是**尚未量測恢復**，不是 EP2 不能執行。原失敗紀錄仍保留在 [attempt1 report](/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_dynamic_ep_recovery_pilot_20260915_attempt1/report.json)。BF16 重複 token 跨 TP/EP 形狀的既有一致性 gate 尚未解決；本表通過的是這個普通 workload，不代表所有 workload 正確。

下一步是先用不同 ordinary prompts 取得每候選至少三次匹配 cache 的 ready、host handoff、service 與實體 rank dispatch 成本；獨立驗證 generic 和 true-route 預測後，使用同一候選集合各跑 Original/MoE-aware 三次 paired 2A、2B 及多事件 overall。四 GPU 的 EP4 形狀另作 startup／expert 覆蓋／恢復 gate，通過後才加入候選集合。
