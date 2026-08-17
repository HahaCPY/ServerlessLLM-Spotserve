# Four-version MoE preemption recovery comparison

## 實驗目的

這份實驗把三個明確定義的 baseline，和本專案的 Modified/NIXL 路徑放在同一個 preemption 情境比較：

1. **No Recovery**：source worker 被移除後，request 直接失敗。
2. **Rerouting**：把 request 的完整 context 送到事先 READY 的既有 replica；不建立新 engine，也不改 TP/EP。
3. **Reparallelization**：停止 source 和 spare worker，依剩餘 GPU 建立新的合法 TP2 vLLM engine；不搬 KV、不做 NIXL restore。
4. **Modified**：source export KV metadata/state，既有 target 透過 NIXL restore，接著繼續 request；不重建 engine。

## 測試環境與方法

- Model：`/work/spotserve-models/Qwen2-MoE-Tiny`
- GPU：GPU0、GPU1、GPU3；GPU2 當時被其他使用者占用，因此沒有使用。
- 每個 worker 都是獨立 container，但都在同一台主機；這是跨 container 模擬，不是實體跨節點。
- source：GPU0、TP1。
- 既有 reroute replica：GPU1、TP1，測試開始前已 READY。
- spare：GPU3、TP1；Reparallelization 會停止它，然後在 GPU0+GPU3 建立 TP2 target。
- trace：3 個事件（add、remove `node-0`、DONE）；remove source 後立刻執行策略。
- 每個模式測試 64、240、480 token 三種 context，各 1 次，共 12 個 cells。
- 每一格都實際啟動 vLLM container；Modified 使用真正 NIXL connector/export/restore。前 3 組不呼叫 export/restore。

測試程式：

- `tests/spotserve_test/run_four_version_recovery_smoke.py`
- `tests/spotserve_test/run_four_version_recovery_matrix.py`

原始結果：`/tmp/four-version-recovery-matrix.json`，每格另有對應的 `.json` 和 `.log`。

## 實驗結果

`recovery_s` 是從 preemption 開始到 request 得到結果的完整觀察時間；`target_recovery_s` 是 target 開始 generate 後到產生第一個 output 的時間。前者包含 container/control socket 的停止與協調成本。注意：目前 Modified 的計時是在 NIXL `export`/`restore` 完成後才開始，因此 `target_recovery_s` 不能單獨代表包含 NIXL 傳輸的完整 latency；NIXL 傳輸需要另外以 phase instrumentation 計時。

| Context | No Recovery | Rerouting | Reparallelization | Modified/NIXL |
|---:|---|---|---|---|
| 64 | request failed | continued；21.863 s；重算 66 | continued；209.083 s；TP1→TP2；重算 66 | continued；21.569 s；還原 5 blocks；重算 0 |
| 240 | request failed | continued；21.656 s；重算 242 | continued；208.307 s；TP1→TP2；重算 242 | continued；21.750 s；還原 16 blocks；重算 0 |
| 480 | request failed | continued；21.538 s；重算 482 | continued；208.714 s；TP1→TP2；重算 482 | continued；21.503 s；還原 31 blocks；重算 0 |

三種可恢復模式的 target 階段細節：

| Context | Rerouting target time | Reparallelization target time | Modified/NIXL target time |
|---:|---:|---:|---:|
| 64 | 0.519 s | 0.523 s（不含新 engine 啟動） | 0.296 s |
| 240 | 0.485 s | 0.489 s（不含新 engine 啟動） | 0.297 s |
| 480 | 0.485 s | 0.467 s（不含新 engine 啟動） | 0.309 s |

## 分析

### 1. No Recovery 是 failure baseline

3/3 No Recovery cells 都觀察到預期的 `preempted_worker_invalid`，request 沒有 target、沒有 retry、沒有重算、沒有 KV transfer。這確認其他版本的「成功」不是把 failure 偷換成普通 retry。

### 2. Rerouting 能恢復，但代價是完整 context 重算

Rerouting 的 target 一直是原先已 READY 的 replica，`engine_created=false`、`placement_changed=false`、TP 維持 1；但 64/240/480 token 分別重新處理 66/242/482 tokens。context 越長，搬到 target 的計算工作也越多。

### 3. Reparallelization 能恢復，但啟動新 engine 是主要成本

三次都真的建立了 GPU0+GPU3 的 TP2 vLLM engine，`engine_created=true`、`placement_changed=true`。request 最終能繼續，但完整 recovery 約 208–209 秒，並且仍要重算 66/242/482 tokens。這正是 generic parallelism reconfiguration 的成本，不包含任何 MoE-aware expert optimization 或 state preservation。

### 4. Modified/NIXL 保留既有 engine，KV blocks 隨 context 成長

Modified 三次都成功 restore，`restore_success=true`，沒有 fallback：

- 64 token：5 blocks，重算 0
- 240 token：16 blocks，重算 0
- 480 token：31 blocks，重算 0

target 仍是事先 READY 的 TP1 replica，沒有建立新 engine、沒有改 placement。Modified 的 post-restore generate phase 約 0.296–0.309 秒；Rerouting 的 generate phase 約 0.485–0.519 秒，但兩者計時起點不同，不能直接宣稱這是 NIXL 的 latency 提升。完整 `recovery_s` 仍約 21.5 秒，是因為本測試的 source stop/控制協調固定成本，會把 export/restore 的差異淹沒。

### 5. 這組實驗能證明什麼

- No Recovery：證明 preemption 確實會讓 request 失效。
- Rerouting：證明「已有 replica」可以救回 request，但要付出完整 context 重算。
- Reparallelization：證明 planner 可以用剩餘 GPU 建立另一個合法 parallel configuration，但 engine startup 很昂貴。
- Modified/NIXL：證明在相同既有 target 上，KV state 可以實際 restore，context 不必重算，而且不需要重新建立 vLLM engine。

## 提升比例

以下比例使用本次 64/240/480 token 三組結果計算；因為每個 context 只跑 1 次，這些是初步觀察值，不是統計信賴區間。

### Modified/NIXL 對 Rerouting（目前只能比較重算量，不能直接比較完整 latency）

| Context | Rerouting generate phase | Modified post-restore generate phase | 公平 latency 提升 | 約加速 |
|---:|---:|---:|---:|---:|
| 64 | 0.519 s | 0.296 s | 43.0% | 1.75x |
| 240 | 0.485 s | 0.297 s | 38.8% | 1.63x |
| 480 | 0.485 s | 0.309 s | 36.3% | 1.57x |
| 平均 | 0.496 s | 0.301 s | 不直接計算 | 不直接計算 |

KV context recomputation 則由 66/242/482 tokens 降為 0，重算量降低 **100%**。上表的 post-restore generate phase 只能當作輔助觀察，不能當作完整 NIXL latency speedup。

### Modified/NIXL 對 Reparallelization

完整 recovery 平均由 208.701 秒降至 21.607 秒，降低約 **89.6%**，約 **9.66x**。這個差距主要來自 Reparallelization 必須重新啟動 TP2 engine。

### 為什麼完整 recovery 對 Rerouting 的差距不大

Rerouting 平均完整 recovery 為 21.686 秒，Modified 為 21.607 秒，只降低約 **0.36%**。原因是目前 harness 的完整時間包含約 21 秒的 source stop/control coordination 固定成本；此外 Modified 的 `target_recovery_s` 起點在 restore 完成後，並沒有把 NIXL 傳輸納入該欄位。因此目前可以確定的是「Modified 不重算 KV、也不重建 engine」，但還不能用這份矩陣宣稱 NIXL 的端到端 latency 提升。

要得到公平 latency 比較，下一輪應在同一支 harness 中記錄：`preemption_seen`、`export_start/end`、`abort_end`、`restore_start/end`、`target_first_output` 和 `source_cleanup_end`，再分別比較完整 request latency 與 NIXL transfer latency。

## 限制與下一步

這是同一主機上的多 container 實驗，不是實體跨節點；GPU2 被占用，所以使用 0/1/3。每個 cell 只有一個長 context request、一次重複，且只觀察一次 preemption 和一個續接 token，因此這不是 production throughput benchmark。下一步應在可用 GPU 主機上增加 20 次以上重複、8 個同時長 request、隨機 remove/add trace，並記錄 p50/p95 recovery latency、throughput、fallback、完整 token sequence 一致性；若要宣稱跨節點，還需另外以真實不同 node/NIXL transport 驗證。
