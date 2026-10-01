# Original／MoE 沒有拉開差距：Migration 與 Reparallelization 診斷

## 最新接續：2026-09-14

以下 2026-09-13 內容是舊 Qwen1.5-MoE harness 的歷史診斷，不能套用成新 Granite MoE runner 的實際路徑。新 Granite v2 已完成三組各版三次、共 18 個正式 runs，使用 **host token handoff + GPU replay**，不是本檔下方舊 NIXL restore；actual runtime／prefix 均另有驗證。見[新正式報告](../../results/moe_v2_complete_execution_20260914_attempt2/report.md)。

新一輪已完成 TP=1／2 獨立 microbenchmark，控制 warm status，分開掃 input／output 長度與 C=1／2／4；共 12 個 fresh engines、1512 個完整計時請求，所有跨 TP outputs 相同。測試的 64／256-token cells 中，23 個 TP1 repeatable winner、1 個 unresolved，沒有可靠 TP2 勝出，因此依事前 gate 停止 fleet 擴充。這支持「現有服務成本容易讓兩版同選 TP1」，但不代表 routing 無效或其他負載不存在交叉。完整矩陣、曲線、限制與下一步見[獨立 microbenchmark 報告](moe_tp12_microbenchmark_results_20260914.md)。不把新增 warmed batches 當成正式 Original／MoE repeats，不修改舊原始數據。

---

診斷日期：2026-09-13（Asia/Taipei）。

## 1. 結論與執行範圍

兩條路徑已一起排查。本次完成的是 **既有 artifact 比對、CPU 控制測試、planner 重播與 CPU profiling**，不是修正後的正式 GPU 實驗。沒有改 source／target、部署引擎或啟動 GPU 推論，也沒有修改既有正式數據。

目前最有證據的解釋是：

- **Migration：MoE 評分有作用，但本次唯一 target 涵蓋全部 experts，評分歸零；三組 Original／MoE 配對的 mapping 與成本完全相同。** 實際走相同的 export／NIXL restore 程式路徑，沒有決策帶來的工作量差異證據。
- **Reparallelization：可用 GPU 與 capability filter 把選擇限制成唯一 TP=2 配置。** MoE 版本多產生 logical expert placement，但沒有取得 current expert placement／routing hotness 作為這個 planner 呼叫的輸入，也沒有驗證實體位置變更。
- **Reparallelization 的額外 CPU 路徑已定位到配置表建構與序列化。** 即使只有一個候選，同一次呼叫仍建構 expert plan 兩次；沒有後續可辨識的配置收益抵銷。
- **部分結果欄位不是獨立量測。** `configuration_applied=true` 是模式標籤，不是 runtime 驗證；`restored_blocks` 是 expected blocks，不是搬移完成數。不能把它們當成已完成配置改變或實測 KV reuse 的證據。

不能由本次結果推論「整個專案的 MoE 優化無效」。更準確的結論是：**這個 benchmark 沒有讓 expert-aware 決策形成不同的、可驗證的實際執行。**

診斷資料：[diagnosis.json](../results/moe_spotserve_gap_diagnostics_20260913/diagnosis.json)。

## 2. 名詞與比較對象

- **Original／MoE**：兩組 GPU 推論都使用同一個 Qwen1.5-MoE-A2.7B checkpoint；不是 dense model 對 MoE model。差別在 planner 的 MoE-aware 輸入／評分。
- **Migration mapping**：安排哪個請求移往哪個 target。
- **Expert**：MoE 內被 router 選來運算的子網路；它的 weights 是模型參數，不是每個請求的 KV cache。
- **Logical placement**：描述 expert 預定位置的配置表，不等於 GPU 記憶體中的實際位置。
- **Rank**：分散式運算中的工作程序。修改配置表的 rank，不代表已把權重傳到另一張 GPU。
- **Locality**：目前 scorer 計算的是 target 配置紀錄對請求所用 experts 的 routing-weighted coverage；不是實測 GPU/rank 局部性。
- **Routing histogram**：請求使用各 layer/expert 的次數。top-k 一個 token 會選多個 experts，因此 assignment 次數不等於唯一 token 數。
- **Actuation**：把選定計畫真正執行成不同部署、資料搬移或 GPU 配置。
- **Profiling**：量測各函式／運算階段耗時，找出成本所在。

## 3. 既有正式數據配對比對

讀取 Experiment 2A／2B 共 12 個 `run-N.json`，各三組配對。用已記錄的 prompt token SHA-256 對齊請求，避免各 run 的 request ID 不同而產生假差異。

| 檢查 | Migration：三組配對 | Reparallelization：三組配對 |
| --- | --- | --- |
| Prompt tokens／hash | 全部相同 | 全部相同 |
| 實際 source 中斷進度 | 全部 261／258 new tokens | 全部 261／258 new tokens |
| Prefix／remaining tokens | 全部相同：4357／4354、251／254 | 全部相同：4357／4354、251／254 |
| Routing histogram hash | 全部相同 | 全部相同 |
| Reference／continued output hash | 全部相同 | 全部相同 |
| 選定 mapping 與成本 | 全部相同：兩請求至 GPU 2、3 群組 | 不適用 |
| 選定配置與節點 | 固定 TP=2 target | 全部相同：TP=2、PP=1、vLLM DP=1、replica=1、EP disabled、node-2/3 |
| Expected blocks（非 ack） | 兩組都 546 | 兩組都 0 |
| Metadata recompute counter（非 GPU 實測） | 兩組都 2 | 兩組都 8711 |
| Runtime 配置差異獨立驗證 | 未收集 | 未收集 |
| Actual KV bytes／ack／GPU 重算量 | 未收集 | 未收集 |

每組中斷設定是 256，但實際 snapshot 是 261／258；本次跨模式實際邊界一致。這排除了「某組多做了不同長度前文／剩餘輸出」的已記錄工作量不公平，但不等於已量到物理工作量。

以下時間是既有三次正式 run 的 mean ± sample SD，沒有修改或扣除任何成本：

| 指標 | Original Migration | MoE Migration |
| --- | ---: | ---: |
| 完整 harness runtime（s） | 413.025 ± 0.916 | 413.183 ± 1.232 |
| Migration flow（s，含首輸出） | 3.415004 ± 0.255488 | 3.535876 ± 0.355326 |
| Migration planner（s） | 0.171111 ± 0.002549 | 0.178908 ± 0.000530 |

| 指標 | Original Reparallelization | MoE Reparallelization |
| --- | ---: | ---: |
| 完整 harness runtime（s） | 398.933 ± 0.426 | 402.434 ± 1.998 |
| Planner（ms） | 6.358 ± 0.416 | 166.820 ± 12.349 |
| Target startup（s，發生在 preemption 前） | 170.062 ± 0.570 | 173.250 ± 2.124 |

## 4. Migration：原因與控制測試

### M1. 不是 MoE 開關失效，而是評分沒有區分度

目前 `plan_context_migration()` 只建立一個 `target`，capacity 為 2。Cost matrix 的兩欄是 **同一個 target 的兩個容量 slot**，不是兩個不同目的地。

MoE 版本的每個 request 都有 histogram、placement available、locality=1、remote ratio=0、expert cost=0；Original 版本未啟用 expert score，不能把其 locality=0 欄位解讀為「實際 0% local」。

三組配對的 request mapping、KV／expert／queue／total cost 全部相同。這支持「有執行 scorer，但沒有可改變的 target 選擇」的原因。

程式：[plan_context_migration](../tests/spotserve_test/run_tiny_batch_recovery.py)、[expert cost](../sllm/spot/context_migration.py)。

### M2. Scorer 看 expert 是否被涵蓋，不看其 GPU/rank 位置

使用一個既有 request 的真實 histogram（418176 assignments），另外建立**人工 counterfactual metadata fixture**，隔離評分行為：

| CPU scorer 測試 | Locality | Expert cost | 結果 |
| --- | ---: | ---: | --- |
| 涵蓋全部 1440 keys，MoE on | 1.0 | 0 | scorer available |
| 相同 keys，全部改寫成另一個 rank，MoE on | 1.0 | 0 | rank 變更不影響評分 |
| 刪除最常使用的單一 expert key，MoE on | 0.996081 | 0.039194 | 可以偵測 coverage 缺口 |
| 相同缺口，MoE off | 不適用 | 0 | scorer disabled |

移除的 key 為 `layer:20/expert:47`，其 1639 次 assignments 被估為 remote。權重為 10，成本是 `10 × 1639 / 418176`。

在 KV 等其他成本保持相同的測試中，off 選第一個 partial target，on 改選 full target。這證明開關與評分能改變測試決策；但將相同 expert keys 改 rank 仍然 cost=0，證明此 scorer 不是實體跨卡通訊成本模型。

**這些 partial/rank fixtures 不是合法部署驗證，也沒有把 expert 從 GPU 搬走。** 它們只驗證函式對輸入的反應，沒有加入正式效能統計。

### M3. 兩組沒有不同的資料搬移策略

Current harness 的兩組使用同一組 worker export、NIXL restore 與 generate 操作；MoE decision 沒有用來改變 transfer payload／expert weight 位置。

已記錄的 prefix 與 expected blocks 一致，支持搬移需求相同。因為未保存獨立 bytes/ack，**不能聲稱已逐 byte 驗證實際傳輸完全一致，或已實测搬移量降低**。

Routing 資料是同一份 `vllm_runtime_topk_calibration`，不是同次 NIXL live capture。其相同 hash 有助公平對照，但本次不能驗證 workload 改變時，MoE 是否會即時調整選址。

### M4. 既有 Phase 2 ablation 通過

重跑既有 synthetic planner ablation：

- KV-only 選 `target-kv-busy-remote-expert`。
- 加 expert cost 選 `target-expert-busy`。
- 加 queue cost 選 `target-idle-remote-expert`。
- KV＋expert＋queue 選 `target-expert-idle`。

四種預期選擇及成本比較全部通過。這排除了「相關 synthetic 情境下 scorer 完全不生效」，但不是 GPU 通訊／效能證據。

資料：[Migration fixture report](../results/moe_spotserve_gap_diagnostics_20260913/migration-fixture/report.json)。

## 5. Reparallelization：原因、CPU 重播與 runtime 邊界

### R1. Harness 限制形成唯一候選

`plan_for()` 的有效條件是：

- remove 後只有 GPU 2、3 可用。
- capability 只保留 EP size=1、TP >= 2。
- model GPU requirement／min TP 都是 2，PP 上限是 1。

因此三次 Original／MoE 全部 candidate_count=1，選定同一個兩卡群組。沒有 runtime 支援的不同候選，自然沒有 configuration-selection 收益可測。

### R2. 本次 MoE planner 呼叫缺少動態資訊

MoE 版 `planner_model_config` 比 Original 多了 checkpoint path，讓程式推得 24 layers × 60 experts 的 topology，並產生 1440-expert logical placement。

但這個 harness 呼叫沒有提供：

- current runtime expert placement snapshot。
- source per-request routing histogram／hot-expert load，作為 repara planner 輸入。
- 已啟用、經量測校準的 workload cost model。

**Source 雖有捕捉 routing 資料，不代表它被傳入 `plan_for()`。** Artifact 顯示兩版 workload cost model 都 disabled；MoE movement observation=false、1440 experts movement unknown。

所以 `moved_weight_bytes=0` 不是「實際零搬移」證明，而是缺少觀察時的輸出。本次沒有驗證針對熱門 experts 的位置調整收益。

### R3. 重播實際 `plan_for()`，確認相同決策並定位 CPU 成本

診斷工具透過 AST 只取出 current harness 的 `plan_for` 函式定義，提供原本的 checkpoint、TP=2、四卡列表及 active={2,3}。沒有執行 harness main、podman 或 GPU engine。

先暖機，再做六組交錯 Original／MoE 呼叫：

| CPU warm replay | 平均 planner wall time |
| --- | ---: |
| Original | 0.409 ms |
| MoE | 14.597 ms |

兩版重播都 candidate_count=1，選 TP=2／node-2/3。MoE 額外 CPU 路徑確實存在。

另外一次 cProfile 的 MoE `plan_for` cumulative time 是 20.781 ms；其中：

- `build_logical_expert_placement_plan` 呼叫 **2 次**，cumulative 14.135 ms：候選評分時建一次，選定後再建一次。
- `ExpertPlacementPlan.to_dict` 呼叫 **4 次**，cumulative 5.462 ms。
- `ExpertShard.to_dict` 呼叫 **8640 次**，cumulative 2.942 ms。

這些 cumulative 時間有包含／重疊關係，不能直接相加。Profile 支持「額外成本主要在配置資料建構、轉成 dictionary 與序列化」，不是 expert tensor 搬移。

**此次 warm CPU replay 沒有重現正式環境多出的完整 160 ms。** 兩個環境、暖機與 CPU 負載不同，且正式 run 未保存等價的 CPU profile；因此不能把原本 160 ms 全額歸因給上述函式，更不能用新的數字改寫舊 raw。原環境額外成本仍需同環境 profiling。

資料：[Original CPU profile](../results/moe_spotserve_gap_diagnostics_20260913/repara-original.prof)、[MoE CPU profile](../results/moe_spotserve_gap_diagnostics_20260913/repara-moe.prof)。

### R4. `configuration_applied` 等標籤沒有驗證 actuation

Current harness 填入：

```python
configuration_applied = uses_reparallelization
engine_created = True
placement_changed = True
restore_success = uses_migration
```

這些不能代替獨立成功量測。例如 2B raw 同時是 `configuration_applied=true`、`target_preexisting=true`：target 在 preemption 前已啟動，而此值只因模式呼叫了 planner 就為 true。

Source→target 群組確實不同，也不代表 MoE 對 Original 有額外位置改善。Raw 未包含 V6 adapter apply 結果或 runtime 實際 parallel shape 驗證；這個 direct-worker harness 不等於完整 V6 deployment-executor 效能測試。

### R5. Patch apply／verify 接口確認是 observe-only

只從 repository patch 取出 placement helpers 與 apply／verify 函式，CPU 執行同一份契約：

```text
apply: applied=false, physical_weight_migration=false
verify: verified=false, contract_seen_by_runtime=true
```

這證明 patch 接口能收到計畫，卻明確不宣稱執行或驗證 expert 位置變更。

此 probe 不是現在運行中的 container runtime audit，不能拿它證明某個既有 image 的精確二進位版本。但它與 repository 文件、harness 缺少實體位置操作的路徑一致。

**V6 的 actor recreate 路徑仍存在；observe-only 限制的是自訂 expert placement 的 live actuation，不是所有部署調整與 KV restore。**

### R6. 既有 Phase 4 movement ablation 通過

Synthetic fixture 的 movement penalty 能改變配置選擇：

- 無 penalty：`split_across_two_ep_ranks`，估計 moved experts=2、bytes=2097152、cost=20 ms。
- 有 penalty：`stationary_single_rank`，估計 moved experts=0、cost=0。

兩種預期選擇與欄位全部通過，證明該成本項在提供 current placement 等測試輸入時能影響決策。**這是 logical movement estimate，不是 GPU weight 搬移量測。**

資料：[Reparallelization fixture report](../results/moe_spotserve_gap_diagnostics_20260913/repara-fixture/report.json)。

## 6. 計時、測試與尚未證實的原因

### 計時問題

- `recovery_s` 在 NIXL target final output 後仍包含 source shutdown；replay 在 target submit 前 shutdown。
- Source freeze/snapshot 先做，`preempt_started` 後設，前置暫停不在 notice/remove-clock 內。
- `reparallelization_s = target_startup + planner`，但 target startup 在 preemption 前，不是事件觸發的 transition 時間。
- 完整 runtime 包含 setup、兩端啟動、reference 與 cleanup，不是純服務請求耗時。

這些特別影響跨 recovery 方法排名。Original／MoE Migration 兩版 lifecycle 相同，因此 **不能只拿 cleanup 差異解釋兩版沒有 MoE-specific 收益**；「決策相同」仍是更直接的原因。

未保存 final emission timestamps，不能從舊 raw 精確扣掉 shutdown；本診斷不估算修補。

### 測試結果：61 passed、1 failed

共 62 個 CPU 測試，包含 migration planner／Phase 2、repara planner／Phase 4／executor／adapter、placement、stateful recovery、runtime audit，以及新增 8 個診斷回歸測試。

唯一失敗：

```text
test_context_migration_planner.py::test_context_migration_metric_contains_summary_fields
expected: moe_locality_definition == "unavailable"
actual:   moe_locality_definition == "target_placement_coverage"
```

目前 disabled scorer 仍輸出固定 locality definition；availability／observation 另為 false，而既有測試預期沒有可用資料時 definition 也為 unavailable。這是 **metric contract／測試預期不一致**，需要決定 definition 是固定描述還是只在 available 時輸出，再統一程式與測試。尚未修正，也不直接證明它造成速度差距。

新增 8 個診斷測試全部通過，確保：request ID 正規化、中斷邊界差異能被偵測、missing physical counters 不被解讀成零、相同 metadata 不被解讀成實體工作減少，以及 patch hook／scorer probes 的行為。

資料：[pytest.log](../results/moe_spotserve_gap_diagnostics_20260913/pytest.log)、[pytest.xml](../results/moe_spotserve_gap_diagnostics_20260913/pytest.xml)。Runner 對這次測試失敗回傳 exit code 1，沒有隱藏失敗或改成全部通過。

### 尚未證實的解釋

- CPU offload、跨 GPU／NUMA 傳輸是否主導成本：沒有 GPU／CPU timeline 證據。
- 不同 preempt 時機與長時間／多次 churn 是否放大收益：本次沒有新增 GPU 情境。
- 原始容器的 CPU contention／初始化是否解釋正式 planner 與 warm replay 的差額：沒有同環境 profile。
- 小幅差異是否有統計顯著性：三次樣本不支援強因果／顯著性結論。

不能把這些假設寫成已定位的硬體瓶頸，也不能單純以換更大模型作為解決方案。

## 7. 修正與下一輪驗證優先序（本次未實作）

1. **先修 benchmark 證據與計時**：把 planner-called、deployment-applied、runtime-config-verified 分開；記錄 notice、freeze、export、transfer completion、first/final emission、cleanup；warm target 與事件後 cold target 分開報告。
2. **Migration 保留固定 source／target**：可先驗證 KV 接續與量測，但在唯一 target 全 expert coverage 下，expert-aware target-selection 收益應標成不可驗證。要測選址需另加真實候選 target，先取得使用者同意；不能靠人工刪 expert metadata 假裝真實部署差異。
3. **Repara 使用真實 adapter 路徑與可執行候選**：先確認 runtime 支援、記憶體可容納、實際配置可回讀，再提供 current placement／workload profiles。Live expert migration 若是待驗證目標，需先補其實體執行能力；不能只調 label。
4. **再做同環境 profiling**：確認 logical plan 的重複建構／序列化成本，以及正式 160 ms 差額來自哪裡。此時再決定是否快取／避免重建；不能先改算法後混用舊數據。
5. **Gate 通過後才跑 GPU 效能矩陣**：預先固定早／中／晚 preempt、成對工作量、執行順序與有效性規則，再測多次 churn。舊 v2 與新數據分開存放。

## 8. 重現與資料保護

工具：[run_moe_gap_diagnostics.py](../scripts/run_moe_gap_diagnostics.py)。

```bash
env PYTHONPATH=/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve \
  /work/containers/s112060021/Qwen3/vllm/.venv/bin/python \
  scripts/run_moe_gap_diagnostics.py \
  --output-dir results/moe_spotserve_gap_diagnostics_recheck \
  --run-tests
```

必須使用新的 output directory。工具拒絕把輸出寫進正式 artifact tree，且不下載模型；只讀本機 config.json。若既有 metric 測試未修正，預期 exit code 1；診斷 probe 的通過狀態另記在 JSON。

本次對正式 v2 tree 的 **55 個 JSON** 做 before／after SHA-256 比對，全部未變。每個分析 run 與 current source／model config 的 hash 保存在 diagnosis.json；正式 runtime 與 current source 的版本等價性沒有獨立證明，CPU 重播不冒充原始環境。

沒有修改 existing production code／測試預期，保留使用者原有 dirty worktree；沒有新增正式 GPU run。


---
# 真實 MoE × TP=1／2：目前實驗方針

日期：2026-09-13。使用者已取消 TP=3 要求；保留同一個真實 MoE、最多四卡、多 target、依目前 workload 選 plan，以及公平的 Original／MoE-aware 對照。

## 0. 執行順序與使用者確認點（最新指示）

**等待完整四卡環境再啟動新的 GPU 實驗。** 必須確認 GPU 0／1／2／3 都可供本實驗使用、資源足夠且不受未受控的既有工作干擾；不能只因為系統列出四張卡就視為符合條件。沒有取得完整環境前，不以兩卡／三卡 fallback 執行，也不停止任何既有 GPU 工作。

四卡環境確認後，先新增一個獨立階段：**SpotServe（未加 MoE 優化）**。使用上述同一份真實 Granite MoE 與原始 SpotServe 路徑，保留原始 migration／reparallelization 機制，關閉 MoE-specific migration／reparallelization 優化；不是改跑 dense model，也不是用手動 TP canary 冒充 SpotServe。

- 執行前固定 source／target GPU 群組、workload、GPU availability／preemption 規則與 warmup policy，保存實際部署與恢復證據。具體四卡 baseline 矩陣尚未註冊，不宣稱已完成多 target 或動態 TP 套用。
- 每個 configuration 完整執行三次，保留失敗與重試紀錄；報告平均完成時間、吞吐、延遲、恢復／停機時間、成功率及實際中斷邊界，不因效能結果不好排除有效 run。
- 結果另寫新檔 `docs/moe_spotserve_baseline_results.md`，搭配獨立 raw JSON／logs；不覆寫現有 pilot 報告，不把前輪 pilot 當成這個階段的結果。
- **Baseline 完成、報告產出後立刻停下來，先讓使用者看結果。即使自動檢查通過，也不自動啟動 2A／2B；須等使用者確認後才繼續。**

後續編號沿用原始要求：**2A = Original／MoE-Optimized Migration；2B = Original／MoE-Optimized Reparallelization**。下方第 3／4 節分別對應 2A／2B，不是 Experiment 1／2。原本 Experiment 1 四策略比較仍是另外的完整矩陣，不因先跑 SpotServe 單組而視為完成。

目前狀態：尚未啟動新增 SpotServe baseline，等待完整四卡環境；2A／2B 暫不執行。

## 1. 什麼叫「目前最佳策略」

不是預設「輕負載 TP=1、重負載 TP=2」，也不是每次讓兩個 TP 各勝一次。先過濾目前真正能部署、放得下 context 的配置，再根據相同目標與實測成本選擇。

候選至少含一個 TP=1 instance 與一個 TP=2 instance；可再加入多個 TP=1 副本，副本數與 vLLM data_parallel_size 分開記錄，不能把兩個單卡副本當成 TP=2。

需要的輸入包括：目前請求到達率、排隊數、prompt／remaining output 長度、同時請求數、GPU 可用性、目前 runtime 配置，以及各候選的實測延遲、吞吐量、啟動／恢復／切換成本。

「最佳」是這些合法候選與預先註冊成本目標下的最佳，不宣稱未知配置的全域最佳。若實測 TP=1 在所有測試負載都較好，就如實選 TP=1；不調整權重強迫 TP=2 勝出。也要確認何時「不切換」比重建更划算，避免只有重建候選而忽略留在原配置的成本。

## 2. 模型與驗證條件

優先 canary：ibm-granite/granite-3.1-3b-a800m-instruct，revision=a02780686e08a03fe0d2679a293b5c74a90efa89。

- 官方設定是 GraniteMoeForCausalLM，每層 40 experts、每 token top-k=8，不是普通 Granite dense checkpoint。
- 同一份原始 BF16 safetensors 權重用於所有模式，初始驗證不使用 CPU offload 或量化，也不修改 expert／head 維度。
- TP=1／2 的必要靜態檢查、live 4K／8K 的 64-token canary，以及後續 512-token 成本 pilot 已通過。手動跨 TP token replay 也通過；這不等於正式負載、planner／actor 自動切換或直接 KV restore 已驗證。
- 初始 GPU canary 檢查兩種 TP 的實際 worker model／expert modules 與真實 routed expert IDs，並測 4K／8K context、兩個同時請求、64 個輸出 tokens。engine max_model_len=8704，保留 8K+512 的配置空間；但 64-token canary 不冒充 512-token 正式工作負載。
- 工具：[run_granite_moe_tp_canary.py](../scripts/run_granite_moe_tp_canary.py)。不允許 dense fallback；有缺失 routing 或實體模組證據就不放行正式 MoE 比較。

Original 與 MoE-aware 兩組使用同一個 MoE checkpoint。Original 指未啟用 MoE-aware 決策，不是 dense model。

## 3. Experiment 2A — Migration：單獨驗證 target 選擇

四卡可排他使用時，固定 GPU 0 為 TP=1 source；GPU 1／2／3 各放一個 TP=1 target。Source preempt 時有三個真正獨立的目的地。

使用可控、可量測的背景負載，記錄每個 target 的容量、queue、KV 狀態與可用 expert 成本資訊。Original／MoE-aware 配對使用相同請求、背景工作、模型與實際中斷 prefix。

目前完整 targets 都涵蓋全部 experts，coverage scorer 仍可能全部為 1。多 target 解決了「沒有選擇」，卻不自動解決「MoE 評分沒有區分度」。需要接入真實 expert 負載／通訊成本；無資料時標記 unavailable，不人工刪除 expert 紀錄製造優勢。

## 4. Experiment 2B — Reparallelization：讓選擇真正改變部署

起始 source 為一張 GPU，target pool 至多三張。TP=1 新 instance 使用一張，TP=2 新 instance 使用兩張；每個 run 只建立被選中的方案，不同時啟動所有候選。

為區分 TP 與副本收益，主比較先固定 replica_count=1，允許 TP=1／2；多副本配置作為另外的對照，不混在「同一份模型多卡加速」解讀。

先以相同 checkpoint、context 與固定硬體群組校準候選成本，再根據目前 workload 呼叫 planner。使用已支援的 actor／engine 重建路徑執行 plan，獨立驗證 runtime TP、GPU 群組與服務恢復；不能把「planner 有產出配置表」當成配置已套用。

TP=1→TP=2 改變 KV 分片，先檢查相容性。若不能直接 restore，使用 token replay 重建並計入成本。自訂 expert placement 接口仍是 observe-only，不宣稱已完成任意 expert 權重的實體搬移。

## 5. 公平性與結果量測

- workload 包含 4K／8K 與不同同時請求數；更高併發需先通過記憶體檢查，不把 OOM 候選當成合法方案。
- 預先固定 preemption 規則；GPU 效能不同時可能到達不同 token 邊界，因此正式配對要驗證實際 prefix／remaining tokens，而不只比 CLI 的設定值。
- 舊 harness 的逐 token 輸出／snapshot 仍有 overshoot；若兩版實際邊界不一致，不放進公平配對統計。
- 分開記錄 notice→freeze、planner、啟動、export／restore 或 replay、第一個恢復 token、完成輸出與切換後穩態服務。cleanup 與事前 setup 不混入停機時間。
- 每次保存候選數、各成本項、選定 TP／副本數、runtime 驗證、成功率與失敗 run；pilot 用來定負載，不算正式優化收益。

目前初始 token 邊界可沿用先前 256 的 pilot 設定；正式多情境矩陣與精確 barrier 需要在新 harness 驗證後註冊，不能聲稱已執行多次 spot churn。

## 6. 本輪完成與未完成

已完成：預檢預設改為 TP=1／2；十候選重播取得四個靜態 canary 候選；修正 capability 路徑 TP 上限過濾；新增相同候選隨 workload 改選 TP 的 CPU 回歸；下載固定 revision 權重並核對 SHA-256；執行 live MoE／context canary。

相關 CPU 測試 **58 passed**。控制 fixture 的低到達率選 TP=1、高到達率選 TP=2；增加 TP=2 的切換成本後改選 TP=1。**這只是 scorer 契約證據，不是 Granite 實測效能或正式 ablation。**

**Live canary：TP=1（GPU 3）與 TP=2（GPU 1／3）皆通過。** 同一份原始 BF16 Granite MoE，無 CPU offload；各測 4096／8192 prompt、兩個同時請求、64-token 輸出，共八個請求。實際 worker 的 GraniteMoeForCausalLM／GraniteMoeMoE／FusedMoE 與 32 層、top-k=8 的 routing 皆已驗證。四組跨 TP prompt／output token hashes 相同，但 routing histogram 不完全相同，不承諾一般情況的 bitwise routing 一致。

這兩次是手動指定 TP 的 correctness canary，不是 planner 自動切換，也不是 Original／MoE-aware 對照。

### 2026-09-13 後續成本與恢復 pilot

新增 full-512-token 量測：4096／8192 prompt × 1／2／4 同時請求 × 每情境三批，TP=1 用 GPU 1、TP=2 用 GPU 1／3。84 個請求全部完成；42 組跨 TP prompt／output hashes 相同。暖機後，本矩陣 TP=2 的完整請求平均延遲比 TP=1 高 15.4–20.6%，TTFT 則低 8.9–17.0%。這是 closed-batch 實測，不是 open-loop SLO 容量，不強迫 TP=2 成為最佳。

手動恢復測試：GPU 0 source 設定第 256 個 token 中斷，EngineCore 實際凍結於 258，overshoot=2。補發 client 尚未收到的兩個 tokens，兩個 targets 共用同一份 4354-token prefix，各續產 254 個；TP=1／2 的合併 512-token 輸出都符合不中斷參考。實際 runtime MoE／TP 已檢查，但使用的是手動新建 engine，不是 planner→正式 actor 套用。

冷恢復 TP=2 雖較早送出首 token，之後有 69.3 秒 delivery gap，同期有 MoE kernel 首次 JIT 警告。正式報告必須同時呈現前四個 tokens、最大停頓與完成時間，不能只用首 token 當作穩定服務恢復。

新增實測成本資料接口：工作量、checkpoint／runtime、實體 GPU 群組、計時稽核與相同 frozen prefix 的 transition evidence 都要匹配，未知成本不可填零。目前 full-request profiles 與 replay 的 remaining work／max_num_seqs 不同，尚不能冒充完整 transition matrix 或直接接入正式 planner。

最後重跑十一個相關 CPU 測試檔案，**99 passed**，保存完整命令與 log；前一輪不同選取集合為 102 passed，前述 58 是上一階段契約測試，數字不相加。詳細：[成本與恢復 pilot 結果](moe_spotserve_tp12_pilot_results.md)。

尚未完成：正式候選／remaining-work 成本矩陣、多 target harness 與真實 expert 成本接入、動態 TP plan 的正式 actor GPU 執行、精確 preemption barrier、跨 TP 直接 KV restore 與 Original／MoE-aware 正式對照。手動 token replay 通過不補足這些缺口。舊 Qwen matrix 的固定 TP／單 target 限制仍在原 harness，尚未把它假報成新設計。所有本輪 artifact 的 formal_experiment_eligible 仍為 false。

本輪下載使用固定 revision 的官方檔案清單。遇到 Xet／鎖等待後只停止本輪 downloader、保留 partial data，改成普通 HTTP；沒有終止任何 GPU 工作。

檢查時 GPU 2 仍有其他高負載工作，四卡排他使用尚未確認；本輪 canary／pilot 不使用 GPU 2，沒有停止任何既有 GPU 工作。Pilot 自建容器已全部退出。小規模 correctness canary 若改用不同群組，會在 raw 記錄，不用它直接替代正式固定群組的效能校準。

證據：[screening.json](../results/moe_spotserve_tp12_preflight_20260913/screening.json)、[strategy_fixture.json](../results/moe_spotserve_tp12_preflight_20260913/strategy_fixture.json)、[pytest.log](../results/moe_spotserve_tp12_preflight_20260913/pytest.log)。

GPU 驗證與限制：[TP=1／2 canary 結果報告](moe_spotserve_tp12_canary_results.md)。
