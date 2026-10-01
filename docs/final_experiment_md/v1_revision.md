# 下一輪實驗的改進設計

最新修訂：2026-09-14（Asia/Taipei）；下方保留 2026-09-13 pilot。使用者已取消 TP=3 要求；保留同一個真實 MoE、最多四卡、多 target、依目前 workload 選 plan，以及公平的 Original／MoE-aware 對照。

### 接續執行（2026-09-14）

使用者後續已核准「成本預測 → 真正執行 → GPU pilots → 正式各三次」。新的 route-cost forecaster、local physical-GPU executor actor 與 sequential runner 已實作；首版獨立 GPU 校準完成，現正統一 target live routing 設定重測。下方「停在前置、尚缺實作」是核准前快照；當前 gates／正式完成數請以 [v2 結果與狀態](1_2A_2B_v2_results.md) 為準。

固定成本模型為六個先前 workload 的 generic mean／live histogram k=3 neighbours，另兩 workload 保留驗證；formal workload 不參與成本訓練。不因初版 routing MAE 較差而調 k。初始 4K+512、C=1；2A/2B native freeze=256，整體比較則依註冊 wall-clock 時刻 pause，保留實際生成邊界並先驗證剩餘長度成本。整體 trace 同時適用兩版，包含 source 0 回收、所有合法 targets 再回收五秒，第二時刻於 pilot 後一次性註冊，不依每次所選 TP 修改。

兩版均開啟 routing capture 供後續中斷取得訊號；Original 不把訊號納入選擇。恢復採 host token handoff + GPU replay，不聲稱 direct KV restore 或真正搬動單一 expert 權重。保留 idle store，不聲稱四卡排他。成本、TP2 correctness、單事件及多事件 pilots 任一失敗先停在該關，不补造正式平均或強迫兩版不同配置。

## 0. 本次採用的範圍與啟動條件（最新指示）

**等待完整四卡環境再啟動新的 GPU 實驗。** 必須確認 GPU 0／1／2／3 都可供本實驗使用、資源足夠且不受未受控的既有工作干擾；不能只因為系統列出四張卡就視為符合條件。沒有取得完整環境前，不以兩卡／三卡 fallback 執行，也不停止任何既有 GPU 工作。

使用者最新要求取代先前「只跑 baseline 後暫停」與四策略矩陣：**目前只做下列三項對照，每個版本、每個情境各三次。** 全部使用同一份真實 Granite MoE，Original 不是 dense model。

| 對照 | 關閉 MoE 優化的版本 | 啟用 MoE 優化的版本 | 固定條件 |
| --- | --- | --- | --- |
| 1：整體比較 | generic migration + generic replanning | route-aware migration + route-aware replanning | 相同請求、GPU availability trace、恢復方式與目標 |
| 2A：migration ablation | generic target 選擇 | route-aware target 選擇 | 全部 TP=1，禁止配置調整；source 0、targets 1/2/3 |
| 2B：reparallelization ablation | generic TP 成本 | route-aware TP 成本 | TP=1/2、單副本、同一非 MoE-aware 恢復政策 |

三項各有兩版，每情境最低 18 個正式 runs；增加情境就分別增加 runs，不把三種不同 workload 當成三次重複。配對順序交錯為 Original→MoE、MoE→Original、Original→MoE；每次回到相同初始部署與 cache policy。平均、sample SD、配對差值、失敗與重試都保留。

**共用修正與 MoE 優化要分開：** 兩版使用相同合法候選、實測 generic 成本與完成時間目標；只有 MoE-aware 增加 source expert routing 條件化訊號。這是「共用成本修正後、未加 MoE-specific 訊號」的 v2 baseline，不冒充完全未修改的歷史 v1 原始 scorer。若改善只來自修正 generic scorer，不歸因給 MoE。

截至本次已在四張實卡完成同一份 4352+256、C=1 的 TP=1 四群組及 TP=2 GPU 1/3 warmed-service 校準，各三批；跨 TP 輸出 hashes 相同。保留 GPU 0/1 的既有 idle sllm-store，未停止其他工作，不宣稱排他正式環境。另已三次驗證 synchronous C=1 source 的 native freeze 恰好為 256 tokens、overshoot=0、KV 仍保留。這些是前置放行，不是三組正式對照。

再以開啟 live routing capture 的新 source engine 做三次相同精準 freeze，均取得真實 4351-token、32-layer、top-k=8 expert histogram。訊號取得已通過；條件化成本預測／獨立校準與正式 GPU runner 尚未完成，不能用同一份 generic profile 補 route hash 冒充 MoE-specific 成本。因此本輪停在前置結果，正式三組仍為零；後續需要補實作，不是只增加三次重複。

正式啟動前仍需通過：受控完整四卡、三次獨立 transition 成本、真實 routing 條件化成本與辨識力、恢復後 client delivery、planner→runtime TP/GPU→輸出正確性，以及整體比較的固定多事件 wall-clock trace。缺任一項就停在準備階段，**不以 fixture 或手動指定 TP 代替正式對照**。

新結果檔：[1_2A_2B_v2_results.md](1_2A_2B_v2_results.md)。目前正式 runs 為零；不覆寫 [v1 歷史結果](1_2A_2B_v1.md) 或下方舊 pilot。

## 1. 什麼叫「目前最佳策略」

不是預設「輕負載 TP=1、重負載 TP=2」，也不是每次讓兩個 TP 各勝一次。先過濾目前真正能部署、放得下 context 的配置，再根據相同目標與實測成本選擇。

2B 的主候選固定 TP=1／2、PP=DP=replica_count=1、EP 關閉；目前不加多副本比較，以免混入副本收益。模型放得下單卡，不代表禁止雙卡：兩個方案都實測後選擇。

第一版成本介面的共用目標為：**engine ready + state install/replay + remaining request 平均完成時間**，單位全部毫秒。啟動與 replay 要分段，不能把 replay→首 token 與完整 request latency 重複相加。engine ready 的實測已包含模型權重載入，不再加 observe-only 的 logical expert movement。closed-batch throughput 只報告，不當成 open-loop queue/SLO 容量，也取消偏好用滿 GPU 的靜態分數。

需要的輸入包括：精確 frozen prompt／remaining output／同時請求數、固定背景 trace、cache policy、GPU 群組、checkpoint revision／runtime、各候選至少三次完整服務與 transition 校準。MoE-aware 另外要求真實 source routes 與獨立事前校準；未知成本拒絕使用，不填零。

「最佳」是這些合法候選與預先註冊成本目標下的最佳，不宣稱未知配置的全域最佳。若 TP=1 在所有測試負載都較好，就如實選 TP=1，不調整權重强迫 TP=2 勝出。本次新 bridge 只處理 source 已不可用的 forced recovery；source 還可服務的自主 replanning 必須另外加入 keep 候選，不能套用本次兩個重建候選。

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

目前完整 targets 都涵蓋全部 experts，coverage scorer 仍可能全部為 1。多 target 解決「沒有選擇」，不自動解決「MoE 評分沒有區分度」。本次改採獨立事前量測的 route-conditioned service cost：在相同背景與 cache policy 下，估計 source 這類 expert routes 在各完整 target 的剩餘服務成本；不人工刪除 expert 紀錄。

這是**條件化整體服務成本**，目前不是逐 expert queue／kernel time 或實體 dispatch traffic。新介面不宣稱 physical locality 優化已實現。校準器／真實多 target 資料仍未完成；沒有資料不放行。也要用保留 workload 與 routing 訊號移除／打亂的診斷，確認優勢不是僅來自更精細的 prompt lookup 或背景負載差異。

## 4. Experiment 2B — Reparallelization：讓選擇真正改變部署

起始 source 固定 GPU 0，target pool 為 1/2/3。首批沿用實測群組：TP=1 用 GPU 1，TP=2 用 GPU 1/3；只建立被選方案，不同時建立重疊群組，也不把 GPU 1/3 的成本拿去套 GPU 1/2。任何群組變動都重新校準。

主比較固定 replica_count=1，允許 TP=1／2，避免把兩個單卡副本當成一個雙卡 instance。TP 會切分各 expert 的權重張量，不等於每張卡只剩一半 experts。

先校準剩餘工作與重建成本，再呼叫相同 planner。新 bridge 已接入 measured scorer，並明確綁定校準的實體 GPU 群組；executor 新增切流量前的 runtime 檢查。實際 actor 的資料收集與 GPU 套用仍需驗證：CPU bridge 的 gpu-* 是本機物理群組標記，不是 Ray node ID，adapter 不能直接拿它當 Ray placement。

Original／MoE-aware 可能都選 TP=1 或都選 TP=2，這不算失敗。要分開判讀「訊號無差異」「分數不同但最佳配置相同」「選擇不同且 runtime 真正不同」「runtime 不同但效益被重建成本抵消」。

TP=1→TP=2 改變 KV 分片，先檢查相容性。若不能直接 restore，使用 token replay 重建並計入成本。自訂 expert placement 接口仍是 observe-only，不宣稱已完成任意 expert 權重的實體搬移。

## 5. 公平性與結果量測

- workload 包含 4K／8K 與不同同時請求數；更高併發需先通過記憶體檢查，不把 OOM 候選當成合法方案。
- 2A／2B 主情境：初始 prompt=4096、output=512、C=1；source 在權威 generated token=256 的 barrier 凍結，故校準的是 prompt=4352、remaining=256，不是 4096+512。8K 情境另測 8448+256。128／384 邊界先作穩健性診斷，不混成三次重複；C=2/4 需先驗證整批實際邊界與成本介面相容。
- 舊 harness 仍有 overshoot，不能只檢查 CLI threshold。獨立 2A／2B 必須共用一致的實際 prefix／remaining；freeze acknowledgement、已算但未送達 token 的 backfill、client 拼接與參考輸出都要驗證。
- 第一組整體比較需要連續請求與至少兩次 availability/preemption 事件。先以小試跑定下固定 wall-clock trace、回收與回歸卡的時間及足夠長的 workload；正式三次前註冊，不依各版本的完成時間移動事件。多次策略分岔後 prefix 不同可能是合法結果，不用強制相同後續 token 邊界來改寫同一個 spot trace。
- 分開記錄 notice→freeze、planner、啟動、export／restore 或 replay、第一個恢復 token、完成輸出與切換後穩態服務。cleanup 與事前 setup 不混入停機時間。
- 每次保存候選數、各成本項、選定 TP／副本數、runtime 驗證、成功率與失敗 run；pilot 用來定負載，不算正式優化收益。

目前已放行受控 C=1 synchronous source 的精確 barrier；整批 freeze、恢復後 client delivery、連續 workload／多事件 actor harness 尚未放行，不能聲稱已執行多次 spot churn。先檢查成本能否區分 targets／TP；若不能，停止正式重複量測並如實報告 null effect，不繼續放大沒有區分度的矩陣。

## 6. 本次共用修正與保護（2026-09-14）

- capability 讀本機 config.json 判斷 MoE，修正 Granite 官方目錄名不含 moe 而被當 dense 的問題；靜態 catalogue 仍不等於任意 shape 的 GPU 驗證。
- [measured bridge](../../sllm/spot/moe_ablation.py) 要求相同 revision/runtime、實體群組、prefix、背景／cache、三次校準與成本分段；2A 有三個真 targets，2B 只允許實測 TP=1/2、單副本。route-aware 成本缺失就拒絕，不 fallback 成同分 coverage。
- [CPU planning gate](../../scripts/plan_moe_ablation.py) 同時檢查兩版；它不啟動 GPU、不能放行正式實驗。
- [executor](../../sllm/spot/reparallelization_executor.py) 的 opt-in runtime verifier 可在切流量前擋下 TP／GPU／MoE／state 不符合的 target；CPU mock 通過不表示正式 actor 已執行。
- migration 無觀測時的 locality 定義改成 unavailable；沒有改寫歷史 raw 結果。

本次進度與測試保存於[新報告](1_2A_2B_v2_results.md)。以下保留前輪紀錄；其中舊路徑、當時 GPU 狀態與測試數字不是本次執行的結果。

## 7. 前輪完成與未完成（歷史紀錄）

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


----

# 試跑 report

日期：2026-09-13（Asia/Taipei）。本輪是正式實驗前的成本與恢復能力驗證，不是 Original／MoE-aware 的優化收益。

## 1. 已完成的服務成本量測

同一個 ibm-granite/granite-3.1-3b-a800m-instruct，revision=a02780686e08a03fe0d2679a293b5c74a90efa89。原始 BF16，32 層、40 experts、top-k=8，無 CPU offload、量化或 dense fallback。兩種 TP 的實際 worker model／expert modules 均已驗證；live routing 證據保留在[前輪 canary](moe_spotserve_tp12_canary_results.md)。

配置：TP=1 用 GPU 1；TP=2 用 GPU 1／3。V1 runner、eager、TRITON MoE、routing capture／prefix cache 關閉，max_model_len=8704、max_num_seqs=4、batch token cap=2048、memory utilization=0.8、OMP_NUM_THREADS=1。控制兩版的 runner，避免 capture 使其中一版切換到不同 runner。

事前[protocol](../results/moe_spotserve_tp12_pilot_20260913/protocol.json)固定 4096／8192 prompt、1／2／4 同時請求、每請求 512 output、每情境三次量測。每個情境先做 64-token shape warmup，warmup 不進統計。使用固定長度的重複 token prompts，不作品質或真實 trace 測試。

共 **84 個量測請求全部完成，每個 512 tokens**，合計 43,008 個生成 tokens。42 組跨 TP prompt／output hashes 都相同，不承諾一般 workload 的 bitwise 一致。兩種 TP 各 18 個計時窗口完整，日誌沒有計時期間的 JIT／autotune 警告；這是 stdout 順序稽核，不是全 kernel profiler。

## 2. 暖機後結果

每個數字依序為 TP=1／TP=2。Latency 是完成整個請求的平均秒數；TTFT 是第一個生成 token 的平均毫秒數；吞吐是整批 output tokens／整批 wall time，不是單一請求的 decode 速度。

| Prompt | 同時請求 | Latency 秒：1／2 | TTFT 毫秒：1／2 | Output tokens/s：1／2 |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 11.818 / 14.074 | 124.3 / 113.2 | 43.32 / 36.38 |
| 4096 | 2 | 12.475 / 14.572 | 196.6 / 169.6 | 81.93 / 70.07 |
| 4096 | 4 | 12.645 / 15.014 | 340.8 / 302.5 | 161.20 / 135.57 |
| 8192 | 1 | 11.842 / 14.280 | 284.8 / 241.9 | 43.24 / 35.85 |
| 8192 | 2 | 12.803 / 14.971 | 439.9 / 364.9 | 79.69 / 68.08 |
| 8192 | 4 | 13.340 / 15.393 | 755.4 / 631.2 | 152.04 / 131.54 |

本矩陣中，TP=2 的平均完整請求延遲比 TP=1 高約 **15.4–20.6%**；但 TTFT 比 TP=1 低約 **8.9–17.0%**。因此「雙卡比較好」不是單一答案：完成時間／吞吐優先時目前傾向 TP=1，第一個 token 則有相反取捨。

這不是把 planner 寫死選 1，也不是為了讓 2 勝出而改權重。未測高併發、不同 remaining work 或不同 GPU 資源限制，不外推所有負載的最佳 TP。

## 3. 冷 engine 與暖機不可混淆

| TP | Engine construction 秒 | Short warmup 秒 | Engine + short warmup 秒 |
| ---: | ---: | ---: | ---: |
| 1 | 77.612 | 111.511 | 189.127 |
| 2 | 88.561 | 112.071 | 200.639 |

這是每個 TP 各一次 fresh ephemeral container 的 engine 計時，不是重複量測的典型啟動成本，也不包含完整 container／Python import 時間。主機權重 file cache 已被前輪讀取；「冷」指新 engine／容器內 kernel cache，不是整台主機冷啟動。

首次 JIT／autotune 都保留在未計時階段，不把三分鐘的暖機混入約十幾秒的服務時間。正式重建要另外預先固定 cache policy，並量測 notice→ready／restore／持續輸出，不能直接拿 engine construction 冒充整個停機成本。

TP=2 日誌顯示 P2P custom all-reduce 不可用，實際走 PYNCCL；SM120 symmetric-memory communicator 也不可用。缺少 E=40 的已調校 kernel config。這些是本機結果的背景，不代表 TP=2 在其他硬體也必定較慢。

## 4. 恢復 pilot 的範圍

[Replay protocol](../results/moe_spotserve_tp12_pilot_20260913/replay_protocol.json)事前註冊，兩個手動 target 的 GPU 恢復測試皆通過；這只放行 token-replay 能力，不放行正式效能對照。

GPU 0 的 source 真正生成到指定進度後，以 engine pause keep／clear_cache=False 凍結。查 EngineCore 的權威 metadata，保留實際 generated prefix；若超過設定的 256 tokens，記錄 overshoot，不稱為精確 256-token barrier。已算出但尚未被 consumer 收到的 tokens 要 backfill，避免漏掉 source 與 target 交界的輸出。

同一份 frozen prefix 依序交給 TP=1（GPU 1）、TP=2（GPU 1／3）以 TokensPrompt 重算 KV 並接續。這是手動的兩種恢復能力測試，不是 planner 選擇、兩次真正的 spot GPU revocation 或 NIXL KV restore。

每個候選分開從自己的啟動時計時，包含 cold JIT；第二個候選的時間不累加第一個候選的執行／cleanup。除了第一個恢復 token，也量前四個 tokens、最大 delivery gap 與完成時間，避免第一個 token 回來後又長時間卡住卻被宣稱服務已穩定恢復。

實際 source：consumer 已收到 256 個 tokens，EngineCore 凍結時已有 **258 個，overshoot=2**；notice→freeze acknowledgement 為 43.6 ms。因此 strict_requested_boundary_verified=false，不能聲稱精確在第 256 個中斷。先 backfill 尚未送達的兩個 source tokens，再把同一份 4354-token prefix（4096+258）交給兩個 targets，各生成剩餘 254 個。兩次 client 合併輸出均為 512 個，token hash 都與不中斷的 TP=1 參考相同。

| Target TP | Container→ready 秒 | Replay→首 token 秒 | Replay→前四 tokens 秒 | 最大 delivery gap 秒 | Replay→完成秒 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 85.439 | 110.484 | 110.540 | 0.029 | 116.220 |
| 2 | 95.077 | 41.324 | 110.645 | 69.266 | 117.466 |

Container→ready 與 replay 是不同區段。從各候選自己的啟動時計，到首 token 約為 195.9／136.4 秒，到完成約為 201.7／212.6 秒；不是從最初 source freeze 累計的兩次 outage。每個 target 只做一次冷恢復，不把這些數字當成平均遷移成本或顯著差異。

**TP=2 首 token 較早，但之後停了約 69 秒。** 日誌同期有 fused_moe_kernel 首次 JIT 警告，前四個 tokens 回來的時間兩者皆約 110.6 秒。這支持「只用第一個恢復 token 判定服務已穩定恢復會失真」；compiler 警告與停頓對應，但未用 GPU profiler 精確分攤因果。正式方案需預先固定 target warmup／cache policy，不能事後只替其中一版暖機。

本 harness 已驗證 metadata→prefix backfill→target 拼接；這不是 production SLLM HTTP client 的 delivery／exactly-once 證明。Source 在兩次 counterfactual 測試期間保持邏輯凍結且 GPU resident，未撤走實體卡，未呼叫 planner 或正式 actor 套用。所有自建容器已退出，既有 GPU 工作未停止。

## 5. 放行限制

- 這是 closed-batch pilot，不是到達率／排隊時間／SLO 容量量測；不能把 batch throughput 當成已證明的 open-loop 容量。
- 固定順序 TP=1→2、每情境三批；P95 小樣本接近最大值，不做顯著性或 production 尾延遲宣稱。
- GPU 2 仍有其他工作；GPU 0／1 也有小型既有 worker。沒有停止任何既有程序，四卡排他使用未確認，正式四卡 migration 尚未跑。
- 原始 4096／8192 + 512 profiles 不能直接套用到凍結後更長 prompt／更短 remaining output。Replay worker 的 max_num_seqs=1 也不同於本輪 profiles 的 4；不能把它冒充相同 runtime 的完整 transition matrix。
- 新[成本資料接口](../sllm/spot/moe_tp_profiles.py)要求完整計時稽核、匹配 prompt／output／concurrency、相同 checkpoint／runtime、相同實體 GPU 群組與 frozen prefix 的已驗證 transition evidence。缺資料不是零成本。
- 尚未完成 planner 的實測選擇→正式 actor／engine 套用、多 target 背景工作、live expert 成本接入、精確 preemption barrier 或 Original／MoE-aware 正式配對。手動 replay 通過不代表上述鏈路完成；所有本輪報告的 formal_experiment_eligible=false。

## 6. 證據與測試

[摘要 JSON](../results/moe_spotserve_tp12_pilot_20260913/service_cost_summary.json)、[TP=1 JSON](../results/moe_spotserve_tp12_pilot_20260913/tp1_profile.json)、[TP=2 JSON](../results/moe_spotserve_tp12_pilot_20260913/tp2_profile.json)、[TP=1 log](../results/moe_spotserve_tp12_pilot_20260913/tp1_profile.log)、[TP=2 log](../results/moe_spotserve_tp12_pilot_20260913/tp2_profile.log)。

[Replay JSON](../results/moe_spotserve_tp12_pilot_20260913/replay_pilot.json)、[Replay log](../results/moe_spotserve_tp12_pilot_20260913/replay_pilot.log)、[共享 prefix／runtime 稽核](../results/moe_spotserve_tp12_pilot_20260913/replay_pair_audit.json)、[執行程式碼 SHA-256](../results/moe_spotserve_tp12_pilot_20260913/code_provenance.json)。

最後重跑十一個相關 CPU 測試檔案，**99 passed**：[verification log](../results/moe_spotserve_tp12_pilot_20260913/final_verification_pytest.log)、[完整命令](../results/moe_spotserve_tp12_pilot_20260913/final_verification_command.json)。這包括新 profile aggregation／lookup／replay／legacy flags 契約，不宣稱整個專案所有測試都通過。前一輪不同選取集合的 102 passed 保留在[原 log](../results/moe_spotserve_tp12_pilot_20260913/pytest.log)，兩輪不相加。

工具：[profile_granite_moe_tp.py](../scripts/profile_granite_moe_tp.py)、[run_granite_moe_replay_pilot.py](../scripts/run_granite_moe_replay_pilot.py)。既有 Qwen 正式 raw 結果沒有被改寫。
