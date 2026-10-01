# MoE × SpotServe v2：正式結果與前置紀錄

日期：2026-09-14（Asia/Taipei）。**三組正式 Original／MoE GPU 對照均完成，每組每版 n=3，共 18 個完整 runs。** 不覆寫 [v1](1_2A_2B_v1.md) 或舊 pilot；下方前置過程保留為歷史紀錄。

## 最新正式結果

[完整 pipeline](../../results/moe_v2_complete_execution_20260914_attempt2/pipeline.json) 為 passed：cost validation、shape validation、single-event pilots、two-event overall pilots、formal ablation 與 formal overall 均通過。

| 組別與指標 | Original：秒 mean ± sample SD | MoE-aware：秒 mean ± sample SD | 完整正式 runs |
| --- | ---: | ---: | ---: |
| 第一組 overall stream complete | 327.1574 ± 0.7852 | 326.2688 ± 1.1085 | 每版 3 |
| 2A notice→complete | 10.7403 ± 0.0378 | 10.6310 ± 0.1249 | 每版 3 |
| 2B notice→complete | 95.6378 ± 0.1195 | 95.7399 ± 0.2487 | 每版 3 |

2A 六次皆選 GPU2 TP1；2B 六次皆選 GPU1 TP1，configuration 實際套用與完整 512-token reference 均驗證。Overall 六次均各有兩次固定 trace 中斷。兩版均用同一真正 Granite MoE；Original 為共用 generic 成本修正、未使用 route-neighbour 評分的基準，不是 dense，也不冒充歷史原版程式完全未改。

兩版選同一方案後共用恢復 executor；恢复是 host token handoff + GPU replay，沒有 direct KV restore、physical expert placement 或少搬 expert 權重的證據。小幅 timing 差不宣稱 MoE-specific 收益。完整 mean／SD、paired differences、raw runs 見[自動生成正式報告](../../results/moe_v2_complete_execution_20260914_attempt2/report.md)。

下一輪已依使用者核准執行[TP1／2 前提 microbenchmark](moe_tp12_microbenchmark_results_20260914.md)，先檢查成本交叉及 warm-status 控制，不重跑已完成的 18 次，不先擴充 fleet。

## 前置過程（歷史紀錄，當時的進行中／0 runs 不代表最新狀態）

## 實驗範圍

### 使用者核准後的接續實作與執行（2026-09-14；進行中）

使用者已核准補成本預測、接通切換、小試跑後正式三次。本次新增 [routing predictor](../../sllm/spot/moe_route_cost.py)、[local GPU adapter](../../scripts/moe_gpu_runtime.py)、[executor actor](../../scripts/moe_recovery_actor.py) 與 [runner](../../scripts/run_moe_calibrated_experiments.py)。這是本機 Podman／實體 GPU adapter，不冒充已部署 Ray cluster。

固定六種 training、兩種 validation，以及完全不參與 target 成本訓練的 formal workload，皆為明示的 synthetic repeated-token 4K prompts。Generic 用同 shape／背景下的 training service mean；MoE-aware 用 live prefix expert histogram 的固定 k=3 nearest-neighbour 成本，沒有把估計值偽裝成三次量測。validation 比較 routing 移除／打亂；不靠結果調 k 或強迫不同 plan。

GPU adapter 切流量前檢查實際 rank GPU UUID、MoE module、TP、runtime signature，以及 worker 真正收到的 host token prefix；GPU prefill／remaining decode 在切換後計時，避免 replay 首 token與完整 latency 重複相加。Native freeze 改為一次性 barrier，另測 resume 後可繼續；TP2 correctness canary 的手動指定只算功能檢查，不算優化版勝出。

第一輪因 callable RPC 被安全序列化拒絕而失敗；未開啟 insecure serialization，改 opt-in named worker-extension RPC 後重跑，保留 [attempt1](../../results/moe_calibrated_execution_20260914/calibration.json)。[attempt2 calibration](../../results/moe_calibrated_execution_20260914_attempt2/calibration.json) 已完成：九種 source workload 的實際 256-token freeze 均正確，四個候選各有三次獨立 cold-ready 成本；八種 calibration/validation workload 各三次 remaining replay 均接回完整 512-token reference，計時中 JIT warnings 為零。這些是獨立校準，不是正式對照 runs。

首版保留工作量診斷的 service latency MAE：generic **39.88 ms**、真 routing **53.04 ms**、打亂 routing **53.70 ms**。目前 routing 沒有較準，打亂後亦相近；這不是 MoE-specific 收益，不據此調 k 或強迫不同選擇。見 [cost validation](../../results/moe_calibrated_execution_20260914_attempt2_validation/cost_validation.json)。

首版 target routing capture 關閉，不能沿用到會再次中斷 target 的整體比較；[unified-capture calibration](../../results/moe_calibrated_route_capture_20260914/calibration.json) 已重測全部 capture=true，匹配的 held-out service MAE 為 generic **19.11 ms**、routing **24.31 ms**、shuffled **22.49 ms**，仍沒有預測改善。

**ready 成本稽核發現問題，未放行正式：** TP1 初輪 cache population + kernel JIT 令 warm-ready 約 196 秒，後兩輪約 85–87 秒；TP2 約 95 秒。直接混合平均會因不同編譯 cache 狀態偏選 TP2，不是有意義的 TP 策略差異。已中斷本實驗自己的第一個 pipeline 子程序，保留 [pipeline failure](../../results/moe_v2_complete_execution_20260914/pipeline.json)，未停止外部 store 或產生正式 runs。已完成 [matched-cache calibration](../../results/moe_matched_cache_calibration_20260914/calibration.json)：**四個候選全部重新量三次獨立 ready trials**；在各候選所有校準形狀完成 cache population 後才量，不按 latency 挑掉某次舊資料，也不覆寫舊數字。原 service 校準／參考輸出保留且明確引用，十二個新 workers 均 exit=0 且恢復輸出正確。

| 候選 | 新 ready 秒：mean ± sample SD | training service 秒（4352+256） |
| --- | ---: | ---: |
| GPU1 TP1 | 85.6554 ± 0.3657 | 6.1979 |
| GPU2 TP1 | 85.3004 ± 0.6436 | 6.1961 |
| GPU3 TP1 | 85.5540 ± 0.4187 | 6.2326 |
| GPU1/3 TP2 | 95.9281 ± 1.2982 | 7.3711 |

在尚未測 target formal output 的 frozen-source 輸入上，兩版目前 2A 都選 GPU2 TP1、2B 都選 GPU1 TP1，整體初始 forced-recovery 也都選 GPU2 TP1；這是事前 planner prediction，不是正式部署或效益。新的 [sequential pipeline](../../scripts/run_moe_experiment_pipeline.py) 已於 attempt2 接續 GPU shape validation，後續須依序通過實際 TP2 correctness、單事件恢復及兩事件 workload pilots，才啟動正式重複。

整體合法候選為 GPU1／2／3 各一個 TP1 與 GPU1/3 TP2，只建立選中的群組；同 TP 的多個實體候選先取該 TP 最低成本，避免 backend configuration 去重覆蓋掉較好 target。2B 保留 GPU1 TP1／GPU1/3 TP2，避免把額外 migration-target 選擇混入 TP ablation。強制回收均先取得 host snapshot、停止本實驗 source，再建立／切至選定 target；native 單事件 gap 使用實際 source boundary token delivery 時戳，非只拿 pause ack 當最後 token。

四卡既有 store 保留，後續兩版均採相同同步 scheduler、routing capture 與 task compiler cache；不是四卡排他環境。本輪相關十八個測試檔已 **220 passed**（後續修改需再回歸）。

**GPU 剩餘 shape 驗證已通過：** 在兩個 held-out workloads 實際 native freeze=128／384，四個 target 候選各做三次 replay，共 48 個完整恢復請求，皆與完整 512-token reference 相同、timed JIT warnings=0。scaled prefill TTFT + calibrated decode-step 的 generic/routes 合併平均相對誤差約 **0.64%**，兩訊號各自亦須通過預註冊 ≤20% gate；未以 validation 調參。這支持固定 wall-clock 中斷時使用實際剩餘長度成本，不是 MoE 預測優於 generic。見 [shape validation](../../results/moe_v2_complete_execution_20260914_attempt2/validate-shape/shape_validation.json)。當前正進行真實 GPU recovery pilots；正式平均值尚未完成。

| 項目 | 比較 | 每版每情境需要 | 本次正式完成 |
| --- | --- | ---: | ---: |
| 1 整體 | 共用成本修正後的未加 MoE 版／MoE-aware 版 | 3 runs | 0 |
| 2A migration | generic／route-aware target 選擇，全 TP=1 | 3 runs | 0 |
| 2B reparallelization | generic／route-aware TP=1/2 選擇，單副本 | 3 runs | 0 |

模型固定 GraniteMoeForCausalLM：ibm-granite/granite-3.1-3b-a800m-instruct，revision `a02780686e08a03fe0d2679a293b5c74a90efa89`；BF16、40 experts、top-k=8，兩版都不是 dense。模型 config 與真實 worker expert modules 均要驗證。

## 已實作

1. [MoE 辨識](../../sllm/backends/vllm_capability.py) 以本機 checkpoint 設定為準，不只看名稱。local dense 設定不會因路徑含 moe 被當 MoE；壞掉或缺失的本機設定拒絕推測。
2. [成本契約](../../sllm/spot/moe_tp_profiles.py) 比對 checkpoint revision、runtime、實體 GPU、remaining workload、prefix 與計時稽核；新增 independent route-conditioned service lookup。
3. [measured planning bridge](../../sllm/spot/moe_ablation.py) 接到現有 TP scorer，兩版都最小化 `ready + install/replay + remaining mean latency`。generic 修正兩版共用；只有 MoE-aware 加 routing 條件化成本。沒有 GPU-count bonus，不拿 closed-batch throughput 推算排隊容量。
4. 2A 必須提供三個不同實體 TP=1 targets 的校準；2B 僅 TP=1／2、PP=DP=replica=1、無 EP，選定 plan 明確綁到校準 GPU 群組。observe-only expert placement 不計入額外權重搬移成本。
5. [executor](../../sllm/spot/reparallelization_executor.py) 新增 opt-in runtime verifier；在切流量前核對 actual TP、GPU、MoE ranks、runtime 與 installed prefix。不符合則清掉本次新 target；預設非資源重用路徑保留舊 worker。新的 local GPU actor 已接入 actual worker named RPC 與 host prefix receipt；仍須 GPU pilot 證明端到端可執行，不能 echo plan 當證據。
6. 舊 migration 指標在沒有 locality 觀測時改標 unavailable，而非 target placement coverage。只改未來輸出的語意，沒有重寫舊實驗數字。

## 驗證結果的意義

CPU 契約測試覆蓋：相同成本仍同選 TP=1、route-conditioned 成本改變時可選 TP=2、重建太貴改回 TP=1、三個完整 targets 可因實测服務成本分出優劣、校準 GPU 不可用時不偷換群組、缺成本／不同 prefix／cache／背景／未滿三次即拒絕、實際 runtime 不符時不切流量。

**CPU 測試數字是人工 fixture，不是 Granite 效能或 MoE 收益。** 早期 route-conditioned lookup 是 exact measured-cost 契約；本次另實作獨立校準的 forecasting 模組，以既有六 workload 的真實成本推估未測 formal prefix，不把預測值偽裝成 exact measurements。它衡量條件化整體服務時間，不是逐 expert queue、expert weight migration 或 physical dispatch traffic。

先前十三個相關測試檔 157 passed；本輪加上實際 frozen-batch／CUDA UUID、native barrier、routing probe 與 scheduler runtime 的保護測試後，十五個測試檔 **187 passed**。初次擴大回歸為 151 passed / 1 failed；失敗可獨立重現於舊 context-migration 無觀測卻標 coverage 的指標。修正 unavailable 語意後全部通過；舊輸出見[CPU regression log](../../results/moe_ablation_readiness_20260914/pytest.log)，本輪見[187-test regression](../../results/moe_spotserve_v2_calibration_20260914/pytest.log)。不同階段的測試數字不相加。

對本機 Granite 權重目錄的 read-only 檢查已確認 config 能被辨識為 MoE；未測 shape 只標 static catalogue。將舊 pilot 嘗試送入新 gate，正確被拒絕，沒有產出虛假的正式候選。見[local readiness log](../../results/moe_ablation_readiness_20260914/local_readiness.log)。

## 本次 GPU 前置量測

本輪使用四張 RTX 5070 Ti，各 16 GB；保留 GPU 0/1 的既有 sllm-store（各 222 MiB），未停止其他工作。不是四卡排他環境。先同時建立四個真實 Granite MoE TP=1 engines，核對 CUDA UUID 與實際 worker model/expert modules；全部完成短暖機及 frozen-shape 暖機後才一起放開計時 barrier。

工作量：同一份真實 MoE 舊輸出的前 256 tokens，加原 4096-token prompt，共 4352-token input，再生成 256 tokens；C=1、BF16、V1/eager/TRITON、max_model_len=8704、max_num_seqs=1、batch cap=2048、memory utilization=0.8，無 offload 或 prefix cache。校準時這份 prefix 是從舊實際 258-token snapshot 裁出，不把裁切冒充 native freeze。**隨後本輪三次 live native freeze 恰好得到同一個完整 4352-token prefix hash**；證據與限制分開記錄如下。

| 配置 | 實體群組 | 完整量測批次 | Latency 秒：mean ± sample SD | TTFT ms | Output tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| TP=1 | GPU 0 | 3 | 5.9443 ± 0.0208 | 162.52 | 43.07 |
| TP=1 | GPU 1 | 3 | 5.9320 ± 0.0117 | 149.88 | 43.15 |
| TP=1 | GPU 2 | 3 | 5.9885 ± 0.0153 | 150.06 | 42.75 |
| TP=1 | GPU 3 | 3 | 5.9833 ± 0.0063 | 149.77 | 42.78 |
| TP=2 | GPU 1/3 | 3 | 6.9604 ± 0.0326 | 131.93 | 36.78 |

共 15 個請求、每個完成 256 tokens，跨四個 TP=1 群組與 TP=2 的 output hashes 全部相同。每組三個計時窗口完整，計時中 JIT/autotune 警告均為 0。這是每 engine 三個 warmed batches，不是各版本三次獨立完整 spot runs。TP=2 在四個 TP=1 engines 完成後另測；TP=1 四群組同時量測，故未宣稱完全一致的外部環境或 causal TP speedup。

相對 GPU 1 的 TP=1，本輪 TP=2 總 request latency 高 **17.34%**、TTFT 低 **11.97%**。對這個 C=1、4352+256 的候選服務工作量，「更快出第一個 token」不等於「更快完成」；不能預設 TP=2 較好。這些不是 MoE-aware／Original speedup，尚缺三次獨立 transition 成本，不能聲稱 planner 的最終選擇。engine construction 與第一次 kernel 暖機也不能直接冒充 notice→ready。

計時 logs 顯示上述 service engines 使用 async scheduling；下方精準 source freeze 則為 synchronous custom scheduler。不能跨這兩種模式直接沿用成本。runtime signature 本輪因此加入 actual `async_scheduling`／`scheduler_cls`，新 profiler 從實際 engine 讀值；本批 raw reports 未包含這兩欄，保留原始結果，不事後編造 scheduler observation。其 latency 可供前置分析，但不通過新版完整 runtime deployment gate。

### 精確 native freeze：三次通過

新增 opt-in [NativeFreezeScheduler](../../sllm/spot/vllm_benchmark_scheduler.py)；在 native scheduler 接收實際生成結果後、排程下一步前，對仍在進行的請求於第 256 token pause。受控條件為 synchronous、C=1、無 speculative decoding；overshoot 或 async placeholders 直接拒絕，不靠事後裁掉輸出。

[Live freeze probe](../../scripts/check_granite_native_freeze.py) 在 GPU 0 的真實 Granite MoE engine 上執行三個不同、原本應產生 512 tokens 的請求。全部實際 freeze=256、overshoot=0、allocated KV blocks>0；消費端已收 256 tokens、backfill=0，並與同 engine 的不中斷 512-token reference 前綴相同。三個 frozen full-prefix hashes 均為 `57bc401dd299825fd21f091fad001c69a84b264f5e4bfe563be7095a8e7c6282`，與上述校準 batch 相符。

本項沒有 target replay／直接 KV restore、沒有 planner-selected deployment，也沒有真實回收 GPU。首次 probe 的 source routing capture 關閉；不能從 MoE 模組推導出真實 expert histogram。

### 真實 runtime routing + native freeze：另三次通過

另建立 source engine，開啟 `enable_return_routed_experts=True`，維持 V1、synchronous 與相同精準 barrier。三個不同 live requests 均 freeze=256、overshoot=0、reference prefix 相同，並從實際 frontend routed-experts chunks 取得 histogram；沒有用模型設定或事前 canary 的 histogram 代替 live 資料。

每次均觀測 **4351 個 routed input tokens × 32 層 × top-k 8**，expert IDs 全落在各層 0..39，所有層的 assignment count 一致，共涉及 1123 個 layer/expert keys。三次 histogram hash 均為 `0418662803fcd05ab2644ef7ed357ac5ec8300d6c5a61e846620f060a7f15105`。4352-token frozen prefix 的最後一個新 token 尚未作下一步 input，因此沒有冒充觀測到它或剩餘 256 tokens 的未來 routing；這不是 freeze overshoot。

證據：[routing + freeze JSON](../../results/moe_spotserve_v2_calibration_20260914/native_routes_freeze.json)、[完整 log](../../results/moe_spotserve_v2_calibration_20260914/native_routes_freeze.log)。三次 live routing 是 MoE／訊號取得的 gate，**不是三次正式 migration 或 reparallelization**；也沒有量到逐 expert queue／GPU kernel time 或 physical dispatch traffic。

第一輪四個容器因 NVML/CUDA UUID 的 GPU- 前綴格式差異，於載入 engine 前被防錯檢查拒絕。保留 attempt1 logs，統一 UUID 正規格式後重跑，不取消群組檢查。新 profiler 可以載入實際 calibration token batch，不再以同長度的任意 synthetic prompts 替代。

證據：[protocol](../../results/moe_spotserve_v2_calibration_20260914/protocol.json)、[actual prefix batch](../../results/moe_spotserve_v2_calibration_20260914/frozen_batch_4352_256.json)、[GPU 0 profile](../../results/moe_spotserve_v2_calibration_20260914/gpu0_profile.json)、[GPU 1 profile](../../results/moe_spotserve_v2_calibration_20260914/gpu1_profile.json)、[GPU 2 profile](../../results/moe_spotserve_v2_calibration_20260914/gpu2_profile.json)、[GPU 3 profile](../../results/moe_spotserve_v2_calibration_20260914/gpu3_profile.json)、[TP=2 profile](../../results/moe_spotserve_v2_calibration_20260914/tp2_gpu13_profile.json)、[cross-TP check](../../results/moe_spotserve_v2_calibration_20260914/cross_tp_summary.json)、[native freeze proof](../../results/moe_spotserve_v2_calibration_20260914/native_freeze.json)。同目錄保留 raw native JSON 與完整 logs。

## 早期放行檢查（歷史快照；最新狀態見上方接續實作）

- 本次已完成上述四卡 warmed-service 校準；沒有正式 Original／MoE runs，也沒有停止既有 GPU 工作。四卡排他或既有服務可受控共存的正式環境條件仍需確認。
- 舊 4096/8192+512 profiles 不等於 freeze 後的 4352/8448+256；舊 replay 每 TP 只有一次、max_num_seqs 不同，不能當成三次匹配的 transition 成本。不能補零。
- 2A 缺三個 target 在同背景/cache 下的 routing 條件化實測與校準器。只有全部 experts 的 coverage 對照仍無辨識力；單純增加 target 不會補足。
- 尚需保留 workload、routing 訊號移除／打亂診斷：即使條件化 lookup 有收益，也要排除只是 prompt lookup 更細、背景負載或其他 generic 效應。不能直接宣稱 MoE-specific 因果。
- C=1 synchronous 精確 native barrier 已三次通過；這不等於 C>1 整批 freeze、target 恢復後 client 拼接、planner→正式 actor GPU deployment 或不同 TP replay 的端到端驗證。
- 第一組連續負載／多次 spot churn 不在目前單事件、單 frozen batch bridge 的能力範圍。需註冊並驗證固定 availability trace、剩餘工作成本與實體群組後才開始三次正式量測；source 可保留的自主 replan 另需 keep 候選。
- 舊 `run_tiny_batch_recovery.py` 的固定 TP／target 假設仍未改成新版 GPU runner；新 CLI 只會讀資料、選 plan，不會啟動 actor。不要用舊 harness 的模式標籤冒充新實驗已套用配置。

因此 `formal_experiment_eligible=false`。不補造平均值、speedup 或顯著性。

## 早期下一步紀錄（已由本次使用者核准接續執行）

**本輪停在放行階段，未啟動正式三組。** 已有同 prefix 的真實服務時間與 live expert routes，但目前 route-conditioned lookup 仍只有契約，沒有能從 routes 推估不同 target／TP 成本的校準器；把相同 service profile 加上 route hash 並不能產生新的 MoE 訊號。重跑舊 modes 三次也不能補上這個缺口。

下一步不是單純增加 repetitions：需要先實作並獨立驗證 routing 成本預測／背景校準與保留 workload 診斷，再接通 planner-selected GPU deployment、非重疊 transition 計時與恢復後 client correctness；整體比較另需連續負載與固定多事件 trace。若不補實作，現有設計只能誠實報告沒有可驗證的 MoE-specific 決策差異，不能宣稱已完成有意義的新版 ablation。[本輪機器可讀狀態](../../results/moe_spotserve_v2_calibration_20260914/status.json) 列出各 gate。

這些需要補校準器／正式 runner，而不只是執行既有實驗指令；先確認後續實作方針再擴充。小試跑只用於檢查意義與放行，不算正式三次。

三項正式對照每版每情境三次，採交錯配對順序；報告 run 級數據、mean、sample SD、配對差值、成功率與失敗，不把不同情境混成重複。保存 source/target、候選成本、實際 TP、preemption 時間與 prefix、notice→freeze、planner、ready、install/replay、首／前四 tokens、最大 delivery gap、完成時間及 trace/logs。

CLI 僅作 CPU 決策前置檢查：

```bash
python -m scripts.plan_moe_ablation --bundle calibration_bundle.json
```

Bundle 根欄位為 `kind`、`calibrations`、`query`、`available_gpu_indices`；各 candidate 含 `profile`、`transition`、MoE 所需 `conditioned_profile`。完整契約與**僅 CPU 範例**見 [tests](../../tests/spotserve_test/test_moe_measured_ablation.py)。CLI 通過仍不等於四卡、actuation 或正式實驗已通過。
