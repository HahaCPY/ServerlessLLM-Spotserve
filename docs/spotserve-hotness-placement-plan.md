# Hotness-aware Expert Placement 實作計畫

日期：2026-10-11

更新：2026-10-12，補充 request routing 差異、公平 latency comparison 與 candidate latency profiling table。

狀態：設計計畫，尚未實作或驗收。此文件補充 [MoE-aware SpotServe Planner](spotserve-moe-aware-planner.md) 的 expert-aware re-parallelization；已有的 physical remap 能力作為執行基礎。

## 1. 目標與範圍

第一階段新增 `hotness_balanced` placement strategy，使用真實 routing statistics 平衡同一個 deployment 內各 EP rank 的預測 expert 負載，並考慮重新放置 experts 的搬動成本。先在固定 EP size、同質 GPU、runtime 已支援的 expert partition/remap 模式上完成。

研究問題：在相同模型、拓撲、workload 與 resource trace 下，hotness-aware placement 能否降低 EP rank load imbalance，且其效益足以抵銷 expert movement 的成本？

本計畫的兩項必要前提：

1. **每個 request 的 routing history 不同。** Placement 使用 workload window 的聚合 hotness；latency 評估則將同一批 requests 在不同策略下重播並逐筆配對，另報告整體分布與 throughput，不預設每筆 request 都會變快。詳細比較方式見第 3.3 節。
2. **Candidate latency 必須透過實測 profiling 建表。** 表格依模型、硬體、parallel configuration、實際 placement 與 workload 條件查詢 latency distribution/throughput，並在獨立 workload 上驗證預測誤差後才啟用 latency ranking。缺少可比 profile 時回到合法 capacity/movement 決策，不用 hotness imbalance 或共用預設值假裝已量得 candidate latency。表格與 lookup 設計見第 4.5 節。

後續再加入 source-rank-conditioned routing statistics，最佳化 expert locality 與 remote dispatch。第一階段的負載平衡不保證 A2A payload 下降；負載平衡與通信最佳化需分別驗證。

本次不新增 expert replication 或 redundant hot experts。既有 EP resize、KV restore 與跨 failure domain remap 可在固定 EP 策略驗收後做整合回歸。

## 2. 現況與待修項目

以下依目前工作目錄的程式碼整理；其中 `hotness` 雛形仍有未提交修改。

| 位置 | 已有能力 | 本計畫需補強 |
|---|---|---|
| `sllm/spot/moe_placement.py` | `ExpertPlacementPlan`、linear/round-robin placement、hotness 排序雛形、movement diff/bytes/cost | 統一 hotness contract、負載平衡分配、選擇理由與 stability gate |
| `sllm/spot/reparallelization.py` | 各 topology candidate 的 workload score 與 expert movement penalty；latency 取自 candidate config 或共用預設值 | 建立實測 latency/throughput profiling table 與 lookup，接入 placement 預測負載，保留各 candidate 的 placement 決策證據 |
| `sllm/routers/roundrobin_router.py` | replan、runtime metadata 與執行協調 | 為同一次 replan 擷取一致的 routing/placement snapshot |
| `sllm/backends/vllm_backend.py` | runtime apply/verify、由 config 推導的 placement metadata | 避免從 strategy 名稱推導出尚未實際套用的 placement |
| `sllm/vllm_expert_remap.py` | DP EngineCore metadata aggregation 與 remap 協調 | 保留統計 provenance，檢查跨 TP/DP 的 aggregation 語意 |
| `sllm_store/vllm_patch/runtime_moe_metadata.patch` | runtime top-k routing、global/recent histograms | 補足 window、sample count、freshness 與 rank aggregation metadata |

目前 `hotness` 是先按熱度排序，再切成連續區段分給 ranks，因此可能把熱門 experts 集中到同一個 rank。planner 會加總 global/recent 原始 counts，backend 推導路徑則讓 recent 覆蓋 global，兩邊尚未共用相同語意。

目前 runtime 的 recent histogram 來自 `_RECENT_EVENTS = deque(maxlen=64)`，代表最近 64 個 routing events；每個 event 的 token 數可能不同，也不能直接稱為最近 64 秒或 64 個完整 decoding steps。

`ExpertPlacementPlan.estimated_load_balance_penalty` 與 `estimated_dispatch_cost` 已有欄位，但目前建構 plan 時未填入 hotness 最佳化結果。新增策略需要接通 candidate scoring，不能只輸出額外 metadata。

## 3. Hotness Contract

### 3.1 統計來源與範圍

placement 使用 deployment-level 的 `global_expert_hotness` 與 `recent_window_expert_hotness`。`per_request_expert_route_histogram` 保留給 migration/recovery target ranking；不以單一 request 的歷史直接代表整個 deployment 的負載。

統一 expert key 為 `layer:{layer_id}/expert:{expert_id}`。無法辨認 layer 的 `expert:{id}` 不可默默套用到所有 layers；需明確標記 granularity，必要時退回合法 baseline。

每次 replan 先凍結一份 snapshot，所有 candidates 使用同一份資料。snapshot 至少記錄下列語意；欄位命名在實作時集中於同一個 contract：

- deployment/engine identity、TP/DP/EP shape、placement epoch/fingerprint。
- routing source、counter unit、aggregation scope、histogram key granularity。
- 最近事件數、各 layer 的 routed-assignment sample count、最後 routing observation time。
- global counters 的起始或 reset marker，避免 actor 重建後接續錯誤的歷史。
- current runtime placement、實際 resident expert bytes，以及資料是否完整。

counter 的基本單位是 top-k routed assignments，一個 token 可以貢獻多個 expert counts。需驗證同一批 token 在多個 TP workers 上是否被重複記錄；獨立 DP token streams 可以加總，重複觀測不可重複加總。重新取得 metadata 的時間不等於最後一次 routing observation time。

### 3.2 正規化

每一層分別正規化 global/recent histogram，避免 actor lifetime 或 window 大小主導 score：

```text
recent_share[l,e] = recent_count[l,e] / sum_e(recent_count[l,e])
global_share[l,e] = global_count[l,e] / sum_e(global_count[l,e])

hotness[l,e] = alpha * recent_share[l,e]
             + (1 - alpha) * global_share[l,e]
```

`alpha` 是可設定參數。`0.8` 可作初始實驗值，需與 recent-only (`alpha=1`) 做 sensitivity check；這個混合比例不是從 runtime 自動學習而來，也不是已驗證最佳值。

global 包含 recent，因此這是兩種歷史分布的加權 predictor，不應當成兩份獨立 samples。若只有一份有效 histogram，使用該分布並記錄來源；資料不足、過期、非有限數或 reset 不明時，不自動執行 hotness remap。

缺少 expert key 只有在 contract 確認「完整 sparse histogram，省略代表零」時才視為零；缺漏 layer 或不完整 worker snapshot 應標記 unavailable。

### 3.3 每個 Request 不同，Placement 以 Workload Window 決策

每個 request 的 expert routing history 都可能不同，同一個 request 在不同 decoding steps 的 routing 也可能改變。deployment hotness 描述近期 workload 的聚合分布，不代表所有 requests 使用相同 experts，也不承諾每一筆 request 的 latency 都降低。

```text
C[q,l,e,W] = request q 在 observation window W 內的 routed assignments
H[l,e,W] = sum_q(C[q,l,e,W])
```

先加總同一 window 內的 counts，再依 3.2 正規化；不要先把每個 request 正規化後等權平均，否則少量 routing 的短 request 會與大量 routing 的長 request 得到相同負載權重。

placement 用聚合 hotness 平衡未來一段時間的預測負載。request-level histogram 仍可協助 migration/recovery 選 target，但不把不同 requests 的 latency 直接相減當成 placement 效果。

公平的 latency comparison 是把同一批 requests、同一份 arrival trace 分別送進 baseline/candidate，讓 request `q` 在兩種策略下配對比較：

```text
latency_delta[q] = latency_candidate[q] - latency_baseline[q]
```

匹配 prompt/tokenization、generation budget、stop policy、sampling/seed 與 cache 初始條件，另記錄實際 output length、routing 與完成狀態。相同 seed 不保證跨 topology 的浮點計算產生完全相同輸出；輸出或長度改變時，先檢查 correctness，並將 latency 結果按長度分層或另外標示。

同時報告整批 workload 的 latency quantiles、throughput，以及按 prompt/output length 分層的配對結果。`p95(candidate) - p95(baseline)` 與 `p95(latency_delta)` 是不同統計，不可混稱。失敗、timeout 與 fallback 另外呈現，不能只留下雙方最快或成功的 requests；若報告成功配對子集，需列出 coverage。

## 4. Placement 演算法

### 4.1 容量限制下的 Greedy Assignment

每一層獨立分配，沿用 runtime 實際支援的每 rank expert slot 數量及 memory 限制：

1. 將 experts 依 `hotness[l,e]` 由高到低排序，平手時以 expert ID 排序。
2. 找出尚有合法 expert slot 與 memory capacity 的 ranks。
3. 將 expert 放到目前預測負載最低的可行 rank。
4. 負載平手時優先保留 current owner，再以固定 rank 順序決定。
5. 更新 rank load、expert count、weight bytes，直到所有必要 experts 被覆蓋。

```text
rank_load[l,r] = sum(hotness[l,e] for e assigned to r)
```

例如 EP2、每 rank 兩個 experts：

| Expert | Hotness | 排序後連續分組 | Greedy 平衡 |
|---|---:|---|---|
| 0 | 100 | rank 0 | rank 0 |
| 1 | 80 | rank 0 | rank 1 |
| 2 | 20 | rank 1 | rank 1 |
| 3 | 10 | rank 1 | rank 0 |
| Rank load | | 180 / 30 | 110 / 100 |

Greedy 是可解釋的 heuristic，並非全域最優解。單一 expert 特別熱門時，固定單份 expert ownership 無法把該 expert 的工作拆到多個 ranks。

### 4.2 合法性

- 所有 required experts 必須被覆蓋，並遵守 runtime 的 partition/replication 語意。
- 不因熱度而新增未受支援的 replication，或改動 rank 的 expert tensor capacity。
- 正確區分 SLLM actor replicas、vLLM DP ranks、TP shards 與 logical experts；不能把整個 SLLM deployment 的多個 replicas 視為同一個 EP group。
- 只對 runtime 支援的 topology 產生可執行 plan；不合法或無法完成分配時回報原因，不輸出部分 placement 供 apply。

### 4.3 同一 Topology 內選擇 Placement

對同一個合法 topology，至少產生 `current`（仍合法時）、`linear`、`round_robin`、`hotness_balanced` candidates。所有 candidates 使用相同 hotness snapshot 與 movement estimator。

逐層計算 load imbalance，避免只加總所有 layers 而掩蓋單層 bottleneck：

```text
imbalance[l] = max_r(rank_load[l,r]) / mean_r(rank_load[l,r]) - 1
B(P) = mean_l(imbalance[l])

J(P) = B(P) + movement_weight * normalized_movement_cost(P)
```

只對有有效 samples 的 MoE layers 計算，並回報其 coverage。MVP 同質 GPU、相同 expert compute shape 下先以 routed assignments 估計負載；量化模式、expert shape 或 GPU compute capacity 不同時，需額外校準。

movement cost 使用已有的 weight bytes 與 bandwidth/per-expert estimate，正規化的 reference cost 必須寫入 config/report。memory capacity 作為硬限制，第一版不額外加入 memory penalty 權重。

若 movement cost 未知，記錄 `unknown`，不能當成已知零成本。自動策略採保守的 move-count/byte budget；需要搬動但連 bytes 都不明時，先只產生建議 plan。

初始 greedy 以負載為主，再以 movement objective 在 candidates 間選擇。若實驗顯示搬動過多，可增加受容量限制的 pair-swap refinement；此項不列為第一版必要條件。

### 4.4 與 Topology Score 整合

`J(P)` 用於相同 topology 下比較 placement。不能單憑 imbalance 跨 EP sizes 排名，因為 EP1 的 imbalance 天然為零，卻不代表 throughput 最佳。

每個 topology candidate 附帶選定的 placement、per-layer/rank load、`B(P)`、movement cost 與 confidence，再進入既有 workload cost model。movement penalty 沿用現有路徑，只計入一次。candidate latency/throughput 採下節的實測 profiling table；`B(P)` 不直接換算成 milliseconds，也不能假設 imbalance 降低 20% 就讓 request latency 降低 20%。

同一次 replan 的最終 `ExpertPlacementPlan` 應重用 candidate scoring 時的內容，避免重新讀 hotness 後產生不同 mapping。report 需保留 strategy、snapshot identity、selection reason 與 score components。

### 4.5 Candidate Latency Profiling Table

此表是待實作項目。現有 `_score_parallel_candidates_for_workload()` 優先使用 `candidate.latency_estimate_ms`，沒有時使用共用 `latency_estimate_ms`；目前這些 config 值不足以證明不同 candidates 的真實 latency 差異。

表格估計的是「候選配置在某種 workload 條件下的 latency 分布」，不是由 expert hotness 推導出某個 request 的確定完成時間，也不為每個 request ID 各建一張表。

#### 表格索引與量測欄位

| 類別 | 必要條件/欄位 |
|---|---|
| Hardware/runtime | GPU type/count、interconnect/deployment domain、vLLM/patch version、dtype/quantization、MoE/A2A backend |
| Model | model ID/revision/config fingerprint、top-k |
| Parallel configuration | TP、PP、vLLM DP、effective EP；SLLM replica count 與 engine-level scope 分別記錄 |
| Placement | actual runtime placement fingerprint、strategy、profiled rank-load distribution |
| Workload bucket | prompt length、output length/budget、prefill/decode mode、controlled concurrency/arrival rate、observed batch distribution、routing skew/distribution |
| Cache condition | warm engine、prefix-cache policy、KV reuse/restore mode與初始 cache 狀態 |
| Measurements | service latency mean/p50/p95/p99、TTFT、inter-token latency、throughput、sample count、repetitions、variation |
| Provenance | profile ID/version/time、workload hash、routing observation window、raw measurement artifact |

`batch size` 以實際量測的 batch 分布與受控 concurrency 記錄；`max_num_seqs` 只是上限，不等於每一步實際 batch size。hotness 需包含 expert IDs 與 placement-induced rank load；只有一個 global skew scalar 無法區分「熱門 experts 集中在一個 rank」與「分散在多個 ranks」。

同一份 workload bucket 下的示意表如下，數字要由 profiling 填入，不預填假設效能：

| Profile | TP | vLLM DP | Effective EP | Placement | Workload | Mean Service Latency | p95 | Throughput |
|---|---:|---:|---:|---|---|---|---|---|
| P1 | 1 | 2 | 2 | linear，actual mapping A | W1 | 待量測 | 待量測 | 待量測 |
| P2 | 1 | 2 | 2 | hotness_balanced，actual mapping B | W1 | 待量測 | 待量測 | 待量測 |
| P3 | 1 | 4 | 4 | linear，actual mapping C | W1 | 待量測 | 待量測 | 待量測 |

`effective EP` 沿用 runtime 對 TP/DP/EP 的語意，不把 EP 當成額外獨立相乘的 GPU 維度。不同硬體、topology、workload、cache 或通信 backend 的 rows 不能任意互換。

#### Profiling 與 Lookup 流程

1. 固定 hardware/runtime/model，列出合法 topology 與代表性的 placement candidates。
2. 用共同 history window 產生各策略的 placement，凍結 mapping，再重播共同的獨立 profiling workload。
3. 每個 candidate warmup 後，在指定 concurrency/load 下量測並保存 raw per-request records、routing、batch、placement 與 counters；load/model startup/remap window 另量。
4. 彙整相同條件的 latency/throughput distribution，保存 sample count 與變異。需要 serving capacity 時另做 load sweep，找出穩定可承受的吞吐量。
5. planner 用目前 workload window 與 selected placement 查表，記錄 `profile_id`、匹配方式與 confidence，再填入 candidate 的 latency/throughput estimate。
6. 在未參與建表的 workload 上驗證 prediction error，通過後才開啟 profile-based latency ranking。

第一版先做指定模型/workload buckets 與已量測 placement 的 exact lookup。動態產生的 mapping 未命中時，標記 `profile_unavailable`；之後可加入經驗證的 bucket interpolation 或校準模型，但不得僅憑 strategy 名稱或 GPU 數匹配。

缺資料時保留已有的合法 capacity/movement 決策，該輪對比較集合一致停用無法公平比較的 latency/throughput score components，並輸出 fallback reason；不能只拿掉缺資料 candidate 的 penalty，讓它因為沒有量測而佔優。不將缺值視為零 latency，也不以共用預設值假裝各 candidate 已有相同的實測效能。

#### 服務時間與 End-to-End Latency

profile 的 service interval 定義為 engine 接受 request 到完成；若其中包含 engine 內的 batching/scheduling wait，需明確記錄。router admission queue、replan startup、weight remap、KV transfer/restore 等另外計時，避免重複加總已包含的時間。

lookup 必須記錄所用的 latency statistic：平均 serving cost 使用 mean，tail-latency objective 才使用指定 quantile；不在不同 candidates 之間混用 mean 與 p95。

```text
normal request latency
  = router admission wait + engine service interval

preemption-affected request latency
  = 實際 request timeline 上的等待、執行與 recovery intervals
```

preemption recovery 與 model loading 可能重疊，不能無條件把所有 costs 加成 request latency。mean/p95 的 profile 是特定條件下的統計；`p95(queue) + p95(service)` 不等於 `p95(end-to-end)`，尾延遲仍用完整 workload replay 的 per-request timeline 驗證。

throughput 優先讀取相同條件下的實測值。`replica_count * batch_size / latency` 僅是簡化估算，對 continuous batching 不能直接當實測 capacity；低 offered load 下的 completed requests/s 也不能當成系統最大 throughput。queue penalty 使用哪種 throughput 必須標示，並沿用一致的 scope。

## 5. Stability 與 Runtime 執行

新增以下可設定 gates，數值先由 synthetic cases 決定，再用 runtime experiments 調整：

| Gate | 用途 |
|---|---|
| `minimum_hotness_samples` | 避免樣本太少就改 placement |
| `hotness_max_age_s` | 避免使用過期 routing |
| `minimum_improvement_ratio` | 預測改善須超過門檻，抑制小幅波動 |
| `placement_cooldown_s` | 限制非必要 remap 頻率 |
| movement budget | 限制 moved experts/bytes 或預估 remap 時間 |

改善比例應固定以 `B(current)` 與 `B(candidate)` 計算；current 已平衡或改善量為零時保留 current。cooldown 與改善門檻用於效能最佳化；resource loss 導致 current placement 已不合法時，優先產生合法 recovery/redeployment plan。

有校準時間模型後，可再使用 amortization gate：預測時間節省乘上預期 request/batch 數，大於 movement time 加上 hysteresis 才執行。沒有時間模型時，不把 dimensionless imbalance 假稱為 latency savings。

執行流程：

```text
resource/preemption event or explicit optimization request
-> freeze routing + actual placement snapshot
-> legal topology candidates
-> hotness placement + movement evaluation
-> selection and stability gates
-> existing coordinated runtime remap / supported recreate path
-> actual placement readback + expert weight hash verification
-> commit placement epoch
-> subsequent migration/recovery target planning
```

第一版沿用現有事件驅動流程與手動觸發，不新增週期性背景重平衡。若啟用 vLLM EPLB，需明確協調 mapping ownership 與 epoch；第一版驗證使用單一 placement writer。

資料缺失且 current topology 未變時，保留已驗證的 current placement；current 已不合法時，選擇 runtime 支援的 deterministic baseline。fallback 需有原因與 metrics。

backend 的 `derived_from_model_config` 只能表示推導資料。策略名稱與預期 mapping 不足以證明 weights 已搬動；apply 後仍以 runtime resident placement、weight hashes 與 inference correctness 為驗收依據。

## 6. 實作分工與順序

| 步驟 | 預計修改位置 | 交付與驗收 |
|---|---|---|
| A. 統一 Hotness Snapshot | runtime patch、`vllm_runtime_metadata.py`、`vllm_expert_remap.py`、router | canonical keys、normalization、window/sample/freshness metadata、TP/DP 去重規則；所有 candidates 使用同一份 snapshot |
| B. `hotness_balanced` | `moe_placement.py` | 容量限制下的 greedy assignment、deterministic tie-break、per-layer/rank load、合法性檢查 |
| C. Candidate Profiling Table | 新 profiling script/schema、`reparallelization.py` | 真實 latency/throughput 建表、workload/placement lookup、缺資料退化、prediction error 驗證 |
| D. Movement 與 Selection | `moe_placement.py`、`reparallelization.py` | current/baseline/balanced 比較、unknown cost 處理、stability gates；選中的 mapping 可追溯至 score/profile |
| E. Runtime 整合 | router、backend、既有 remap hooks | 真實 physical remap、placement/hash verify、epoch 更新與失敗處理 |
| F. Metrics 與 Benchmark | `metrics.py`、analyzer、benchmark runner/config | predictor、movement、measured serving performance 分開輸出；通過下節驗收 |

追蹤 checklist：

- [ ] A. Hotness snapshot contract 與 aggregation 驗證完成。
- [ ] B. `hotness_balanced` 分配與合法性測試通過。
- [ ] C. Candidate latency/throughput profiling table、lookup 與 prediction error 驗證完成。
- [ ] D. Movement-aware selection、stability gates 與 candidate score/profile 整合完成。
- [ ] E. Runtime remap、actual placement、weight hash 與 inference correctness 驗證通過。
- [ ] F. 相同 requests 的配對 latency comparison、ablation/benchmark、metrics 與結果文件完成。

新增 config/metric 名稱與驗證入口都屬本計畫的待實作項目，尚不能當成目前可直接執行的選項。既有 `linear`/`round_robin` 行為應有回歸保護；現有 `hotness` 雛形需明確記錄遷移語意，實驗使用顯式 `hotness_balanced` 名稱。

## 7. 驗證計畫

### 7.1 不使用 GPU 的 Planner 驗證

擴充 `tests/spotserve_test/test_moe_placement.py`、`test_reparallelization_planner.py` 與 phase4 ablation。至少涵蓋：

- 上述 skewed case 從 180/30 改成 110/100，coverage 與每 rank slot 數不變。
- uniform/all-zero/missing/stale/invalid histograms；global/recent normalization 與單一來源退化。
- 多層 expert IDs 不混用、TP 重複觀測不放大 samples、counter reset 不造成錯誤熱度。
- EP1、runtime 支援的非整除 expert counts、slot/memory 不足、不支援的 replication。
- 相同 snapshot/config 多次輸出相同 mapping；平手保留 current owner。
- 搬動成本過高或改善不足時保留合法 current placement；unknown cost 有明確狀態。
- 候選 score、序列化 plan、執行 mapping 一致，movement penalty 無重複計算。
- profiling table 的 exact match、workload/placement mismatch、過期/版本不符/缺資料，不會產生虛構的零 latency 或重複計入 queue；分位數不直接相加。

新增預計入口 `scripts/run_hotness_placement_ablation.py`，輸出 baseline/candidate placement、per-layer load、movement diff、selection reason 與 assertions。

### 7.2 Runtime 功能驗證

先用 Qwen2-MoE-Tiny、固定 EP2 的已支援 remap topology，收集 routing 後產生 plan，再走既有 apply/verify。驗收：

1. routing 來源是 runtime observed top-k，snapshot samples/freshness 有效。
2. actual rank ownership 等於 selected plan，expert weight hashes 保持正確。
3. remap 前後固定 prompt、seed、sampling 設定的輸出符合既有 correctness tolerance。
4. 選擇搬動的案例有實際 moved shards/bytes；選擇不搬動的案例有可解釋原因。
5. active-request 模式沿用既有 step-boundary coordination，request completion、state 與錯誤處理通過回歸。

功能驗證通過後，再跑既有 DP2 coordinated remap、elastic EP resize 與 simulated cross-host 回歸；各路徑是否支援新策略分別記錄，不以單一路徑通過代表全部完成。

### 7.3 Performance 與 Ablation

預計新增 `benchmark_matrix_hotness_placement_performance.yaml`，比較以下 strategies：

| 組別 | 目的 |
|---|---|
| `linear` | runtime 預設 baseline |
| `round_robin` | deterministic 分散 baseline |
| 現有排序分組的 `hotness` | 診斷熱門 experts 集中的效果；以固定案例保留其舊行為 |
| `hotness_balanced`，movement weight 為零 | 隔離負載平衡效益 |
| `hotness_balanced`，啟用 movement/stability gates | 評估成本與穩定性 |

synthetic histogram 用於證明決策正確；performance claim 使用真實模型 routing。每組保持相同硬體、模型、TP/DP/EP、workload、trace、warmup、sampling 與量測長度。用獨立的相同 warmup/history 建立 training window，固定策略參數後再量測後續 workload；不可用 candidate evaluation 的未來 routing 反過來挑 placement。

先依 4.5 建立 candidate profiling table，再使用另一份 evaluation workload 做 end-to-end comparison。每筆 request 依 3.3 配對，保留 token lengths、完成狀態與 routing 差異；不能拿 baseline 的 request A 與 candidate 的 request B latency 宣稱 placement 效益。

比較 uniform routing、skewed routing、hotness shift 三類情境。實際模型若無法產生足夠 routing skew，應如實報告，不靠注入 histogram 宣稱 end-to-end 效能改善。多次重複並報告變異；remap window 與 post-remap steady state 分開呈現。

必要 metrics：

- 每層/每 rank 的 predicted load、max/mean ratio、load CV，以及 remap 後實際 observation window 的同類統計。
- hotness source、window、sample count、age、layer coverage、fallback/no-op 原因。
- latency profile ID、workload/placement match、estimate source/confidence、held-out prediction error 與 profile-miss rate。
- moved experts/weight bytes、estimated movement cost、measured remap duration、apply/verify 結果。
- request success、clean success、fallback rate、p50/p95/p99、throughput。
- request-level paired latency delta、配對 coverage、prompt/output-length 分層結果。
- remap window latency 與 post-remap steady-state latency。
- baseline/candidate A2A counters 的相同區間 delta；流量計數與 predictor 分開標示。

完成判準是策略影響了真實 runtime placement、correctness 通過且量測可重現。若 predicted imbalance 降低但 throughput/latency 沒改善，保留此結果並分析計算/通信瓶頸；不把功能驗收當成 performance conclusion。

## 8. 後續：Routing Locality 與 A2A

global hotness 沒有 token source 的資訊。要減少 remote dispatch，需保留 source rank 與 destination expert 的 joint statistics：

```text
route_count[source_rank][layer][expert]

estimated_remote_assignments(P)
  = sum(route_count[s,l,e] where owner_P(l,e) != s)
```

這個簡式先限 TP1、單一 owner 的 topology；TP sharding、replication 與跨 domain 情況需用 runtime 實際 dispatch semantics 計算 locality。資料應在 dispatch 前觀測，避免搬動後把 expert execution rank 誤當 token source。

之後在負載與容量限制下加入 remote dispatch objective，利用現有 destination-aware sparse A2A backend 重播相同 workload，比較 collective payload delta。先定義 payload 包含哪些 tensor/metadata、是否以 sender bytes 計數，避免 send/receive 重複加總。internode/network wire bytes 與 collective tensor payload 分開報告。

該階段的 claim 必須由實測 reduction 與 output correctness 支持。第一階段可宣稱使用 runtime hotness 做 movement-aware expert load balancing；A2A reduction 則等待這一階段的獨立驗收。
