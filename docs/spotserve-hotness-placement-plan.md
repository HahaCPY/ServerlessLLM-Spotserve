# Hotness-aware Expert Placement 實作計畫

日期：2026-10-11

更新：2026-10-14，補充 request routing 差異、公平 latency comparison、bounded profiling MVP、movement 單次計分、三階段驗收，以及完成後的 sensitivity/load/elastic/cross-domain/state-restore 回歸。

實作進度更新：2026-10-14，canonical hotness snapshot、`hotness_balanced` greedy、stability/movement gates、bounded exact profile lookup、movement 單次計分、metrics、無 GPU ablation、GPU physical-remap smoke、實測 profiles、paired latency benchmark、Phase 3 source-rank locality，以及列於本文件的完成後整合回歸均已完成。

狀態：**本計畫定義的 Phase 1–3、A–F 與完成後整合回歸均已完成。** 實測未顯示 `hotness_balanced` 有 latency/throughput 優勢，因此不宣稱 serving performance 改善；Phase 3 locality 則在輸出一致的前提下量得 15.76% sparse A2A payload reduction。Expert replication、多 GPU 型號、完整 K8s topology sweep 與 profile interpolation 是第 1 節明確排除的研究延伸，不是本計畫的未完成項目。此文件補充 [MoE-aware SpotServe Planner](spotserve-moe-aware-planner.md) 的 expert-aware re-parallelization。

### 0.1 2026-10-13 實測結果

主要 profiling 實驗環境固定為 Qwen2-MoE-Tiny、2 張 NVIDIA GeForce RTX 5070 Ti、vLLM 0.11.2、TP1/PP1/DP2/EP2、FP16、單一 SLLM replica。完整機器可讀證據在 [runtime-experiment-complete.json](../results/hotness_placement/runtime-experiment-complete.json) 與 [physical-remap-smoke.json](../results/hotness_placement/physical-remap-smoke.json)，逐 request routing/latency traces 位於同一個 results 目錄。runtime report 狀態為 `passed`。

GPU physical-remap smoke 實際搬動 4 個 expert shards、786,432 bytes；apply/verify 都成功，weight SHA-256 digests 在搬動前後一致，runtime ownership 與 plan 一致，remap 前後 inference 均完成。獨立 smoke 的 remap 時間為 587.78 ms；完整 candidate 實驗中的 round-robin、hotness-balanced、hotness-locality remap 分別為 217.75、183.09、237.79 ms，三者也都通過 physical apply/verify。

Bounded profile manifest `hotness-ep2-db272ba0db9d` 共 9 個條件、10 samples/條件，三種策略產生三個不同 actual mapping；第三個 `hotness-shift` bucket 未參與 calibration。另以 held-out workload 驗證 prediction error：linear 1.104%、round-robin 1.204%、hotness-balanced 0.929%。

| Strategy | Workload bucket | Mean latency (ms) | p95 (ms) | Throughput (req/s) |
|---|---|---:|---:|---:|
| linear | short-factual | 1066.89 | 1090.73 | 0.9373 |
| linear | structured-reasoning | 1098.59 | 1108.50 | 0.9103 |
| linear | hotness-shift | 1116.28 | 1124.25 | 0.8958 |
| round_robin | short-factual | 1068.58 | 1092.48 | 0.9358 |
| round_robin | structured-reasoning | 1096.16 | 1107.96 | 0.9123 |
| round_robin | hotness-shift | 1115.61 | 1121.46 | 0.8964 |
| hotness_balanced | short-factual | 1070.30 | 1092.18 | 0.9343 |
| hotness_balanced | structured-reasoning | 1103.79 | 1112.77 | 0.9060 |
| hotness_balanced | hotness-shift | 1114.04 | 1121.96 | 0.8976 |

Paired benchmark 使用 30 個相同 request/repetition IDs，coverage 100%，所有 response signatures 相同。相對 linear 的平均 latency delta 為 round-robin -0.471 ms、hotness-balanced +2.125 ms、hotness-locality +0.050 ms。這支持 benchmark/lookup 功能驗收，但不支持 `hotness_balanced` 的 latency 或 throughput 改善 claim。

Phase 3 的 pre-dispatch `source_rank/layer/expert` joint histogram 在 runtime 可用；`hotness_locality` 產生第四個不同 actual mapping。相同 workload 與 13,952 次 sparse collectives 下，linear 的 dispatch+combine payload 為 16,950,912 bytes，locality 為 14,279,232 bytes，減少 2,671,680 bytes（15.76%），且 paired outputs 全部一致，因此本次 A2A reduction claim 成立。這是 communicator 邊界的 sparse tensor payload，不等同實體 wire bytes。

### 0.2 2026-10-14 完成後回歸

| 回歸 | 結果 | 機器可讀證據 |
|---|---|---|
| Active-request physical remap | 2/2 requests 成功、零 fallback；4 shards / 786,432 bytes 實際搬動；step-boundary barrier 與 active-request marker 均成立；remap 226.16 ms | [active-request-remap-summary.json](../results/hotness_placement/active-request-remap-summary.json) |
| `alpha` sensitivity | `alpha=0.8` 與 recent-only `alpha=1.0` 產生不同 mapping；所有 assertions 通過 | [sensitivity-ablation.json](../results/hotness_placement/sensitivity-ablation.json) |
| Movement-weight sensitivity | weight 0 接受 2 experts / 200 ms 預估搬動；weight 1 保留 current、0 搬動 | [sensitivity-ablation.json](../results/hotness_placement/sensitivity-ablation.json) |
| Load/arrival sweep | concurrency 1/2 與 offered load 0.5/1.5 req/s 已對四種策略實測；高負載下 latency 上升但沒有策略性 serving 優勢 | [runtime-experiment-complete.json](../results/hotness_placement/runtime-experiment-complete.json) |
| Elastic EP2→EP4 | in-place resize、actor 未重建、active request 完成、drain=1、runtime placement verified | [elastic-ep-resize.json](../results/hotness_placement/elastic-ep-resize.json) |
| Simulated cross-host/failure-domain | 2 個 Ray nodes / 2 failure domains；8 experts、1,572,864 bytes 跨 domain 搬動；推論後新增 512 次 inter-container A2A | [simulated-cross-host-remap.json](../results/hotness_placement/simulated-cross-host-remap.json) |
| KV/state restore | 3/3 requests、1/1 restore、16 restored tokens、6 restored KV blocks、true-KV rate 100%、零 fallback | [stateful-recovery-regression-summary.json](../results/hotness_placement/stateful-recovery-regression-summary.json) |

跨 failure-domain 測試使用同一台實體主機上的兩個獨立 Ray/container domains，因此證明 placement、weight movement 與 inter-container A2A 路徑，不宣稱已量得實體跨機網路效能。硬體只有同型 RTX 5070 Ti，故沒有虛構第二種 GPU 型號的結果。

## 1. 目標與範圍

第一階段新增 `hotness_balanced` placement strategy，使用真實 routing statistics 平衡同一個 deployment 內各 EP rank 的預測 expert 負載，並考慮重新放置 experts 的搬動成本。先在固定 EP size、同質 GPU、runtime 已支援的 expert partition/remap 模式上完成。

研究問題：在相同模型、拓撲、workload 與 resource trace 下，hotness-aware placement 能否降低 EP rank load imbalance，且其效益足以抵銷 expert movement 的成本？

本計畫的兩項必要前提：

1. **每個 request 的 routing history 不同。** Placement 使用 workload window 的聚合 hotness；latency 評估則將同一批 requests 在不同策略下重播並逐筆配對，另報告整體分布與 throughput，不預設每筆 request 都會變快。詳細比較方式見第 3.3 節。
2. **Candidate latency 必須透過實測 profiling 建表。** 表格依模型、硬體、parallel configuration、實際 placement 與 workload 條件查詢 latency distribution/throughput，並在獨立 workload 上驗證預測誤差後才啟用 latency ranking。缺少可比 profile 時回到合法 capacity/movement 決策，不用 hotness imbalance 或共用預設值假裝已量得 candidate latency。表格與 lookup 設計見第 4.5 節。

Phase 3 已加入 source-rank-conditioned routing statistics，最佳化 expert locality 與 remote dispatch。第一階段的負載平衡不保證 A2A payload 下降，因此負載平衡與通信最佳化仍分別驗證；結果見 0.1 與第 8 節。

本次不新增 expert replication 或 redundant hot experts。EP2→EP4 resize、KV restore 與跨 failure domain remap 已在固定 EP 策略驗收後完成整合回歸，結果見 0.2。

### 1.1 三階段目標與 Claim

以下 Phase 1–3 僅指本次 hotness placement 優化，不改動原 MoE-aware planner 的 Phase 編號。

| 階段 | 核心能力 | 獨立驗收與可支持的 Claim |
|---|---|---|
| Phase 1 | Runtime hotness 的 movement-aware load balancing | 驗證實際 expert ownership、weight/inference correctness、rank load 與 moved bytes；只依結果報告負載平衡，不以 imbalance 代替 latency 或 A2A 效益 |
| Phase 2 | 實測 candidate profiles 的 latency/throughput ranking | 固定 profile 條件、held-out prediction error、同批 requests 的配對結果；只有實測支持時才報告 serving performance 改善 |
| Phase 3 | Source-rank-aware routing locality | 收集 source-rank/expert joint statistics，再驗證 remote dispatch 與 sparse A2A payload delta；只有實測流量下降且 correctness 通過時才報告 A2A reduction |

三階段可以分別通過或未通過，不能用其中一種 metric 的改善替代另一階段的驗收。

### 1.2 第一版固定實驗範圍

為避免 profiling state-space explosion，MVP 固定以下條件：

| 維度 | MVP 限制 |
|---|---|
| Model | 一個模型，先用 Qwen2-MoE-Tiny，revision/dtype/quantization 固定 |
| GPU/environment | 一種 GPU type、一份固定 runtime/interconnect/deployment manifest |
| Parallel shape | 核心 profiling 固定 TP1、PP1、vLLM DP2、effective EP2、SLLM replica count 1；EP2→EP4 另做整合回歸，不展開完整 topology sweep |
| Placement strategies | 三種：linear、round_robin、hotness_balanced；同一份 calibration snapshot 下凍結實際 mappings |
| Workload buckets | 三個有完整 manifest 的固定 workload suites；前兩個用於 calibration，第三個 `hotness-shift` 作 held-out stress/prediction-error 驗證 |
| 其他條件 | 固定 sampling/stop policy、generation budget、arrival/concurrency 設定、cache policy 與 warmup；記錄實際 token lengths/batch/routing |

完成版 profile table 有 `3 placements * 3 workload buckets = 9` 個條件組合，每個條件 10 次 samples。表格不為每個 request、每個新 arrival rate 或每次 replan 的 fingerprint 自動新增 row。

不同策略可能產生相同 actual mapping，這種情況需去重或明確標示，不能把相同 placement 當作不同物理候選的改善證據。

完整 schema 保留 hardware/runtime/placement 等 provenance 與 invalidation checks。固定的維度存於 dataset manifest；MVP 查詢入口縮成 frozen candidate ID 與 workload bucket ID，再驗證其 actual placement fingerprint/manifest 相符。fingerprint 作為一致性檢查，不展開所有可能 expert mappings。

舊的排序分組 `hotness` 只保留為 synthetic 診斷案例。MVP 通過後已完成 movement weight、`alpha`、arrival/concurrency 與 EP2→EP4 sensitivity/integration checks；更多模型/GPU shapes、完整 topology sweep 與 profile interpolation 仍屬明確排除的研究延伸。真實模型的 routing 分布與 serving 無顯著改善結果均保留在 canonical report。

## 2. 實作前現況與完成項目

以下保留實作前的差距與本次對應完成項目，方便追溯原始 plan 與交付內容。

| 位置 | 實作前能力 | 本次完成項目 |
|---|---|---|
| `sllm/spot/moe_placement.py` | `ExpertPlacementPlan`、linear/round-robin placement、hotness 排序雛形、movement diff/bytes/cost | 統一 hotness contract、負載平衡分配、選擇理由與 stability gate |
| `sllm/spot/reparallelization.py` | weighted score 與 expert movement penalty；另有 `spotserve_algorithm1` 的 profile-only latency/throughput ranking | 沿用 profiling 基礎，加入固定 EP2 下的 placement/workload matching 與單次 movement 計分 |
| `sllm/routers/roundrobin_router.py` | replan、runtime metadata 與執行協調 | 為同一次 replan 擷取一致的 routing/placement snapshot |
| `sllm/backends/vllm_backend.py` | runtime apply/verify、由 config 推導的 placement metadata | 避免從 strategy 名稱推導出尚未實際套用的 placement |
| `sllm/vllm_expert_remap.py` | DP EngineCore metadata aggregation 與 remap 協調 | 保留統計 provenance，檢查跨 TP/DP 的 aggregation 語意 |
| `scripts/run_k8s_moe_f1_f2.py` | `ensure_candidate_profiles()` 的 shape profiling、cache hash 與 repeated measurements | 先確認可共用的 schema/量測流程，再擴充 bounded placement profiles；MVP 不依賴完整 K8s topology sweep |
| `sllm_store/vllm_patch/runtime_moe_metadata.patch` | runtime top-k routing、global/recent histograms | 補足 window、sample count、freshness 與 rank aggregation metadata |

實作前的 `hotness` 是先按熱度排序，再切成連續區段分給 ranks，可能把熱門 experts 集中到同一個 rank；planner/backend 對 global/recent counts 的語意也不一致。本次新增 canonical snapshot/normalization contract 與 capacity-constrained greedy，planner、backend 與 runtime aggregation 已共用相同語意。

目前 runtime 的 recent histogram 來自 `_RECENT_EVENTS = deque(maxlen=64)`，代表最近 64 個 routing events；每個 event 的 token 數可能不同，也不能直接稱為最近 64 秒或 64 個完整 decoding steps。

`ExpertPlacementPlan.estimated_load_balance_penalty` 與 `estimated_dispatch_cost` 原有欄位先前未填入 hotness 最佳化結果；本次已接通 candidate scoring、序列化 plan、selection reason 與實際執行 mapping，而非只輸出額外 metadata。

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

`current` 是 no-op/reference candidate，不一定是第四種策略；若它不在已凍結的 profile mappings 中，仍可作 Phase 1 的保留決策，但不得聲稱已有 Phase 2 的 latency profile。

逐層計算 load imbalance，避免只加總所有 layers 而掩蓋單層 bottleneck：

```text
imbalance[l] = max_r(rank_load[l,r]) / mean_r(rank_load[l,r]) - 1
B(P) = mean_l(imbalance[l])

J(P) = B(P) + movement_weight * normalized_movement_cost(P)
```

只對有有效 samples 的 MoE layers 計算，並回報其 coverage。MVP 同質 GPU、相同 expert compute shape 下先以 routed assignments 估計負載；量化模式、expert shape 或 GPU compute capacity 不同時，需額外校準。

movement cost 使用已有的 weight bytes 與 bandwidth/per-expert estimate，正規化的 reference cost 必須寫入 config/report。memory capacity 作為硬限制，第一版不額外加入 memory penalty 權重。

`J(P)` 只用於同一 topology 內挑選 placement，屬於局部比較的 objective。它可保留在 diagnostics，但不作為一個額外 penalty 疊加到 existing workload score；最終 movement 計分規則見 4.4。

若 movement cost 未知，記錄 `unknown`，不能當成已知零成本。自動策略採保守的 move-count/byte budget；需要搬動但連 bytes 都不明時，先只產生建議 plan。

初始 greedy 以負載為主，再以 movement objective 在 candidates 間選擇。若實驗顯示搬動過多，可增加受容量限制的 pair-swap refinement；此項不列為第一版必要條件。

### 4.4 與 Topology Score 整合

`J(P)` 用於相同 topology 下比較 placement。不能單憑 imbalance 跨 EP sizes 排名，因為 EP1 的 imbalance 天然為零，卻不代表 throughput 最佳。

Phase 1 用 J 選擇 placement；Phase 2 的候選單位是 `(parallel shape, frozen placement, workload bucket)`。即使 MVP 固定 EP2，仍保留三種合法 mappings 的可比 profiles 供 latency/throughput ranking，不能先用 J 把同一 topology 裁成唯一 mapping，讓 profiling 失去比較對象。J 在 Phase 2 保留作 diagnostics，movement 則依下述單次計分或既有 policy 的 budget gates 處理。若只有一個有效且不同的 actual mapping，記錄候選不足，不宣稱已驗證 profile ranking 效益。

每個 topology candidate 附帶選定的 placement、per-layer/rank load、`B(P)`、movement cost 與 confidence，再進入既有 workload cost model。movement penalty 沿用現有路徑，只計入一次。candidate latency/throughput 採下節的實測 profiling table；`B(P)` 不直接換算成 milliseconds，也不能假設 imbalance 降低 20% 就讓 request latency 降低 20%。

對 weighted score 路徑固定以下責任：

```text
placement_J(P) = B(P) + lambda * normalized_movement_cost(P)
  -> only selects a placement within the same topology

final_score = workload_score_excluding_expert_movement
            - expert_weight_movement_penalty_weight * movement_cost_ms
```

`final_score` 不再扣一次 `placement_J(P)`，也不扣 J 內的 normalized movement 分量。局部比較與最終排名可以參考同一份成本資料，但同一最終 objective 只能包含一份 movement penalty；raw movement bytes/time 可以同時出現在多個觀測欄位。

目前 weighted score 的 `replan_window_cost_ms` 已包含 expert movement，若 `replan_window_penalty_weight` 與 `expert_weight_movement_penalty_weight` 同時非零，movement 會經兩個 score components 扣分。實作本計畫時需將「完整 window estimate」與「供 aggregate penalty 計分的 window 分量」分開，movement 的 score ownership 固定給 `expert_weight_movement_penalty`。其他 load/migration/queue 分量也需避免 aggregate penalty 與細項重複計入。

`spotserve_algorithm1` 目前按 profile latency/throughput 排名，不使用 weighted objective；保留其 policy 語意。movement-aware placement selection 與 budget gates 在 placement 層處理，不自動把 `J(P)` 或 weighted penalty 加進該 ranking。

同一次 replan 的最終 `ExpertPlacementPlan` 應重用 candidate scoring 時的內容，避免重新讀 hotness 後產生不同 mapping。report 需保留 strategy、snapshot identity、selection reason 與 score components。

### 4.5 Candidate Latency Profiling Table

本次已實作 bounded placement/workload table 與 exact matching。實作沿用 `spotserve_algorithm1` 的 profile-only ranking 與既有 shape profiling/cache 基礎，補上 placement fingerprint、hotness snapshot、workload bucket、provenance 與 invalidation 驗證。

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

同一份 `short-factual` workload bucket 的實測表如下：

| Profile | TP | vLLM DP | Effective EP | Placement | Workload | Mean Service Latency | p95 | Throughput |
|---|---:|---:|---:|---|---|---|---|---|
| P1 | 1 | 2 | 2 | linear，actual mapping A | short-factual | 1066.89 ms | 1090.73 ms | 0.9373 req/s |
| P2 | 1 | 2 | 2 | hotness_balanced，actual mapping B | short-factual | 1070.30 ms | 1092.18 ms | 0.9343 req/s |
| P3 | 1 | 2 | 2 | round_robin，actual mapping C | short-factual | 1068.58 ms | 1092.48 ms | 0.9358 req/s |

`effective EP` 沿用 runtime 對 TP/DP/EP 的語意，不把 EP 當成額外獨立相乘的 GPU 維度。不同硬體、topology、workload、cache 或通信 backend 的 rows 不能任意互換。

#### Profiling 與 Lookup 流程

1. 先通過 A/B 的 contract、hotness semantics 與 actual placement smoke gates，再依 1.2 固定 hardware/runtime/model、EP2 shape、三個 placement candidates 與三個 workload buckets。
2. 用共同 history window 產生各策略的 placement，凍結 mapping，再重播共同的獨立 profiling workload。
3. 每個 candidate warmup 後，在指定 concurrency/load 下量測並保存 raw per-request records、routing、batch、placement 與 counters；load/model startup/remap window 另量。
4. 彙整相同條件的 latency/throughput distribution，保存 sample count 與變異。需要 serving capacity 時另做 load sweep，找出穩定可承受的吞吐量。
5. planner 用目前 workload window 與 selected placement 查表，記錄 `profile_id`、匹配方式與 confidence，再填入 candidate 的 latency/throughput estimate。
6. 在未參與建表的 workload 上驗證 prediction error，通過後才開啟 profile-based latency ranking。

完成版僅在 1.2 的九個條件組合內做 exact lookup，不展開完整 schema 的笛卡兒積。動態產生的 mapping 未命中時，標記 `profile_unavailable`；profile interpolation 或校準模型仍未納入，且不得僅憑 strategy 名稱或 GPU 數匹配。

缺資料時保留已有的合法 capacity/movement 決策，該輪對比較集合一致停用無法公平比較的 latency/throughput score components，並輸出 fallback reason；不能只拿掉缺資料 candidate 的 penalty，讓它因為沒有量測而佔優。不將缺值視為零 latency，也不以共用預設值假裝各 candidate 已有相同的實測效能。

上述保留決策屬於明確選用的 Phase 1/weighted mode。既有 `spotserve_algorithm1` 的 profile-only mode 繼續拒絕缺有效 profile 的 candidates；不在同一輪默默切回 heuristic，report 必須標明實際 policy/phase。

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

工程順序固定為 **A -> B -> C -> D -> E -> F**。A/B 先凍結 key、counter/window、rank ownership 與 plan serialization 語意，並用既有 apply/verify hooks 做最小 actual placement/hash smoke；通過後才開始 C 的量測與建表，避免 profiles 因 contract 尚未固定而作廢。這個 B 的 smoke 不新增另一套 remap engine；E 負責完整 controller 整合與回歸。

Phase 1 的驗收可關閉 profile ranking，獨立證明 movement-aware load balancing。Phase 2 再開啟 C 的 profiles 並驗證 serving performance；D/E/F 的整合需保留兩種 mode 的獨立結果。Phase 3 的 source-rank instrumentation 與 A2A optimizer 已在 A–F 後完成，結果見 0.1 與第 8 節。

| 步驟 | 修改位置 | 交付與驗收 |
|---|---|---|
| A. 統一 Hotness Snapshot | runtime patch、`vllm_runtime_metadata.py`、`vllm_expert_remap.py`、router | canonical keys、normalization、window/sample/freshness metadata、TP/DP 去重規則；所有 candidates 使用同一份 snapshot |
| B. `hotness_balanced` | `moe_placement.py`、既有 runtime hooks | 容量限制下的 greedy assignment、deterministic tie-break、per-layer/rank load、合法性檢查與 actual placement smoke |
| C. Candidate Profiling Table | 既有 profiling 基礎、bounded schema/driver、`reparallelization.py` | 固定 EP2/三種 mappings/兩個 workload buckets 的建表與 lookup；缺資料處理、prediction error 驗證 |
| D. Movement 與 Selection | `moe_placement.py`、`reparallelization.py` | current/baseline/balanced 比較、unknown cost 處理、stability gates；movement 單次計分、policy 語意保留；mapping 可追溯至 score/profile |
| E. Runtime 整合 | router、backend、既有 remap hooks | 真實 physical remap、placement/hash verify、epoch 更新與失敗處理 |
| F. Metrics 與 Benchmark | `metrics.py`、analyzer、benchmark runner/config | predictor、movement、measured serving performance 分開輸出；通過下節驗收 |

追蹤 checklist：

- [x] A. Hotness snapshot contract 與 aggregation 驗證完成。
- [x] B. `hotness_balanced` 分配、合法性測試與 actual placement smoke 通過。
- [x] C. Bounded EP2 candidate profiling table、exact lookup 與 prediction error 驗證完成。
- [x] D. Movement-aware selection、stability gates 與 candidate score/profile 整合完成；movement 在最終 objective 只計一次。
- [x] E. Runtime remap、actual placement、weight hash 與 inference correctness 驗證通過。
- [x] F. 相同 requests 的配對 latency comparison、ablation/benchmark、metrics 與結果文件完成。

驗證入口為 `scripts/run_hotness_placement_ablation.py`（無 GPU）與 `scripts/run_hotness_placement_runtime_experiment.py`（真實 DP2/EP2 runtime）。既有 `linear`/`round_robin` 行為有回歸保護；實驗使用顯式 `hotness_balanced` 與 `hotness_locality` 名稱。

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
- 固定所有其他成本，只增加已知 movement cost；weighted final score 的變化僅為一次 `-weight * delta_movement_ms`。同時啟用 replan-window penalty 時仍成立；J 不額外扣分，raw window estimate 仍完整記錄。
- Bounded profile manifest 限制 model/GPU/EP shape 與 9 個完成版條件組合；未列入的 mapping/workload 不自動增列或近似命中，profile-only policy 不默默轉成 heuristic。
- 固定 EP2、至少兩個不同 actual mappings 時，Phase 2 保留所有有效 profiles 參與排名；mapping 重複或候選不足會被標記，不由 J 預先刪掉所有替代 placements。
- profiling table 的 exact match、workload/placement mismatch、過期/版本不符/缺資料，不會產生虛構的零 latency 或重複計入 queue；分位數不直接相加。

驗證入口 `scripts/run_hotness_placement_ablation.py` 會輸出 baseline/candidate placement、per-layer load、movement diff、selection reason 與 assertions。

### 7.2 Runtime 功能驗證

先用 Qwen2-MoE-Tiny、固定 EP2 的已支援 remap topology，收集 routing 後產生 plan，再走既有 apply/verify。驗收：

1. routing 來源是 runtime observed top-k，snapshot samples/freshness 有效。
2. actual rank ownership 等於 selected plan，expert weight hashes 保持正確。
3. remap 前後固定 prompt、seed、sampling 設定的輸出符合既有 correctness tolerance。
4. 選擇搬動的案例有實際 moved shards/bytes；選擇不搬動的案例有可解釋原因。
5. active-request 模式沿用既有 step-boundary coordination，request completion、state 與錯誤處理通過回歸。

MVP 使用固定 DP2/EP2 coordinated remap 做 runtime 驗收；另已完成 EP2→EP4 in-place resize 與 simulated cross-host/failure-domain 回歸，各路徑的結果分別記錄在 0.2，且沒有因此擴大 bounded profile lookup 範圍。

### 7.3 Performance 與 Ablation

實際驗收由 `scripts/run_hotness_placement_runtime_experiment.py` 使用固定 manifest 比較以下 strategies：

| 組別 | 目的 |
|---|---|
| `linear` | runtime 預設 baseline |
| `round_robin` | deterministic 分散 baseline |
| `hotness_balanced`，固定 movement/stability 設定 | 評估 movement-aware 負載平衡與 serving performance |
| `hotness_locality` | Phase 3 source-rank-aware remote dispatch/A2A 驗收；不列入 9-row Phase 2 profile table |

Phase 2 對前三種策略與三個 workload buckets 建立 9-row profiles；`hotness_locality` 另作 Phase 3 驗收。舊 `hotness` 的連續分組問題由 synthetic cases 驗證；MVP 通過後也已完成 movement weight 0/1 與 `alpha` 0.8/1.0 sensitivity ablation，而沒有把參數組合乘進 bounded profile lookup。

synthetic histogram 用於證明決策正確；performance claim 使用真實模型 routing。每組保持相同硬體、模型、TP/DP/EP、workload、trace、warmup、sampling 與量測長度。用獨立的相同 warmup/history 建立 training window，固定策略參數後再量測後續 workload；不可用 candidate evaluation 的未來 routing 反過來挑 placement。

先依 4.5 建立 candidate profiling table，再使用另一份 evaluation workload 做 end-to-end comparison。每筆 request 依 3.3 配對，保留 token lengths、完成狀態與 routing 差異；不能拿 baseline 的 request A 與 candidate 的 request B latency 宣稱 placement 效益。

比較三個固定 workload suites 實際觀測的 routing 分布，其中 `hotness-shift` 未參與 calibration，作為 held-out stress/prediction-error test。實際結果如實保留 routing skew 與沒有 end-to-end serving 優勢的結論，未以 synthetic histogram 取代 performance claim；各條件有 10 次 samples，remap window 與 post-remap steady state 分開呈現。

必要 metrics：

- 每層/每 rank 的 predicted load、max/mean ratio、load CV，以及 remap 後實際 observation window 的同類統計。
- hotness source、window、sample count、age、layer coverage、fallback/no-op 原因。
- latency profile ID、workload/placement match、estimate source/confidence、held-out prediction error 與 profile-miss rate。
- moved experts/weight bytes、estimated movement cost、measured remap duration、apply/verify 結果。
- request success、clean success、fallback rate、p50/p95/p99、throughput。
- request-level paired latency delta、配對 coverage、prompt/output-length 分層結果。
- remap window latency 與 post-remap steady-state latency。
- baseline/candidate A2A counters 的相同區間 delta；流量計數與 predictor 分開標示。

Phase 1/2 中的 A2A counters 僅作 observation；global/recent hotness 不含 token source rank，無法直接估計 remote dispatch，也不能因 load CV 或 latency 下降就宣稱 A2A reduction。A2A reduction claim 已由 Phase 3 的 joint routing statistics 與獨立實測 gate 支持。

完成判準是策略影響了真實 runtime placement、correctness 通過且量測可重現。若 predicted imbalance 降低但 throughput/latency 沒改善，保留此結果並分析計算/通信瓶頸；不把功能驗收當成 performance conclusion。

## 8. Phase 3：Routing Locality 與 A2A（已完成）

global hotness 沒有 token source 的資訊。要減少 remote dispatch，需保留 source rank 與 destination expert 的 joint statistics：

```text
route_count[source_rank][layer][expert]

estimated_remote_assignments(P)
  = sum(route_count[s,l,e] where owner_P(l,e) != s)
```

這個簡式先限 TP1、單一 owner 的 topology；TP sharding、replication 與跨 domain 情況需用 runtime 實際 dispatch semantics 計算 locality。資料應在 dispatch 前觀測，避免搬動後把 expert execution rank 誤當 token source。

已在負載與容量限制下加入 remote dispatch objective，利用 destination-aware sparse A2A backend 重播相同 workload，比較 collective payload delta。runtime patch 在 dispatch 前記錄 `source-rank:{rank}/layer:{layer}/expert:{expert}`，DP aggregation 對獨立 source streams 加總並對 TP 重複觀測取每 key 最大值；contract 不完整或 rank semantics 不符時 fail closed，不啟用 locality placement。

實測以 communicator 實際傳輸的 sender/receiver sparse tensor bytes 計數，未把它描述成 internode/network wire bytes。linear 與 `hotness_locality` 的相同 workload payload 分別為 16,950,912 與 14,279,232 bytes，降低 2,671,680 bytes（15.76%）；collective calls 都是 13,952，paired output signatures 全部一致。完整 counters 與 mappings 保存於 [runtime-experiment-complete.json](../results/hotness_placement/runtime-experiment-complete.json)。
