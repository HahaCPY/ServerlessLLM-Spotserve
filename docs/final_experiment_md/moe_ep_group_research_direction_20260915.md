# MoE × SpotServe：EP-group 與研究方向門檻

日期：2026-09-15。這是架構／正確性／容量診斷，**沒有做 Original vs MoE-aware 正式 A/B**，沒有修改 planner、expert placement 或 workload。以下 GPU 數據為探索性 microbenchmark；EP 的全模型 correctness gate 尚未放行。

## 1. Instance 與 EP-group：原始碼到底代表什麼

`InstanceHandle` 有 `instance_id`、`num_gpu`、`node_id`、一個可路由的 `backend_instance` Ray actor、`ready` 與 request concurrency，**不是物理 GPU，也不是 vLLM rank**；它是服務一份邏輯完整模型的後端 endpoint。一份模型可由一個單卡完整 replica 實作，也可在同一 actor 內由 TP×DP×PP 個 GPU workers 切分。[InstanceHandle](../../sllm/utils.py)、[actor 啟動](../../sllm/inference_instance.py)、[deployment adapter](../../sllm/spot/vllm_deployment_adapter.py)。

`ParallelPlan.replica_count` 是彼此獨立的完整服務副本數；每份 actor 需要 `TP×DP×PP` 張 GPU。Adapter 每個 replica 建立一個 actor、一個 `InstanceHandle`，並把這份 actor 的全部 GPU 資源綁到**同一個 scheduler worker node**。這是 endpoint 的合理抽象；不合理之處是用 endpoint 當成最細的搶佔、rank 故障與遷移單位，使 vLLM 內部 rank／expert ownership／KV ownership 對 SpotServe 不可見。[plans](../../sllm/spot/reparallelization.py)、[adapter](../../sllm/spot/vllm_deployment_adapter.py)。

原 [SpotServe 論文](https://arxiv.org/abs/2311.15566) 把動態平行化、跨 preemptible instances 的 device mapping／通訊成本和 stateful recovery 作為系統核心。**推論：**本專案目前可以重建不同 TP/replica endpoints，但 actor 內 GPU ranks 被隱藏，並未實作論文那種對每張／每節點資源的 shard 級 device mapping。這不是說「一個 endpoint 有完整模型」本身錯誤，而是控制平面缺一層 distributed group。

目前 `handle_preemption(node_id|instance_id)` 比對整個 handle、將它標為 PREEMPTING 並捕捉 token；migration target 是 instance/deployment，reparallelization 可以選 TP/DP/replicas 並建立新 actor，但沒有 per-rank health 或重建 expert coverage 的驗證。[preemption](../../sllm/routers/roundrobin_router.py)、[migration granularity](../../sllm/spot/context_migration.py)。原生 EP2 的每層 40 experts 在兩 ranks 上分成 0–19／20–39；失去一個 rank 立即失去半數 expert，剩下的一個 rank **不能當完整 model 繼續服務**。既有 logical expert placement contract 是 `observe_only_contract`，`live_migration=false`，不是物理 expert 跨卡搬運。[adapter placement](../../sllm/spot/vllm_deployment_adapter.py)。

最小架構提案是保留現有 `InstanceHandle` 為可路由 model endpoint，在其下加 `ParallelGroup/EPGroup` 與 `RankHandle`：記錄物理 GPU UUID／node／fault domain、TP/DP/PP/EP coordinates、各 layer 真實 expert IDs、KV owner、rank health、coverage epoch。只有**所有 ranks 存活且每層 expert coverage 完整**的 group 才 ready；rank 故障先把整個 group 從流量移除、保住可復原的 request token，再在新 group 啟動／placement 驗證後切流量。這是安全控制平面設計，**不是聲稱 pinned vLLM 已支援原址 elastic rank join**；先用重建整個 group，不硬改 vLLM。跨 Ray nodes 的 rank placement 或物理 expert live relocation 另需實作／驗證，不能由新增 metadata 自動取得。

## 2. EP correctness：守恆檢查與仍未放行的部分

本輪沿用同一 Granite MoE checkpoint（32 MoE layers、每層 40 experts、top-8）、相同 calibration prompts；沒有替換 dense model。[前一輪完整報告](moe_ep_correctness_jit_overhead_diagnostics_20260915.md)。三配置：`TP2`（全部 experts 的 TP-shards）、`TP2+EP2`（每 rank 20 個完整 experts；attention TP2）、`DP2+EP2`（每 rank 20 個完整 experts；attention DP2、AllGather/ReduceScatter）。

- 原有 BF16 gate：四個非重複的正常 4096-token prompts，各輸出 512 tokens，TP/兩種 EP **逐 token 完全一致**；兩個沿用的 repeated-token 壓力案例在 TP2+EP2 的第 12／7 token、DP2+EP2 第 4／9 token 開始不同。這兩個案例只用於查 correctness，不當作展示 speedup 的正式 workload。
- 新的 32-token、小輸入 C=2 原生 collective 取樣：TP 的 final reduction、TP2+EP2 的 final reduction、DP2+EP2 的 AllGather(hidden/weights/IDs) 與 ReduceScatter combine 均通過 host-side 守恆比較；IDs／AllGather 完全一致，reduction／combine 的 BF16 relative L2 約 0.17–0.18%，低於事前 3% oracle 容忍值。Router 全 rows 有效、unique、top-8 非 tie mismatch=0；實際 GPU expert kernel 的 FP32 local oracle 通過。這只驗證**首個小 input 的各 native collective 與各層首個 kernel**，不是長輸出每個 assignment 的完整證明。[原始 BF16 資料](../../results/moe_ep_correctness_gate_bf16_20260915_attempt2)、[守恆分析](../../scripts/analyze_moe_phase2_diagnostics.py)。
- FP16 轉型敏感度診斷：四個正常 prompts 仍一致；TP2+EP2 的 repeated17／18 分別第 11／19 token 不同，DP2+EP2 的 repeated17 前 32 tokens 相同、repeated18 第 19 token 不同。小輸入 native collective 檢查也通過。這是**另外一種 dtype 的診斷**，不能用來事後取消 BF16 差異；checkpoint BF16 byte-hash 對轉型後 FP16 不適用，已單獨標記為未驗證。[FP16 原始資料](../../results/moe_ep_correctness_gate_fp16_20260915_attempt2)。
- FP32 vLLM native comparator 無法啟動：Triton 在這張 GPU 的 default FP32 MoE config 請求 shared memory `131072`，硬體上限 `101376`。因此沒有 FP32 pinned vLLM 對照，不得謊稱 FP32 gate 通過。[失敗日志](../../results/moe_ep_correctness_gate_fp32_20260915/tp2-reference.log)。
- 補做**普通 train-code／story／science／serving** 各 P512、C8、128 output tokens 的同輸入＋top-20 logprob gate，無 skew 設定。TP2 與 TP2+EP2 對**每個請求** 128/128 tokens 完全一致，包括兩個 train-code 相同 prompt 的 request 在第 5 token 彼此不同；DP2+EP2 與 TP2 只有該第 5 token 受不同 DP attention/batch shape 影響，其餘七請求相同。TP2 兩份相同 prompt 的第 5 token 分別為 2538／5950，前四 tokens 相同；一份兩個候選 logprob 幾乎同分（margin=0），另一份 margin≈0.5，兩份 shared returned candidate logprob 最大差≈0.281。TP2+EP2 同批亦呈同樣選 token 模式；這排除了「該正常案例的 EP 專有 expert mapping」作為直接原因，**不會因此宣告較早 repeated-token 路徑正確**。[原始 normal gate](../../results/moe_ep_normal_batch_shape_gate_20260915)、[共同 prefix 分析](../../scripts/analyze_moe_phase2_diagnostics.py)。

前一輪相同 prefix 的 native layer trace 顯示 layer0 router logits／選定 IDs 相同、layer0 MoE output 有 BF16 小差異；往後 hidden 差異累積，後段 expert routing 才改變；既有 HF FP32 參考與 TP 自身 prefill/decoder 的形狀差異也說明 logits 對數值／batch shape 敏感。[layer 與 teacher-forced 證據](moe_ep_correctness_jit_overhead_diagnostics_20260915.md)。結論是：**目前最符合「數值／batch shape 差異累積，再經 routing／greedy 放大」，未觀察到小輸入 token drop、expert map 錯誤或 AGRS 守恆失敗；但同 prefix logprob 差可達 0.28，仍不能排除尚未覆蓋的長路徑 correctness 問題**。native capacity-factor/drop 政策沒有開啟，不把其他 MoE 的 capacity overflow 直接套進解釋。正式 EP performance **不放行**。

## 3. 相同兩張實體 GPU：容量、併發與 throughput

使用 GPU2/3，三個 group 模式各起一個原生 vLLM engine；完整 replica 模式在 GPU2 與 GPU3 **同時**各起一個 TP1 engine，雙進程 sample 經檔案 barrier 同步。BF16、`gpu_memory_utilization=0.7`、`max_model_len=8704`、`max_num_seqs=64`、prefix cache off，FlashInfer/Triton cache hit。輸入只用先前四個正常校準 prompts 的前 512 或 4096 tokens，沒有重複單一 token 製造 skew；每 cell 暖跑一次、同一 fresh engine 量兩次，無 MPS。記錄每 rank model parameter bytes、實體 KV storage、allocator peak／free bytes、每請求完成／TTFT／TPOT／p95 latency、每個 batch 的 generated tokens/sec。[protocol 與逐筆結果](../../results/moe_ep_capacity_normal_20260915/summary.json)。**限制：**同一個 batch 會循環使用僅四個普通 prompts、prefix caching 刻意關閉，並非大量不同使用者的 open-loop arrival trace；不從這小組輸入宣稱「多數真實 MoE workloads」的全面結論。

| 兩張 GPU 的配置 | 每 rank 參數 | 每 rank KV storage | group 可用 KV tokens | 解讀 |
| --- | ---: | ---: | ---: | --- |
| TP2（無 EP） | 3.07 GiB | 7.08 GiB | 231,888 | 同一 TP-group 的 token 容量，不把兩 rank 重複加總 |
| TP2+EP2 | 3.07 GiB | 7.08 GiB | 231,888 | expert 雖改成每 rank 20 個完整 experts，**沒有多出 KV** |
| DP2+EP2 | 3.33 GiB | 6.40 GiB | 104,848/rank；均衡分流合計 209,696 | 兩 ranks 各有自己 DP KV，合計假設平均送流 |
| 兩個完整 TP1 replicas | 6.14 GiB | 3.97 GiB | 64,992/replica；均衡分流合計 129,984 | 每張卡重複存一份完整模型 |

下表為同一 workload／兩張卡／同一 engine 各 cell 的兩個 sample 平均；generated tokens/sec 包含完整 batch 的 prefill wall time，**不能當單純 decode kernel TPS**。`C` 是兩卡合計的同時請求數；兩個 replicas 各接一半。列出 C=8／48，其他 C=16／24／32 在 raw JSON。

| 正常 workload | C | TP2 無 EP | TP2+EP2 | DP2+EP2 | 兩個完整 replicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| prompt512／output128 | 8 | 272 | 262 | 243 | **327** |
| prompt512／output128 | 48 | 1,442 | 1,386 | 1,322 | **1,771** |
| prompt4096／output32 | 8 | 153 | 146 | 147 | **208** |
| prompt4096／output32 | 48 | 262 | 249 | 292 | **358** |

在此正常輸入、C≤48 上，**兩個完整 replicas 是所有已測 cell 的最高 throughput**；DP2+EP2 在長 prompt／C48 比無 EP 的 TP2 好約 11%，但仍低於完整 replicas，而且同時改動 attention 的 TP/DP 切法，不能把它歸因於 expert placement。TP2+EP2 的記憶體／KV 與 TP2 一樣、服務曲線沒有顯示優勢。KV bytes 多 ≠ serving 一定更快：正常 prefill/decode 的計算、DP 排程與 rank 同步也同時改變。兩次同一 engine 的 sample、部分模式與 FP16 correctness run 共用主機 CPU；這**不是可重複的部署級顯著性或正式 EP 快於 TP 證明**。

同一 TP2 engine 對一個正常的 512-token train-code prompt 在同 batch 的不同 request 也出現兩種輸出 SHA；第二輪在**另一個新 TP2 engine** 同樣重現正常 P512/C8 下兩個相同 prompt 第 **5** 輸出 token 不同（前四個 tokens 相同）。無計時的第三輪正常 logprob gate 在 TP2／TP2+EP2 都重現完全相同的逐請求模式，DP2+EP2 有一份不同；這顯示 BF16 結果有 batch/scheduling shape 敏感性，不能把 TP 與 EP 輸出不同的 SHA 誤稱 EP token drop。該 gate 的 logprobs **只作 correctness 分析，未倒填前兩輪吞吐**。[首次 hash](../../results/moe_ep_capacity_normal_20260915/tp2.json)、[重現的 output tokens](../../results/moe_ep_capacity_normal_20260915_highc/tp2.json)、[第三輪 gate](../../results/moe_ep_normal_batch_shape_gate_20260915)。

第二輪用**各模式新建獨立 engine**、相同 cache-hit 條件，測短 context 的 C8／64／96／128，長 context 的 C64／96；同樣每 cell 暖跑一次、量兩次，額外保存完整輸出 tokens。[第二輪 protocol、TPS／TTFT／TPOT／p95](../../results/moe_ep_capacity_normal_20260915_highc/summary.json)。兩次部署的 C8 是交叉核對點，沒有混成同一平均。

| 正常 workload | C | TP2 無 EP | TP2+EP2 | DP2+EP2 | 兩個完整 replicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| prompt512／output128 | 64 | 1,822 | 1,743 | 1,691 | **2,293** |
| prompt512／output128 | 96 | 1,417 | 1,378 | 2,389 | **3,239** |
| prompt512／output128 | 128 | 1,823 | 1,768 | 2,966 | **4,048** |
| prompt4096／output32 | 64 | 263 | 250 | 280 | **350** |
| prompt4096／output32 | 96 | 271 | 258 | 302 | **364** |

四配置皆完成上列測試點（短最大 C128、長最大 C96）的 batch；每 cell 兩次樣本的 request count 與 stream/length 驗證皆通過，這是**最大已測完成併發**，不是無限時間負載的最大穩定併發。`max_num_seqs=64` 是**每個 engine** 的 scheduler 上限：TP2／TP2+EP2 各只有一個 engine，DP2+EP2 有兩個 DP scheduler，兩完整 replicas 也有兩個獨立 scheduler。因此短 context C96／128 的 TP2 queue 增大是配置差異，**DP2+EP2 對 TP2 的 C96 優勢不能單獨歸因 MoE**。短 context full replicas 到 C128 throughput 仍上升，**尚未實測其真正 saturation peak**；長 context C64→96 普通 TP2 與 replicas 的 TPS 趨平，但仍須在真實 arrival/SLO 約束下查 queueing、KV recompute 才能訂「最大穩定」門檻。雖然 DP2+EP2 長 context 比 TP2 勝出，**所有已測 C 與兩類普通 workloads 的吞吐冠軍仍是兩個完整 replicas**，沒有得到「EP 高 C 容量導致整體反超」的 crossover。

延遲也沒有替 EP 反轉：短 context C128 的兩完整 replicas p95 約 **4.05 s**、平均 TPOT **27.1 ms**，DP2+EP2 約 **5.46 s／36.5 ms**，TP2 約 **8.93 s／32.5 ms**（單 TP scheduler 排隊會進入 p95）；長 context C96 的 replicas 約 **8.34 s／61.4 ms**，DP2+EP2 約 **10.1 s／116.9 ms**，TP2 約 **11.24 s／143.3 ms**。這是本輪 batch/request 指標，不是跨獨立部署的信賴區間。

## 4. Actuator：先證實可以做什麼，別把通用 TP 收益算作 MoE-specific

若未來 EP 通過全模型 correctness 並在正常情境（不是挑一個 skew 案例）穩定位於 TP/replicas 的服務成本 Pareto 前緣，最小可執行 actuator 才是**選擇整份 group 部署**：`TP1 full replicas`／`TP2`／`TP2+EP2`／`DP2+EP2`，依 arrival/concurrency、prompt＋output 長度、可用卡數／warm state 與**實測**自然 routing imbalance 修正 cost。Original 選出的 plan 必須是共同候選／fallback：沒有信心或沒有優勢就沿用 Original，不以讓兩版選擇不同為目的。

目前已錄到正常 Granite trace 的 layer/expert histogram，四類 workload 每層 top5 experts 佔 assignment 約 39.7–43.4%，固定 0–19／20–39 的每層兩側 assignment 比值平均約 1.32–1.43。這是**歷史真實 routing counts，不是本輪各 GPU 觀察到的 compute 時間／all-to-all bytes**，也不是人為 skew。下一個診斷是把當次可量測 rank kernel／dispatch cost 與這份分佈對齊，確認 routing correction 是否改善成本排序。[calibration 真實 routes](../../results/moe_matched_cache_calibration_20260914/calibration.json)。

重要限制：pinned Granite class 沒有已驗證的 runtime expert live-relocation／EPLB protocol；現有 placement plan 是 observe-only，不能現在宣稱「優先搬 hot experts」可改變 serving。[舊 EP feasibility 診斷](moe_ep_correctness_jit_overhead_diagnostics_20260915.md)。目前已量的 **TP1 recovery** 在 FlashInfer/JIT cache hit 後約 31 秒，其 checkpoint＋weight loading 約 1 秒；這**不是**已量到的 EP-group configuration switch time。動態切 group 要另量建置／中斷成本，按**預測負載持續時間、回本門檻與 hysteresis**決策，不能對每個 request 即時重啟。若所有正常 profile 的 measured-best 是完整 replicas 或無 EP TP2，應停止以 EP capacity 作為主要 MoE-aware actuator；改研究有可驗證執行手段的 routing-aware scheduling／rank load，而非修改 workload 逼 EP 勝出。

## 5. Decision diversity：正式 A/B 的必要 gate

目前 `vllm_capability.py` 的 static MoE shape catalogue 有 `TP2+EP2`，但**沒有** native 已測得的 `TP1×DP2+EP2` shape；planner 雖理解 DP 與 replica_count 的不同，這份 candidate list 會先刪掉重要 EP-group 選項。[capability catalogue](../../sllm/backends/vllm_capability.py)。這是「作決策前的候選公平性」問題，**本輪沒有動手修改 catalogue**。同時 `TP2+EP2` 雖然已在列表，曲線／KV 與普通 TP2 幾乎重疊，不能只擴候選就假設兩版會分歧。

要先建立以普通真實 workload traces 加權的 `concurrency × input/output × routing profile × preempt 後可用 cards × warm/cold state` offline grid：每格附所有物理可行 candidates 的實測 TPS、p95、TTFT/TPOT、resident KV／重建成本，以及數值 correctness；用**完全相同資源／負載／候選**離線跑 Original 與 MoE-aware，保留 Original fallback。報告 `decision_change_rate`、有實測勝出的變更比率、提升的 CI／變異、無優勢但仍切換的誤判率，以及 workload-weighted 預期改善。**目前沒有合法的 Original-vs-MoE-aware decision change rate**：EP gate 未放行、部分 candidate 尚不存在、還沒有對四維 routing/資源格的對稱校準；不能先設定「要選不同」再改 cost 權重。

## 6. 四張 GPU 與 MPS

四張真實卡足夠做單機最小 gate、在固定兩卡上比較 TP2／EP2／兩個獨立完整 replicas，以及搶佔一張後的有限 fleet 切法：3×TP1 或 TP2/EP2+TP1（而不是假設能熱機所有替代雙卡 groups）。限制是真實跨節點／網路失效域與同時 warm 的配置數，**不是當前最早的 EP 正確性／服務成本瓶頸**。上述 [SpotServe 論文](https://arxiv.org/abs/2311.15566) 的雲端 distributed instance migration claim 不能由同一台 PCIe 機器直接驗證。

MPS 僅在先證明物理卡數卡住某個必要的 *logical control-plane decision* 後，作候選／planner 壓力測試；[NVIDIA 的 MPS 說明](https://docs.nvidia.com/deploy/mps/architecture.html)是讓多個進程共享同一張實體 GPU 的計算資源，並不創造新的實體 GPU／獨立故障域。共用一張物理 GPU 的多個 MPS workers 可以建立更多**邏輯進程**，卻不是更多獨立 EP 物理計算資源；不得拿它們的 throughput、KV／故障獨立性當正式 EP 多卡數據。`tc netem` 只有確認真正的 group 傳輸走 socket/network 後才可能是網路成本實驗；目前主要 handoff 為 host token replay，套網路延遲不會憑空產生 expert weight migration 收益。

## 7. 五個明確判斷與停止條件

1. **架構先改什麼？** 在現有 endpoint 下增加 per-group／per-rank／expert coverage 的可驗證 metadata 與全 group 故障處理；保留原 actor 重建、token replay 的安全路徑，不先實作 live expert migration／elastic EP。
2. **TP vs EP 是否有可用 crossover？** 在兩種正常 prompt/output、短 C≤128、長 C≤96 下，**沒有觀察到 EP-group 對共同候選（特別是兩完整 replicas）的吞吐 crossover**；`TP2+EP2` 與 `TP2` 的 KV 完全同容量，`DP2+EP2` 雖有比 full replica 更大 KV，也未勝過兩個完整 replicas。這是探索性的 tested-range negative result，不能推廣所有 workload；更不能拿尚未通過 correctness 的數字作正式收益結論。
3. **最值得實作的 MoE-aware actuator？** 條件式 group 配置選擇原本是首個*候選*，但其必要條件是 EP correctness、對含非 EP TP／full replicas 的 Pareto 優勢與 routing-information 增益；本輪高 C 仍由完整 replicas 勝出，故**目前沒有已證實值得實作的 EP 配置 actuator，暫不實作**。接下來先找現有 runtime 是否存在可執行的 routing-aware serving／expert-load scheduling 操作空間，而非 recovery loading 順序或逼 EP 勝出的特殊流量。
4. **4 GPU 夠嗎？** 夠當前 correctness／capacity gate 和小型 3GPU 搶佔後配置對比；不夠獨立多節點 spot fleet／網路／elastic-rank claim。MPS 不優先。
5. **正式 Original vs MoE-aware 怎麼改？** Correctness、normal-workload measured crossover、可執行 actuator、對等候選和 decision grid 分歧先過關；然後在相同 4GPU、預先固定正常 workloads／preemption／warm state 下做配對 runs（各版本至少三個**獨立 fresh engines**，視變異再加重複），報 TPS／p95／TPOT／KV／rank imbalance／重建成本與負載加權結果。若門檻沒過就如實報 negative finding，**不強迫兩版選不同方案**。

CPU 分析測試與重算：[診斷測試](../../tests/spotserve_test/test_moe_ep_capacity_diagnostics.py)。上述四張 GPU 的工作只讀取本機 checkpoint／vLLM code，僅寫新診斷 JSON、log、這份報告；沒有使用 MPS、改正式 baseline、停止他人的 GPU 進程或刪除舊結果。
