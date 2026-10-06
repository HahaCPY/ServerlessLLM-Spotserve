# SpotServe for MoE Inference

> 報告版本：2026-10-02
>
> 程式基準：Git commit `287f7dc`
>
> 實驗範圍：single-host、4× NVIDIA GeForce RTX 5070 Ti 16 GB
>
> 研究主軸：MoE-aware Reparallelization、Migration 與 Recovery

---

## 1. 摘要

Spot GPU 能降低大型語言模型的部署成本，但 GPU 可能在推論期間被回收（preemption），使既有平行配置失效、進行中的 request 中斷，並造成 KV cache 等推論狀態遺失。SpotServe 透過 reparallelization、request migration 與 state recovery，在資源變動後重新配置服務並恢復未完成請求。

本專題研究如何將上述機制延伸至 Mixture-of-Experts（MoE）推論。MoE 每個 token 只啟用部分 experts，因此服務成本不只取決於 GPU 數量與 KV cache，也受到 expert placement、token routing、queue pressure 與 all-to-all communication 影響。本專題在 ServerlessLLM 控制平面與 vLLM runtime 間建立 MoE metadata、planner、state hooks 與 runtime verification，使系統能依據 GPU topology、expert locality、KV compatibility 與 queue 狀態選擇恢復方案。

實驗顯示，NIXL KV restore 能將 context recomputation 降為 0；但早期 Original 與 MoE-aware 正式比較因兩版選到相同 target 與 TP=1，end-to-end latency 未出現可歸因於 MoE-aware planning 的明顯差距。最新版本已進一步完成固定 EP 的實體 expert remap、destination-aware sparse all-to-all，以及同一 actor 的 quiescent EP2→EP4 resize。這些結果證明 MoE 特性已進入實際 runtime 操作，但大多仍是 single-host 驗證，不能宣稱已完成 physical multi-host recovery。

**關鍵詞：** SpotServe、Mixture-of-Experts、Preemption、Reparallelization、Context Migration、Stateful Recovery、Expert Placement、NIXL、vLLM

---

## 2. 研究目的與問題

本專題的目的，是研究 SpotServe 應用於 MoE 推論時，如何在 GPU 因 preemption 發生變動的情況下降低服務恢復與請求遷移 latency，並利用 MoE 的 sparse routing 特性，設計比 generic SpotServe 更適合 MoE inference 的資源調整策略。

核心研究問題為：

1. GPU 數量或 topology 改變後，如何同時考慮 parallel shape 與 expert placement？
2. Request migration 時，如何在 KV reuse、expert locality 與 target queue 之間取捨？
3. 如何判斷 KV／inference state 能否在不同 TP／DP／EP 配置上安全恢復？
4. Expert remap 是否能降低實際 all-to-all payload，而不只是改變 logical placement？

本專題不把 vLLM EPLB 重新實作在 SpotServe 中。兩者的責任區分如下：

- **SpotServe MoE planner：** 處理 preemption／resource change 後的 topology、target、migration 與 recovery 決策。
- **vLLM runtime／EPLB：** 執行 deployment 內的 expert ownership、weight remap、collective communication 與 load balancing。

---

## 3. 系統設計與具體做法

### 3.1 MoE-aware Reparallelization

原始 reparallelization 主要依可用 GPU 與 TP／PP／DP shape 重新部署模型。本專題加入以下資訊：

- Runtime-observed TP／DP／EP shape 與 GPU topology。
- 每層 expert ownership、placement epoch 與 fingerprint。
- Global／recent expert hotness 與 per-request routing histogram。
- Model loading、expert movement、queue、communication 與 remaining-workload cost。

Planner 先由 backend capability 產生合法候選，再建立 `ExpertPlacementPlan` 並估計：

```text
Replan Cost
  = model/engine startup
  + expert movement
  + expected serving latency
  + queue pressure
  + communication / load-imbalance penalty
```

選定方案後依變更種類執行：

- **一般 shape 變更：** 建立新 vLLM actor，ready 後切流並 drain 舊 actor。
- **相同 EP size：** 在同步 step boundary 使用 vLLM expert-remap hook 搬動 expert tensors，更新 `expert_map`，並以權重摘要、placement fingerprint 與 coverage 驗證。
- **EP size 變更：** 一般情況走 controlled actor recreation；目前另已驗證受限條件下、同一 actor 的 quiescent Elastic EP2→EP4 resize。

這使 reparallelization 從「剩餘 GPU 能否容納模型」提升為「在目前 workload 與 expert routing 下，切換到哪個配置最值得」。

### 3.2 KV + Expert Locality + Queue Migration

Migration planner 從 patched vLLM runtime 取得 request 的 routed-expert histogram，並結合 target 的 KV metadata、expert placement 與 queue snapshot。對 request `r` 與 target `t`，概念上的成本為：

```text
C(r, t)
  = w_kv × KV migration/recompute cost
  + w_exp × remote expert routing cost
  + w_q × queue cost
  + target warmup cost
```

其中：

- **KV cost：** target 可重用的 blocks 越多，重算成本越低。
- **Expert cost：** request 歷史上常用的 experts 若不在適合的 target rank，remote dispatch 成本較高。
- **Queue cost：** 避免所有 requests 因 KV 或 expert locality 集中到同一個 busy target。

如果 runtime 無法提供 routing 或 placement metadata，planner 會 fail closed，退化成 KV／queue-only 決策，不捏造 expert locality。

### 3.3 MoE-aware Stateful Recovery

Recovery 不直接以「模型名稱相同」判定 state 可用，而是分層檢查：

1. Model revision、dtype、tokenizer 等 semantic compatibility。
2. State serialization schema 與 backend restore capability。
3. KV block size、layout、rank mapping 與 cache geometry。
4. EP layout、placement epoch／fingerprint 與必要 expert coverage。

若 target 通過必要的 KV／state gates，系統透過 vLLM state hooks 與 NIXL export／restore GPU KV blocks；若 runtime state 為空、hook 不存在或 layout 不相容，則安全退回 token replay／retry。KV restore correctness 與 expert locality 分開回報，避免把 EP mismatch 一律誤判成 KV 不可恢復。

### 3.4 Expert Remap 與 Sparse All-to-All

最新版本加入 physical expert remap。固定 EP remap 透過 vLLM `rearrange_expert_weights_inplace()` 搬動未量化 FusedMoE expert tensors；所有 ranks 先通過 preflight，remap 後再核對 expert map、權重 SHA-256 與完整 coverage。

僅改 placement 不一定會降低 communication。傳統 `allgather_reducescatter` 對相同 shape 可能維持固定 payload，因此專題加入 `spotserve_sparse` backend：

1. 由 runtime `topk_ids` 與 `expert_map` 找出每個 token/expert row 的 destination rank。
2. 本地 expert rows 直接 bypass network。
3. 只有 remote rows 進入 variable-size `all_to_all_single` dispatch／combine。
4. Runtime counter 記錄 collective calls 與 tensor payload bytes，前後輸出也必須一致。

只有 candidate payload 嚴格低於 baseline，才允許回報 A2A reduction。

### 3.5 Preemption 後的整體流程

```text
Preemption / capacity event
        ↓
收集可用 GPU、live requests、KV 與 MoE runtime metadata
        ↓
產生合法 TP／DP／EP 與 expert-placement candidates
        ↓
Reparallelization：選擇 topology + placement
Migration：以 KV + expert locality + queue 分配 requests
Recovery：執行 compatibility gates
        ↓
direct KV restore / token replay / retry
        ↓
runtime readback、placement verification、correctness 與 metrics
```

---

## 4. 實驗設定與解讀原則

主要實驗使用 single-host 4-GPU testbed。不同實驗使用的 checkpoint 與 harness 不完全相同，因此數字只在各自表格內比較，不能跨表直接計算 speedup。

| 實驗 | 主要模型／配置 | 用途 |
|---|---|---|
| 四版本 recovery | Qwen2-MoE-Tiny，TP1 | 比較 failure、rerouting、recreate 與 KV restore |
| Original vs MoE-aware | Granite 3.1 3B A800M MoE | 比較 generic 與 route-aware planner |
| Expert remap／A2A | Qwen2-MoE-Tiny，DP2/EP2 | 驗證 physical remap 與 collective payload |
| Elastic EP resize | TP1×DP2×EP2 → TP1×DP4×EP4 | 驗證同一 actor 的 quiescent resize |

目前 preemption 由 JSONL trace replay 模擬，不是真實 cloud provider termination notice；same-host multi-container 或 two-failure-domain simulation 也不能當成 physical multi-host 結果。

---

## 5. 重點實驗結果

### 5.1 四版本 Recovery 小成果

以下擇 Tiny 240-token context、每格三次平均，呈現四種策略的主要差異。

| Version | Recovery time | P99 proxy | Success | Recovery data |
|---|---:|---:|---:|---|
| No Recovery | 21.457 s | — | 0% | Request failed |
| Rerouting | 21.670 s | 0.493 s | 100% | 重算 243 tokens，0 restored blocks |
| Reparallelization | 162.015 s | 0.466 s | 100% | 新 engine，重算 243 tokens |
| Modified/NIXL | 21.415 s | 0.301 s | 100% | 重算 0 token，restore 18 KV blocks |

**分析：** No Recovery 無法完成 request；Rerouting 與 Reparallelization 都要重算完整 context，其中 Reparallelization 又包含新 engine startup，因此時間約 162 秒。Modified/NIXL 成功將重算量降為 0，證明 KV state 確實被 restore。不過 Rerouting 與 NIXL 在此 controlled harness 都使用事前 READY target，完整 wall-clock 主要被 source pause、container 與控制成本主導，不能只由 21.670 vs 21.415 秒宣稱 production speedup。表中的 P99 也是單請求 proxy，不是線上流量統計 P99。

較長的 Qwen 4K／8K controlled runs 同樣觀察到 NIXL `recomputed_tokens=0`，分別 restore 257／497 blocks；最新 standalone V8 benchmark 則回報 1/1 true KV restore、16 restored tokens、6 restored blocks、0 fallback 與 `true_kv_rate=100%`。

### 5.2 Original SpotServe vs MoE-aware

正式比較固定相同 Granite MoE model、workload、trace 與硬體。Original 是不使用 route-neighbour score 的 generic baseline；兩版都不是 dense model。每組每版三次，共 18 個完整 GPU runs。

| Experiment | Original（mean ± SD） | MoE-aware（mean ± SD） | 相對差 |
|---|---:|---:|---:|
| Overall stream complete | 327.1574 ± 0.7852 s | 326.2688 ± 1.1085 s | −0.27% |
| Migration notice→complete | 10.7403 ± 0.0378 s | 10.6310 ± 0.1249 s | −1.02% |
| Reparallelization notice→complete | 95.6378 ± 0.1195 s | 95.7399 ± 0.2487 s | +0.11% |

**分析：** 兩版的差距很小，不能解讀為 MoE-aware optimization 的穩定收益。2A Migration 六次都選 GPU2 TP1，2B Reparallelization 六次都選 GPU1 TP1；當 planner 最後做出相同決策，兩版便會使用相同 executor，MoE signals 只增加少量 planning 工作而沒有改變實際執行。

後續 TP1／TP2 microbenchmark 也支持此解釋：在 64／256-token output、C=1／2／4 的 24 個 cells 中，23 個由 TP1 穩定勝出、1 個 unresolved，沒有可靠的 TP2 multi-token winner。這表示目前候選配置缺少成本 crossover，而不是僅靠增加 MoE cost term 就一定能產生收益。

### 5.3 新版 Runtime 能力驗證

| 能力 | 實測結果 | 可宣稱範圍 |
|---|---|---|
| Active fixed-EP remap | 同一 TP2/EP2 actor 於同步 step boundary 搬動 4 shards、0.75 MiB；最近一次 runtime remap 133.18 ms，apply／verify 通過 | Same-host fixed-EP physical remap；不是 EP-size resize 或 cross-host |
| Sparse A2A | 相同 workload、matching outputs；payload 1,157,376 → 937,728 bytes，降低 18.98% | Same-host DP2/EP2 collective payload reduction；不是 NIC traffic reduction |
| Elastic EP resize | 同一 actor 由 2 workers／EP2 擴成 4 workers／EP4；coverage 不變、6/8 expert owners 改變，resize 前後 inference 通過 | Quiescent same-host resize；不是執行中 GPU step 的 zero-pause resize |
| Failure-domain simulation | 8 expert shards、1,572,864 bytes 跨兩個模擬 failure domains；觀察到 512 次 inter-container A2A calls | Single physical host simulation；不能稱為 physical multi-host |

**分析：** 新版已從 logical placement 推進到可驗證的 physical expert operation。尤其 sparse A2A 的結果說明「搬動 expert」本身不夠；只有 communication backend 能讓 local rows bypass、remote rows 採 variable-size transfer 時，placement 才能轉化為 payload reduction。Elastic resize 則證明 actor identity 可以保留，但目前 resize boundary 仍需 quiescent 或先 drain active request。

Physical cross-host scheduling、host identity、moved-byte 與 internode-A2A fail-closed gates 已實作，但尚缺兩台實體 GPU hosts 上的 passing report，因此本報告不宣稱 cross-host expert migration 或 cross-node KV restore 已完成。

---

## 6. 討論與限制

### 6.1 為何早期 MoE-aware 沒有明顯更快

主要原因不是 planner 沒有執行，而是候選缺乏可辨識度：完整 replicas 都涵蓋相同 experts、queue 差異不足，且 TP1 在多數 decode workload 已較快。當 KV、expert locality 與 queue signals 都不會改變 target／shape 時，MoE-aware 版本自然難以產生 end-to-end 優勢。

### 6.2 Recovery latency 不只由 recomputation 決定

KV restore 可避免重新計算 prompt，但 total recovery 還包含 engine startup、runtime imports、compiler cache、export／transfer／restore 與控制面協調。故應同時報告 restored blocks、recomputed tokens、startup、first-token 與 completion latency，而不能只報 success rate。

### 6.3 目前結果的外部效度

- 多數結果來自 single-host 4-GPU 環境。
- 不同表格使用不同模型與 harness，只能做表內比較。
- Four-version controlled benchmark 的 target warm policy 不完全相同。
- Active fixed-EP remap 只涵蓋同步 step boundary；Elastic EP resize 仍需 quiescent／drain。
- A2A counter 量的是 runtime tensor payload，不等同 physical NIC wire bytes。
- Physical multi-host 與真實 cloud spot interruption 尚未完成。

---

## 7. 結論與下一步

本專題已將 SpotServe 的三個核心流程接入 ServerlessLLM／vLLM MoE serving，並將其延伸為：

- **Reparallelization：** GPU topology → topology + expert placement + workload cost。
- **Migration：** KV target selection → KV + expert locality + queue cost。
- **Recovery：** Dense state checks → KV／state／EP／placement-aware compatibility。

實驗證明 NIXL 能避免 context recomputation，並證明 physical expert remap、sparse A2A payload reduction 與受限條件下的 Elastic EP resize 可執行。然而，Original 與 MoE-aware 正式 A/B 尚未呈現明顯 latency 優勢，因為目前候選常收斂到相同 target 與 TP1。換言之，專題已建立必要的 runtime ability，但仍需讓 planner 在具有真實成本差異的候選中做出不同決策，才能證明 MoE-aware policy 的端到端價值。

後續優先工作為：

1. 建立具有不同 expert ownership、KV reuse 與 queue pressure 的真實 targets，重跑 paired Original／MoE-aware A/B。
2. 使用 routing-skew、短輸出、高 concurrency 與長 context workload，建立可重複的 configuration crossover。
3. 在兩台實體 GPU hosts 完成 cross-host expert migration、internode A2A 與 KV recovery gate。
4. 將 Elastic EP resize 從 quiescent／drain 推進到具 rollback 的低停頓切換。
5. 以至少三次 paired repeats 分段量測 notice→ready、restore、first token、completion、A2A payload 與 correctness。

---

## 參考文件

1. [MoE-aware planner](../spotserve-moe-aware-planner.md)
2. [SpotServe × MoE gap analysis](../spotserve-moe-gap-analysis.md)
3. [Performance summary](../spotserve-performance-summary.md)
4. [Performance experiment guide](../spotserve-performance-experiment-guide.md)
5. [Cross-host experiment guide](../spotserve-cross-host-experiment.md)
6. [Four-version recovery results](old_four_version_exp.md)
7. [Original／MoE-aware formal results](1_2A_2B_v2_results.md)
