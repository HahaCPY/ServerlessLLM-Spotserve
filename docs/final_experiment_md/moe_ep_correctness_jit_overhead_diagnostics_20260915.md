# Granite MoE：EP correctness、JIT cache 與 recovery／overhead 診斷

日期：2026-09-15，Asia/Taipei。這是診斷報告，**不是 Original vs MoE-aware 正式比較**。

## 1. 結論與本輪範圍

- 原生 EP 能分散 expert 權重並執行，但 **whole-model correctness 尚不放行**。四個既有非重複-token 案例各生成 512 tokens，兩種 EP 都逐 token 等同 TP；重複-token 壓力案例仍分歧。Router 與抽樣 expert 計算核對通過，不等於整個模型已證明正確。
- 約 55 秒 FlashInfer JIT **可以透過跨程序持久 cache 避免**。同 model/config、每次重新建立容器：FlashInfer cache miss 為 86.636 秒；三次 hit 為 **31.254 ± 0.256 秒**。
- 如果 FlashInfer、Triton、CUDA cache 全部從空目錄開始，另一次啟動為 **197.677 秒**；同目錄第二次為 **30.751 秒**。因此原先約 85 秒不是「所有 compiler cache 全冷」的普遍基準。
- 31.254 秒的 cache-hit recovery 中，checkpoint＋weight loading 約 **0.969 秒，3.10%**；整個 model-loading 階段約 1.961 秒，6.27%。約 **76.93%** 在 imports／engine、worker bootstrap。現有條件下，expert loading 不是大幅改善 recovery time 的主戰場。
- EP 成本不是一個固定百分比。量到真正的 expert GEMM、token alignment、AllGather／ReduceScatter／AllReduce；但 profiler 明顯干擾 rank 同步，**不能把端到端慢幾%精確分攤成 PCIe、dispatch、compute 各幾%**。
- 本輪沒有實作 optimization、skew workload、fleet planner，也沒有更改 vLLM 源碼、權重、router 或 expert placement。

主要可重算資料：[diagnostic_summary_v2.json](../../results/moe_ep_phase2_20260915/diagnostic_summary_v2.json)。原始結果、logs、前一版分析均保留；JSON 的 `status: passed` 表示程序／診斷執行成功，**不是 EP 數值等價驗證已通過**。

## 2. 固定環境與公平性

| 項目 | 本輪設定 |
|---|---|
| 模型 | 本機 `granite-3.1-3b-a800m-instruct`，revision `a02780686e08a03fe0d2679a293b5c74a90efa89` |
| MoE 證據 | `GraniteMoeForCausalLM`；32 MoE layers，每層 40 experts、top-k=8；未替换 dense model |
| vLLM | HEAD `3483240b7ea3d4372b6c79369ea36617f8b1fbb2`；實際 `0.23.1rc1.dev562+g3483240b7.d20260716` |
| Python／Torch | vLLM `.venv` Python 3.12；Torch 2.11.0+cu130 |
| GPU | 4× RTX 5070 Ti 16 GB，無 NVLink；主要 EP 診斷 GPU2/3，layer trace GPU0/1，recovery GPU1，FP32 oracle GPU0 |
| 通訊 | 本機 NCCL logs 為 `SHM/direct/direct`；GPU peer-read 顯示 chipset 不支援。不是 TCP 多節點實驗，也未量測純 PCIe 頻寬 |
| EP engine | 原生 V1、BF16、eager、TRITON MoE；max length 8704、max sequences 8、batched tokens 2048、memory utilization 0.7 |
| Recovery engine | 既有 NativeFreeze recovery 路徑、TP=1、max sequences 1、memory utilization 0.8、routing capture 開啟；其餘主要設定相同 |
| 共同限制 | CPU offload=0、prefix caching 關閉、async scheduling 關閉；模型和源碼唯讀掛載、network=none；只關閉自己的診斷容器 |

三種配置在同一組 GPU2/3 上**順序**執行，每種只有一個 engine；每個 performance cell 先暖跑一次，再量三次。這三次 SD 是同一 engine 內的 batch 變動，不是三次獨立部署，也不是正式統計顯著性檢驗。獨立冷啟動的三次 cache hit 則確實每次建立新容器／程序。輔助診斷在另一對卡上曾同時執行，可能共享 CPU、主機記憶體資源，故 latency 只作探索資料。

| 配置 | 實際 expert 權重位置 | 實際 MoE 通訊機制 |
|---|---|---|
| TP2 reference：TP=2、DP=1、EP off | 每 rank 全部 40 experts，但每個 expert 的 intermediate dimension 切為 256 | MoE tensor-parallel 最後 AllReduce |
| TP2+EP2：TP=2、DP=1、EP on | rank0 experts 0–19、rank1 20–39，每個 local expert 為完整 intermediate=512 | MoE 內部 TP=1／EP=2，最後跨 EP ranks 合併；**沒有 AGRS dispatch/combine** |
| DP2+EP2：TP=1、DP=2、EP on | 同樣每 rank 20 個完整 experts | `NaiveDPEPModular` → `AgRsAll2AllManager`，AllGather dispatch＋ReduceScatter combine |

前兩種 attention 都是 TP2，較能隔離 MoE 切法；第三種 attention 變成 TP1／DP2，包含 attention 與 DP scheduler 的差異，**不是只改一個 MoE communication kernel 的消融**。Placement 與 checkpoint scalar 核對來自本輪每 rank 的 runtime 証據，不是只看 flag；詳見 [tp2-reference.json](../../results/moe_ep_phase2_20260915/tp2-reference.json)、[tp2-ep2.json](../../results/moe_ep_phase2_20260915/tp2-ep2.json)、[dp2-ep2.json](../../results/moe_ep_phase2_20260915/dp2-ep2.json)。

## 3. EP correctness：通過了什麼，還不能宣告什麼

### 3.1 相同輸入的逐 token 比較

六個案例都是 4096 prompt tokens，greedy、固定 seed、ignore EOS、生成 512 tokens、C=2。四個非重複案例直接使用先前 calibration 的 prompt，沒有根據 expert hotness 選新輸入。兩個重複-token 案例沿用前一輪 `[17]×4096`／`[18]×4096`，是壓力診斷，不是新造的 skew workload。

| 相同輸入案例 | TP2+EP2 第一次不同的 output token | DP2+EP2 第一次不同的 output token |
|---|---:|---:|
| previous-repeated17 | 12 | 3 |
| previous-repeated18 | 7 | 9 |
| train-code | 無，512/512 相同 | 無，512/512 相同 |
| train-story | 無，512/512 相同 | 無，512/512 相同 |
| validation-science | 無，512/512 相同 | 無，512/512 相同 |
| formal-serving | 無，512/512 相同 | 無，512/512 相同 |

另有 C=8、每個輸出 128 tokens 的高 routing 負載。四個既有非重複案例仍一致，重複案例仍不同，而且同一重複輸入的兩份併發請求可能在不同位置分歧。所有已量請求的 stream prefix、完成長度與 finish reason 驗證通過；這只排除了可見的串流增刪／改寫，**沒有證明內部 token→expert assignments 全部無遺漏**。

### 3.2 Router、expert kernel 與 capacity／drop 檢查

- 每次檢查遍歷全部 routing rows：IDs 必須在 0–39、同一 row 不得重複、weights 必須有限且和為 1；與實際 router logits 的 FP32 softmax top-8 核對。三種配置 32 個 router objects 全覆蓋，invalid／duplicate／nonfinite／non-tie top-k mismatch 均為 0，weight 誤差低於事先設定的 `1e-5`。
- 高負載快照中，TP 與 TP+EP 每 rank 各核對 1,081,088 routing rows，DP+EP 各 540,608 rows；每 row 8 assignments。計數跨 layers 且可能含 runtime padding／dummy，不是唯一使用者 token 數。
- 對每個 Triton expert-kernel object 用**實際 rank-local weights** 建立 FP32 tensor oracle，抽取至多 16 個均勻位置（含第一／最後 row），核對 up/gate→SiLU→down→routing 加權的 local output。32 objects/rank 全檢查；最大 relative L2：TP 0.621%、TP+EP 0.792%、DP+EP 0.699%。全部低於事先設定的 3%，且 max-abs／nonfinite 檢查通過。
- 原生 [TritonExperts](../../../vllm/vllm/model_executor/layers/fused_moe/experts/triton_moe.py) 路徑使用 `moe_align_block_size`，以 block padding 排列 token assignments；本輪沒有開啟 capacity-factor／overflow token-drop 政策。因此不能套用其他模型訓練的 capacity-factor 解釋。
- 抽樣 oracle 能排除已抽樣位置明顯 mapping／expert 計算錯誤，**不能代替每個 assignment 的完整 dispatch→compute→combine 守恆驗證**。未看到 token drop 證據不等於已證明零 drop。

### 3.3 相同前綴 logprobs 與 FP32 參考

不能在第一個不同 token 後繼續直接比較 autoregressive logprobs，因為兩版前綴已不同。因此本輪將 TP 的 512-token continuation 放入每個模式的 prompt，teacher-force 相同 4608-token 輸入，逐位置比較 shared returned top-k／true-token logprobs，共 3072 positions/mode。

| 比較 | 最大 shared logprob 絕對差 | top-1 sets 不相交位置 | 超出原定 abs=0.15 的位置 |
|---|---:|---:|---:|
| TP+EP teacher vs TP teacher | 5.891 | 18 | 2420/3072 |
| DP+EP teacher vs TP teacher | 8.812 | 26 | 2395/3072 |

**沒有事後調寬門檻讓結果通過。** 此測量只涵蓋 returned candidates，非全 vocabulary KL。也核對 TP 自身 cached decode vs teacher-forced prefill：最大差 4.273，top-1 不相交共 16 個位置，集中在重複案例。這表示 BF16 的 prefill／decode 計算形狀與 KV 重算本身就敏感，不宜把所有 teacher 差異都歸咎 EP；但 strict whole-model numerical gate 仍未通過。

進一步在第一次不同 token 的**相同完整前綴**上跑 HF `GraniteMoeForCausalLM` FP32、TF32 關閉、eager attention，使用相同 checkpoint。為符合單卡記憶體，按 256 tokens 分塊處理完整前綴，不縮短 context。

| 比較／案例 | TP token | EP token | FP32 top-1 | FP32 logp(TP token)−logp(EP token) |
|---|---:|---:|---:|---:|
| TP+EP，repeated17 token12 | 20901 | 34 | 34 | −3.396 |
| TP+EP，repeated18 token7 | 34 | 32 | 34 | +1.065 |
| DP+EP，repeated17 token3 | 34 | 47763 | 34 | +5.418 |
| DP+EP，repeated18 token9 | 203 | 36 | 203 | +3.978 |

這些不是都屬於「top-1 幾乎同分」，不能以 near-tie 一句話放行。FP32 參考是 **HF cross-implementation oracle，不是完整 vLLM FP32 ground truth**，本身亦有 attention／prefill chunk 與 native BF16 差異。原始資料：[TP+EP FP32](../../results/moe_ep_phase2_20260915/tp2-ep2-fp32.json)、[DP+EP FP32](../../results/moe_ep_phase2_20260915/dp2-ep2-fp32.json)。

### 3.4 逐層定位：目前最有力的原因證據

額外對上述 TP+EP 兩個分歧前綴，分別以 native TP2 與 native TP2+EP2 **重新 prefill 相同完整輸入，C=1、只生成下一 token**，複製 32 層最後 query 的 MoE input、router logits／top-8、MoE output、decoder output，不替換任何計算。

| native TP vs TP+EP | repeated17，4107-token 前綴 | repeated18，4102-token 前綴 |
|---|---:|---:|
| Layer0 input／router logits／selected IDs | 完全相同 | 完全相同 |
| Layer0 MoE output relative L2 | 0.407% | 0.378% |
| 首次 router-logit 不同（layer index 從0算） | 1 | 1 |
| 首次 top-8 expert-set 不同 | 20 | 29 |
| 當層 reference top8−top9 logit gap | 0.003906 | 0.005859 |
| 當層最大 router-logit 差 | 0.027344 | 0.037109 |
| Layer31 decoder output relative L2 | 27.294% | 8.593% |

觀察支持：**相同 router 輸入的第一層並未先選錯 expert，而是 TP 切分與 EP 完整 expert 的 BF16 計算／加總先出現小差異；後面差異大於 top-8 邊界，才改變 expert set，形成不連續放大**。這是目前有證據支持的數值累積機制，尚不是證明所有分歧皆為可接受浮點差異。

這次重算前綴兩版下一 token 反而相同（repeated17=20901、repeated18=32），與原 cached decode 的分歧不同，進一步顯示 prefill／cached decode 路徑敏感。HF FP32 輔助逐層 trace 亦顯示 TP 和 EP 都可能與 FP32 的 expert set 不同；不能把 TP 視為數值真值。PyTorch 官方亦說明不同批量／計算順序的浮點運算不保證 bitwise 相同；但這項一般原理**不能直接證明本輪的大幅後層差異正常**。[PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

原始 trace：[native TP](../../results/moe_ep_phase2_20260915/tp2-layer-trace.json)、[native TP+EP](../../results/moe_ep_phase2_20260915/tp2-ep2-layer-trace.json)、[HF FP32 trace](../../results/moe_ep_phase2_20260915/tp2-ep2-fp32-layer-trace.json)。此逐層追蹤尚未覆蓋 DP+EP 的所有分歧，也未完整追蹤 cached decode 的歷史 KV 誤差。

**Correctness 決定：原生 EP capability 成立；已量 operator checks 通過；全模型數值等價／差異可接受性尚未確認。不得使用下面 EP latency 作正式勝負結論。**

## 4. FlashInfer JIT：cache miss 與 hit 分開量

主要系列只建立一個新的空 FlashInfer workspace；保留原有 Triton／CUDA cache，checkpoint 位於本機且不清 OS page cache。每次啟動相同 model/config、新程序／容器；一個 cold-cache run，接三個 hit runs。

| Cache 狀態 | Recovery 到固定 warm request 完成（秒） | Dummy sampler（秒） | FlashInfer build/load（秒） |
|---|---:|---:|---:|
| FlashInfer miss；其他 compiler cache 保留 | 86.6356 | 55.0764 | 54.9104 |
| Hit，新容器1 | 31.4998 | 0.1435 | 0.0141 |
| Hit，新容器2 | 30.9889 | 0.1452 | 0.0145 |
| Hit，新容器3 | 31.2744 | 0.1449 | 0.0143 |
| **Hit 平均 ± sample SD，n=3** | **31.2544 ± 0.2560** | **0.1445 ± 0.0009** | **0.0143 ± 0.0002** |

證據不是只有總時間變短：三次 hit 的 cache `.so` **bytes／SHA256／mtime 全不變**；`run_ninja` 從 54.907 秒變為平均 0.01125 秒。Hit 仍呼叫 build/load 與 no-op ninja，但沒有再花約55秒編譯。這證明**本輪測試的同 model/config** 可以跨程序重用；不保證不同 CUDA architecture、版本、dtype、shape 或 backend 也 hit。

資料：[cache_diagnostics.json](../../results/moe_cache_phase2_20260915/cache_diagnostics.json)。

### 補充：全部 compiler cache 空，不與以上平均混合

另外建立全新 `moe_all_jit_cache_phase2_workspace_20260915`，FlashInfer、Triton、CUDA cache root 原先不存在：

| 全新 compiler-root 系列 | Recovery（秒） | Dummy sampler（秒） | Application warm request（秒） |
|---|---:|---:|---:|
| 所有 compiler cache cold，n=1 | 197.6768 | 54.8111 | 111.5343 |
| 同 root 第二次啟動，n=1 | 30.7512 | 0.1434 | 1.8346 |

Cold 的 native JIT monitor 記錄首次 `_compute_slot_mapping_kernel` 與 `fused_moe_kernel` Triton compilation，發生在 application warm request 內。**111.534 秒是整個該 request 的實測時間，並非逐 compile-call 精確加總**。因此 eager／torch.compile 關閉不代表沒有 FlashInfer／Triton JIT，engine constructor 完成也不代表已覆蓋實際 serving shapes。

這個系列只有一冷一熱，沒有可用 SD，不混入主要 hit n=3 平均。[補充原始結果](../../results/moe_all_jit_cache_phase2_20260915_attempt2/cache_diagnostics.json)、[cold worker log](../../results/moe_all_jit_cache_phase2_20260915_attempt2/workers/cache-cold-r0.log)。

資料稽核：補充系列使用當時共用 driver，raw protocol 的 `cache_cold_definition` 仍寫成「僅 FlashInfer 空」；**對這個全新 root 補充系列不正確**。以上分類按實際新目錄與啟動命令，raw 不覆寫；driver 已補初始 cache-root empty 記錄供後續診斷。補充第一個 attempt 因建立新 cache 的 parent directory 缺少而在 GPU 啟動前失敗，沒有納入數據；有效資料在 `attempt2`。

## 5. Cache-hit recovery breakdown 與 MoE-specific 上限

沿用原 NativeFreeze recovery 設定，warm request 是既有 train-code frozen prefix **4352 input tokens＋64 output tokens**。計時從 host 發起 owned container 到該固定 warm request 完成；不是移除 source／恢復整個 fleet 的總流程。Engine constructor ready 平均 **29.442 秒**，完成固定 warm request 的 ready 定義為 **31.254 秒**，兩個邊界分開保存。

以下為三個 cache-hit run 的**互不重疊主階段**，同一 monotonic clock，sum 等於每次 total：

| 主階段 | 平均秒數 | Sample SD |
|---|---:|---:|
| Container／Python entry | 0.7857 | 0.0008 |
| Frontend imports／configuration | 6.2560 | 0.0853 |
| Engine-core spawn／pre-worker bootstrap，扣 CUDA span | 10.4526 | 0.2633 |
| Worker Python imports，扣 CUDA lazy-init | 7.3339 | 0.0793 |
| 所有已捕捉程序的 CUDA `_lazy_init`，outermost | 0.1786 | 0.0091 |
| Worker constructor | 0.0008 | 0.0000 |
| Worker device init，**包含下表 distributed init** | 0.4684 | 0.0308 |
| Whole model loading，**包含下表 checkpoint／weight** | 1.9606 | 0.0632 |
| Engine memory profiling／dummy runs | 1.4116 | 0.0068 |
| KV cache initialization | 0.0079 | 0.0002 |
| Engine kernel warmup | 0.3849 | 0.0041 |
| Worker 後的 engine bootstrap 殘餘 | 0.2007 | 0.0026 |
| Control connection／runtime verification | 0.0093 | 0.0005 |
| Application warm request | 1.8034 | 0.0059 |
| **合計** | **31.2544** | **total SD=0.2560** |

`engine-core spawn/pre-worker bootstrap` 包含 core imports、IPC 等等待，**不是純 OS process fork 時間**。CUDA init 在 worker constructor 前已被 imports 觸發，透過 opt-in `sitecustomize` 提前捕捉；frontend 也初始化 CUDA，所以合併所有已捕捉 PID，重入 nested lazy-init 不重複計。這個 0.1786 秒只是 `_lazy_init` API span，不是所有 CUDA library／driver 相關初始化成本。較早的 `refined_breakdown.json` 僅有 GPU worker 的約0.034秒 CUDA span；**本報告以 v2 all-process 分析為準**，舊衍生檔保留供稽核。

下面是**巢狀**子階段，不能再加到主表：

| 子階段 | Hit 平均秒數 | 說明 |
|---|---:|---|
| Distributed initialization | 0.0680 | 包在 worker device init；TP1 仍建立 group，但不是2卡EP init成本 |
| Checkpoint discovery | 0.0026 | 檔案定位／index 等，不是讀完全部權重 |
| Checkpoint iterator advance | 0.00895 | 290 calls 的 iterator wall time；mmap 的實際取頁可能在後續 copy 發生 |
| Checkpoint＋weight loading | **0.9688 ± 0.1118** | 包含 safetensors／mmap 訪問、view／切片、weight-loader callbacks／copy；不是純 disk I/O |
| Expert weight-loader callbacks | **0.8046** | 3840 calls 的 aggregate CPU wall time，是上一列子集；不等於 GPU傳輸bytes或可消除時間 |
| Model structure／allocation | 約0.1580 | Whole model-loading 子集 |
| Weight postprocessing | 約0.0070 | Whole model-loading 子集 |
| Dummy sampler | 0.1445 | Memory profiling 子集，包含 FlashInfer cache-hit library load |

每次 loading 的 `read_bytes`、`input_blocks`、`major_faults` 增量均為 **0**。有 minor faults；這是本機 **storage/page-cache warm** 的測量，不能稱為 remote checkpoint 冷讀，也不能把 iterator 的9ms当成 checkpoint 全部 I/O。資料已分開保留 iterator、callbacks、I/O counters，但 mmap 取頁與 copy 仍交錯，**未得到純 disk／純 H2D 的完全互斥分割**。

改善上限需按可改變部分計算：

- 假設不可能但樂觀地將**全部 checkpoint＋weight-loading**刪成零，最多約減少 **3.10%** 的31.254秒（約0.969秒）；expert-only 更小。
- 假設刪掉整個 model-loading（含 allocation／postprocess／其他非expert開銷），最多約 **6.27%**。這不是可達的 MoE optimization claim。
- Imports／engine／worker bootstrap 合計約24.043秒、**76.93%**。持久 JIT cache 能先改善共同固定成本，但不屬於 MoE-aware 專屬收益。
- 完整模型仍須所有必要權重到齊才服務；本輪沒有 lazy expert／部分載入可服務機制。**改成 hot-expert-first 的順序本身，不能降低 ready critical path**。更慢遠端 storage 可能改變比例，需另量，不能從本機結果推定。

## 6. EP overhead profiling 與負載變化

各模式掃 P∈{512,4096}、N∈{1,128}、C∈{1,2,4,8}。P 是 prompt 長度、N 是 output 長度、C 是同時提交請求數；不是固定每一步 decode 的有效 batch size。N=1 近似 prefill-heavy，但仍含 scheduler／sampler／IPC，不是純 prefill kernel 時間。

以下是**未啟動 kernel profiler 的三次 batch wall 平均**，括號為相對 TP2 的時間差。資料只用於診斷，不作 EP 正式效能 claim：

| P | N | C | TP2 秒 | TP2+EP2 秒（差） | DP2+EP2 秒（差） |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1 | 0.0300 | 0.0309 (+3.12%) | 0.0629 (+110.03%) |
| 512 | 1 | 2 | 0.0582 | 0.0607 (+4.27%) | 0.0605 (+3.93%) |
| 512 | 1 | 4 | 0.0651 | 0.0682 (+4.87%) | 0.0691 (+6.25%) |
| 512 | 1 | 8 | 0.1099 | 0.1173 (+6.75%) | 0.1094 (−0.38%) |
| 512 | 128 | 1 | 3.5555 | 3.8505 (+8.30%) | 4.0863 (+14.93%) |
| 512 | 128 | 2 | 4.2342 | 4.4318 (+4.67%) | 4.6861 (+10.67%) |
| 512 | 128 | 4 | 4.2733 | 4.4910 (+5.09%) | 4.8281 (+12.98%) |
| 512 | 128 | 8 | 4.3586 | 4.5386 (+4.13%) | 4.9229 (+12.95%) |
| 4096 | 1 | 1 | 0.1032 | 0.1099 (+6.50%) | 0.1532 (+48.44%) |
| 4096 | 1 | 2 | 0.2029 | 0.2163 (+6.61%) | 0.2091 (+3.08%) |
| 4096 | 1 | 4 | 0.4024 | 0.4293 (+6.70%) | 0.3762 (−6.51%) |
| 4096 | 1 | 8 | 0.8031 | 0.8568 (+6.68%) | 0.7384 (−8.06%) |
| 4096 | 128 | 1 | 4.1293 | 4.4122 (+6.85%) | 4.7880 (+15.95%) |
| 4096 | 128 | 2 | 4.4244 | 4.6055 (+4.09%) | 4.8410 (+9.42%) |
| 4096 | 128 | 4 | 4.6029 | 4.8465 (+5.29%) | 5.1181 (+11.19%) |
| 4096 | 128 | 8 | 5.0669 | 5.2989 (+4.58%) | 5.4950 (+8.45%) |

三次 SD、generated tokens/sec 均在 JSON。觀察：TP+EP 慢幅約3–8%；DP+EP 的 C1 需要沒有本地請求的另一个 DP rank 仍參與 EP collectives／dummy，短工作量放大協調固定成本；prefill-heavy、C4/8 反而可能受 attention DP 分工獲益。**不能將它全部解釋為 EP expert placement 收益，也不能沿用上一輪1–13%為固定定律。**

### 6.1 真正執行的 kernel／通信路徑

用每 rank 的 Torch CPU＋CUDA profiler，另捕捉 router、expert apply、AGRS dispatch/combine、MoE final reduce 的 CUDA-event intervals。採用原生輸入和權重，不改 kernel。

代表性 **P512/N128/C2 的 profiled batch**，單位為每 rank 累積 kernel ms（非端到端 stage 秒）：

| 配置／local rank | Expert GEMM | Dispatch alignment | Router kernels | AllGather | ReduceScatter | AllReduce |
|---|---:|---:|---:|---:|---:|---:|
| TP2 / 0 | 144.59 | 15.26 | 28.03 | 2.71 | 0 | 127.93 |
| TP2 / 1 | 144.12 | 15.25 | 27.95 | 39.36 | 0 | 3752.04 |
| TP2+EP2 / 0 | 145.93 | 15.37 | 28.07 | 2.37 | 0 | 131.54 |
| TP2+EP2 / 1 | 136.64 | 15.36 | 28.00 | 33.76 | 0 | 2118.99 |
| DP2+EP2 / 0 | 147.86 | 15.66 | 28.56 | 297.74 | 272.10 | 14.27 |
| DP2+EP2 / 1 | 138.60 | 15.66 | 28.50 | 719.63 | 615.96 | 27.28 |

如何解讀：

- **Expert compute**：`fused_moe_kernel` 是上／下兩次 GEMM；完整 expert 與 TP-half expert 的單次形狀不同、local token assignment 數不同，不能預設 EP 必然少算。同案例 C1 的 rank0 expert GEMM 為 TP117.50ms、TP+EP126.03ms、DP+EP210.59ms；但 DP idle/dummy 和 routing 差異也是因素。
- **Dispatch／alignment**：全局→local expert map、token block padding／排列 kernel 確實執行。這和跨卡 dispatch 是兩件事；TP+EP 也有 alignment，並不代表它走 AGRS all-to-all。
- **Combine／all-to-all**：DP+EP 量到真實 AllGather 和 ReduceScatter kernels，及對應 native manager 呼叫。它將全部 DP hidden states、top-k IDs／weights gather，再 reduce-scatter outputs；不是只把指定 expert 的 token 稀疏發给該 rank。[本機 pinned AgRsAll2AllManager](../../../vllm/vllm/distributed/device_communicators/all2all.py:42)
- **TP+EP 的通訊**：AGRS callback 為0；MoE final reduction 加上 attention TP AllReduce。表中少量 AllGather 不是 MoE dispatch，可能來自其他 engine／sampling 操作。DP+EP 表中小型 AllReduce 含 scheduling metadata，不是 TP2 的 BF16 expert-output 合併。
- **其他 kernel／CPU 開銷**：dense GEMV/GEMM、attention、RMSNorm、copy、sampling／IPC、Python kernel launch 也有成本。尤其短 decode kernel 很快，不能把 Python wrapper 的CUDA interval全部稱為expert純GPUcompute。

### 6.2 Profiler 限制與還不能精確歸因的部分

- P512/N128 的 profiled batch 比普通 batch 慢 **1.43–1.71×**；P4096/N1 約0.85–1.04×，短 batch 的形狀／dummy／測量變動也存在。兩種觀測時間窗不能直接互減成 overhead 的精確 critical-path 分割。
- NCCL kernel 可包含等另一 rank launch／到達的時間；明顯 per-rank skew 隨 case 改變。上表 TP rank1 的3.752秒不是單純 PCIe 資料傳輸時間，也不是無 profiler 的TP服務時間。
- Profiler export／stop 本身常花約一分鐘，**未算入上表普通或 profiled batch wall**；它是診斷工具成本，不是服務／recovery 成本。
- 聚合排除 `MoEDiag:*` 與 `nccl:*` CUDA annotations，避免和實際 `ncclDevKernel*`／expert kernels 重複加總；只保存每 rank top80 CUDA event types，總數屬已記錄 kernel 的聚合，不是完整 critical-path。
- 因正確性仍未完全放行、部分性能输入输出會分歧，表中並非嚴格逐步相同 activations 的因果比較。**已完成路徑與 kernel 機制定位，但尚不能精確回答「慢13%中有幾%是純通訊」**。這需要後續固定共同前綴／activation 的輕量量測與更低干擾 tracing，而非事後湊百分比。

完整 per-cell/per-rank kernels、calls、CPU totals、CUDA intervals 與 profiler multiplier 在 v2 JSON 的 `kernel_breakdown`；原始 `profiles` 同時保留。

## 7. 下一步方向：現在不做 expert placement 實作

本輪的決策是：**先不啟動 MoE-aware expert placement，也不以 cold recovery time 當主要 MoE-specific 優化目標。**

1. **Correctness gate 仍未放行**：下一個若獲准的診斷，應補 native cached-decode 的逐層／KV追蹤、同形狀更高精度參考與完整 assignment 守恆核對；區分可接受 numerical sensitivity 和真正 combine／mapping bug。不得靠移除壓力案例、放寬門檻或只看四個相同輸出来宣告通過。
2. **先控制共同 cache／warmup 成本**：記錄 model revision、GPU arch、Torch/vLLM/FlashInfer/Triton、dtype/backend、warmup shapes 等 cache identity；cold/hit 分層，ready 要包含實際服務 shape。維持 cache 能改善兩版共同成本，不包裝成 MoE-aware 成效。本輪只有診斷用持久目錄，未部署新的 cache 管理策略。
3. **EP 支援 ≠ 可動態搬 experts**：pinned Granite 類別沒有 `MixtureOfExperts` runtime protocol 所需的 `num_moe_layers`、`expert_weights`、`moe_layers`、`set_eplb_state` 等介面；runner 只對通過該 protocol 的 `_moe_model` 接上 EPLB。[Granite class](../../../vllm/vllm/model_executor/models/granitemoe.py:473)、[protocol](../../../vllm/vllm/model_executor/models/interfaces.py:847)、[runner gate](../../../vllm/vllm/v1/worker/gpu_model_runner.py:5236)。這是源碼能力檢查，本輪沒有試跑 `enable_eplb`，不能聲稱已驗證動態 relocation。
4. **現有 placement 操作空間有限**：native experts 是 linear 0–19／20–39。Pinned round-robin 僅支援 multiple expert groups 等條件，AGRS 也不屬於支援其 routing table 的 backend；不應假設改 flag 就能替 Granite 任意重排。[placement gate](../../../vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py:123)。本專案 observe-only placement 介面未改成 actuator。
5. **即使能重排，AGRS 不保證少傳 bytes**：它 gather 全部 states，簡單換 expert ownership 不會減少這種 activation bytes；或許能改善 local expert 負載／kernel batch 形狀，需先在既有真實輸入下觀察 per-expert counts、per-rank compute、throughput／queueing，不需要先造 skew。[AGRS 實作](../../../vllm/vllm/distributed/device_communicators/all2all.py:85)
6. **不要預設 EP weight-filter 或 hot-first 能省 recovery**：Granite checkpoint 是 packed 3D expert tensors；pinned `enable_ep_weight_filter` 文件指出3D fused checkpoint不生效，不能直接宣稱各rank只讀半份expert檔案。[flag 說明](../../../vllm/vllm/config/parallel.py:164)。完整權重到齊才能ready的前提下，單改順序也不縮短關鍵路徑。

因此，若 correctness 後續獲確認，更合理的下一個研究目標是**既有輸入下的 serving throughput／expert load imbalance／routing-aware scheduling 操作空間診斷**，先證明有可執行的控制手段，再決定是否值得實作 optimization。網路延遲模擬不會消除 imports／JIT，也不能让 token 交接變成真實 expert-weight migration；本輪未採用 `tc`、MPS 或人工限速。

## 8. 可重算與驗證

新增 diagnostic-only workers、opt-in early timestamp hooks、cache／EP／FP32／layer-trace probes 與 CPU analysis tests。原生運算先執行，診斷只讀取／核對輸入輸出；profiling 的性能擾動如上列明。沒有改 planner、routing policy、expert ownership 或底層 vLLM 源碼；既有 dirty worktree 保留。

可用 vLLM `.venv` 重算分析；輸出採 exclusive-create，請使用新的檔名，不覆蓋 raw：

```bash
../vllm/.venv/bin/python -m scripts.analyze_moe_phase2_diagnostics \
  --ep-dir results/moe_ep_phase2_20260915 \
  --cache-dir results/moe_cache_phase2_20260915 \
  --output results/moe_ep_phase2_20260915/diagnostic_summary_recomputed.json
```

主要 GPU scripts：[EP driver](../../scripts/run_moe_ep_phase2.py)、[EP probe](../../scripts/probe_moe_ep_phase2.py)、[correctness worker](../../scripts/moe_ep_correctness_worker.py)、[cache driver](../../scripts/run_moe_cache_diagnostics.py)、[early hooks](../../scripts/diagnostic_site/sitecustomize.py)、[layer trace](../../scripts/run_moe_ep_layer_trace.py)、[FP32 oracle](../../scripts/probe_granite_fp32_prefix.py)、[analysis](../../scripts/analyze_moe_phase2_diagnostics.py)。重新跑 GPU 診斷須使用新 cache/output 路徑，不清除既有 cache。

CPU regression checks：phase2、architecture diagnostics、GPU-runtime、TP microbenchmark、native-freeze probe 五組，共 **50 passed**。這驗證 accounting／analysis／artifact 行為，不能替代上述 GPU correctness gate。所有 owned containers 已結束；沒有停止其他使用者的 GPU 工作。
