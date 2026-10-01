# Granite MoE EP feasibility 與 cold recovery breakdown

日期：2026-09-15（Asia/Taipei）。狀態：EP feasibility、基本 communication microbenchmark、三次 cold recovery accounting 均完成。

## 結論先說

1. **目前 Granite checkpoint 與 vLLM runtime 確實可以使用原生 EP。** 不只接受 flag：32 個 MoE layers 中，兩個 ranks 各持有不重疊的 20 個完整 experts，GPU 權重樣本符合 checkpoint；engine 與 inference 均完成。`TP2/DP1 + EP` 和 `TP1/DP2 + EP` 的通訊機制不同，後者才實測到 AG/RS dispatch／combine。
2. **無 NVLink 沒有讓最小 EP 測試無法運作，但不能宣稱 EP 有效能收益。** 在本輪 C=2 四個 shapes，TP+EP 比 TP2 控制組慢約 5.1–7.0%；DP+EP 慢約 1.2–13.1%。DP+EP 的 instrumented dispatch／combine 占比不小，約 13–49%，且包含等待與 profiling overhead。
3. **約 85 秒 recovery 的主要成本不是載入 experts。** 三次平均 `86.008 ± 0.688 s`；整段 model loading 約 `1.791 s`，checkpoint／weight loading 約 `0.822 s`（總時間約 **0.96%**）。約 `55.009 s` 集中於 dummy sampler；補充 CPU profile 定位到 FlashInfer sampling JIT build/load 與另一 rank 的鎖等待。
4. **有物理 expert partition 的研究入口，但還不是已完成的 MoE recovery optimization。** 下一步先驗證 EP 數值正確性、再確認真實 workload 是否有值得解決的 rank/expert 不平衡。通用 FlashInfer cache 問題應另外處理；本輪沒有實作 optimization，也沒有開始正式 Original／MoE-aware 實驗。

## 範圍

本輪只做 EP 可行性、基本通訊 microbenchmark、既有 cold recovery 分段與下一步建議。沒有新 MoE optimization、偏態 workload、fleet planner、正式 Original／MoE-aware 比較、網路整形或 vLLM 底層架構修改。

新診斷 worker 是 opt-in subclass／process-local timing hooks，原生函式仍執行原本運算；不改 expert 載入順序、weights、routing 或模型完整性。Cold 測試使用既有控制 worker、NativeFreezeScheduler、GPU adapter 與同一 warmup prefix，而不是另一個簡化 engine 的 cold time。

## 環境與方法

- 同一 `ibm-granite/granite-3.1-3b-a800m-instruct`，revision `a02780686e08a03fe0d2679a293b5c74a90efa89`；32 layers、40 experts/layer、top-k=8，BF16。
- 本機 vLLM HEAD：`3483240b7ea3d4372b6c79369ea36617f8b1fbb2`；既有實驗 runtime version 為 `0.23.1rc1.dev562+g3483240b7.d20260716`，本輪另外保存實際 runtime／module paths。
- 四張 RTX 5070 Ti，無 NVLink；EP 固定 GPU2/3，cold 固定原本選中的 GPU1。開始前四卡均無 compute processes；不停止其他工作。
- 同一 container image `localhost/spotserve-python312-nixl:latest`、V1/eager/TRITON、Torch `2.11.0+cu130`、CPU offload=0、prefix caching=false。只清理本輪自建容器，模型與 vLLM source 唯讀掛載；本輪不使用 NIXL。
- EP 依序測 `TP2/DP1/EP off` 控制組、`TP2/DP1/EP on`、若可行再測 `TP1/DP2/EP on`。三者皆最多兩張 GPU；DP2 不等於建立兩個完整 MoE replicas，必須檢查實際 expert partition。
- EP baseline 沿用 synthetic repeated-token shape 測量，不製作 domain skew、強制 hot expert 或修改 router。Input=512/4096、output=1/64、C=2，每 cell 三個 warmed batches；一個 fresh engine/config，屬基本可行性小試，非獨立部署三次或統計顯著性證明。
- 每 rank 需驗證實際 GPU UUID、32 個 expert modules、global→local expert map、GPU w13/w2 shapes，並核對每個本地 expert 的三個投影權重樣本與原 checkpoint。權重樣本核對不是全 tensor bytewise hash。
- Native `allgather_reducescatter` backend 是以 all-gather／reduce-scatter 實現 dispatch／combine，不能寫成已使用 NVLink 或某個高效 sparse all-to-all kernel。
- 基本 warmed-service 測量關閉 CUDA-event tracing；通訊另做一個 instrumented batch/cell，逐 rank 保存 CUDA events 與 CPU callback wall times，不跨 ranks 加總成總 wall time。
- Cold：三個 fresh containers，同一 GPU1、routing capture=true、NativeFreezeScheduler、max_model_len=8704、max_num_seqs=1、max_num_batched_tokens=2048、memory utilization=0.8；沿用舊 calibration 的 **4352-token train-code frozen prefix + 64 output tokens** 暖機。
- Compiler cache 延用既有路徑，沒有清除全機 OS page cache。此處 cold 指 process/engine cold，不代表 checkpoint 儲存體 cold；另記錄 physical read_bytes、major faults 與 input blocks，不能把 mmap iterator elapsed time當成全部磁碟 I/O。

## 診斷可靠性

第一次雙卡控制組遇到 stdout 的兩個 ranks JSON 接在同一行，收集器出現 `JSONDecodeError: Extra data`。該次由收集器中止，**不是 EP 不支援證據**，不列入有效 EP microbenchmark。原始失敗 logs／JSON 保存；修正 log 容錯後另用新目錄重跑。

Spawn 子程序會重新 import frontend script；入口 timestamp 必須依 PID 區分。首次 cold raw accounting 誤取 child entry，造成 frontend-import 負耗時。三次 inference 與 worker spans 有效，但**該 raw 檔的原始 major partition 不使用**。現已按真正建立 engine 的 frontend PID、原始 logs 與 host monotonic timestamps 重算，全部主要階段非負、加總吻合總時間；原始資料不覆寫，另存 [cold_accounting_verified.json](../../results/moe_architecture_diagnostics_20260915/cold_accounting_verified.json)，包含來源 SHA-256 與 clock provenance。入口 guard 與 accounting 測試亦已補上。

## 1. EP 是否真的生效

TP（tensor parallel）把**同一 expert 的矩陣切小**；EP（expert parallel）則把**不同的完整 experts 分給不同 ranks**。兩者不能僅由 command-line 的 TP 值判定。

下表是實際每一層 GPU tensors，`w13` 合併兩個輸入投影，所以完整 expert 的中間維度是 `2 × 512`。

| 實測配置 | 每個 rank 的 expert IDs | 每 rank 的 w13／w2 shape | MoE 內部 TP／EP | 真正 runtime 路徑 |
| --- | --- | --- | --- | --- |
| TP2 / DP1 / EP off | 兩邊都有 0–39 | `[40,512,1536]` / `[40,1536,256]` | TP=2 / EP=1 | 每個 expert 分矩陣，MoE final all-reduce |
| TP2 / DP1 / EP on | EP rank0：0–19；rank1：20–39 | `[20,1024,1536]` / `[20,1536,512]` | TP=1 / EP=2 | 完整 expert 分散；`NoDPEP` prepare/finalize，最後 all-reduce；**沒有 AG/RS dispatch** |
| TP1 / DP2 / EP on | EP rank0：0–19；rank1：20–39 | `[20,1024,1536]` / `[20,1536,512]` | TP=1 / EP=2 | `NaiveDPEPModular` → `AgRsAll2AllManager.dispatch/combine`；`use_all2all_kernels=true` |

三種配置均驗證實際 GPU2/3 UUID、32 個 expert modules、local/global maps 與每個本地 expert 的 w1/w3/w2 樣本。EP 的兩個 ownership sets 每層不重疊、聯集恰為全部 40 experts；不是兩個完整 expert replicas，也不是只在 metadata 宣告分散。TP+EP 的 attention 等非 MoE 部分仍可維持 TP，不應把 MoE 內部 TP=1 說成整個 engine 都單卡。

這與目前 pinned code 一致：[Granite MoE 使用 FusedMoE](../../../vllm/vllm/model_executor/models/granitemoe.py:67)、[EP config 將 MoE TP 改為 1](../../../vllm/vllm/model_executor/layers/fused_moe/config.py:1190)、[non-local expert 不載入該 rank 的 parameter](../../../vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:601)。其中 [use_all2all_kernels](../../../vllm/vllm/model_executor/layers/fused_moe/config.py:1040) 明確要求 DP>1；因此 `--enable-expert-parallel` 不代表每個配置都使用相同 all-to-all。

DP+EP logs 實際出現 `Using AgRsAll2AllManager all2all manager`、`EP Rank 0/2 ... Local/global ... 20/40`；instrumented inference 的每 rank dispatch 與 combine 次數分別為 **96、2112、128、2144**（四個 cells），不是只檢查初始化物件。其實作是 [prepare/finalize dispatch/combine](../../../vllm/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:70) 呼叫 [AG/RS manager](../../../vllm/vllm/distributed/device_communicators/all2all.py:42)：all-gather hidden states/top-k，再 reduce-scatter outputs。這是 all-to-all 語義的通用實作，**不是 selective sparse all-to-all，也沒有測 DeepEP／NIXL EP**。

注意 DP2 的 worker `rank` 在各自 TP world 中都是 0；實際 EP rank 是 0/1。本報告以 `parallel.ep_rank`、`local_rank`、GPU UUID 共同識別，沒有將兩份 rank=0 當成同一張 GPU。

### 推論核對的界線

全部 warm、warmed-service、profile batches 都完成指定長度；stream prefix 未被改寫，各配置的三次 warmed outputs 各自一致。單 token 的輸出全部與 TP2 控制組一致。

但 **64-token 輸出不是全部跨配置 byte/token 相等**：TP+EP 四個 prompt/request 組合中三個不同，首次不同 token 分別為第 7、12、7 個；DP+EP 也有三個不同，首次差異為第 5、3、9 個。原始 token IDs/hashes 全部保留。BF16 reduction/order 改變是可能原因，**本輪沒有 logits／誤差容忍／品質驗證，不能直接把差異判為正常，也不能宣稱輸出等價已通過**。後續 EP 研究需先加入獨立數值正確性 gate；這些差異沒有被隱藏或改寫。

## 2. 基本服務與 communication 成本

### 關閉 profiling 的 warmed batches

下列時間是同時兩個 requests 全部完成的 batch wall time，單位秒，mean ± sample SD；每 cell 三個 warmed batches。DP2 每 rank 接一個 request，TP2 則同一 engine 接兩個 requests，因此包括部署模式本身的排程差異；不是純 expert kernel benchmark，也不是 Original／MoE-aware ablation。

| Input / Output / C | TP2 控制組 | TP2 + EP2 | TP1 / DP2 + EP2 | TP+EP／DP+EP 相對控制組 |
| --- | ---: | ---: | ---: | ---: |
| 512 / 1 / 2 | 0.05594 ± 0.00031 | 0.05898 ± 0.00053 | 0.05954 ± 0.00068 | +5.4% / +6.4% |
| 512 / 64 / 2 | 1.80037 ± 0.01205 | 1.90770 ± 0.00870 | 2.03603 ± 0.00713 | +6.0% / +13.1% |
| 4096 / 1 / 2 | 0.20226 ± 0.00014 | 0.21640 ± 0.00006 | 0.20460 ± 0.00053 | +7.0% / +1.2% |
| 4096 / 64 / 2 | 1.99007 ± 0.00540 | 2.09175 ± 0.00486 | 2.18189 ± 0.00931 | +5.1% / +9.6% |

只有一個 fresh engine/config，SD 是 **engine 內 batch 變異**，不是三次獨立部署變異；不同配置依序執行，沒有交錯配對控制。因此以上只是機制與基本成本篩檢，不能據此做顯著性宣稱或推論所有 workload 都不適合 EP。

### DP+EP 的 dispatch／combine

另行一次 instrumented batch/cell，單位 ms；rank0/1 對應 EP rank0/1。

| Input / Output | Instrumented batch wall | Rank0 dispatch / combine | Rank1 dispatch / combine | 各 rank 的 dispatch+combine / profile wall |
| --- | ---: | ---: | ---: | ---: |
| 512 / 1 | 68.554 | 9.402 / 6.833 | 17.988 / 15.577 | 23.7% / 49.0% |
| 512 / 64 | 2349.172 | 169.773 / 139.052 | 383.389 / 323.350 | 13.1% / 30.1% |
| 4096 / 1 | 210.374 | 21.212 / 27.943 | 28.723 / 42.256 | 23.4% / 33.7% |
| 4096 / 64 | 2475.113 | 287.280 / 261.575 | 357.984 / 273.200 | 22.2% / 25.5% |

CUDA events 包住 current-stream 的函式區間，包含 collective 等待與 CPU launch gaps，不是「純 PCIe 在傳資料」的精準時間；沒有量 physical link bandwidth 或 bytes。Instrumentation 本身使 DP+EP 的 batch 比 ordinary warmed mean 高約 **2.8–15.4%**；不能把表中百分比直接當成可刪除的端到端成本。兩個 ranks 同時工作，**不跨 ranks 加總**；`moe_final_reduction` 和內層 `actual_moe_all_reduce` 也不能重複加總。DP+EP 的 TP world size=1，後者 callback 是 [單 rank early-return](../../../vllm/vllm/distributed/parallel_state.py:637)，不是額外跨 GPU all-reduce。

TP+EP 的實際 MoE final all-reduce 亦有非零事件：四個 cells 的逐 rank cumulative ranges 約 **7.1–15.8、169.5–199.6、39.0–46.3、199.7–218.5 ms**；AG/RS dispatch/combine 未執行。完整 counters/CPU/GPU event totals 在 raw JSON，無需把「EP flag 開了」誤說成同樣的通信路徑。

GPU2/3 topology 是 `NODE`、同 NUMA，沒有 `NV#`；`nvidia-smi topo -p2p r` 跨卡為 `CNS`，即該環境 peer-read 不支援，**不等於已量到某個頻寬數值**。NCCL logs 顯示 `via SHM/direct/direct`，即本機 shared-host-memory transport，不是 NVLink 或 direct GPU peer transport；也不是雲端網路。證據見 [topology 與結束狀態](../../results/moe_architecture_ep_20260915_attempt2/environment_final_topology.json)、[DP+EP log](../../results/moe_architecture_ep_20260915_attempt2/dp2-ep2.log:206)。

**判斷：可以繼續做有邊界的 EP 研究，不需要硬改 vLLM 才能測；但低併發下 communication/同步成本值得警戒，目前沒有 EP 加速證據。** 本輪沒有用網路限速放大差距。對當前 SHM collective 任意加 TCP `tc` 不會等價模擬這條資料路徑。

## 3. 約 85 秒 cold recovery 拆解

### 定義與主要階段

沿用既有 recovery GPU adapter：host 啟動 `podman run` 的 monotonic 時點 → native engine/control ready、runtime inspection → 原來的 4352-input/64-output warmup 完成。**不包含 source 停止、preemption planner 或之後完整 replay completion**，因此不是全流程中斷時間。

三次總時間為 **86.767、85.831、85.425 s**，mean ± sample SD = **86.008 ± 0.688 s**。Ready 的界線分開保存：engine constructor 返回約 **83.857 ± 0.236 s**，host 完成 runtime inspection、可開始 application warmup 約 **84.192 ± 0.688 s**，application warmup 完成才是此流程採用的 warmed-ready endpoint **86.008 ± 0.688 s**。以下是非重疊主要階段；各 run 精確加總吻合總時間，表格因 rounding 有些微差。

| 主要階段 | Mean ± sample SD（s） | 總時間比例 |
| --- | ---: | ---: |
| Podman/process/Python 入口前啟動 | 0.803 ± 0.026 | 0.93% |
| Frontend imports/configuration | 6.039 ± 0.024 | 7.02% |
| Worker constructor | 0.000865 ± 0.000067 | <0.01% |
| Worker device initialization（含 distributed） | 0.474 ± 0.011 | 0.55% |
| Model loading 整段 | 1.791 ± 0.004 | 2.08% |
| Engine memory profiling（含 dummy model/sampler） | 56.284 ± 0.072 | 65.44% |
| KV cache initialization | 0.007912 ± 0.000064 | 0.01% |
| Engine kernel warmup | 0.396 ± 0.032 | 0.46% |
| Engine bootstrap/import/IPC 等尚未細分殘差 | 18.061 ± 0.186 | 21.00% |
| Control connection/runtime verification | 0.335 ± 0.563 | 0.39% |
| Application warmup（4352 input / 64 output） | 1.816 ± 0.016 | 2.11% |
| **總計** | **86.008 ± 0.688** | **100%** |

此處 startup、device、bootstrap 是觀測區間，**沒有完全隔離 CUDA driver/context initialization**：三次在 `init_device` 前 CUDA 都已 initialized，故不能把 `0.474 s` 寫成純 CUDA 初始化，或把缺少 lazy-init span 寫成 CUDA 耗時 0。更早初始化包含於 imports/bootstrap 等區間。18.061 秒也是明確保留的殘差，不假冒完整拆成可歸因的編譯／I/O。

### Nested spans：哪些才與 weights 有關

以下已包含於上表，**不能再相加至總時間**：

| 子階段 | 三次平均（s） | 所屬主要階段／解讀 |
| --- | ---: | --- |
| Checkpoint/weight loading | **0.822** | Model loading；約總時間 **0.96%** |
| Model structure/weight allocation | 0.162 | Model loading |
| Weight postprocessing | 0.006917 | Model loading |
| Distributed initialization | 0.073842 | Worker device initialization |
| Dummy model run | 0.535 | Engine memory profiling |
| **Dummy sampler run** | **55.009** | Engine memory profiling；約總時間 **64.0%** |

Expert weight-loader callbacks 的 CPU cumulative time 平均約 `0.679 s`，checkpoint iterator advance 約 `0.00893 s`。前者包括 native copy/processing/launch，後者為 mmap iterator 行進，**都不是可獨立刪掉的 physical I/O/傳輸時間**。在整段 model loading 結束只增加一次 CUDA synchronize，以捕捉未完成 GPU 工作，沒有每個 expert 強制同步。

三次 load-phase `read_bytes`、`input_blocks`、`major_faults` 的 delta 都為 **0**，表示 checkpoint 已受 OS page cache 支援；mmap 不會完整反映在 `rchar`。所以這是 **process/engine cold、storage warm** 的結論，不能外推成網路儲存體或真正 storage-cold loading 也只需 0.822 秒。本輪沒有清除共享主機的 page cache。

### 55 秒究竟在做什麼

Cold 三次 spans 一致指向 `_dummy_sampler_run`（54.911–55.069 秒）。另在相同 pinned runtime/image 的 TP2 控制組做一次補充 cProfile，觀察到：

- `_dummy_sampler_run` → FlashInfer `top_k_top_p_sampling_from_logits` → `top_k_mask_logits` → `get_sampling_module` → **`jit.core.build_and_load`，約 54.9 秒**。
- 一個 rank 的 `build`／`run_ninja` 約 54.87 秒；另一 rank 主要在 `FileLock` context 等待約 54.92 秒。這是 build/load 與鎖／子程序等待，不可寫成 CPU 實際計算 55 秒，亦不可把 nested cumulative times 相加。

補充 CPU profile 不是三次 cold 的逐次 CPU profile；它是用來佐證同版本 sampling 路徑的定位，不能忽略配置差異。Cold 的 55 秒 dummy-sampler spans 是直接量測結果。

即使 workload 是 greedy、engine 設 `enforce_eager`，目前 native [_dummy_sampler_run](../../../vllm/vllm/v1/worker/gpu_model_runner.py:6062) 還會建立 top-p/top-k sampling metadata 去暖機，所以 eager/TRITON MoE **沒有排除 FlashInfer sampling JIT**。不能把這段錯認為 expert weight loading、MoE all-to-all 或 TorchInductor graph compilation。

當前 GPU adapter/EP runner 只持久化 Triton 與 CUDA caches；沒有設定 `FLASHINFER_WORKSPACE_BASE` 或持久化對應 FlashInfer workspace。[FlashInfer cache 預設路徑](../../../vllm/.venv/lib/python3.12/site-packages/flashinfer/jit/env.py:51) 位於容器 home 下，而 fresh `--rm` containers 沒有保留該路徑；[build_and_load](../../../vllm/.venv/lib/python3.12/site-packages/flashinfer/jit/core.py:307) 的鎖與 ninja 路徑符合補充 profile。**持久化 matching version/architecture 的 FlashInfer cache 或預建 kernels 是有證據的下一輪通用改善假說，尚未實作／驗證，也不是 MoE-specific 優化。**

## 4. MoE-specific optimization 空間與下一步

### 現在能說什麼

EP 成功表示目前模型/engine **已有「不同 ranks 擁有不同完整 experts」的物理基礎**，不再只能建立多個全 expert replicas；但這次只是固定原生 linear ownership，`enable_eplb=false`。沒有實作動態 expert placement、routing-aware rank replacement、部分模型先服務或 expert 搬移。既有 observe-only placement 介面仍不能自動 actuator 權重。

此外目前 AG/RS **會 gather 所有 states**，不是按 local/remote expert 決定傳不傳，因此**改 expert placement 不會自然降低這個 backend 的 activation communication bytes**。可研究的潛在收益主要是實際 expert/rank compute load 是否更平衡、或 native loader 真正少載入哪些必要權重；必須另證明收益機制，不能從 EP flag 推出「少搬資料」。EP group 失去一個 rank 也不能讓剩餘半模型直接獨立正常服務；group/必要權重的恢復 barrier 是下一輪需先釐清的限制。

### 建議順序（未執行）

1. **先做獨立 EP 數值正確性 gate。** 使用合理真實 prompts、logits/容忍度與 TP baseline 交叉核對，處理本輪跨配置 token 差異；不得用修改 router 或強制不同決策掩蓋差異。通過後，才有研究 routing-aware execution 的可信基礎。
2. **另案驗證通用 cold 啟動 cache。** 固定 runtime/GPU/image、保持兩版同等 cache 狀態，對照 FlashInfer cache 未持久化／持久化或預建；重跑 cold breakdown 確認是否真的降低約 55 秒區段。這項若有效，Original/MoE-aware 都應取得同樣 cache；不能只給 MoE 版以製造差距。
3. **若 EP 正確性通過，先診斷自然 workload 的 expert/rank load 與 AG/RS 占比，再決定 placement/routing-aware recovery 設計。** 不新增 skew，不預設需要搬 hot experts；只有存在可改善的不平衡且目前原生 execution 可 actuator，才值得擴充。不因本輪 C=2 沒贏就否定所有 EP 場景，也不因能啟動就開始正式 SpotServe。
4. **Weight-specific recovery 暫不列為主要速度 claim。** 在這個 storage-warm 設定，假設整個 checkpoint/weight loading 完全免費，樂觀上限也約 **0.96%**；連整段 model loading 都完全移除，上限約 **2.08%**。這是刪掉整段的保守改善天花板，不是已可達成的 expert optimization 收益。若未來研究 storage-cold/remote checkpoints，先量那種狀態的 I/O/ready barrier，再討論 selective loading。
5. **不採用未經驗證的 hot-expert-first loading。** Native [base loader](../../../vllm/vllm/model_executor/model_loader/base_loader.py:52) 需完成 load_weights/postprocessing 才返回 model，之後還有 profile/warmup/ready。只改 expert 讀取順序、最後仍全部載入，不能縮短此 barrier，也不能解決 FlashInfer sampling JIT。

因此本輪決策是：**EP 可行、基本成本允許繼續小規模診斷，但先停在 feasibility/breakdown；正確性與通用 cold cache 是下一輪前置事項。沒有足夠證據直接開始新的 MoE optimization 或正式 Original／MoE-aware 實驗。** 若後續自然 workload 沒有可改善的 EP load/必要權重成本，也應如實報告當前平台的研究空間有限，而非強迫兩版選不同方案。

## 程式與 artifacts

- [Host 診斷 runner](../../scripts/run_moe_architecture_diagnostics.py)
- [EP probe](../../scripts/probe_granite_ep.py)
- [Opt-in diagnostic worker](../../scripts/moe_architecture_diagnostic_worker.py)
- [Timestamp／CUDA event 工具](../../scripts/moe_diagnostic_trace.py)
- [CPU accounting／gating tests](../../tests/spotserve_test/test_moe_architecture_diagnostics.py)
- [有效 EP 控制組（TP2）](../../results/moe_architecture_ep_20260915_attempt2/tp2-control.json)
- [有效 TP+EP evidence](../../results/moe_architecture_ep_20260915_attempt2/tp2-ep2.json)
- [有效 DP+EP evidence](../../results/moe_architecture_ep_20260915_attempt2/dp2-ep2.json)
- [核對後的統計／communication／output summary](../../results/moe_architecture_ep_20260915_attempt2/verified_summary.json)：含來源檔案 bytes 的 SHA-256、所有四個 cells 的 mean/SD、逐 rank events、輸出差異及限制。Cold accounting 的 `source_row_sha256` 則沿用專案 `digest()`，對 parsed JSON 的 `json.dumps(sort_keys=True)` 計算 SHA-256，兩者 hash 定義不同。
- [Cold 修正後 accounting](../../results/moe_architecture_diagnostics_20260915/cold_accounting_verified.json)、[cold r0](../../results/moe_architecture_diagnostics_20260915/cold-r0.json)、[r1](../../results/moe_architecture_diagnostics_20260915/cold-r1.json)、[r2](../../results/moe_architecture_diagnostics_20260915/cold-r2.json)。r0/r1/r2 的舊 major partition 不用，只保留 raw traces；以上修正 accounting 才是表格來源。
- [首次失敗 collector 紀錄（排除，不是 EP failure）](../../results/moe_architecture_diagnostics_20260915/tp2-control.json)。沒有覆寫歷史正式 ablation 或舊結果。

驗證：相關診斷／既有 GPU adapter／native freeze／microbenchmark CPU tests **44 passed**；新增與修改的 Python 檔案 `py_compile` 通過，`git diff --check` 通過。CPU fixtures 不是 GPU EP 支援證據。所有本輪自建容器與 GPU compute processes 均已結束；結束核對四卡各約 2 MiB，無 compute process。
