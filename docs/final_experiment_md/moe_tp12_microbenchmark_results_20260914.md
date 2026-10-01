# Granite MoE TP=1／TP=2 前提驗證

日期：2026-09-14（Asia/Taipei），GPU 測量於 23:14:50 完成。狀態：**C=1／2／4 全部量測通過，但多 token 配置交叉門檻未通過；停止 fleet 擴充**。

## 本輪結論

同一個真實 Granite MoE，在控制暖機狀態後，共完成 **12 個 fresh engines、648 個計時 batches、1512 個完整計時請求**。所有測量點的跨 TP output hashes 相同，計時區間 JIT warnings=0；所有請求均以指定輸出長度正常結束。實驗容器已退出，結束時四張 GPU 均無 compute processes。

在測試的 **64／256-token 輸出、C=1／2／4** 中：24 個 cells 有 23 個是 TP1 的 repeatable winner，1 個 unresolved，**沒有可靠 TP2 勝出，也沒有多 token crossover**。這支持「目前服務成本容易讓兩版都選 TP1」的解釋；單純多放一個 TP2 target，並不能保證有值得選的不同答案。TP2 在長輸入、單 token 輸出確實較快，但不能將此優勢外推成長 decode 收益。

這不是 Original／MoE-aware 收益實驗，也不證明 routing cost model 無效；它驗證的是進入新配置實驗前的服務成本前提。本輪不新增正式 1／2A／2B runs、不強迫兩版分歧。下一輪應先補測短輸出或容量壓力，再決定是否擴充 fleet。

## 目的與範圍

先回答同一 MoE 在目前硬體下，哪些工作量 TP=1 較好、哪些 TP=2 較好。只有單 token／TTFT 優勢不足以支持一般 decode 或 fleet 配置收益；若 C=1 的 64／256-token cells 沒有可重複的成本交叉，接續測 C=2／4。

本輪是獨立 warmed-service microbenchmark，不是 Original／MoE-aware 對照，不是正式 spot experiment。沒有 source preemption、KV／expert 權重搬移、網路整形或 fleet planner；不把共用 executor 說成兩種獨立資料搬移機制，也不強迫 TP2 勝出。

## 固定環境

- 模型：`ibm-granite/granite-3.1-3b-a800m-instruct`，revision `a02780686e08a03fe0d2679a293b5c74a90efa89`。
- 確認 `GraniteMoeForCausalLM`、sparse top-k、真實 runtime expert modules；32 layers、每層 40 experts、每 token top-k=8。兩種 TP 使用相同 checkpoint，沒有 dense fallback。
- 四張 RTX 5070 Ti，各 16 GB；執行前四卡均無 compute processes。之後每個 worker 記錄前後 GPU／process snapshots，不停止其他工作。
- TP1 固定 GPU2，TP2 固定 GPU2/3；不是 topology-rotation 結論，其他 physical groups 須另測。
- [Topology 觀測](../../results/moe_tp12_microbenchmark_20260914/hardware_topology.txt)：GPU2/3 均在 NUMA1，關係為 NODE；跨 NUMA pairs 為 SYS，沒有 NVLink 標記。不能將 topology 當成實測 P2P 頻寬。
- BF16、V1/eager/TRITON、synchronous standard scheduler、routing capture=true、prefix cache off、CPU offload=0；只比較相同 C runtime 下的兩個 TP。
- 實際 worker versions：vLLM `0.23.1rc1.dev562+g3483240b7.d20260716`、PyTorch `2.11.0+cu130`；NVIDIA driver `595.80`。Container image 為 `localhost/spotserve-python312-nixl:latest`，本輪沒有使用 NIXL handoff，containers 的 network=none。
- 載入與同形狀／同輸出長度暖機不計入 warmed service。Compiler cache 共用，但不將首次 cache population 混入 cold-ready 平均。
- 語料是 synthetic repeated-token prompts；不推論一般聊天品質、domain routing 因果或線上 SLO capacity。
- 啟動 log 顯示 TRITON 使用 default MoE config，缺少對應 RTX 5070 Ti 的 tuned config；結論限定本輪 backend／runtime，不推論模型在其他 kernel tuning 下必然沒有交叉。

已重新計算兩個權重 shard 的 SHA-256，與先前已驗證 checkpoint 相同：

| Shard | SHA-256 |
| --- | --- |
| 1 | `c4467e3124323e483c31ef2ab8afed70d2734bdb2bb03a5b05da4d4a529735b1` |
| 2 | `065160c51d07aa9614aad8e5f6e43ac5fdbfa0320dec47cf0a0c0f0a0d3cc17c` |

## 預先固定的矩陣與放行門檻

| 變數 | 值 |
| --- | --- |
| Input | 512／2048／4096／8192 tokens |
| Output | 1／64／256 tokens |
| 第一階段 C | 1 |
| 條件式第二階段 C | 2／4 |
| 每 TP independent engine blocks | 3，順序 TP1→TP2、TP2→TP1、TP1→TP2 |
| 每 cell／block warmed batches | 3；不是九次獨立部署 |

曲線以完整 batch wall time 為主；另保存每請求 latency、TTFT、首 token 後 decode 時間與完整 output token IDs。TTFT 是 frontend 收到首 token 的時間，不冒充 pure GPU prefill kernel time。不同 TP 接收的 prompt hashes 必須一致；輸出差異須揭露，不能將不同序列的 timing 當成已驗證同一工作量的 crossover。

每個 cell 的三個配對 engine-block 平均差都須同號，且差值平均超過 `max(1 ms, 2 × 配對差值 sample SD)`，才標描述性的 repeatable winner。多 token crossover 另外要求 64／256-token cells 中，存在 TP1 與 TP2 各自勝出的情境，且兩 TP 的 output hashes 相同。這是小試跑 gate，不是統計顯著性聲明。

若 C=1 未通過，自動接續 C=2／4 的獨立 service sweep；若仍未通過，停止 fleet 擴充，僅報告本測試範圍內沒有觀測到可靠交叉。即使找到交叉，也仍須獨立驗證 generic／routing／shuffled 的排序與選擇，不能直接宣稱 MoE 收益。

## 實作與證據

- [Host runner](../../scripts/run_moe_tp_microbenchmark.py)：unique owned containers、timeout／清理、完整矩陣 gate、三個 engine blocks、配對統計、native SVG 曲線。
- [Service profiler](../../scripts/profile_granite_moe_tp.py)：opt-in output-length sweep、sync scheduling、routing capture、實際 engine runtime 與 GPU UUID 驗證，舊單長度 CLI 保留。
- [CPU tests](../../tests/spotserve_test/test_moe_tp_microbenchmark.py)：缺 block／partial batches／input mismatch 不補造平均，single-token advantage 不冒充 multi-token crossover。
- [本輪 protocol](../../results/moe_tp12_microbenchmark_20260914/protocol.json) 與 [初始環境](../../results/moe_tp12_microbenchmark_20260914/environment_initial.json)。
- 原始 artifacts 目錄：`results/moe_tp12_microbenchmark_20260914/`；成功或失敗均保存新 worker JSON／logs，不覆寫舊正式數據。

## C=1 完整結果

六個 fresh workers 均 passed；共 **216 個完整計時請求**，所有跨 TP outputs 相同，timed JIT warnings=0。以下是三個 independent engine-block 平均值的 mean ± sample SD；每 block 內含三個 warmed batches。

| Input | Output | TP1 秒：mean ± SD | TP2 秒：mean ± SD | TP2 相對變化 | Repeatable winner |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 1 | 0.0260 ± 0.0007 | 0.0317 ± 0.0001 | +21.57% | TP1 |
| 512 | 64 | 1.5152 ± 0.0215 | 1.8067 ± 0.0174 | +19.24% | TP1 |
| 512 | 256 | 6.0427 ± 0.0828 | 7.2036 ± 0.0551 | +19.21% | TP1 |
| 2048 | 1 | 0.0612 ± 0.0004 | 0.0560 ± 0.0002 | −8.46% | TP2 |
| 2048 | 64 | 1.5519 ± 0.0145 | 1.8264 ± 0.0140 | +17.69% | TP1 |
| 2048 | 256 | 6.0900 ± 0.0537 | 7.2465 ± 0.0549 | +18.99% | TP1 |
| 4096 | 1 | 0.1294 ± 0.0006 | 0.1112 ± 0.0003 | −14.13% | TP2 |
| 4096 | 64 | 1.6142 ± 0.0146 | 1.8859 ± 0.0130 | +16.83% | TP1 |
| 4096 | 256 | 6.1768 ± 0.0977 | 7.3051 ± 0.0328 | +18.27% | TP1 |
| 8192 | 1 | 0.2975 ± 0.0008 | 0.2382 ± 0.0005 | −19.93% | TP2 |
| 8192 | 64 | 1.7856 ± 0.0257 | 2.0060 ± 0.0103 | +12.34% | TP1 |
| 8192 | 256 | 6.3621 ± 0.0752 | 7.4345 ± 0.0288 | +16.86% | TP1 |

正號表示 TP2 較慢，負號表示 TP2 較快。取樣的 64／256-token 八個 cells 全部是 TP1 較快；TP2 優勢只出現在本輪三個長輸入／單 token cells。這不推論未測的 2..63-token 短尾、其他 C 或其他 kernel tuning 沒有交叉。

![C=1 完整成本曲線](../../results/moe_tp12_microbenchmark_20260914/c1-c1-curves.svg)

以 4096+256 為例：TP2 TTFT 平均約 110.87 ms，TP1 約 128.65 ms；但首 token 後每剩餘 token 平均時間，TP2 約 28.21 ms、TP1 約 23.72 ms。約 17.77 ms 首 token 優勢被後續約 4.49 ms/token 差值累積抵銷，完整 batch 約慢 1.13 秒。這是 frontend／服務階段時間，不把全部差值直接歸因給 all-reduce 或 pure GPU kernel；還需 GPU tracing 才能作更細因果拆解。

本輪 C1 gate 為 `multi_token_crossover_observed=false`、TP2 多 token 勝出 cells=0、TP1=8、`single_token_only_tp2_advantage=true`。這裡「multi token」限定取樣的 64／256；gate 是實驗放行門檻，不是統計顯著性測試。已依事前條件接續並完成 C=2／4，沒有開始 fleet。

見 [C1 summary 與完整 raw workers](../../results/moe_tp12_microbenchmark_20260914/c1-summary.json)。

## C=2／4 完整結果

六個 fresh workers 均 passed，C2 共 **432 個完整計時請求**、C4 共 **864 個**。兩個 C 都在 max_num_seqs=4 engine 內量測，與 C1 max_num_seqs=1 的 runtime 分開記錄，不拼接成同一部署 profile。以下同樣是三個 independent engine-block 平均值的 mean ± sample SD。

| Input | C | Output | TP1 秒：mean ± SD | TP2 秒：mean ± SD | TP2 相對變化 | Repeatable winner |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 2 | 1 | 0.0510 ± 0.0010 | 0.0603 ± 0.0008 | +18.29% | TP1 |
| 512 | 2 | 64 | 1.6208 ± 0.0136 | 1.8911 ± 0.0193 | +16.68% | TP1 |
| 512 | 2 | 256 | 6.3832 ± 0.0551 | 7.4784 ± 0.0684 | +17.16% | TP1 |
| 512 | 4 | 1 | 0.0703 ± 0.0003 | 0.0702 ± 0.0002 | −0.22% | unresolved |
| 512 | 4 | 64 | 1.6413 ± 0.0127 | 1.9100 ± 0.0191 | +16.37% | TP1 |
| 512 | 4 | 256 | 6.4339 ± 0.0723 | 7.5209 ± 0.0722 | +16.90% | TP1 |
| 2048 | 2 | 1 | 0.1188 ± 0.0002 | 0.1048 ± 0.0002 | −11.81% | TP2 |
| 2048 | 2 | 64 | 1.7087 ± 0.0189 | 1.9774 ± 0.0229 | +15.73% | TP1 |
| 2048 | 2 | 256 | 6.5122 ± 0.0508 | 7.6141 ± 0.0849 | +16.92% | TP1 |
| 2048 | 4 | 1 | 0.2343 ± 0.0003 | 0.2043 ± 0.0003 | −12.78% | TP2 |
| 2048 | 4 | 64 | 1.8413 ± 0.0190 | 2.0808 ± 0.0179 | +13.01% | TP1 |
| 2048 | 4 | 256 | 6.6657 ± 0.0733 | 7.7425 ± 0.0566 | +16.15% | TP1 |
| 4096 | 2 | 1 | 0.2531 ± 0.0003 | 0.2147 ± 0.0004 | −15.17% | TP2 |
| 4096 | 2 | 64 | 1.8482 ± 0.0206 | 2.0867 ± 0.0110 | +12.90% | TP1 |
| 4096 | 2 | 256 | 6.6593 ± 0.0726 | 7.7384 ± 0.0581 | +16.20% | TP1 |
| 4096 | 4 | 1 | 0.5019 ± 0.0003 | 0.4238 ± 0.0007 | −15.56% | TP2 |
| 4096 | 4 | 64 | 2.1082 ± 0.0147 | 2.3094 ± 0.0095 | +9.54% | TP1 |
| 4096 | 4 | 256 | 6.9519 ± 0.0705 | 7.9715 ± 0.0530 | +14.67% | TP1 |
| 8192 | 2 | 1 | 0.5860 ± 0.0003 | 0.4684 ± 0.0006 | −20.06% | TP2 |
| 8192 | 2 | 64 | 2.1935 ± 0.0195 | 2.3423 ± 0.0111 | +6.78% | TP1 |
| 8192 | 2 | 256 | 7.0217 ± 0.0818 | 7.9983 ± 0.0676 | +13.91% | TP1 |
| 8192 | 4 | 1 | 1.1647 ± 0.0009 | 0.9285 ± 0.0015 | −20.28% | TP2 |
| 8192 | 4 | 64 | 2.8050 ± 0.0181 | 2.8347 ± 0.0180 | +1.06% | unresolved |
| 8192 | 4 | 256 | 7.6647 ± 0.0588 | 8.5219 ± 0.0597 | +11.18% | TP1 |

![C=2 完整成本曲線](../../results/moe_tp12_microbenchmark_20260914/c2-c4-c2-curves.svg)

![C=4 完整成本曲線](../../results/moe_tp12_microbenchmark_20260914/c2-c4-c4-curves.svg)

最接近的多 token cell 是 8192+64、C4：TP2−TP1 平均約 **29.72 ms（+1.06%）**，但三組配對差值為 **+34.05／+63.34／−8.24 ms**，方向不一致；paired sample SD 約 35.98 ms，描述性 guard 約 71.97 ms。故標為 unresolved，不能宣稱兩者完全相同，也不能宣稱 TP2 或 TP1 在此點有可靠優勢。

第二階段 gate：`multi_token_crossover_observed=false`、TP2 多 token 勝出 cells=0、TP1=15。連同 C1，共 23 個 TP1 勝出多 token cells、1 個 unresolved；沒有符合事前門檻的 TP2 多 token 勝出，因此依計畫停止 fleet 擴充。沒有事後更改樣本數或放行門檻。

完整證據：[第二階段 summary](../../results/moe_tp12_microbenchmark_20260914/c2-c4-summary.json)、[全部 raw runs 與結束環境](../../results/moe_tp12_microbenchmark_20260914/microbenchmark.json)、[自動產生的結果報告](../../results/moe_tp12_microbenchmark_20260914/report.md)。

## 驗證與限制

本輪十九個相關 CPU 測試檔共 **237 passed**，包含 output-length sweep、三個 engine blocks、single-token gate 與既有 planning／freeze／runtime 保護測試。CPU fixture 不當成 GPU 效能證據。

結束後另對完整 artifact 重新核對：1512 個 output token lists 的長度與 SHA-256、648 個完整 batches 的請求數、TTFT 不晚於完成時間、各 cell 的三個 block mean／sample SD 與 gate 均相符。三張 SVG 的 XML 可解析，本報告的本機 links 均存在。

C>1 的 service 測試不需要 native freeze；現有 NativeFreezeScheduler 仍限制 synchronous C=1，因此 C>1 service 成功不等於併發恢復已通過。

## 後續決策原則

1. **先看實測成本，不把不同配置直接當成有效決策空間。** TP=1 與 TP=2 的確是不同 runtime，但若服務成本一直偏向同一個配置，單純增加這兩種 target 仍可能讓兩版都選 TP=1。
2. **TP 不等於 EP。** 本輪 EP=false；TP2 把模型運算／張量分到兩個 ranks，沒有把不同 experts 分配成不同 instance 的私有集合。每個完整 instance 仍涵蓋全部 experts，不能用它宣稱 expert coverage／實體 placement 已變得異質。
3. **若多 token gate 沒通過，先停 fleet，提出下一輪前提驗證。** 可另測未取樣的 2..63-token 短輸出；或在確認記憶體安全後，測更高併發與 KV 容量壓力。也可另做 kernel tuning／GPU tracing。這些都屬於新矩陣，不能事後改本輪門檻或把未量測情境寫成已存在的交叉。
4. **若 gate 通過，還需獨立 routing 驗證。** 固定相同候選集合、負載資訊與 warm status，在 held-out workloads 對照 generic、真實 routing、shuffled routing 的預測排序、實測排序與選擇成本。必須控制 input／remaining output 長度及 C；若僅由這些一般負載資訊就能選對 TP，不能稱作 routing 的額外收益。只有路由訊號在這一步增加價值，才有依據進入 Original／MoE-aware 正式比較。
5. **網路模擬不是本輪缺乏配置交叉的直接解法。** 現有新 runner 交接的是 host tokens，不是 expert 權重或直接 KV transfer；對兩版相同 payload 注入相同延遲，只會增加共同成本。必須先確認實際傳輸路徑與不同決策會改變的 bytes／links，才設計局部限速；不修改全機介面，也不把 `tc` 當成 GPU P2P／NCCL 的通用限速工具。

## 重現方式

使用相同本機 checkpoint、container image 與 vLLM 環境，輸出到**新的**目錄；既有 artifact 目錄拒絕覆寫。需四卡環境，GPU2/3 不可有其他 compute jobs。

```bash
/work/containers/s112060021/Qwen3/vllm/.venv/bin/python -u -m scripts.run_moe_tp_microbenchmark \
  --model /work/containers/s112060021/Qwen3/spotserve-models/granite-3.1-3b-a800m-instruct \
  --model-revision a02780686e08a03fe0d2679a293b5c74a90efa89 \
  --output-dir results/moe_tp12_microbenchmark_NEW_RUN \
  --compiler-cache /work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve/results/moe_route_task_compiler_cache_20260914
```
