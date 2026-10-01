# 新 MoE checkpoint × Elastic EP partial recovery feasibility（2026-09-15）

本輪只檢查候選 checkpoint、最小 EP2 serving、graceful rank removal；Phase 2 crash 與 Phase 3 replacement 僅在前一階段通過時執行。沒有修改 Granite、vLLM assertion、正式 SpotServe planner/cost model 或 workload；這不是正式 Migration Ablation。

## Phase 0：目前釘住的 source 與候選

實際 runtime 為 vLLM `0.23.1rc1.dev562+g3483240b7.d20260716`、local HEAD `3483240b7ea3d4372b6c79369ea36617f8b1fbb2`，Transformers `5.14.1`。vLLM 工作樹原本已有其他未提交修改；本輪沒有改 vLLM source。四張 RTX 5070 Ti 各顯示 16,303 MiB；權重以 BF16、每參數 2 bytes 估算，表內不是 engine 峰值記憶體。官方 checkpoint 規格與檔案大小分別見 [Trinity Nano Preview 卡與 config](https://huggingface.co/arcee-ai/Trinity-Nano-Preview)、[Trinity Nano Base](https://huggingface.co/arcee-ai/Trinity-Nano-Base)、[LFM2-8B-A1B config](https://huggingface.co/LiquidAI/LFM2-8B-A1B/blob/main/config.json)、[OLMoE config](https://huggingface.co/allenai/OLMoE-1B-7B-0924-SFT/blob/main/config.json)、[Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

| 候選 checkpoint | total／active | routed experts／top-k（另列 shared） | BF16 checkpoint 權重 | TP1／EP2 在 16GB 的判斷 | 目前 source 的 Elastic EP／EPLB interface | 四卡配置 |
| --- | --- | --- | --- | --- | --- | --- |
| `arcee-ai/Trinity-Nano-Preview` | **6.120B**（index）／約 **1B**（官方卡） | 128／8，另 1 shared | **12.240 GB = 11.40 GiB** | TP1 可嘗試；EP2 預估約 6.34 GiB 權重/rank，仍須 runtime 驗證 | `AfmoeForCausalLM` 原生 registry、有 `MixtureOfExperts`、`moe_layers`、EPLB metadata 更新；**source-level 通過，runtime 待測** | `EP2(GPU0/1) + 2 spare(GPU2/3)`；或 `2 × EP2`，但後者無 spare |
| `arcee-ai/Trinity-Nano-Base` | 約 6.120B／約 1B | 128／8，另 1 shared | 約 12.240 GB = 11.40 GiB | 同上 | 同一 `AfmoeForCausalLM`，**source-level 通過，runtime 待測** | 同上；與 Preview 是相同架構的 backup，不是獨立機制驗證 |
| `LiquidAI/LFM2-8B-A1B` | 約 8.3B／1.5B | 32／4 | **16.680 GB = 15.53 GiB** | EP2 權重可分散；TP1 BF16 已幾乎吃滿單卡，**EP2→EP1 graceful removal 很可能 OOM** | `Lfm2MoeForCausalLM` 原生 registry、有 `MixtureOfExperts`、`moe_layers`、EPLB metadata 更新；runtime 未測 | `EP2 + 2 spare` 可建，但不能假定縮成 EP1；或 `2 × EP2` 無 spare |
| `allenai/OLMoE-1B-7B-0924-SFT` | 約 7B／約 1B | 64／8 | **13.839 GB = 12.89 GiB** | TP1 緊但可嘗試；一般 EP2 可分 expert，仍需實測 | 目前 `OlmoeForCausalLM` **沒有 `MixtureOfExperts`／EPLB interface**，和 Granite 有相同 Elastic gate 風險，因此不選 | Elastic EP 配置不列為可行 |
| `Qwen/Qwen3-30B-A3B` | 約 30B／3B | 128／8 | **61.067 GB = 56.87 GiB** | TP1／EP2 BF16 都不適合 16GB，需更多 GPUs 或量化 | `Qwen3MoeForCausalLM` 有 interface，但尺寸 gate 失敗 | 四卡 BF16 不適合作本輪 `EP2 + spare` |

來源檔案：[AFMoE model／FusedMoE／EPLB fields](../../../vllm/vllm/model_executor/models/afmoe.py)、[LFM2 MoE](../../../vllm/vllm/model_executor/models/lfm2_moe.py)、[OLMoE model](../../../vllm/vllm/model_executor/models/olmoe.py)、[MixtureOfExperts protocol](../../../vllm/vllm/model_executor/models/interfaces.py)、[Elastic EP config gate](../../../vllm/vllm/config/parallel.py)、[model registry](../../../vllm/vllm/model_executor/models/registry.py)。AFMoE/LFM2 都直接用 EP group 的 rank/size 算本地 physical experts、註冊 `FusedMoE(enable_eplb=...)`；`AfmoeForCausalLM` 將每個 MoE layer 的 runner 放入 `moe_layers` 並能更新 physical-expert metadata。這是 source-level 支援，不可替代實際 inference/coverage 證據。OLMoE 一般 FusedMoE inference 不能推論為 Elastic EP 可用。

**選擇：**先用 chat-tuned `Trinity-Nano-Preview`（revision `b576b32b8bba7f5218312c9509e494537f8d282d`）做 Phase 1；`Trinity-Nano-Base` 是同架構、同尺寸的 backup。Preview checkpoint index 記錄 6,120,003,328 total parameters／12,240,020,480 BF16 weight bytes。按 54 個 MoE layers、128 routed experts、hidden 1024、expert intermediate 256 估算，EP2 每 rank 權重約 6.8 GB；這只估權重，CUDA context、KV、communicator、warmup 仍要以 runtime peak 查核。

## Phase 1／2／3 gate

Phase 1 在獨立 Podman 容器中，只使用 GPU0/1；GPU2/3 保持 spare。TP1、DP2、EP2、Ray backend、同步 EPLB、零 redundant experts、AG/RS all-to-all。必須先確認兩 ranks 正常推論、runtime model 被 `is_mixture_of_experts` 接受、每層兩 rank 的 `expert_map` 聯集完整，再呼叫正常的 `scale_elastic_ep(1)`。before/after 紀錄 PID、GPU UUID、model object ID、EP/DP group identity、weight pointer、coverage、inference token IDs 與 reconfiguration latency。若失敗，不 kill rank。

Phase 2 只在 Phase 1 passed 後重建 EP2 group，以專用容器內**精確確認的 Rank1 PID**模擬 crash；查 Rank0/engine/process group 存活與 reconfiguration。若 survivor 無法保留，停止，不進 replacement。Phase 3 只在 Phase 2 survivor 可保留且能安全重組時，用 GPU2 加 replacement rank，核對 missing-expert coverage 與 Rank0 PID/model state 未 restart；未通過 coverage 前不恢復 routing。

## 實測狀態

權重下載與 Phase 1 runtime gate 進行中；此處尚無可用的 Elastic EP serving／rank removal 結果，不能預先宣稱 partial recovery 可行。
