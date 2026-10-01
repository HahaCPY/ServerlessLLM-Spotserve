# EP rank-level partial recovery：Phase 1 graceful removal feasibility

日期：2026-09-15。狀態：**Granite + 目前釘住的 vLLM，Phase 1 未通過前置 serving gate；沒有執行 rank removal。** 這不是正式 Original／MoE-aware Migration Ablation。依分階段要求，本輪在 Phase 1 回報後停止，沒有 kill rank、加入 replacement rank 或做 stateful/cost comparison。

## 實驗範圍與環境

- 本機 vLLM HEAD `3483240b7ea3d4372b6c79369ea36617f8b1fbb2`，工作樹有既存 NIXL／KV 等修改，本輪沒有修改 vLLM source。
- 同一 Granite MoE checkpoint `granite-3.1-3b-a800m-instruct`；runtime log 確認 `GraniteMoeForCausalLM`，不是 dense model。EP2 使用 TP1、DP2、Ray DP backend、`enable_expert_parallel=True`、`enable_elastic_ep=True`、`enable_eplb=True`、同步 EPLB、零 redundant experts、AG/RS all-to-all。
- 專用 Podman 容器只使用 GPU2/3、模型與原始碼唯讀掛載；不發布端口。既有 `sllm-store` 基線程序保留。實驗完成／失敗後 GPU2/3 均回到約 242 MiB 基線；只停止本輪自建失敗容器。
- 唯讀 rank probe 打算在 before/after 取得 PID、GPU UUID、model object ID、EP/DP process-group identity、每層 runtime `expert_map`、weight device/shape/data pointer 與 40-expert coverage，再在同一 engine 前後各做 inference。這些欄位因前置 engine 啟動失敗，**沒有成功的 before/after runtime snapshots**。

## 實際嘗試與結果

| Attempt | 到達階段 | 結果 | 原始資料 |
| --- | --- | --- | --- |
| 1 | Ray local startup | 容器 worker registration 失敗；未進模型。後續以相同映像獨立重現時，Raylet log 顯示預啟動過多 workers，最後 `pthread_create` resource unavailable。 | [attempt1](../../results/moe_elastic_ep_phase1_20260915_attempt1/container.log)；[獨立 Raylet 重現日誌](/tmp/spotserve-ray-phase1.3ZUkbM/session_2026-09-15_06-35-29_074690_1/logs/raylet.err) |
| 2 | Ray placement-group 建立 | 將 Ray CPU 額度限制為 4 後 Ray 可啟動；但 DP master `127.0.0.1` 與 Ray bridge node IP 不符，被 `create_dp_placement_groups` 拒絕；未進模型。 | [attempt2 report](../../results/moe_elastic_ep_phase1_20260915_attempt2/report.json)、[container log](../../results/moe_elastic_ep_phase1_20260915_attempt2/container.log) |
| 3 | Granite 權重載入後的 profile/dummy warmup | 使用 Ray 實際 node IP 後建立兩個 EP ranks；兩 ranks 各約 3.36 GiB model memory，weight-loading 約 0.70／0.88 秒。**兩 ranks 均在 `GPUModelRunner.eplb_step()` 的 `_moe_model is not None` assertion 失敗**；engine 未 ready、未完成 inference、未呼叫 2→1 `scale_elastic_ep()`。故障後自建容器卡住並由本輪明確停止。 | [attempt3 report](../../results/moe_elastic_ep_phase1_20260915_attempt3/report.json)、[container log](../../results/moe_elastic_ep_phase1_20260915_attempt3/container.log)、[rank0 Ray log](../../results/moe_elastic_ep_phase1_20260915_attempt3/ray/session_2026-09-15_06-38-45_626320_1/logs/worker-9efec4486cebee09459d2738c3e606c601768b6d9ebd90899a6ca2d8-01000000-1969.out) |

Attempt 3 的 `report.status=failed` 是判準；較早版本的 driver 只看容器 exit code，因 probe 曾捕捉 `SystemExit` 而留下 `exit.json returncode=0`，**不得以該 exit code 宣稱實驗成功**。後續 runner 已改為同時檢查 report status 並對失敗返回非零；本輪沒有把失敗 attempt 重跑或倒填成成功數據。

## 阻塞層級：為何「正常 EP」不等於「Elastic EP」

以前的最小 EP 診斷已證明 Granite 在**未開 Elastic EP／EPLB**時可以 serving、每層 40 experts 分成 rank0 0–19／rank1 20–39；見 [既有 EP feasibility](moe_ep_feasibility_cold_recovery_20260915.md)。但目前 [vLLM ParallelConfig](../../../vllm/vllm/config/parallel.py) 明確要求 `enable_elastic_ep` 必須同時開 `enable_eplb`；關掉 EPLB 不是合法的 Elastic EP 配置。

EPLB 的 [GPUModelRunner](../../../vllm/vllm/v1/worker/gpu_model_runner.py) 先以 `is_mixture_of_experts(model)` 尋找實作 [MixtureOfExperts protocol](../../../vllm/vllm/model_executor/models/interfaces.py)，再把它保存為 `_moe_model`。目前 [GraniteMoeForCausalLM](../../../vllm/vllm/model_executor/models/granitemoe.py) 使用 `FusedMoE`，但不提供該 protocol 要求的 `moe_layers`、expert-weight metadata、`set_eplb_state`／`update_physical_experts_metadata` 等介面。結果 `_moe_model=None`，profile/dummy run 進入 `eplb_step` 時直接 assertion；實際 runtime logs 的兩個 ranks 都在此失敗。

因此**最早需要修改的是 vLLM 的 Granite model↔EPLB interface/metadata 層**，不是 SpotServe planner、Migration cost model、Ray process-group 或 EP all-to-all 的延遲參數。這可能需要正確註冊 32 層 expert weights 與 EPLB mapping，不能只刪掉 assertion 或 monkeypatch dummy warmup；Graceful scale-down 本身在 [ElasticEPScalingState](../../../vllm/vllm/distributed/elastic_ep/elastic_state.py) 還要進行 EPLB expert reshuffle。這一輪沒有實作該介面，也沒有硬改 vLLM。

## Phase 1 四項問題的誠實答案

1. **Surviving rank 是否不 restart？** 未知；因 engine 不能 ready，沒有正常 scale-down。
2. **Process group／communicator 如何重建？** 尚無 Granite runtime 觀測。原始碼有 standby group 建立、EPLB reshuffle、切換與舊 group destroy 的預定路徑；不能把原始碼路徑當成實測完成。
3. **Survivor model state 與 `expert_map` 是否保留／改變？** 未知；未得到 before/after snapshots。
4. **Reconfiguration 實際耗時？** 未測得；`scale_elastic_ep(1)` 未呼叫，不能拿 engine 啟動失敗所花時間代替。

## 下一步 gate

在目前 Granite checkpoint 上，**Phase 1 未完成，因此不進 Phase 2 crash kill**；否則只會量到已知的 EPLB 啟動缺口，不是「Elastic EP 能否承受突然死亡」。若要繼續，先由使用者選擇：在不做大型 workaround 的前提下，針對 Granite 補足並獨立驗證 vLLM `MixtureOfExperts`／EPLB model interface；或提供本機已存在、原生 EPLB-compatible 的 MoE checkpoint 只做 vLLM 機制 feasibility。兩者都需先讓 Elastic EP2 正常 serving、驗證 runtime coverage，才能重跑 graceful removal。即使該 gate 通過，也不預設 crash replacement、KV 保留或較低 JIT/total recovery cost 成立。
