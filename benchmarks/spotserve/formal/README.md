# K8s 多節點 Qwen MoE F1/F2 實驗工具

請在 ServerlessLLM 的 head pod 內執行
`scripts/run_k8s_moe_f1_f2.py`。該 pod 必須與 router 共用實驗結果儲存空間、
能連線至 ServerlessLLM HTTP endpoint，並與所有 GPU worker 連接至相同的
Ray namespace。

預設實驗使用 Qwen1.5-MoE-A2.7B-Chat。K8 testbed 是一個不配置 GPU 的 head pod，
加上八個各配置一張 GPU 的 Ray worker pods。八張 GPU 是同一個 resource pool，不是
八個固定 serving instances。正式 trace 在 model deployment 前先將 pool-slot 6、7 標為
不可用，因此初始可服務容量是 6 GPU；其中兩個獨立 EP2 serving deployments 使用 4 GPU，
保留 2 GPU 作 make-before-break reconfiguration/migration headroom。Workload 使用彼此
不同的 8192-token prompt、產生 2048 tokens、關閉 prefix caching，並根據離線 profile
校正 anchor request 的抵達時間，使固定的 preemption notice 發生在約 512 個 generated
tokens（允許範圍 384--640）。

八個 worker pods 在整個實驗期間都已由 Kubernetes/Ray 配置完成；trace 的 `add` 與
`preempt` 是 scheduler 可用性事件，不是量測 Kubernetes autoscaler 或 pod 啟停時間。
因此結果必須稱為同一個 8-GPU pool 上的 trace-driven logical availability experiment。
若要宣稱真實 cloud instance add/remove，還需平台 adapter 將相同事件接到 pod lifecycle。

固定 trace 位於
[`k8s_qwen15_moe_a27b_8gpu_trace_plan.json`](k8s_qwen15_moe_a27b_8gpu_trace_plan.json)：

| Trace time | 事件 | deadline 後可用 GPU | 實驗語意 |
| ---: | --- | ---: | --- |
| 0 s | 8 個 workers 中先開放 slots 0--5 | 6 | 兩個 EP2 sources + 2-GPU migration headroom |
| 570 s | slot 3 收到 preemption notice | 6 | grace 期間 source 尚活著，開始 freeze/export/transfer/attach/ack |
| 600 s | 30 秒 grace 到期 | 5 | slot 3 正式不可用 |
| 660 s | add slot 6 | 6 | 新容量加入並觸發一次 runtime replanning |
| 780 s | add slot 7 | 7 | 再加入一張 GPU 並重新規劃 |
| 900 s | add slot 3 | 8 | pool 回到上限 |
| 1170 s | slots 3、6 同時收到 notice | 8 | 測試較高容量下的雙 GPU churn |
| 1200 s | 第二次 grace 到期 | 6 | trace 最後保持 6 張可用 GPU |

這條路徑是 `6 → 5 → 6 → 7 → 8 → 6`，同時包含縮容、逐步擴容、回復到滿載以及
多 GPU 回收，比單純 `8 → 7 → 8` 多出可觀察的 planner 選擇點。每個 event 的容量單位
都是一張 GPU；`pool-slot-N` 會在正式 run 前解析成實際 Ray worker ID。

名稱中的 `A2.7B` 是 active-parameter scale，不可當成 dense 2.7B checkpoint 的顯存需求；
EP2/EP3/EP4 都必須用正式 GPU 與相同 model snapshot 個別通過 peak-memory/profile gate。

Preemption 後的 planner 候選為 EP2、EP3 與 EP4。候選是以 TP/PP/DP/EP
平行形狀區分；只更換實體 GPU ID 不會被視為不同候選。

未來 K12 應使用另一份 config、profile、trace 與 scenario manifest；只需把 worker pool
擴成十二個單 GPU pods，不應把 worker 數硬編碼成十二個 serving instances，也不能直接
重用 K8 的 throughput/profile 數字。

## 正式宣稱與目前 readiness

這份 runner 是 formal pipeline 骨架，但目前仍有下列 blockers。在全部通過前，輸出只能
標為 K8 pilot / logical-preemption evidence，不能標為完整 SpotServe stateful recovery：

1. worker-subset allocation、distributed member tracking、vLLM reserved pod-IP placement、
   worker-ID trace 與獨立 grace deadline 已通過 CPU 測試，仍需實機 integration gate。
2. 跨 pods KV restore capability 與 persistent context ownership 尚未通過。vLLM engine
   restart 後的 KV/weight buffers 不可被假設仍存在；沒有 actual restored blocks 就必須
   降級命名為 `SpotServe-planner` 或 live make-before-break migration。
3. KM device mapping、逐層搬運、token-budget JIT 與 transfer ACK 尚未完成；global engine
   recreation 仍可能中斷 unaffected instance。不能稱為完整原版 SpotServe。
4. 每次呼叫 runner 都必須提供 `--repeats R`；F1 四組加 F2 四組，共執行 `8 × R`
   個 formal runs。`R=1` 只能作 exploratory evidence；`R>=2` 才能估計 sample variance
   與 paired Student-t confidence interval。

請先閱讀 [implementation status](../../../docs/spotserve-k8s-implementation-status.md)，
不要直接以 `--phase all` 的骨架存在就認定 stateful recovery 可在 9 pods 上成立。

KubeRay/RayCluster manifest 由實驗環境端建立並維護，因此不列為 repository blocker。
正式環境仍須由外部 K8s deployment 提供一個無 GPU Ray head、八個單 GPU workers、相同
image、共享唯讀模型與 persistent result volume；runner 的硬體盤點 gate 仍必須通過。

## EP 比較規則

Original SpotServe 與 MoE-aware SpotServe **兩組都要使用相同 EP 設定與 EP2/EP3/EP4
candidate set**。Original 組只關閉 expert placement、route histogram、A2A/expert movement
等 MoE-specific planner/migration costs；MoE-aware 組再打開這些 inputs/weights。若只讓
MoE-aware 組開 EP，量到的是「有無 EP + 有無 MoE optimization」的混合效果，不是公平的
MoE optimization 效果。

EP 並非 generic KV migration/reparallelization 的必要條件；但 expert locality、expert
remapping 與 MoE all-to-all 是本研究要比較的機制，因此 F1 的 Original/MoE-aware pair
以及 F2 四格 ablation 都應維持 EP enabled。

## Worker 必要 metadata

head 必須宣告 `control_node: 1` 且沒有 GPU；每個 GPU Ray worker 必須宣告
`worker_node: 1` 與唯一的 `worker_id_<id>: 1`，並提供一張 GPU。
同一 Kubernetes node 上的不同 pods 也要有不同 Ray pod IP（不使用共用 hostNetwork IP）。
`spotserve_physical_host_<hash>` 為建議的實體拓撲 metadata，不以八個 pods 推論八台主機。
請在每個 K8s worker pod 設定穩定的
實體主機識別資訊，例如：

```yaml
env:
  - name: WORKER_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SPOTSERVE_PHYSICAL_HOST_ID
    valueFrom:
      fieldRef:
        fieldPath: spec.nodeName
  - name: SPOTSERVE_FAILURE_DOMAIN_ID
    valueFrom:
      fieldRef:
        fieldPath: spec.nodeName
  - name: IMAGE_DIGEST
    value: "<head 與所有 worker 共用的 immutable image digest>"
```

所有 worker 必須使用相同的 immutable image digest 與相同的唯讀 model
snapshot。請在 head pod 與每個 GPU worker 中，透過 `IMAGE_DIGEST` 匯出相同的
digest。

預設設定會拒絕混用不同 GPU 型號、GPU 記憶體容量、driver、Torch/CUDA 或
vLLM 版本，避免這些差異干擾 Original SpotServe 與 MoE-aware SpotServe 的比較。

## 執行方式

第一次在 9 pods 上執行，請先照 [smoke 驗證文件](../../../docs/spotserve-k8s-smoke-verification.md)
跑 `--phase smoke`；不要直接啟動完整矩陣。smoke 不執行離線 profile 或 preemption，
也不代表跨 pod KV restore 已通過。

首先在不連接 cluster 的情況下檢查靜態設定：

```bash
python scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --output-dir /results/qwen-moe-f1-f2 \
  --dry-run
```

接著在 head pod 執行完整的 gated pipeline：

```bash
python -u scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 3 \
  --output-dir /results/qwen-moe-f1-f2 \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --ray-address auto \
  --ray-namespace sllm
```

若從遠端工作站操作，請透過 `kubectl exec` 在 head pod 內執行相同指令；
不要在每個 GPU pod 各自啟動一個 benchmark driver。

實驗會依照下列固定順序執行：

1. 盤點恰好八張閒置 GPU，並檢查實體主機、軟體版本、model architecture、
   context limit、expert 數能否被 EP 整除、GPU 記憶體可行性，以及各節點的
   checkpoint SHA-256。
2. 量測 pod-to-pod TCP 頻寬。此數值只作為 cost model 的 proxy，不會被宣稱為
   NIXL/GPUDirect 的實際傳輸效能。
3. 重用輸入條件完全相符的離線 profile；若缺少某個 EP shape，則執行兩點式
   decode 校正。每一份 profile 都必須從 runtime 讀回完整的 EP rank。
4. 保持 trace 的 preemption notice 固定在 570 與 1170 秒；根據初始 EP2 profile 推導
   兩個 anchor requests 的抵達時間，並要求 notice 當下的 runtime token snapshot 落在
   384--640 generated tokens。
5. 執行配對的 Original/MoE-aware pilot。若 pilot 的正規化 planner 決策沒有
   分歧，仍會保留為有效證據，不會因此中止 formal runs。
6. 能力驗證通過後，執行 F1（4 treatments × `R` runs）與 F2（MoE replan × MoE
   migration 的 4 cells × `R` runs）；`R` 來自必填的 `--repeats`，KV recovery policy
   在 F2 四格保持相同。
7. 產生 `/results/qwen-moe-f1-f2/results.md` 與 JSON evidence ledger。

Runner 不會默默重試無效的 GPU run。請先修正記錄下來的 blocker，再重新執行
相同指令；ledger 中已通過驗證的 runs 會直接沿用。若只要重新產生 Markdown
報告而不連接 cluster，可執行：

```bash
python scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 3 \
  --output-dir /results/qwen-moe-f1-f2 \
  --phase report
```

第一次從零開始執行時，最多會包含 12 個 profile workloads（3 個 shapes ×
2 種輸出長度 × 2 次 repeats）、2 個 pilots，以及 `8 × R` 個 formal runs。由於每次
都可能需要載入 model，且 ready/request timeout 設為 1800 秒，整體會是長時間
工作。請使用 persistent volume 保存結果，讓工作中斷後能根據 ledger 繼續執行。

## 結果判讀

只有在正規化後的配對 formal evidence 顯示 parallel shape、source→target migration 或
expert→rank placement 發生差異，且每組 paired repeat 的 P95 latency 或 throughput 朝
預期方向變化時，`results.md` 才會回報 `supported`。`R=1` 時這只表示單次機制證據符合
預先方向，不等於統計上成立，SD 與 CI 會標為 unavailable；`R>=2` 時報告 sample SD 與
依實際 R 計算的 paired 95% Student-t CI。

若不符合上述條件，報告會記錄 `not_supported`。程式不會在看過 formal 結果後，
再回頭調整 workload、preemption 時機或 planner candidates。
