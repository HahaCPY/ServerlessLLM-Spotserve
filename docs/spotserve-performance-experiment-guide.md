# SpotServe MoE Performance Experiment Plan And Runbook

這是 SpotServe x MoE performance evaluation 的正式實驗契約，整合研究問題、
claim boundary、preliminary plan、正式 matrix、執行命令、success gates 與單次 pilot
判讀規範。目標是在同一個 Kubernetes Ray cluster、相同模型、workload 與 preemption
trace 下，比較 SpotServe-style baseline 與本專案的 MoE-aware 實作。

## Claim Boundary

目前正式 K8s testbed 的邏輯拓撲是：

```text
1 x head pod (Ray head + ServerlessLLM controller/router + benchmark driver; 0 GPU)
└── 8 x Ray worker pods
    └── 1 GPU / pod, 1 stable worker ID / pod
```

`8` 是目前一次 formal run 可發現的 GPU pool 上限，不是八個固定 serving instances。
planner 必須從所有 `READY` worker pods 的 GPU 建立同一個 resource pool，並為一個或多個
deployment 動態選擇合法形狀。初始 serving shape 不必占滿八張 GPU；目前 K8 config 使用
兩個不重疊的 EP2 deployments（共 4 GPUs）。正式 trace 初始開放 6 GPUs，因此保留兩張
GPU 作 make-before-break reconfiguration/migration headroom。未來 K12 是
相同契約下的獨立 scenario：一個 head pod 加十二個單 GPU worker pods，使用自己的
cluster manifest、profile、trace 與 frozen scenario manifest，不能把 K8 profile 直接當成
K12 profile。

K8/K12 的 `worker pod`、Ray `node` 與實體 Kubernetes host 不一定一對一。正式紀錄必須同時
保存 pod UID、Ray node ID、GPU UUID、`spec.nodeName` 與 failure domain。若多個 worker pods
位於同一台實體主機，cloud/provider preemption 應以該實體 failure domain 為失效單位，
不可宣稱為八台或十二台獨立 machines。

因此可以做的公平比較是：

```text
SpotServe-style baseline
vs.
MoE-aware context migration / recovery / re-parallelization
```

兩者必須使用相同硬體、模型、GPU 數量、workload、trace 與 benchmark 設定。

可以使用的結論：

> On a Kubernetes Ray testbed with one non-GPU head pod and eight one-GPU
> worker pods, the MoE-aware extension is compared against a SpotServe-style
> baseline under identical trace-driven logical-preemption workloads.

在尚未通過實體 pod eviction 與 provider failure-domain gate 前，必須寫
`logical preemption replay`，不能寫成真實 cloud spot interruption。不可直接將本專案的
latency、throughput 或加速比與 SpotServe 論文在 AWS instances 上公布的絕對數字比較；
pod 間網路、每 pod 一張 GPU 與論文每 instance 多 GPU 的拓撲也必須在報告揭露。

### Current implementation boundary

2026-10-05 的控制面修正已加入 worker-subset allocation、全部 rank 的 member bookkeeping、
vLLM Ray-DP 的 reserved pod-IP constraints、最新 scheduler snapshot、序列化重配置、
切換前 target READY 重檢，以及不因 migration handler 變慢而延長的 grace deadline。
正式 trace 指定 worker ID；deadline 到期撤銷 source engine，不刪除 Kubernetes pod。
共同初始 topology 與 F1/F2 主矩陣也已對齊。這些僅通過 CPU/mocked-runtime 測試。

formal config 啟用 `selection_policy=spotserve_algorithm1`：在當下可配置的 GPU pool 內，
有候選滿足 throughput 時選最低 profile latency，否則選最高 throughput；缺少 profile
不會猜測吞吐量。`queue_model=none` 使用固定 workload 的實測 latency；可明確選用
`md1_aggregate` 近似，但不能聲稱它適用於 bursty continuous batching。原有 weighted
policy 保留供非 paper-mode callers 使用。這不是原版完整 cloud acquisition optimizer。

**尚未補齊**：GPU allocation owner 仍是 engine；persistent context daemon、GPU→topology
KM device mapping、逐層權重搬運、runtime token-budget JIT interruption、跨 pods KV restore
及 transfer-ACK 驅動的 source lease release 仍需實作或正式硬體驗證。目前 whole-deployment
actor recreation 也尚未改成保留 unaffected engines 的增量交易。不能將 token replay、
logical placement 或控制面 deadline 修正稱為以上機制已完成。

完整差異、最小驗證與後續順序見 [K8s implementation status](spotserve-k8s-implementation-status.md)。

KubeRay/RayCluster manifest 由實驗環境端建立並維護，不是 repository 的演算法或 runtime
blocker。正式 runner 仍會 fail closed 驗證實際連入的是一個無 GPU head pod、八個單 GPU
worker pods、共享模型與 persistent results volume；不能只因 manifest 存在就略過 P0/P1
硬體盤點與 GPU identity gate。

## Experiment Groups

目前已存在的 mechanism/performance gates 至少包含以下四組：

| Group | Matrix | 目的 |
| --- | --- | --- |
| E1 | `benchmark_matrix_reparallelization_performance.yaml` | V6 re-parallelization baseline vs applied |
| E2 | `benchmark_matrix_context_migration_performance.yaml` | V7 migration baseline vs MoE-aware target selection |
| E3 | `benchmark_matrix_stateful_recovery_performance.yaml` | V8 token replay vs true state/KV restore |
| E4 | `benchmark_matrix_spotserve_core_performance.yaml` | baseline vs combined migration/recovery flow |

另外執行下列機制實驗：

| Group | Matrix | 目的 |
| --- | --- | --- |
| M1 | `benchmark_matrix_spotserve_core_trace_sweep.yaml` | 多種 preemption/recovery trace 的穩定性 |
| M2 | `benchmark_matrix_expert_remap_a2a_reduction_performance.yaml` | 相同 workload remap 前後的 A2A payload reduction |
| M3 | `benchmark_matrix_elastic_ep_resize_performance.yaml` | 同 actor、active-request drain 的 EP2 -> EP4 resize |
| M4 | `benchmark_matrix_simulated_cross_host_expert_remap_performance.yaml` | legacy 單機 2+2 GPU failure-domain simulation |

E1-E4 與 M1-M4 用來確認個別機制、runtime hooks、trace replay 與 metrics 是否可信。
它們是 legacy/local mechanism smoke，不是下列 K8 formal comparison 或 ablation 的替代品；
不得把不同 testbed、matrix、workload 或 trace 的數值拼成正式比較表。

## Formal Evaluation Design

正式 evaluation 分成兩個實驗。每個 `configuration` 是以下欄位的完整組合：

```text
experiment + scenario + strategy/ablation flags + model + workload
+ context length + output length + concurrency + arrival pattern
+ initial GPU configuration + preempted GPU + preemption condition
+ runtime flags + random seed
```

只要其中一個欄位不同，就視為不同 configuration。每次執行必須由使用者提供
`--repeats R`，同一個 F1/F2 configuration 會各自執行 R 次；不能由 checked-in config
默默決定次數。`R=1` 不得宣稱統計顯著或提供信賴區間。

### Common model and runtime contract

F1 與 F2 固定使用同一份本機 model snapshot：

| Item | Formal setting |
| --- | --- |
| Model | `Qwen1.5-MoE-A2.7B-Chat` (`Qwen2MoeForCausalLM`) |
| Shared pod path | `/models/Qwen1.5-MoE-A2.7B-Chat` |
| Architecture contract | 從 frozen `config.json` 讀取並記錄；EP degree 必須可被 expert 數整除 |
| Precision | `bfloat16` |
| Load format | `auto` |
| Maximum sequence length | 16384 tokens |
| Sampling | `temperature=0`, `ignore_eos=true` |
| Formal seed | `20261002` |
| Prefix caching | disabled for every strategy |
| Server sequence capacity | `max_num_seqs=16` |
| Formal request | exact 8192 input tokens + exact 2048 output tokens |

`A2.7B` 表示每 token 的 active parameter scale，不代表 checkpoint 只需容納 2.7B dense
weights。正式部署仍須以實際 snapshot 大小、每 rank 載入方式與實測 GPU peak memory 做
EP2/EP3/EP4 feasibility gate，不能只用名稱中的 2.7B 推算顯存。

正式開始前保存 `config.json`、tokenizer files 與 weight files 的 manifest/SHA256；之後
不得只用 model directory 名稱判斷模型相同。input length 指套用 chat template 後送進
模型的實際 token 數，不是字元數或空白切詞數。workload generator 必須對每個 request
重新 tokenize 並 assert 實際 input tokens 等於 scenario 設定，且
`input_tokens + max_output_tokens <= 16384`。所有 head/worker pods 必須 mount 同一份唯讀
snapshot，並驗證 checkpoint SHA-256 與 immutable image digest 一致。

每個 scenario 先執行 4 個不納入統計的 warm-up requests，再執行 24 個 measured
requests。所有 strategy 使用 byte-identical workload 與相同 seed；output 必須驗證實際
generated token count。若 backend 不接受 `ignore_eos` 或無法產生指定長度，必須在
preliminary 階段解決，不能在不同 strategy 使用不同 output 長度。

### Dynamic planning contract

Formal manifest 只能固定 model、合法 candidate set、planner weights、安全限制、失效
GPU/failure domain 與 preemption condition。不得在 run 前固定以下 planner outputs：

```text
selected target nodes
selected TP / PP / DP / EP
selected replica count
selected migration target
selected expert placement / placement fingerprint
```

每次 `preempt` event 必須先把受影響 instance/node 標成 `PREEMPTING`，再由 planner 讀取
事件當下的狀態並重新產生 candidates 與 decision：

```text
ready / preempting / dead nodes and currently available GPUs
active and in-flight requests
recent request arrival rate and observed latency
queue depth / queue pressure
KV/context reuse state
runtime MoE top-k routing histogram
current expert placement and runtime capability metadata
```

static candidate constraints 不等於 static plan；planner 可以在合法集合內再次選到和上一
次相同的 configuration，但必須留下新的 input snapshot、candidate scores 與選擇理由。
`execution_status=unchanged` 只有在確實重新規劃後才有效。

目前程式的 `handle_preemption()` 已在 event 內呼叫 `_replan_after_spot_event()`，並重新
取得 worker-node、runtime metadata 與 recent-workload snapshot；但現有 aggregate
summary 尚不足以證明每個 preemption 都各自觸發一次 planner。正式 F1/F2 前必須在
trace/router metrics 加入並保存：

```text
preemption_event_id / trace_event_index
planner_invocation_id
planner_started_at / planner_finished_at
planner_input_snapshot or deterministic snapshot hash
available GPU and worker-state snapshot
active-request / queue / KV / route / placement snapshot availability
candidate_count and top candidate component costs
selected plan and execution status
```

需要 dynamic planner 的 configuration，必須逐 event 通過：

```text
number of planner invocations = number of preemption events
planner event_id = triggering preemption event_id
planner_started_at >= instance marked PREEMPTING timestamp
planner input availability excludes the failed capacity
candidate_count > 0 when legal capacity remains
selected target does not contain preempting/dead nodes
selected plan came from the recorded runtime candidate set
execution_status in {applied, unchanged}
no precomputed target plan exists in matrix/config/trace
```

只檢查 `replanning_events >= 1` 不足以通過。任何一個需要 planner 的 preemption 缺少
對應 decision，整個 run 即 invalid。Rerouting 本來就不使用 topology planner，不套用
invocation-count gate；它仍須證明 preemption 確實發生。F1 runner 不包含 `No Recovery`；
失敗請求仍須保留在每個 treatment 的結果中，不得以排除 failed runs 的方式改善指標。

### F1: Four-strategy comparison

每個正式 scenario 比較下列四種 treatment。表中的「動態」表示只能固定 candidate
constraints，target plan 必須在 preemption event 後才產生。

| Strategy | Request/recovery behavior | Topology planner | Migration planner | MoE-aware inputs |
| --- | --- | --- | --- | --- |
| Rerouting | 新 request 與允許 retry 的 request 送到 remaining ready replica；不搬 KV/state | OFF | OFF | OFF |
| Reparallelization | token replay；每次 preemption 後動態選新 parallel plan | Dynamic, non-MoE cost | OFF | OFF |
| Original SpotServe | capability-gated shared recovery mode、standard context/KV cost、動態 non-MoE topology/migration planning | Dynamic, non-MoE cost | Dynamic, KV/queue cost | OFF |
| MoE-aware SpotServe | 相同 recovery mode、動態 topology/migration、expert placement/locality cost | Dynamic, MoE-aware cost | Dynamic, KV/queue/expert cost | ON |

Rerouting 可以 retry，但必須從頭重算並證明
`KV reused = 0`。Reparallelization、Original SpotServe 與 MoE-aware SpotServe 都必須
通過 per-preemption dynamic-planning gate。Original 與 MoE-aware 的合法 capacity、
candidate shapes 與 non-MoE weights相同；差異只能來自表中明列的 MoE information 與
planner/migration cost。Original 與 MoE-aware pair 的 KV restore、token replay、retry、
transition order 與 source lifetime 必須相同；若 capability gate 不支援 true restore，兩組
都降級為 token replay 並改稱 planner comparison，不能只讓其中一組使用 stateful restore。

因此 F1 是 `4 strategies × R valid runs = 4R runs`。R 由必填 CLI 參數決定；R=1
只作 exploratory comparison，不宣稱統計顯著。

#### EP fairness rule

Original SpotServe 與 MoE-aware SpotServe 都必須使用同一個 MoE model、同一組
EP-enabled candidate shapes、相同 vLLM `enable_expert_parallel` 與相同 serving capacity。
基準組可以使用 EP 執行，但 topology/migration ranking 不得讀取 expert placement、top-k
route histogram、A2A dispatch 或 expert movement cost；MoE-aware 組則只打開這些
MoE-specific inputs/weights。若只在 MoE-aware 組開 EP，結果會同時混入 parallelism mode
與 MoE optimization，無法把差異歸因於本專案的優化。

EP 不是一般 KV migration 或 generic reparallelization 的必要條件；但若研究問題是
expert locality、expert remapping 或 MoE A2A，則兩個比較組都需要 EP，否則 MoE-specific
optimization 幾乎沒有可作用的機制。

### F2: MoE optimization ablation

F2 只改變兩個 MoE optimization，其他 recovery semantics 必須完全相同。四組都啟用
dynamic topology planner、context migration、KV migration 與相同的 token-replay
recovery policy，避免把 true state restore 變成第三個 ablation factor。

| Configuration | MoE-aware reparallelization | MoE-aware migration | Required treatment |
| --- | --- | --- | --- |
| Original SpotServe | OFF | OFF | dynamic base planner；`disable_logical_expert_placement=true`；migration 不使用 expert locality |
| Reparallelization Only | ON | OFF | dynamic expert-placement/movement-aware planner；migration 不使用 expert locality |
| Migration Only | OFF | ON | dynamic base planner；migration 使用 runtime top-k、expert locality/dispatch cost |
| Full MoE-aware SpotServe | ON | ON | dynamic expert-aware topology 與 expert-aware migration |

所有四組固定 `enable_reparallelization=true`、`enable_workload_cost_model=true`、
`enable_context_migration=true`、`enable_kv_cache_migration=true`。Migration optimization
OFF 時固定 `enable_moe_expert_locality=false` 且 `expert_dispatch_weight=0`；ON 時使用同一
個預先凍結的 positive weight。Reparallelization optimization OFF 時禁止 logical expert
placement/live remap cost 影響 ranking；ON 時啟用 runtime placement hooks 與相同的 expert
movement cost weights。正式 config 完成後需輸出 resolved flags diff，證明每個 pair 只差
預定的 treatment。

F2 四組全部必須對每次 preemption 動態執行 planner；OFF 代表不使用 MoE-specific cost，
不是事先固定 plan，也不是關閉 planner。

因此目前唯一的正式 scenario 是 `4 configurations × R valid runs = 4R runs`。F1 與
F2 合計 `8R` 個 configuration-runs；smoke、warm-up、preliminary 與 invalid replacement
不計入這 8R 次。F1 runner 已移除 `No Recovery` 並對齊四組
主比較；F2 主 runner 也已改為四格。四個 KV replay variants 保留為可呼叫的 diagnostic
flags，但不列入預設 matrix，也不能混入 `8R`-run 主分析。

### Primary formal scenario: combined stress

目前只執行一個最可能讓兩種 planner 產生不同 decision 的 combined-stress scenario。
選擇依據是在同一次 preemption 同時啟動 KV、decode、queue 與 MoE placement/routing cost，
不是先看 performance 結果再挑對 MoE-aware 有利的 workload。

K8 scenario 使用一個 head pod 與八個單 GPU worker pods。正式共同初始 topology 是兩個
獨立 EP2 serving deployments（共使用 4 GPUs）；trace 初始可用容量為 6 GPUs，另保留
2 GPUs 為 make-before-break headroom。trace 鎖定
其中一個 EP2 deployment 的非 primary member worker，邏輯 pool transition 為 8 -> 7；
另一個 EP2 必須維持 `READY` 供 Rerouting 使用。需要 planner 的 treatments 再從當下七個
`READY` GPUs 動態選 EP2/EP3/EP4 或記錄無合法 plan。失效 worker
是 controlled input，但替代 worker、parallel shape、migration source/target 與 expert
placement 都不是預先固定的 planner outputs。JSON 已改為雙 EP2；worker-subset allocation
已有 CPU 驗證，正式硬體上的 rank placement 與獨立失效仍須 preliminary gate 驗證。

K12 不和 K8 混在同一 configuration。未來 K12 scenario 使用一個 head pod、十二個單 GPU
worker pods、K12 專用 add/remove trace 與實機 profile；只有 profile 和 trace 都凍結後，
才能把它加入 sensitivity study。K8 與 K12 的絕對 request rate 不直接互相比較，除非使用
同一固定 arrival trace；若各自按 reference capacity 正規化，只能比較 normalized load 下
的 treatment effect。

下表是 preregistered candidate。必須先通過 preliminary feasibility/timing gates；一旦
通過便原值凍結，不能根據正式 performance 結果修改。

| ID | Purpose | Input tokens | Output tokens | Client concurrency / arrivals | Preemption phase | GPU transition |
| --- | --- | ---: | ---: | --- | --- | --- |
| K8-CS | KV + decode + queue + MoE combined stress | 8192 | 2048 | `max_num_seqs=16`；24 requests；6 trace-aligned bursts x 4 | middle：anchor 已生成 384--640/2048 tokens，target 512，且 queue depth > 0 | pool 6 -> 5 -> 6 -> 7 -> 8 -> 6；最多 8 GPU |

選用 `8192 + 2048 <= 16384` 是為了建立可量測的 KV state，並讓 request 橫跨分鐘級
add/preempt/replanning 階段。六個 bursts、實際 arrival timestamps 與所有 request bytes
必須由 frozen workload 產生，
不能依 treatment 調整。middle preemption 讓系統在已有可重用 state、但仍有足夠剩餘工作
時做 migration/reparallelization decision。

Preemption phase 以 anchor request 的 generated-token progress 定義。trace replayer
使用 wall-clock time，因此 timing preliminary 先量測到 512 generated tokens 的時間，
再反推 anchor request 的固定抵達時間；570 與 1170 秒的 resource-event time 不移動。
resolved arrival trace 凍結後由所有 strategy 共用。正式 run 同時記錄實際 token progress；若偏離目標
超出 384--640 tokens、anchor 已完成，或 preemption 時 queue depth 為 0，該
run invalid。不能替每個 strategy 個別調整 preemption time。

所有 strategy 的 initial placement、target worker/failure domain、workload bytes、request IDs、arrival
timestamps、seed 與 trace 完全相同。

正式 trace plan 位於
`benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu_trace_plan.json`。它以 GPU 為容量
單位；runner 會產生含實際 worker ID 的 `generated/formal-trace.jsonl`。capacity event
時間固定，profile 只校準 anchor request 的 arrival time：

```text
anchor_arrival_s = fixed_preemption_notice_s
                 - estimated_time_to_512_tokens_s
preemption_deadline_s = fixed_preemption_notice_s + 30
```

若第一個 anchor 必須在正式 workload 開始前發送，runner 會停止；不會移動 trace event，
也不會為個別 treatment 改時間。8 張 Ray GPU workers 實際都已配置，但 runner 在部署前
把 slot 6、7 標成 scheduler-unavailable，所以服務容量從 6 GPU 開始；這不包含真實 pod
provisioning latency。

light-load、KV-only、decode-only 與 early/late preemption 暫時延後為 sensitivity
experiments，不納入目前 `8R` 個正式 runs。只有 primary scenario 完成且需要解釋結果時，
才另立新版 preregistered plan 執行，不能事後挑其中表現最好者補進主表。

目前 matrix 是 F1 4 組、F2 4 組，worker IDs 與 distributed member nodes 已有明確語意。
正式執行前仍須逐項證明 controlled variables、實際 rank placement 與 KV capability；
CPU 測試不能替代 K8s GPU integration gate。

## Preliminary And Scenario Freeze

Preliminary 不納入 F1/F2 主矩陣，也不作正式統計。依下列固定順序執行：

| Gate | Minimal experiment | Pass condition |
| --- | --- | --- |
| P0 Model/token gate | 在每個 pod 驗證 model；送 1 request | 相同 SHA-256；exact 8192 input / 2048 output；總長不超過 16384；無 OOM |
| P1 Pool/topology gate | 發現 8 workers；部署前將 2 workers 設為 unavailable；建立兩個不重疊的 EP2 deployments；鎖定並 preempt slot 3 | planner 依序看到 6、5、6、7、8、6 READY GPUs；受影響 deployment 被標記；candidate set 非空 |
| P2 Rerouting gate | 使用和所有 treatments 相同的雙 EP2 initial topology | unaffected EP2 繼續服務；無 KV reuse；沒有停止所有 deployments |
| P3 Load gate | K8-CS 跑 1 次、無 preemption | bursts 形成預定 concurrency/queue；24 measured requests 全完成 |
| P4 Timing gate | K8-CS 做 middle-preemption calibration | resource-event time 固定；所有 treatment 共用校準後的 anchor arrival，使 progress 落在 384--640 tokens |
| P5 Dynamic-planner gate | 單 preemption 與 double-preemption 各 1 次 | 每個 event 都有新 snapshot/ranking/decision；失效 worker 不在 target |
| P6 Capability gate | cross-node KV/state restore 與 greedy token equality | capability/pass evidence 明確；失敗時降級 treatment 名稱，不得宣稱 stateful recovery |
| P7 Treatment smoke | F1 四組與 F2 四組各跑 1 次 smoke | resolved flags 符合 treatment；兩組都使用相同 EP；workload/trace hash 相同 |

P3 只校準 workload 是否形成預定壓力，不用 Original/MoE latency 差異挑 scenario。P4
只校準 phase，不用哪個版本比較快決定 timing。P5 可以觀察 planner 是否選不同 plan，
但 primary scenario 的去留由 feasibility 與 mechanism coverage 決定，不由 improvement
方向決定。

若 candidate 值因 VRAM/runtime 限制失敗，只能按以下預先規則做最小降階，並重新通過
P0-P7：

```text
K8-CS input: 8192 -> 4096 -> 2048
K8-CS output: 2048 -> 1536 -> 1024
K8-CS burst size: 4 -> 2
```

不得任意換成只對 MoE-aware 有利的值。若 member-specific 8 -> 7 logical preemption、
cross-node restore 或 rerouting topology 不受目前 runtime 支援，停止相應的正式 claim 並
記錄 blocker；不要把 deployment-level stop 冒充 worker-level preemption。

P0-P7 全部通過後產生 frozen scenario manifest。manifest 至少記錄 model/file hashes、
所有 configuration 欄位、resolved runtime flags、legal candidate set、planner weights、
workload/trace/config SHA256、calibrated trace time、seed、commit 與 image ID。正式 run
中途不得因結果不理想而更換 scenario；任何變更建立新版本，舊結果只保留為 pilot。

## Required Formal Artifacts And Execution Order

P0-P7 完成時必須建立並 review 下列 artifacts；目前既有 E1-E4 matrices 不能取代：

```text
benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json
benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu_trace_plan.json
benchmarks/spotserve/formal/scenario_manifest.json
benchmarks/spotserve/formal/workloads/k8_combined_stress.jsonl
<results-volume>/generated/formal-trace.jsonl
<results-volume>/generated/configs/<treatment>.json
<results-volume>/generated/profiles/<shape>.json
<results-volume>/evidence-ledger.json
```

執行順序固定為：

1. 記錄 repository、image、model hashes 與 GPU 環境。
2. 在 head pod 啟動 runner；完整實驗前只執行一次與 frozen environment 相符的 profile。
3. 執行 P0-P7，只做 feasibility、load/timing calibration 與 treatment smoke。
4. Review 並 commit frozen scenario manifest、workloads、traces、resolved configs 與 hashes。
5. 執行 F1：K8-CS 的四個 strategies，每個 configuration 執行 R 次，共 4R valid runs。
6. 執行 F2：K8-CS 的四個 ablations，每個 configuration 執行 R 次，共 4R valid runs。
7. 每個 run 立即驗證 trace、preemption phase、dynamic-planner audit、success gates，
   備份後先把該次數據貼到目前對話框，才可開始下一個 run。
8. 完成 8R 個 valid runs 後彙整 raw difference、improvement 與因果證據；R=1 不計算
   CI，R>=2 計算 paired Student-t CI。

任何正式 run 開始後才修改 workload、trace、planner weights 或 candidate set，都必須停止
該版本 evaluation、遞增 manifest version，並從受影響的 configuration 重新執行一次。

## Metric Availability Gate

目前 `run_benchmark.py` 與 analyzer 已直接提供：

- request mean latency、P50、P95、P99 與 requests/s；
- raw request 的 send/completion timestamp、success 與完整 response；
- phase-specific latency/throughput；
- replan execution duration、placement 與 moved bytes/shards；
- context migration plan、KV reuse、queue cost、expert locality/routing counters；
- state/KV restore、fallback 與 restored-block counters。

下列正式欄位目前不能由主要 non-streaming benchmark 直接、嚴格地取得：

- TTFT、TPOT/inter-token latency、prefill latency、decode latency；
- tokens/s；
- scenario makespan/total experiment time 的直接 summary field（可由 raw timestamps 推導）；
- 精確的 preemption-to-service-restored/downtime；
- 統一定義的 context migration wall-clock duration；
- 可將每個 preemption、planner invocation、input snapshot 與 selected plan 一對一連結的 event ID；
- topology planner 所需的完整 active-request、queue、KV、runtime route 與 placement snapshot。

正式 F1/F2 開始前，必須為這些必填欄位補上 timestamp/streaming instrumentation，
或在 preregistered schema 中標成 unavailable 並從主要 claim 移除。不得用完整 response
latency 冒充 TTFT，也不得用 planner estimated cost 冒充實際 migration/recovery time。

## Formal K8s preflight

在 head pod 的 repository 根目錄記錄程式版本、時間、image/model digest、Ray nodes 與 GPU
環境。下列命令只盤點資源，不會啟動 profile 或 formal runs：

```bash
git status --short
git rev-parse HEAD
date --iso-8601=seconds
nvidia-smi
test -f /models/Qwen1.5-MoE-A2.7B-Chat/config.json && echo MODEL_OK
python scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --output-dir /results/qwen-moe-f1-f2 \
  --dry-run
```

將完整輸出存進實驗紀錄。所有正式 run 必須使用相同 commit 與 image；working tree
狀態必須記錄，模型 `config.json` 必須存在。每個 worker pod 應提供穩定 `WORKER_ID`、
`SPOTSERVE_PHYSICAL_HOST_ID`、`SPOTSERVE_FAILURE_DOMAIN_ID` 與相同 `IMAGE_DIGEST`。

確認八個 worker pods 各暴露一張未被其他 workload 使用的 GPU；head pod 不配置 GPU。
GPU 型號、顯存、driver、CUDA/Torch/vLLM、checkpoint hash 不一致時 fail closed。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv
```

不要終止其他使用者的程序。如果 GPU 不足，等待資源釋放後再跑。

## Legacy local mechanism runbook (not formal K8s evidence)

以下 Podman/Qwen2-MoE-Tiny/E1--M4 命令只保留供既有單機 mechanism smoke 使用。它們的
model、topology、failure semantics 與正式 Qwen1.5-MoE-A2.7B K8 study 不同，不得把結果
納入 F1/F2 主表，也不是目前 K8/K12 部署指南。

### Image And Source Setup

如果目前 image 已包含最新 vLLM patches，後續只需使用 `--skip-build`。
在全新機器或 vLLM patch 有變更時，第一次 prepare 必須省略 `--skip-build`，完成
一次 image rebuild。一般 Python/controller 修改可使用 `SPOTSERVE_SYNC_SOURCE=1`
同步到 container。

每次切換 deploy set 都要重新執行對應的 prepare command。prepare 可能重建
container，所以切換前先保存上一組結果。

## E1: Re-parallelization

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set reparallelization-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效 run 必須符合：

```text
trace_replay_success = 1
replanning_events >= 1
replanning_execution_applied >= 1
replanning_execution_failed = 0
```

主要報告 `success_rate`、`latency_p95_ms`、`throughput_req_s`、
`phase_replan_window_latency_p95_ms`、`phase_post_replan_latency_p95_ms` 與
`replanning_avg_execution_duration_ms`。

## E2: Context Migration

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set context-migration-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效的 MoE runtime run 應包含：

```text
context_migration_events >= 1
context_migration_plan_count >= 1
route_source = vllm_runtime_topk
route_kind = runtime_observed_topk
```

同時保存 planner 的 cost breakdown：

```text
KV migration cost
expert dispatch cost
queue penalty cost
hot expert locality ratio
estimated remote routing ratio/tokens
```

`route_source=request_instrumentation` 或 fixture 只能算 planner sanity check，不能算
真實 vLLM routing instrumentation 實驗。

## E3: Stateful Recovery

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT=auto \
SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION=1 \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set stateful-recovery-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效的 true KV restore run 必須符合：

```text
state_restore_attempts_total >= 1
state_restore_successes_total >= 1
true_kv_restore_successes_total >= 1
true_kv_restored_blocks_total > 0
supports_state_restore_requests >= 1
state_restore_fallback_count = 0
```

只看到 `recovered_tokens_total > 0` 不足以證明 true KV restore，因為 token replay
也可能產生 recovered token。

## E4 And M1: Combined Core

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_CORE_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_CORE_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set spotserve-core-performance
```

先跑主要比較：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

再跑 multi-trace sweep：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_trace_sweep.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 300 \
  --ray-address auto \
  --ray-namespace sllm
'
```

每個 baseline/candidate pair 都必須確認 `trace_replay_success=1`。如果 trace replay
失敗，即使 request success rate 是 100%，該 run 仍然無效。

## M2: Physical Remap And A2A Reduction

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
VLLM_SPOTSERVE_EXPERT_REMAP=1 \
VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
VLLM_SPOTSERVE_A2A_TRACE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set expert-remap-active-performance
```

執行相同 workload 的 remap 前後比較：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_expert_remap_a2a_reduction_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

只有同時符合下列條件才能宣稱 A2A traffic reduction：

```text
traffic_reduced = true
claim_supported = true
candidate observed_payload_bytes < baseline observed_payload_bytes
payload_reduction_ratio > 0
```

這裡量到的是 same-host sparse collective payload，不是 internode NIC traffic。

## M3: Elastic EP2 To EP4

這組測試獨占四張 GPU。先確認每張 GPU 都有至少 8192 MiB free memory。

準備環境：

```bash
MODEL_FOLDER=/work/spotserve-models \
SPOTSERVE_REPARALLELIZATION_MODEL_PATH=/models/Qwen2-MoE-Tiny \
SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT=auto \
SPOTSERVE_SYNC_SOURCE=1 \
scripts/prepare_spotserve.sh --skip-build \
  --deploy-set reparallelization-performance
```

執行：

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_elastic_ep_resize_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效 run 必須符合：

```text
before_worker_count = 2
after_worker_count = 4
runtime_verified_placement = true
actor_recreated = false
active_request_observed = true
active_request_completed = true
elastic_ep_admission_drained = true
changed_experts 非空
```

這證明 same-host live in-place EP2 -> EP4，不代表 cross-host EP resize。

## Repetitions

Runner 不再從 checked-in config 默認正式次數；每次命令都必須傳入 `--repeats R`，其中
R 是正整數。同一結果目錄會記住 R，續跑必須使用相同值；若要改變 R，請使用新的結果
目錄，避免混用不同實驗契約。R=1 時只報 raw value，SD 與 CI 標為 unavailable；R>=2
時計算每個 configuration 的 sample SD，以及 Original/MoE 配對差值的 95% Student-t CI。

先跑 smoke 並確認所有 success gates；smoke 不算正式 run。所有正式實驗中的每一個
configuration 固定獨立執行 R 次，取得 Run 1 到 Run R。baseline 與 candidate
必須保留在同一個 matrix/cycle 中，不要在不同環境分開執行。

同一時間只執行一個 benchmark。每個 run 完成後必須先檢查 success gate、立即回報並
備份結果，才能開始下一個 run；不要用未檢查中間結果的背景迴圈一次跑完全部輪次。每次
切換 deploy set 前，也必須先保存目前 container 中的結果。

各次之間必須重新建立相同 initial state：刪除並重新部署 model/worker、等待 cleanup
完成、清空上一輪 metrics、重設 scheduler/queue、確認 worker count 與 placement，執行
固定且不納入統計的 warm-up，再開始 measurement。禁止沿用上一輪的 KV cache、request
queue、actor state 或未完成 request。若 teardown/redeploy 後無法證明狀態乾淨，該次
不得開始或必須標成 invalid。

每次 measurement 前逐項確認：

```text
上一輪 requests 全部完成或已清除
上一輪 model/actors 已 delete，cleanup 已完成
model/workers 依相同 deploy config 重新建立
router metrics、scheduler state、queue 與 KV cache 已清空
worker count、GPU mapping、TP/EP 與 expert placement 符合 manifest
GPU memory/utilization 回到可接受的基準
固定 warm-up 已完成，warm-up requests 不納入正式 summary
trace 尚未提前 replay
```

正式 matrix 的 run config 應明確設定 `delete_models_before_run`、`delete_after_run` 與唯一
的 metrics/output path。現有部分 workload 含 `benchmark_phase=warmup`，但 analyzer 的
overall summary 仍可能包含這些 rows；建立 F1/F2 前必須把 warm-up 移出 measurement，
或明確修改 aggregation filter，不能讓初始化成本只污染某些 configuration。

每次執行前記錄：

```text
git commit
日期與 run index
GPU 型號與空閒記憶體
matrix/config 名稱
model path
任何非預設環境變數
random seed
workload / trace / config SHA256
initial and target GPU/TP/EP configuration
```

R 次必須使用完全相同的 model、workload、context/output length、concurrency、arrival
pattern、initial GPU configuration、preempted GPU、preemption timing/condition 與 runtime
flags。所有可控隨機來源固定 seed；無法固定者必須記錄實際 seed 與來源。
若 trace 使用 `instance_selector=busy/ready`，每次必須從 trace log 記錄實際解析出的
instance ID、physical GPU 與 preemption timestamp；只記 selector 不足以證明各次命中
相同 preempt target。

### Per-run and aggregate statistics

每次完成後立即保存並回報該 run，不等 R 次全部結束才檢查。下表以 R=3 示意；實際欄位
依使用者輸入的 R 展開：

| Metric | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| End-to-End Latency | | | |
| Mean Request Latency | | | |
| P95 Latency | | | |
| TTFT | | | |
| Throughput | | | |
| Recovery Time | | | |
| Migration Time | | | |
| Reparallelization Time | | | |

本文件將 End-to-End 定義為 scenario makespan（第一個 measured request sent 到最後一個
measured request completed）；Mean Request Latency 則是各 measured request 的
send-to-completion latency 平均。兩者不得混用。

需要 dynamic planner 的 run 另附逐事件 audit table：

| Preemption Event | Planner Invocation | Available GPUs | Active/Queue Snapshot | KV/Route/Placement Snapshot | Candidates | Selected TP/PP/DP/EP | Target Nodes | Status |
| --- | --- | ---: | --- | --- | ---: | --- | --- | --- |
| | | | | | | | | |

F1 的 Reparallelization、Original SpotServe、MoE-aware SpotServe，以及 F2 全部四組，都
必須填滿此表。欄位 unavailable 或無法和 preemption event 對應時，該 run 不得作為
dynamic-planning 證據。

### Mandatory chat report after every run

每一個 smoke、preliminary、formal 或 replacement run 結束後，都必須先把結果貼到目前
對話框；禁止等全部 repeats 或整個 matrix 跑完才一次回報。回報必須包含實際數字，不能只提供
結果目錄或寫「成功」。訊息送出前不得開始下一個 run。

每次使用以下格式：

```markdown
## Run Result — <Experiment>/<Treatment>/<Run index>

Status: valid / invalid / pending verification
Scenario: K8-CS
Commit / image / config hash: ...
Preemption: target=<worker/failure-domain>, expected pool=8->7,
observed=..., phase=... tokens (...%)

| Metric | Value |
| --- | ---: |
| End-to-End Latency | |
| Mean Request Latency | |
| P50 / P95 / P99 | |
| TTFT | |
| Throughput | |
| Recovery Time | |
| Migration Time | |
| Reparallelization Time | |
| Success / fallback rate | |

Planner audit: event_id=..., invocation_id=..., available_gpus=...,
candidate_count=..., selected_plan=..., target_nodes=..., status=...

Result files: <host paths>
Invalid reason / warnings: none or explicit evidence
```

尚未具備 instrumentation 的 metric 必須寫 `unavailable`，不能填 0。invalid run 也必須
回報原始數據、log evidence 與 invalid 原因，再以新的 attempt ID 補跑；replacement 不得
覆寫或隱藏 invalid attempt。Run 1 到 Run R 全部 valid 後，再於對話框追加全部
raw-value table、mean、sample standard deviation、improvement 與 variance 判讀。

R 次 valid runs 完成後，對每一個 per-run metric 計算算術平均與 sample standard
deviation：

```text
mean = sum(xi) / R
sample_stddev = sqrt(sum((xi - mean)^2) / (R - 1)), R >= 2
```

主要比較表中的所有 `Avg.` 都使用 R 次 per-run metric 的算術平均，包括 Avg. P95；
不可挑最好的一次。合併 `raw_requests.jsonl` 後重算的 pooled P50/P95/P99 可以作為次要
request-level distribution，但必須標成 pooled，不能取代主要的 three-run mean/stddev。

### Excessive variance and invalid replacement

各次差異過大時不得刪除數字不好看的 run。保留 raw data，檢查 system log、GPU
utilization/memory、request arrival、preemption condition、selected configuration 與
decision path。只有具體證據證明程式錯誤、OOM、worker crash 或條件未按規格執行時，
才能標成 invalid；記錄原因後補跑，直到取得 R 次 valid runs。

若 R 次都 valid 但 variance 仍大，原 R 次仍是主要結果；若要增加 confirmation runs，
必須建立新的 repeat count 與結果目錄。
除非對所有比較 configuration 採用同一個預先聲明的擴充規則，否則額外 runs 不得悄悄
併入主要平均。當 improvement 約 1%，但單次波動約 3-5%，或各次改善方向不一致時，
結果必須標成 inconclusive，需要更多 runs，不得宣稱效能提升。

## Preserve Results

benchmark 結果位於 `sllm_head` 的 `/tmp/spotserve-work/results`。切換 deploy set 或
重建 container 前，先複製到 host。先建立結果根目錄：

```bash
mkdir -p performance-results
```

每次 matrix cycle 跑完後，使用包含 experiment、scenario 與 run index 的唯一名稱：

```bash
RUN_TAG="${EXPERIMENT_ID}-${SCENARIO_ID}-run${RUN_INDEX}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "performance-results/$RUN_TAG"
podman cp sllm_head:/tmp/spotserve-work/results \
  "performance-results/$RUN_TAG/results"
git rev-parse HEAD > "performance-results/$RUN_TAG/git-commit.txt"
nvidia-smi > "performance-results/$RUN_TAG/nvidia-smi.txt"
sha256sum "$WORKLOAD_PATH" "$TRACE_PATH" "$CONFIG_PATH" \
  > "performance-results/$RUN_TAG/input-sha256.txt"
```

其中 `EXPERIMENT_ID`、`SCENARIO_ID`、`RUN_INDEX`、`WORKLOAD_PATH`、`TRACE_PATH` 與
`CONFIG_PATH` 必須在執行前設成 manifest 中的明確值，不可使用模糊或重複標籤。

每個 run 至少保留：

```text
summary.json
raw_requests.jsonl
router metrics JSONL
trace_replayer.log
report.html
特殊 verifier 的 report.json
git commit、image ID 與 nvidia-smi 紀錄
workload / trace / config SHA256
```

## Invalid Run Rules

以下任一情況發生時，不得把該 run 放入效能統計：

- `trace_replay_success=0` 或 `trace_replay_failed>0`。
- benchmark setup/cleanup 出現 error，且模型狀態無法確認乾淨。
- baseline 與 candidate 使用不同模型、GPU 數、workload 或 trace。
- true KV restore claim 只有 token replay counter，沒有 restored blocks。
- runtime routing claim 使用 fixture，而不是 `vllm_runtime_topk`。
- physical placement claim 沒有 `runtime_verified_placement=true`。
- elastic EP run 發生 actor recreate，或 active request 未完成。
- GPU 同時被其他 workload 佔用，造成 OOM、scheduler pending 或異常 throttling。
- formal scenario 的 model/workload/trace/config hash 與 frozen manifest 不同。
- 實際 preemption token progress 超出目標 +/-10 percentage points，或指定 anchor 已完成。
- K8-CS preemption 時沒有形成 queue pressure。
- 需要 dynamic planner 的 strategy 未做到每個 preemption 對應一個 runtime decision。
- planner input 未反映已 preempt 的 capacity，或 selected target 包含 preempting/dead node。
- matrix/config/trace 預先寫死 selected TP/PP/DP/EP、target node、migration target 或 placement。

cleanup warning 不一定代表 run 無效；需進一步確認後續 model deployment、trace replay
與 success gates 都成功。若無法確認，將該次標成 invalid 並重跑。

## Final Tables

每個 strategy/configuration 先列 Run 1 到 Run R，再列 R-run mean 與 sample
standard deviation。主表列出：

```text
success rate
clean success rate
fallback rate
mean / P50 / P95 / P99 latency
TTFT
throughput
recovery / migration / reparallelization time
replan/migration/failure window latency
post-replan/post-migration/post-recovery latency
```

Original SpotServe 與 MoE-aware SpotServe 的 latency improvement 使用 R 次平均計算：

```text
Improvement (%) = (Original mean - MoE-aware mean) / Original mean * 100%
```

建議主要 comparison 表使用：

| Strategy | Avg. Latency | Std. Dev. | Avg. TTFT | Avg. Throughput | Avg. Recovery Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rerouting | | | | | |
| Reparallelization | | | | | |
| Original SpotServe | | | | | |
| MoE-aware SpotServe | | | | | |

Throughput 等「越高越好」的 metric 應使用相反方向的公式，並在表頭寫清楚定義。
除了數值改善，還必須驗證 selected configuration、migration target、KV reuse、expert
placement 或 recovery path 確實改變，且變化與效能差異具有合理因果鏈。若 decision
相同或差異小於 run-to-run variability，結論應標為 inconclusive，而不是 optimization
有效。

MoE mechanism table 另外列出：

```text
expert locality ratio
estimated and observed remote routing ratio
expert remap duration and moved bytes
A2A baseline/candidate payload and reduction ratio
EP transition, changed experts, and actor identity preservation
per-preemption planner input snapshot/hash, candidates, selected plan, and status
```

K8 formal 圖表與報告標題中明確註明：

```text
Kubernetes Ray cluster: 1 non-GPU head pod + 8 one-GPU worker pods;
logical preemption replay; Qwen1.5-MoE-A2.7B-Chat
```

## Legacy Failure-domain And Cross-host Gates

只有一台 4 GPU 主機時，可先執行已提供的 2+2 GPU failure-domain simulation：

```text
benchmarks/spotserve/benchmark_matrix_simulated_cross_host_expert_remap_performance.yaml
```

M4 使用獨立容器，把同一台主機拆成 GPU 0-1 與 GPU 2-3。執行前先保存一般 benchmark
結果，停止一般 SpotServe stack，並確認四張 GPU 都空閒：

```bash
podman stop sllm_head sllm_worker_0 sllm_worker_1 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.free --format=csv
```

啟動模擬環境：

```bash
SPOTSERVE_REPO_ROOT="$PWD" \
MODEL_FOLDER=/work/spotserve-models \
podman-compose \
  -f "$PWD/examples/spotserve/docker-compose.simulated-cross-host.yml" \
  up -d
```

執行 M4：

```bash
podman exec sllm_sim_head bash -lc '
cd /workspace/spotserve &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_simulated_cross_host_expert_remap_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

有效條件：

```text
host_mode = simulated_failure_domain
physical_cross_host_verified = false
simulated_cross_host_verified = true
physical_host_count = 1
failure_domain_count = 2
ray_node_count >= 2
verified_placement = true
physical_weight_migration = true
cross_failure_domain_migration = true
cross_failure_domain_moved_shards > 0
cross_failure_domain_moved_bytes > 0
inter_container_a2a_observed = true
```

立即保存 M4 結果：

```bash
RUN_TAG="$(date +%Y%m%d-%H%M%S)-simulated-domains"
mkdir -p "performance-results/$RUN_TAG"
podman cp sllm_sim_head:/workspace/spotserve/results/spotserve_simulated_cross_host_expert_remap_performance \
  "performance-results/$RUN_TAG/"
```

完成後停止模擬環境並釋放 GPU：

```bash
SPOTSERVE_REPO_ROOT="$PWD" \
MODEL_FOLDER=/work/spotserve-models \
podman-compose \
  -f "$PWD/examples/spotserve/docker-compose.simulated-cross-host.yml" \
  down
```

它驗證兩個 Ray worker nodes、兩個 failure domains、跨 domain 的 physical expert
movement，以及 inter-container A2A activity。報告標題必須註明：

```text
single-physical-host, two-failure-domain simulation
```

這個結果不等於 physical multi-host，也不能用來代表實體 NIC latency/bandwidth。

single-host 主實驗完成後，若另有兩台可互通的 GPU hosts，再執行 physical
cross-host expert remap gate。部署拓樸、映像同步、完整命令與成功條件見：

```text
docs/spotserve-cross-host-experiment.md
```

這是獨立的機制驗證，不可把尚未執行的 cross-host gate 混入 single-host 主表。
