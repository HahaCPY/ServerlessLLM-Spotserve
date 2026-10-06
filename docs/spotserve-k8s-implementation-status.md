# SpotServe / MoE：9-pod 實作狀態與驗收契約

更新：2026-10-05。此文件區分控制面修正、GPU runtime 能力與尚未完成的研究機制。
本次沒有啟動 GPU profile、benchmark、Ray cluster 或 Kubernetes pods，也沒有改寫既有
5070 pilot 結果。CPU 測試通過不等於 K8s 的正式 F1/F2 已可完成。

先在 head 執行 [兩個 EP2 的小驗證](spotserve-k8s-smoke-verification.md)。新增的
`--phase smoke` 不 profiling、不送 preemption，只驗證初始部署／EP readback／短請求；
成功與失敗皆保存 JSON、Markdown 和每個 attempt 的原始 artifacts。

動態 target／native KV 另由 [core-probe](spotserve-k8s-core-verification.md) 驗收：一個
無中斷 baseline 配對一個 worker preemption run，查新 instance/member workers、native
cached-prefix、完整 token 一致和 source release authorization。只重用已凍結的實測
profile，不自動 profiling，也不啟動正式矩陣。現有 cross-worker capability 仍未成立，
因此不能宣稱它已在 K8s 通過。

stateful 正式 cells 已設定 `require_native_kv_restore=true`：export／scope／restore／
cached-prefix 不成立便失敗，不再靜默以 replay 成功充當 native recovery。失敗的請求
從 inflight 表消失也不算 source 可釋放；須有 native receipt。native response 會接回
已提交 output tokens，避免只回 suffix；目前新增 correctness probe 限單次中斷。

## 最終環境

- 由使用者建立 YAML：一個 0-GPU head pod，八個各持有一張 GPU 的 worker pods。
- head 執行 ServerlessLLM controller/router 與唯一一份 experiment script。
- worker pods 可以在相同或不同 Kubernetes nodes；Ray worker ID、pod IP、實體 host
  與 serving instance 是四個不同概念。不能用 pod 數冒充物理 host 數。
- 每個 worker 宣告 `worker_node: 1`、唯一的 `worker_id_<id>: 1`、`GPU: 1`；head
  宣告 `control_node: 1`、`GPU: 0`。各 worker 必須有不同的 Ray pod IP。
- 模型、image digest、NCCL/NIXL 依賴與結果 volume 必須一致。worker 要留足 CPU 給
  backend actor、DP engine control actor 與 GPU worker；head 也須有獨立 CPU 預算，避免
  workload generator 成為瓶頸。NetworkPolicy 必須允許實際使用的 Ray/NCCL/NIXL 通訊。
- 未來十二張 GPU 是另一個 scenario。pool allocation 不硬編碼八個 workers，但 K12
  的硬體 fingerprint、profile 與 workload/trace 必須重新凍結。

共同初始 topology 是兩個 EP2 instances：每個 TP=1、PP=1、vLLM DP=2、EP enabled；
合計四張 GPU，另四張作 overlap headroom。這是共同起點，不是 planner 永遠只能選 EP2。
目前 profile 候選仍是單 deployment 的 EP2/EP3/EP4，不是完整 paper D/P/M/B 探索空間。

## 本次已實作並以 CPU 測試驗證

| 邊界 | 改變 | 驗證範圍 |
| --- | --- | --- |
| GPU pool | target workers 是允許的 pool；scheduler 只保留 shape 需要的 GPU 子集合 | 8/12 workers、occupied/preempting workers、雙 EP2 不重疊 |
| instance membership | 保存每個 rank 的 member worker IDs 與 allocations；primary 只是 actor 的位置 | 非 primary worker 搶佔也能找到 instance |
| vLLM ranks | 將實際保留名單轉成 pod-IP placement constraint；使用受支援的 DP address 參數 | resolution/uniqueness 的 CPU 測試；實際 Ray ranks 仍需 GPU 驗證 |
| planner | opt-in `spotserve_algorithm1` 兩分支、profile-only ranking、固定 tie-break | underload/overload、缺 profile、GPU/shape consistency、batch 往 plan 傳遞 |
| resource snapshot | 每次優先讀 scheduler，避免沿用初始 free-GPU 數 | CPU/mocked scheduler |
| transaction | replan 序列化、切換前 READY 重檢、scaler 遵守 committed replica count；舊 source 切換後仍可找到 | CPU/mocked actors；未提供 persistent GPU ownership |
| grace deadline | Ray/HTTP trace 在 notice 發出時排 deadline；後續 events 不等待 loading/migration handler | 慢 notice handler 不會延長 source lifetime |
| source revocation | auto-dead 撤銷舊 engine owner 並釋放 scheduler reservation；不刪 pods | CPU/mocked actors；GPU rank 的實際回收仍需驗證 |
| recovery isolation | export timeout 不重新呼叫同一個 hook；replay controls 不因有 snapshot 就啟用 restore；strict native 不容許 replay fallback | CPU 回歸測試；未驗證真實 KV transport |
| experiment matrix | F1 四組、F2 四格、各三次，共 24 個主 runs；replay variants 不混入主矩陣 | runner / config CPU 測試 |

vLLM 的 `VLLM_RAY_DP_PLACEMENT_NODE_IPS` patch 必須存在。缺少它會拒絕啟動，不能在
scheduler 假記 worker-0/1，讓真正的 ranks 任意去 worker-6/7。TP=PP=1 時，跨 pods 的
DP/EP groups 使用 `fill`；`span` 是單一 TP/PP world 跨 Ray nodes 的另一個問題。

formal runner 會一次準備所缺的離線 profile，後續使用相同 frozen inputs 的 artifacts；
每個 shape 可以有預先指定的 repetitions。換硬體、pod placement、image、workload 或
profile protocol 會失效，不是每個 treatment 都重新 profiling。正式實驗前仍需在正確
硬體執行這一階段；本次沒有執行。
相同 frozen inputs 下失敗的 shape 也保留，不因再次啟動 runner 而隱式重跑；只有 inputs
改變或明確使用 `--force-profile` 才重做。

不同 repeats 使用預先固定的不同 prompt-order seeds，同一 repeat 的 treatments 共用
相同 workload bytes 與 trace。每個 repeat 留下獨立 workload artifact，不覆寫舊 matrix
引用的輸入。這仍是固定 trace 的三次 paired pilot，不代表對所有 spot traces 的泛化。

## 與原版仍不相符的部分

| 原版核心 | 目前實際情況 | 完成條件 |
| --- | --- | --- |
| configuration optimizer | 固定 pool 內兩分支已加入；不向雲端申請/釋放 instance | 若要完整 D 探索，增加不同 outer replica counts 的實測 profiles；不能把單 instance throughput 隨意線性放大 |
| GPU device mapper | node selection 仍非 GPU→topology 的 KM context matching | live context inventory + 明確可重用的 owner/lease + rank placement 執行與讀回 |
| migration planner | 已有 request/target cost planner；prefix warmup 不是 KV block transfer，更不是逐層權重搬運 | transfer callbacks、KV-first 與逐層 layer-ready ACK、peak memory budget |
| interruption arranger | 目前 bounded export/abort + 獨立 deadline，不是 runtime JIT token budget | scheduler/worker 在 decode boundary 凍結，依剩餘 grace 減 migration 預算安排 token 數 |
| persistent context daemon | 權重與 KV 仍由 vLLM engine/rank 擁有 | allocator/model loader 真的向獨立 owner 取得 GPU tensors；engine kill 後 surviving owner 的 tensors 保持有效 |
| stateful recovery | native hooks/NIXL 已有零件與既有同機 pilot；正式跨 worker/pod scope 仍未通過 | compatible target、真實 bytes/blocks、已提交 token boundary、全輸出正確性與 deadline |
| unaffected instances | global plan 仍可 recreate 全部 ready instances | per-instance keep/create/migrate/release diff；shape 不變且 placement 可保留時不能無故重啟健康 engines |
| MoE migration cost | 目前 locality 主要是 deployment expert coverage；每個 deployment 全覆蓋時沒有 rank-locality 鑑別力 | observed request DP/EP rank、physical expert owner、可執行的 target-rank steering 與 A2A 實測 |

paper 的 D 是獨立 inference pipelines；本專案另有 `replica_count` 和 vLLM 的
`data_parallel_size`。EP 開啟時 EP size 由 TP×vLLM-DP 推得，GPU 使用量為
`replica_count × PP × TP × vLLM-DP`；不能把 worker pod 數當作 D 或 instance 數。

paper-mode 的排序是：有 φ≥α 候選，按 profile latency 最小、GPU 數最少、固定 shape
順序選；沒有則按 φ 最大選。profile throughput 必須是整個 candidate 的 throughput。
目前 `queue_model=none` 不外加未校正的 queue delay；`md1_aggregate` 的選用必須事先
註冊，它只是 Poisson/constant-service 單 queue 近似，不是 Gamma CV=6 的準確模型。
vLLM 的 B 對應 frozen `max_num_seqs` 上限，不代表每次 decode 有固定 B 個 requests。

`enable_live_expert_remap` 在 MoE-replanning cells 才啟用；它要求真實 runtime patch，
不是只留下 placement metadata。paper-mode 的 configuration ranking 本身不使用舊版
weighted-score weights，所以不能宣稱這些 weights 已讓 Algorithm 1 選出不同 shape。
MoE remap 成功與 target migration cost 生效也必須分開驗證，不能只看 flags。

## 為何目前不能直接說正式 F1/F2 已可跑通

有三個尚未解除的主要障礙：

1. source 能存活 30 秒不代表新的 engine 能在 30 秒內 ready。必須先量 engine-ready
   與 migration 時間；若超出 deadline，使用既有 READY compatible target 或事先計入
   pool 的 prewarmed target。不能延長 grace、偷偷借 GPU，或移除 startup 時間。
2. cross-worker KV scope 仍是 fail-closed。token replay 可恢復服務但重算 prefix，不能
   通過正式 stateful treatment 的 restored-block gate。不能直接改 capability=True。
3. source waiter 現在仍保守等待 request 完成，而不是 connector transfer ACK。新增 deadline
   限制只能防止超時被隱藏；KV 已收完但 decode 未完的情況仍需要 ACK 才能正確提早
   釋放 source。不能把 timeout 當作成功的 migration completion。

此外，KM 在沒有 persistent allocation owner 或保留舊 engine 時，不能把「本來有 KV」
當作新 engine 可免費 attach 的 bytes。sllm host/page cache 加速重載也不是 GPU context
survival。所有 F1 treatments 必須有相同 cache policy 和 headroom accounting。

## 最小下一步與後續順序

先在使用者正式 K8s 上做一個有界的 integration probe，而不是直接跑 24 runs：

1. head 讀回兩個 EP2 的四個實際 ranks，證明 worker subsets 不重疊，且不包含 head。
2. 選第一組的非 primary worker；先 notice，第二組仍 READY。source 於 grace 內 freeze，
   target 恢復同一 request；保存 native block transfer、transfer ACK 與 greedy 輸出比較。
3. 分開標記 same-host pod pair 與 different-host pod pair；量測 transfer/startup 並確認
   source 在 deadline 後不能再提供 buffers。same-host 成功不推論 cross-host 成功。

若失敗，保存原因與 artifacts，先停該 capability；最多三次有新假設的診斷，不反覆重跑
formal benchmarks。接著依已支持的原因完成：

- transfer ACK API 與既有 READY target 路徑，避免在 grace 內等待 cold engine。
- keep/create/migrate/release 的增量 deployment transaction，先保留 unaffected engines。
- 以 observed rank/physical placement 取代 deployment coverage 的 MoE cost，接上可驗證
  的 target-rank dispatch，完成 F2 Migration factor。
- daemon 真正持有權重 buffers，再修改 KV allocator/block table/request state；只有通過
  engine-restart survival、lease expiry 與 greedy correctness 才打開 persistent capability。
- 以此 inventory 實作 KM mapper 和逐層 KV-first migration，並量測實際 peak memory。

原版設計依據：[SpotServe paper](https://arxiv.org/abs/2311.15566) 的 §3–§4 與
[官方 artifact](https://github.com/Hsword/SpotServe)。上述 K8s/vLLM 邊界是本 repository
的實作分析，不是原 paper 已提供的 vLLM 功能。
