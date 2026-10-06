# K8s 核心驗證：動態 target 與 live KV recovery

更新：2026-10-05。這是驗收關卡，不是「完整 SpotServe 已實作」的宣稱。
head 與八個 GPU worker pods 由使用者建立；本次修改沒有啟動 GPU 實驗。

## 先說目前能不能通過

目前不能承諾通過：現有 runtime export 仍回報 `can_restore_cross_node=false`，
backend 會拒絕不同 worker 的 restore。source 有 grace time 只解決「還活著」，
不等於 target 已 ready、layout 相容或傳輸完成。

還有一個具體的跨 pod 前置條件：這份 vLLM 的 NIXL side-channel host 預設為
`localhost`。每個真正持有 KV 的 GPU rank 必須 advertise 自己可被其他 pods 存取的
位址，而不是 head／primary actor／所有 ranks 共用的固定 IP。須讀回 export lease 的
`remote_host` 並驗證可達性；只設定 pod env 而沒確認 Ray runtime-env 傳遞的結果也不夠。

因此先做 [啟動 smoke](spotserve-k8s-smoke-verification.md)。在 cross-worker runtime
尚未補好前，不建議為了看到已知失敗而反覆執行下方 GPU probe，更不要啟動完整 F1/F2。
本次沒有直接改 capability=True，也沒有把 token replay 當成 KV 搬移。

## 與 smoke 有什麼不同

| 關卡 | 怎麼驗證 | 可以下的結論 |
| --- | --- | --- |
| `smoke` | 兩個 EP2 啟動、EP rank/size 讀回、四個短請求 | 初始 serving 路徑可用；沒有搶佔與 KV 搬移證據 |
| `core-probe` | 一組無中斷 baseline，配對一組 worker preemption + 動態重新配置 | 僅在全部證據成立時，證明這次 live-KV recovery 與 target membership 路徑可用 |
| 完整 SpotServe | 再驗證 engine restart 後存活 GPU context 保留、context-aware mapping、搬運排程與 token boundary | 不能只靠前兩關宣稱完成 |

`core-probe` 不是 F1/F2 縮小版，不測 MoE 效能改善；不寫 formal ledger，僅執行兩個 runs。
兩者使用相同 checkpoint、初始 topology、workload bytes 和 NIXL connector。
baseline 不送 preemption、不重配置；recovery 才在 notice 後選 config/worker subset、建立
新 target。沒有預先建立 B/C serving targets，也不建立或 eviction Kubernetes pods。

## 每個 check 的意義

1. `paired_requests_and_full_greedy_output_equal`：相同 request IDs、prompt token IDs；
   兩組都成功且完整 output token IDs 逐一相等。比較的是「已提交 output + 新 suffix」，
   不是只比較恢復後生成的後半段，也不是只相信 `usage` 計數。
2. `dynamic_new_target_members_match_plan`：notice 後的 planner invocation、實際 applied
   execution、新 instance IDs、最終 READY member-worker 集合，都與 plan target workers
   一致；不包含被搶佔 worker。只留下 plan JSON、使用舊 target 或 allocation 失敗都不通過。
3. `native_cached_prefix_observed`：被中斷的量測請求有 native restore receipt，source 是
   初始 instance、target 是新 instance，且 target 回報 `cached_tokens>0`、restored blocks
   非零。`expected_blocks` 或 staged hook 本身不足以通過。
4. `no_token_replay_fallback`：所有量測請求的 router metrics 明確回報沒有 replay／restore
   fallback。即使 replay 產生了完全相同的 tokens，也不算 native recovery。
5. `source_release_after_native_receipt_before_deadline`：每個 receipt 都對應到保留 source
   的 release authorization，時間順序為 receipt → release authorization → grace deadline。
   不能把 timeout 或請求因失敗消失當作搬移完成。

目前 receipt 來自 target **native request 完成**，不是 connector 的早期 transfer ACK。
`restored_blocks` 計數由 source lease 的 expected block count 加 target cached-prefix 訊號
組成，不是獨立量測的網路 bytes。這一關也沒有驗證每個 exported block 都已接收、source 實際釋放 GPU allocation 的
時間、rank→pod 身分、request 尚未 decode 完就可提早釋放 source，或多次連續中斷。
因此成功時仍會輸出 `full_spotserve_verified=false` 和 `not_verified`。

## 執行方式

在正式 head pod、repository 根目錄操作。須已有相同 patched image、可讀 checkpoint、
HTTP 服務、八個空閒單 GPU workers 與結果 volume，條件同 smoke 文件。

先做不連 HTTP／Ray、不使用 GPU 的靜態檢查：

```bash
python scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase core-probe --dry-run \
  --output-dir /results/k8s-core
```

預期 `dry_run_passed`，只代表 config 與命令可解析。

以下實機命令應等相關 runtime 能力補好後再執行。先在正確硬體準備一次 profile：

```bash
python -u scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase profile --output-dir /results/k8s-core
```

`profile` 會準備缺少／失效的候選 profile 與 preflight artifacts，不執行 F1/F2。
這不是短 smoke，載入模型與各 shape 的預先指定 repetitions 可能耗時。
模型路徑如需變更，請寫入獨立 config，所有 pods 必須可讀；不要沿用不同硬體的 profiles。

接著執行配對驗證：

```bash
python -u scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase core-probe \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --ray-address auto --ray-namespace sllm \
  --output-dir /results/k8s-core
```

這一關只接受相同 hardware/worker placement/model/workload 的 measured profile cache；
缺少或失效時 exit=2，**不會自行重跑 profile**。同一 fingerprint 下已失敗的 shape 不重試，
仍須至少兩種 passing shapes。`--force-profile` 不能與 `core-probe` 一起使用。

為了與 profile 校準的 preemption timing 一致，probe 使用 config 的正式長度。
目前預設每 run 是 4 個 warmup + 24 個量測請求，8192 input／1024 output tokens，
temperature=0、ignore_eos=true。這不是 smoke 的 512/64，也不是 paper 原始 token 配置。
若因跨 shape 浮點差異造成 greedy token 不一致，先診斷首個分歧點，不得直接放寬 gate
或宣稱一定是 KV transport 損毀。

## 結果與失敗處理

最新結果為 `core-probe-results.json`／`.md`；每次執行保存到
`core-probe-attempts/<attempt-id>/`，包含 baseline/recovery 的 deploy、matrix、workload、
metrics 路徑和 `benchmark.log`。benchmark 保存初始 `instance_states.json`、
`final_instance_states.json`、`raw_requests.jsonl`、runtime EP audit 和 cleanup 紀錄。
每個 run 使用自己的唯一 model name，只 cleanup 本次 model，不刪除其他 models 或 pods。
cleanup 不等於重設 trace 的 worker-state overlay。再次使用同一 cluster 前，還須確認
先前 logical-dead worker 已按新 run 的初始資源契約重設；不能把 7-GPU 的殘留狀態當成
8-GPU 共同起點。多個正式 runs 的 reset/initial-membership 一致性也須另行驗收。

- exit=0 且 `live_kv_core_verified`：五項 gate 通過，僅限這個 trace／配對的 live-KV 路徑。
- `dry_run_passed`：沒有實機證據。
- exit=2／`blocked`：保留失敗與 artifacts；不得生成「SpotServe 遷移成功」的結論。

已執行的 probe 不自動重試。只有 code/config 已變更，或提出新診斷假設時，才使用
`--force-core-probe` 建立新 attempt。三次有意義嘗試仍未解決就停止，提供最新 JSON、
benchmark log、router metrics、初末 membership、trace status 和相關 worker logs。

## 達成原版核心還需要哪些程式修改

目前動態 worker selection 是 READY/free-GPU + 固定排序，不是 KV/expert-aware KM。
原版 context daemon 也不是只保存 metadata：它是權重與 KV 的 GPU allocation owner，
engine 為使用者；因此 surviving GPU 的 engine 重啟不會銷毀那些 allocations。
見 [SpotServe 原始論文](https://arxiv.org/html/2311.15566) 的 §3.1–3.4、§4。

後續順序與每一步的必要證據：

1. 完成不同 pods 的 NIXL reachable address、block-layout/TP compatibility 與實際
   transfer ACK。先在單一有界 request 上驗證；不能只改 scope flag。cold target 若晚於
   grace，該 run 判失敗；任何 prewarm/headroom 都必須計入同一 pool。
2. 將 global recreate 改成 keep/create/migrate/release diff，保留不受影響的 instance；
   allocator owner 尚未分離時，只能將仍活著的 engine-owned allocations 算作重用。
3. 提供有 owner、lease、rank、layout 的真實 context inventory，KM 邊權只計算可保留
   或可搬運的 GPU bytes，再將 mapping 套到實際 rank placement 並讀回。
4. 實作 persistent daemon 的 CUDA allocation／IPC attach，先權重再 KV allocator，
   同時移轉 block table、request token boundary 與 expert tensors/placement metadata。
   engine kill 後 buffers 仍有效，worker 真正 dead 時 lease/buffers 必須撤銷。
5. 再補 KV-first／逐層權重搬運與 token-level interruption；所有 F1/F2 共用相同 runtime，
   只改機制開關。rank-aware MoE locality 也須有實際 dispatch/placement 證據。

這些 runtime 工程尚未完成；本次新增 gate 和 response 修正，沒有用假 daemon 或假
reuse bytes 掩蓋缺口。完整進度見 [實作狀態](spotserve-k8s-implementation-status.md)。
