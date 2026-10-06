# K8s 小驗證：兩個 EP2 的啟動與短請求

更新：2026-10-05。先跑這份驗證，不要直接啟動 `--phase all`。
head pod、Ray cluster 與 ServerlessLLM HTTP 服務由你啟動；script 不建立、刪除或重啟 pods。

## 這次只驗證什麼

一個 0-GPU head + 八個單 GPU worker pods。worker pods 可以同機或跨機。
使用 Qwen1.5-MoE-A2.7B-Chat，建立兩個獨立 EP2 instances，每個 TP=1、PP=1、DP=2、
EP enabled；總計保留四張 GPU，另外四張不配置給 serving instances。

送出 1 個 warmup（16 output tokens）和 4 個量測請求（512 input、64 output tokens，
temperature=0、ignore_eos=true）。512/64 是本次 smoke 的短工作負載，不是 paper 配置宣稱。
短 prompt 不會降低 checkpoint 權重的顯存需求，兩個 EP2 仍需各自載入完整模型的分片。

驗證範圍：

- 八個 worker 的 GPU／軟體／checkpoint inventory 符合環境契約。
- 兩個 EP2 都 READY，各有兩個不同 member workers，兩組 reservation 不重疊。
- 每個 instance 從 runtime 讀回 EP size=2、ranks=[0,1]，不是只相信 config。
- 四個量測請求均成功，且每個完成恰好 64 tokens。
- benchmark 對自己建立的暫時 model 執行 cleanup，保存 cleanup 紀錄。

**不驗證** preemption、cross-pod KV restore、persistent daemon、MoE 性能提升或完整 F1/F2。
動態 target／KV 搬移另有 [核心驗證關卡](spotserve-k8s-core-verification.md)，不能用 smoke
的成功取代。該關卡也不代表 persistent daemon 或完整 paper mechanisms 已實作。
member-worker 清單是 scheduler/router bookkeeping；加上 EP rank/size 讀回，仍不能當成
每個實際 GPU rank→pod IP 的 placement 正確性證明。這個限制會出現在結果 JSON 中。

## 執行前的條件

1. head 和八個 workers 已連到同一個 Ray cluster。head 宣告 `control_node: 1`，沒有 GPU。
   每個 worker 宣告 `worker_node: 1`、唯一的 `worker_id_<id>: 1`、`GPU: 1`。
   `id` 是 Ray custom resource 的 ID，不要求等於 Kubernetes node 名稱。
2. 同機的不同 pods 也必須有不同 Ray pod IP；不使用共用同一 IP 的 hostNetwork 配置。
3. 所有 pods 使用同一個 patched image，並設定相同 `IMAGE_DIGEST`。需要現有的
   `VLLM_RAY_DP_PLACEMENT_NODE_IPS` placement patch 與 runtime EP metadata patch。
4. checkpoint 在 head 和所有 workers 的同一絕對路徑可讀。預設是
   `/models/Qwen1.5-MoE-A2.7B-Chat`；不是 active 2.7B 的 dense 小模型。
5. 八張 GPU 沒有其他 workloads；盤點會暫時使用每張 GPU 的 probe actor，檢查後釋放。
6. head 的 Python 環境已有這個 repository、Ray、Torch、patched vLLM、Transformers 與
   benchmark 所需依賴。結果目錄是你已掛載且可寫的 persistent volume。
7. head 上已有可用的 ServerlessLLM HTTP endpoint。只啟動 Ray head 並不足夠。
   同機／跨機的 Ray/NCCL/all-to-all 通訊不能被 NetworkPolicy 阻擋。

physical-host markers 建議提供，用來記錄 pod placement；不要求八個 pods 分別位於八台主機。

## 怎麼跑

以下命令都在 head pod 內、repository 根目錄執行。`python` 必須是該 image 中裝有
patched runtime 的 interpreter；`/results` 請換成你真正掛載的 persistent volume。

先確認既有 HTTP 服務：

```bash
curl --fail http://127.0.0.1:8343/health
curl --fail http://127.0.0.1:8343/v1/models
```

`/health` 應回傳 `{"status":"ok"}`。如果失敗，先處理服務／port，script 不會替你啟動它。

### 1. 不使用 GPU 的靜態檢查

```bash
python scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase smoke \
  --dry-run \
  --output-dir /results/k8s-smoke
```

預期 exit code=0，`smoke-results.json` 的 status 是 `dry_run_passed`。
它不連 HTTP／Ray，不載模型、不 profiling；**不等於實機 smoke 通過**。

### 2. 真正的小驗證

```bash
python -u scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase smoke \
  --model-path /models/Qwen1.5-MoE-A2.7B-Chat \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --ray-address auto \
  --ray-namespace sllm \
  --output-dir /results/k8s-smoke
```

如果模型掛載位置或 HTTP port 不同，只調整對應參數。不要在八個 workers 各跑一次。
`--model-path` 僅供 smoke 使用；正式實驗的模型路徑要寫進獨立 config，並使用新結果目錄，
避免模型已變更卻沿用舊正式 ledger。
script 先檢查 HTTP，再盤點 GPU/checkpoint，然後啟動兩個 EP2。checkpoint hashing、載入
模型、NCCL 初始化仍可能花數分鐘；ready/request timeout 沿用 config 的 1800 秒。
本次沒有 bandwidth profile、離線 shape profile、trace 或正式實驗。

terminal 會印出這次 attempt 的 `benchmark.log` 路徑。可在第二個 head shell 看進度：

```bash
tail -f /results/k8s-smoke/smoke-attempts/<attempt-id>/benchmark.log
```

## 怎樣算通過

exit code=0，並同時確認最新 `/results/k8s-smoke/smoke-results.json`：

```json
{
  "status": "passed",
  "scope": "startup_and_short_inference_only",
  "checks": {
    "reserved_worker_count": 4,
    "pool_worker_count": 8,
    "unreserved_worker_count": 4,
    "ep_rank_size_readback_verified": true,
    "measured_requests": 4,
    "output_tokens_per_measured_request": 64
  }
}
```

實際 JSON 還有 `member_workers` 與其他 artifacts。這裡的 unreserved 是 membership 差集，
不是實測 GPU 顯存全空閒的證明；若要驗證 rank/pod placement，下一步要讀 actual worker identity。
`dry_run_passed`、`blocked` 或只有 HTTP 200 都不算通過。

輸出：

- `smoke-results.md`、`smoke-results.json`：最新驗證狀態與未驗證項目。
- `smoke-attempts/<attempt-id>/hardware.json`：GPU、Ray worker IDs、pod IP 與 checkpoint inventory。
- 同一 attempt 的 `smoke-config.json`、`deploy.json`、`matrix.json`、`workload.jsonl`、`benchmark.log`。
- `runs/<run-id>/run_metadata.json`：EP runtime audit、ready latency、trace 狀態。
- `runs/<run-id>/instance_states.json`：cleanup 之前的完整 instance/member-worker 清單。
- `runs/<run-id>/raw_requests.jsonl`、`actor_cleanup.json`：請求原始資料與暫時 model 的 cleanup 紀錄。

smoke 不寫 formal ledger，也不建立或重用正式 profiles。請把後續正式實驗放在不同結果目錄。

## 如果失敗

exit code=2，結果標為 `blocked` 並保存原因。先提供：

1. `smoke-results.json` 與 `smoke-results.md`。
2. 該 attempt 的 `benchmark.log`、`hardware.json`（若已產生）。
3. `run_metadata.json`、`instance_states.json`、`actor_cleanup.json`（若已產生）。
4. head／受影響 workers 的 container logs 和是否同機的 placement 資訊。

HTTP 失敗時不會啟動 GPU probe；engine 啟動／EP metadata 失敗時，可能尚未有 raw requests。
模型註冊、服務失聯或 cleanup 失敗可能留下暫時 actor；先查 artifact 中的唯一 `smoke-ep2-*`
model 名稱和 cleanup 紀錄，不要廣泛 kill actors、刪 pods 或啟動另一份實驗搶 GPU。

預設不重試已執行過的 smoke。只有相關 code/config 已改變，或有明確新診斷假設時，才在
相同命令加 `--force-smoke`。它會建立新 attempt，保留舊 artifacts；不是靜默重跑直到通過。
同一問題三次有意義的診斷仍未解決就停止，整理證據再決定下一個最小實驗。

## target 是動態的嗎

正式 dynamic reparallelization 路徑會在事件後選 shape 與 target workers，再建立 target
engine；並非固定遷移到預先建立的 B/C。相同 plan 可保留既有 engine，特定 runtime
conditions 才有 in-place resize/remap；Rerouting 則轉給既有健康 instance。

目前 target worker 選擇主要依 READY/free GPU 與固定排序，不是完整 KV/expert-aware KM；
也沒有自動 prewarm 動態 target。cold engine 若超過 grace，仍不能完成正式 stateful recovery。
本次 smoke 只驗證兩個共同初始 EP2，不送 preemption，不證明這條動態路徑已可用。

通過後依 [core-probe 文件](spotserve-k8s-core-verification.md) 檢查動態 instance/worker
membership、native cached-prefix、完整 greedy output 與 source release authorization。
目前 cross-worker capability 未成立，不反覆跑已知會失敗的 GPU probe。
rank→pod identity、早期 transfer ACK、persistent GPU owner 仍是另外的必要驗證；在這些
能力成立前，不把結果叫完整 SpotServe，也不直接跑完整 F1/F2。
