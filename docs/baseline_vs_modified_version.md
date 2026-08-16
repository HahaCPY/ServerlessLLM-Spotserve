# Baseline vs. Modified Version

本報告中的 **modified version** 是目前加入 NIXL KV-cache export/restore
的 SpotServe runtime；baseline 是同一個 container、同一份 trace、同一個
request，但停用 KV restore，讓 target 重新計算完整 source context。這是
控制變因後的功能與 recovery 效能比較。

## 實驗設定

- Model: `/work/spotserve-models/Qwen2-MoE-Tiny`
- Trace: `examples/spotserve/spot_trace_local_long_context_single_migration.jsonl`
- Trace 狀態：initial `add`、一次 source `remove`、`DONE`
- GPU：0、1、3；每次最多 3 個同主機 containers
- GPU2 當時由其他使用者的 container 佔用
- 每個 context 長度各跑 2 次；每次都包含 baseline 與 modified version
- 共 3 × 2 = 6 個 comparison cells，也就是 12 次 recovery case
- Physical cross-node：否，這是同主機 container 模擬

Qwen2-MoE-Tiny 的模型設定最大位置長度是 512，因此本次使用 64、240、
480 三種 context 長度。每一個 case 都在 source 被 remove 後，驗證 target
是否能繼續處理同一個 request。

## 實驗結果（每個 context 長度 2 次的平均值；中位數相同）

| Context | Baseline 重算 tokens | Modified 重算 tokens | Modified restore blocks | Baseline target recovery | Modified target recovery | Recovery 改善 | Baseline migration | Modified migration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 66.5 | 0 | 5 | 0.502 s | 0.303 s | 39.6% | 22.122 s | 21.986 s |
| 240 | 242 | 0 | 16 | 0.472 s | 0.306 s | 35.2% | 22.079 s | 21.865 s |
| 480 | 483 | 0 | 31 | 0.479 s | 0.305 s | 36.1% | 22.209 s | 22.053 s |

### 成功率

- Baseline：2/2 cells per context passed；restore intentionally disabled。
- Modified version：6/6 restore successes。
- Modified fallback：0/6。
- 兩種模式的 target 都在 source 停止後繼續服務：6/6。
- 所有 cell 的 trace 都執行完畢，沒有失敗 cell。

## 分析

### 1. KV cache 優勢已在 recovery 階段顯現

Baseline 的重算量會隨 context 增加：66.5 → 242 → 483 tokens；modified
version 在三種情況都是 0。相對應地，NIXL 還原的 blocks 從 5 → 16 → 31
增加，表示搬移的是實際 KV state，而不是只記錄 metadata。

Target recovery 平均縮短約 35～40%，其中 480-token case 從 0.479 秒降到
0.305 秒。這是目前最能直接反映 KV restore 效益的數值。

### 2. 完整 migration 時間差距仍小

完整 migration 只快約 0.14～0.21 秒，因為這個數值包含 source container
停止、Podman inspect polling、socket 控制與 target continuation；這些控制
流程約佔 22 秒，會掩蓋 context recompute 與 KV restore 的差異。因此完整
migration 時間不能直接當成 KV copy throughput。

### 3. 這次實驗證明的範圍

這次已證明 modified version 能在 source 被 preempt 後，將實際 KV blocks
交給 target 並繼續 request，而且沒有 fallback。它仍不是 production
throughput benchmark，也不是 planner 真正重新選 `ParallelPlan`、重建
vLLM worker、切換流量的 end-to-end benchmark；本次 source migration 是由
trace 的 `remove` 事件觸發。

## Reproduction

單組比較：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 \
PYTHONPATH=/work/containers/s112060021/Qwen3/vllm:/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve \
/work/containers/s112060021/Qwen3/vllm/.venv/bin/python -u \
tests/spotserve_test/run_trace_baseline_vs_spotserve.py \
  --model /work/spotserve-models/Qwen2-MoE-Tiny \
  --trace examples/spotserve/spot_trace_local_long_context_single_migration.jsonl \
  --gpus 0 1 3 --prompt-tokens 480 --max-model-len 512 \
  --output /tmp/spotserve-trace-long-context-vs-nixl-tiny.json
```

矩陣比較：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 \
PYTHONPATH=/work/containers/s112060021/Qwen3/vllm:/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve \
/work/containers/s112060021/Qwen3/vllm/.venv/bin/python -u \
tests/spotserve_test/run_trace_baseline_vs_modified_matrix.py \
  --model /work/spotserve-models/Qwen2-MoE-Tiny \
  --trace examples/spotserve/spot_trace_local_long_context_single_migration.jsonl \
  --gpus 0 1 3 --prompt-tokens 64 240 480 --repeats 2 \
  --max-model-len 512 \
  --output /tmp/spotserve-baseline-vs-modified-matrix.json
```

矩陣原始 JSON：`/tmp/spotserve-baseline-vs-modified-matrix.json`；每個
cell 的 JSON 與 runner log 也以 `p<context>.r<repeat>` 儲存於同一個目錄。
