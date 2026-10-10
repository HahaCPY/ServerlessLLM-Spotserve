# CSCC Managed Ray 小型 F1 recovery canary

前置證據：8-GPU image smoke、兩 worker TCP、兩 worker NCCL all-reduce 已通過。
本 runner 接著直接驗證 NIXL KV recovery，並與 prefix 重算比較；每組一次，失敗不自動重試。

此版本針對 image 內的 **vLLM 0.11.2 + SpotServe KV patches**。既有本機 recovery worker
依賴較新的 pause API，因此本 runner 使用獨立、opt-in、同步 C=1 scheduler barrier。
不需要修改已安裝的 vLLM、不需重建 image，也不改 production planner 的 cross-host gate。

## 工作負載與範圍

- 自動建立固定 seed 0 的隨機權重 Qwen2-MoE tiny checkpoint，約 400 萬 parameters。
  2 layers、4 experts/layer、top-2、hidden size 256、vocab 4096；不是預訓練模型。
- 兩個不同 Ray GPU workers；每個 engine TP1，無 EP，GPU 數量高峰為 2。
- Prompt 512 tokens、output 384 tokens、生成 256 tokens 後精確凍結。
- 先執行 `kv_restore`，通過後執行 `recompute`；每組各自建立 fresh engines，含不中斷 reference。
- 機制 canary 使用 120 秒 logical grace period，不刪除 Kubernetes pods、不注入真實平台 eviction。
- NIXL 成功的必要條件：`finished_recving`、無 invalid KV blocks、target 實測 prefix
  compute 小於完整 prefix，以及完整輸出與 reference 相同。Stage 成功本身不算 restore 成功。
- Target 在首 token 再凍結；source engine process group 終止後才解除 barrier，確認後續 decode
  不再依賴 source。兩個 actor 的 GPU reservations 在該 treatment cleanup 時釋放。
- 原始 export 的 `can_restore_cross_node=false` 仍保留；本次為明確 opt-in 的 direct-API
  探索性驗證，成功也不自動等於 production planner 宣告支援跨 physical hosts。

## 在 CSCC Workspace 同步

先將新程式 commit/push 到你使用的 repository，再在平台 terminal：

```bash
cd /home/ps2026004/ServerlessLLM-Spotserve
git pull --ff-only
test -f deploy/cscc-ray/run_f1_canary.py
test -f deploy/cscc-ray/f1_canary_scheduler.py
```

## 在 WSL 提交

```bash
cscc-ray-login
cscc-ray-use 8

ray job submit \
  --no-wait \
  -- \
  bash -lc 'cd /home/ps2026004/ServerlessLLM-Spotserve && python -u deploy/cscc-ray/run_f1_canary.py --model /home/ps2026004/ServerlessLLM-Spotserve/models/cscc-f1-tiny-moe-seed0 --prepare-tiny-model --allow-experimental-cross-worker-kv --prompt-tokens 512 --output-tokens 384 --preempt-tokens 256 --grace-s 120 --timeout-s 900 --output-dir results/cscc_ray_20261011/f1_canary_01'
```

沿用已驗證的 8-GPU session，上述工作實際最多使用 2 張 GPU。
`--prepare-tiny-model` 不需下載 Hugging Face weights；第一次在 shared storage 建立 checkpoint，
之後用 manifest checksums 驗證並重用。每個 worker 啟動前也會核對這份 checkpoint。
為保留失敗證據，output directory 必須不存在；不要刪除舊結果來重跑。

```bash
ray job status <JOB_ID>
ray job logs <JOB_ID>
```

## 結果

平台 Workspace terminal：

```bash
python3 -m json.tool /home/ps2026004/ServerlessLLM-Spotserve/results/cscc_ray_20261011/f1_canary_01/result.json
```

同一目錄的 `report.md` 為中文報告，含 elapsed、recovery、first-token downtime、實際 prefix
重算量、hash 結果。每個 treatment 另保存 source metadata、export state、scheduler audit、
engine log tail。若失敗，報告保留具體階段且 Job 回傳 nonzero；只排查第一個失敗，不自動 fallback。

Recovery 從 driver 收到 source 凍結邊界開始，至 target 完成所有剩餘 tokens，包含 source
終止確認與控制通道往返。Engine startup、reference 和 preemption 前生成僅計入 elapsed。
這是機制驗證，不能據 tiny 模型的單次時間宣稱 SpotServe/MoE 效能加速；也不包含 F2 selection。

## 本地檢查

```bash
python3 tests/spotserve_test/test_cscc_f1_canary.py
python3 deploy/cscc-ray/run_f1_canary.py --help
```

CPU tests 檢查 exact boundary、abort/resume、connector evidence、拒絕 prefix fallback 與
control event ordering；實際 GPU/NIXL 成功與否仍以平台此次 Job 的結果為準。
