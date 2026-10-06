# CSCC Managed Ray／自訂 Image 申請清單

請協助確認下列需求；目標是先執行 8 GPU Ray smoke test，再執行 SpotServe F1/F2。

## 專案與映像

- Project 名稱／ID：`________________`
- 申請人：`________________`
- 是否已啟用 Project Build：`是 / 否`
- 是否已啟用 Ray Jobs：`是 / 否`
- 我的 Project role 是否能提交 Ray Jobs：`是 / 否`
- Image name/tag：`spotserve-cscc-ray:ray258-torch290-v1`
- Image digest：`________________`
- 官方文件未提供 Ray Job 自選 image 的 header。是否可讓 Ray head 與所有 GPU workers
  使用這個 project-built digest：`是 / 否`
- 若否，能否由管理員依附上的 Dockerfile、requirements 與 vLLM patch 建立平台 Ray image：
  `是 / 否；申請方式：________________`

映像主要版本：Ray 2.58.0、Python 3.11、PyTorch 2.9.0+cu128、vLLM 0.11.2、NIXL、
ServerlessLLM。映像沿用官方 Ray base，不覆寫 Ray/KubeRay entrypoint。

## GPU 與排程

- 單一 Ray Job 需要的 GPU worker 上限：`8`
- 每個 worker GPU 數：`1`
- 可用 GPU model：`________________`
- Queue：`________________`
- 個人 GPU quota：`________________`
- Project GPU quota：`________________`
- 是否允許 8 個 GPU workers 同時存在：`是 / 否`
- 若八張 GPU 跨 physical nodes，是否仍能加入同一 Ray cluster：`是 / 否`
- Ray session／queue 最長執行時間：`________________`（需求：至少 8 小時）
- 能否停用約一分鐘的 idle-worker scale-down，讓 8 個 worker IDs 在實驗期間保持穩定：
  `是 / 否；替代方式：________________`

## 儲存與上傳

- Shared storage 掛載路徑：`________________`
- Ray head 與 workers 是否看到相同路徑：`是 / 否`
- 建議 model 唯讀路徑：`________________`
- 建議 result 可寫路徑：`________________`
- Ray working-dir upload 上限是否為 500 MiB：`是 / 否 / 其他：________`

## 網路與 runtime

- Worker-to-worker TCP 是否允許：`是 / 否`
- NCCL 是否可跨 worker/pod 使用：`是 / 否 / 未驗證`
- NIXL side-channel／data-plane 是否可跨 worker/pod 使用：`是 / 否 / 未驗證`
- Pod `/dev/shm` 大小：`________________`
- 是否可提供至少 8 GiB `/dev/shm`：`是 / 否`
- 是否提供 physical host/node ID：`是 / 否；欄位：________`
- 是否提供 failure-domain／zone ID：`是 / 否；欄位：________`
- 是否允許 Ray worker custom resources：`是 / 否`
- 若不允許，是否可用 Ray Node ID 作 stable logical worker ID：`是 / 否`

## Preemption 與 grace period

- 是否會提供 termination/preemption notice：`是 / 否`
- 通知形式（signal、檔案、metadata endpoint、event 等）：`________________`
- Grace period 長度：`________________ 秒`
- Grace period 內 source pod/container 是否仍可用 GPU 與網路：`是 / 否`
- Grace period 到期後的終止行為：`________________`
- 可否用測試事件安全模擬 preemption，而不等真實資源回收：`是 / 否`

## 建議驗收順序

1. 自訂 image build 與 security scan 成功。
2. 單一 GPU Ray actor 成功 import 所有套件。
3. 8 個一-GPU actors 同時 READY，並取得 8 個不同 Ray Node IDs。
4. Shared storage 讀寫成功。
5. 兩個 workers 間進行最小 NIXL/KV transfer、attach、ack。
6. 模擬一次 grace-period preemption。
7. 才執行完整 F1，最後執行雙 target 的 F2。
