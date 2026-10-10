# CSCC Managed Ray 映像與從零操作手冊

這個目錄提供 CSCC AI Platform Managed Ray 使用的 SpotServe 實驗映像、八張 GPU smoke
test，以及正式 F1/F2 的單一 Ray Job wrapper。

> 目前狀態：專案 image 已在 CSCC 以 `X-CSCC-Image` 驗證；8-GPU／8-node smoke 與
> 2-worker TCP probe 已通過。`run_formal_f1_f2.py` 會用 Ray Node ID 取代平台沒有提供的
> `worker_id_*` resources、固定 8 個 workers、啟動本次 Job 專用的 ServerlessLLM control
> plane，然後執行 profile 與單次 F1/F2。trace 是受控的 logical graceful preemption，不能
> 說成平台真的刪除了 Pod 或證明了不同 physical host。

## 1. 這個映像包含什麼

- Ray 2.58.0（與 CSCC 平台文件要求相同）
- Python 3.11
- PyTorch 2.9.0 + CUDA 12.8 wheels
- vLLM 0.11.2
- NIXL 與 ServerlessLLM Store
- 本專案的 ServerlessLLM、SpotServe benchmark 與 vLLM patches
- 映像檢查程式 `verify_image.py`
- 八張 GPU／八個 Ray worker smoke test `smoke_ray_image.py`

映像以官方 `rayproject/ray:2.58.0-py311-gpu` 為 runtime base，而且不覆寫 Ray/KubeRay
的 entrypoint，讓平台仍可自行啟動 Ray head 與 workers。

## 2. 先向學長姐或平台管理員確認

在平台上按任何 Build 或啟動 Ray Job 之前，先確認下列項目：

1. 你的帳號已加入正確的 Project。
2. Project 已啟用 Build 與 Ray Jobs 功能。
3. 你有 Project Manager/Admin 或可執行 Build 的權限。
4. 個人與 Project GPU quota 都允許同一個 Ray Job 使用 8 張 GPU。
5. 指定 queue 中同時有 8 張同型 GPU；若 GPU 分散在不同 physical node 也可以做
   distributed smoke test。
6. Managed Ray runtime 採以下哪一種：管理員允許 head/workers 使用本專案 image；或管理員
   依 Dockerfile/requirements 把套件與 vLLM patch 併入平台 Ray image。Project Build 成功
   本身不代表 Ray workers 會使用該 image。
7. 已掛載 shared storage，例如 `/mnt/shared/<storage-name>`，供 model 與結果共用。
8. worker 間允許 Ray、NCCL 與 NIXL 所需的 TCP 通訊。
9. 平台是否提供 termination notice／grace period，以及通知形式和秒數。
10. 平台是否允許在 Ray worker 設定自訂 resources，或至少能查到 stable worker/node ID。
11. 是否能讓 8 個 workers 在整個實驗期間常駐；官方預設 idle worker 約一分鐘會停止，
    但本實驗的 trace 必須保留 stable worker ID 並在稍後執行 `add`。
12. Ray session 的執行時間上限至少 8 小時；完整 profile 與 6 個唯一 formal runs 不能被
    queue/session lifetime 截斷。

## 3. 建置 reference image，再由管理員確認如何接到 Managed Ray

CSCC 的 Project Build 可從壓縮檔建立 private image，且 `Dockerfile` 必須位於壓縮檔最上層。
本專案已確認可在 Ray submission headers 以 `X-CSCC-Image` 指定完成的 private image；請在
repository 根目錄執行：

```bash
python3 scripts/package_cscc_ray_image.py \
  --output dist/spotserve-cscc-ray-build-context.tar.gz
```

這只會封裝 image build 必要的 source、patches、formal config 與 CSCC Ray 工具，不會封裝
models、results、`.git` 或 secrets。產物如果已存在，程式會拒絕覆寫；要更新請改用另一個
檔名，保留每次申請 Build 的可追溯性。

接著在網頁介面：

1. 連上校內網路或 VPN，登入 CSCC AI Platform。
2. 進入目標 Project。
3. 開啟 **Build**，新增一個 image build。
4. 上傳 `dist/spotserve-cscc-ray-build-context.tar.gz`。
5. Image name 建議填 `spotserve-cscc-ray`。
6. Tag 建議填不可變版本，例如 `ray258-torch290-v1`，不要只使用 `latest`。
7. 啟動 Build，保存 build log、image tag 與 image digest。
8. 等安全掃描完成；若掃描阻止部署，把完整報告交給管理員，不要任意降版套件規避。
9. 把 build log、image digest、Dockerfile 與 `requirements-runtime.txt` 交給管理員；請其
   回覆是直接把該 digest 指定給 Managed Ray，還是建立平台版 SpotServe Ray image。

若 Project 頁面沒有 Build 表單，通常代表專案未啟用這項功能或你的角色權限不足，不是
Dockerfile 本身的問題。

## 4. 若管理員要求先自行驗證 Dockerfile

這一節需要一台已安裝 Docker、可以連外下載 base images 的 Linux 主機；不是使用 CSCC
平台 Build 的必要步驟。

```bash
docker build \
  --file deploy/cscc-ray/Dockerfile \
  --tag spotserve-cscc-ray:ray258-torch290-v1 \
  .
```

先做不需要 GPU 的 package/patch 檢查：

```bash
docker run --rm \
  spotserve-cscc-ray:ray258-torch290-v1 \
  python /app/deploy/cscc-ray/verify_image.py --build-time
```

再於有 NVIDIA Container Toolkit 的機器檢查 GPU：

```bash
docker run --rm --gpus all \
  spotserve-cscc-ray:ray258-torch290-v1 \
  python /app/deploy/cscc-ray/verify_image.py --require-gpu
```

兩個指令都應輸出 JSON 且 `status` 為 `passed`。

## 5. Model 與結果放哪裡

不要把 model weights 放進 image，也不要放進 `ray job submit --working-dir` 的目錄。八個
worker 都應從 shared storage 讀相同路徑，例如：

```text
/mnt/shared/spotserve/models/Qwen1.5-MoE-A2.7B
/mnt/shared/spotserve/results/<run-id>
```

建議先由一個受控下載工作把 model 寫入 shared storage，確認大小、revision 與 checksum；
正式各組實驗使用同一份唯讀 model。不要因為要 `git pull` 而刪除 model，只需確保 model
沒有被 Git 追蹤或封裝進 build context。

## 6. 從零安裝 Ray command-line client

這些步驟是在你用來送出 Ray Job 的 terminal 執行，可以是 Workspace terminal，也可以是
已連 VPN、能連平台 endpoint 的個人電腦。

### 6.1 安裝 Python 與建立虛擬環境

Ubuntu/Debian 範例：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git ca-certificates
python3 -m venv .venv-ray-client
source .venv-ray-client/bin/activate
python -m pip install --upgrade pip
python -m pip install "ray[default]==2.58.0"
ray --version
```

預期版本是 `2.58.0`。之後每次開新 terminal，都先執行：

```bash
source .venv-ray-client/bin/activate
```

若你在 CSCC Workspace 中無法使用 `sudo`，通常 Python 已存在，直接從
`python3 -m venv ...` 開始即可；缺少系統套件時再請管理員安裝。

### 6.2 準備程式碼

```bash
git clone <你的-repository-URL>
cd ServerlessLLM-Spotserve
git checkout <實驗用-commit-or-tag>
```

若 Workspace 已經有 repository，改成：

```bash
cd ServerlessLLM-Spotserve
git status --short
git pull --ff-only
```

只有在沒有未提交修改且分支正確時才 pull。正式報告要記錄 `git rev-parse HEAD`，不要只記
branch 名稱。

### 6.3 安裝 CSCC 根憑證

先從平台文件下載
[CSCC root CA](https://nthu-cscc.github.io/CSCC_AI_Platform_Documentation/assets/cscc-root-ca.pem)，
假設存到 `~/Downloads/cscc-root-ca.pem`：

```bash
export REQUESTS_CA_BUNDLE=~/Downloads/cscc-root-ca.pem
export SSL_CERT_FILE=~/Downloads/cscc-root-ca.pem
```

兩個都要設定：Ray submission client 與後續讀取 log 的連線會使用不同的底層 client。
不要用 `--verify=false` 長期繞過憑證驗證。

### 6.4 取得 Ray Jobs endpoint 與 API token

在平台的 Ray Jobs 頁面取得：

- Jobs API address，例如 `https://<ray-endpoint>`
- API token
- queue 名稱
- GPU model 名稱

在 terminal 設定；以下值只是格式示例，必須換成平台實際提供的內容：

```bash
export RAY_ADDRESS='https://<ray-endpoint>'
export RAY_JOB_HEADERS='{"Authorization":"Bearer <api-token>","X-CSCC-GPUs":"8","X-CSCC-Queue":"<queue>","X-CSCC-GPU-Model":"<gpu-model>"}'
```

不要把 token 寫進 repository、shell script、Markdown 或實驗結果。若平台提供的 header 名稱
與上例不同，以 Ray Jobs 頁面產生的指令為準。`X-CSCC-GPUs: 8` 代表最多建立 8 個 GPU
workers；依平台文件，每個 worker 配一張 GPU。

這些 GPU headers 只在建立新的 Ray session 時讀取。如果專案已經有 session 正在執行，先在
專案的 Ray 分頁確認可安全結束，再建立使用 8 GPU 上限的新 session。

## 7. 第一次只跑八 GPU smoke test

正式 F1/F2 很昂貴，而且目前尚欠 Managed Ray adapter。第一次提交只驗證 image、排程、
八個 GPU workers 與 shared storage：

```bash
mkdir -p /mnt/shared/spotserve/results/ray-smoke

ray job submit \
  --address "$RAY_ADDRESS" \
  --working-dir . \
  -- \
  python deploy/cscc-ray/smoke_ray_image.py \
    --expected-workers 8 \
    --output /mnt/shared/spotserve/results/ray-smoke/smoke.json
```

這個程式會建立 8 個各自要求一張 GPU 的 Ray actors，並保持 actors 同時存活直到全部資料
收集完成，因此可以迫使 autoscaler 要求八個 GPU workers。成功條件是：

- 8 個 actor 都能 import Ray、Torch、vLLM、NIXL、ServerlessLLM。
- Ray 為 2.58.0、Torch 為 2.9.0、vLLM 為 0.11.2。
- 每個 actor 都能看到 CUDA GPU。
- 取得 8 個不同 Ray Node IDs，符合平台「一張 GPU 一個 worker」的配置。
- JSON 結果成功寫入 shared storage。

查看 Job 狀態與 log：

```bash
ray job list --address "$RAY_ADDRESS"
ray job status --address "$RAY_ADDRESS" <job-id>
ray job logs --address "$RAY_ADDRESS" <job-id>
```

若需停止仍在執行的測試：

```bash
ray job stop --address "$RAY_ADDRESS" <job-id>
```

不要在失敗後立刻重跑八 GPU job。先依 log 判定是 image、quota、queue、storage、network
或程式問題；變更對應設定後才進行下一次最小測試。

第一次建立叢集通常需要約 30–60 秒；若平台明確回 `503` 並附 `Retry-After`，等指定時間後
重送同一個 smoke job 一次。若持續數分鐘仍為 `503`，停止重試並交由管理員查看叢集啟動
原因。

## 8. 正式單次 F1/F2

正式 protocol 固定使用同一個 Qwen1.5-MoE checkpoint、同一份 8192-input／2048-output
workload 與 `8 → 7 → 5 → 6 → 7 → 8` logical capacity trace。F1 執行 Rerouting、
Reparallelization、Original SpotServe、MoE-SpotServe；F2 對 MoE-aware Reparallelization 與
MoE-aware Migration 做 2×2 ablation。F1 Original 與 MoE-SpotServe 分別等同 F2 的
off/off 與 on/on，所以 F2 引用這兩列，只需 6 次唯一 formal GPU runs。

完整命令、輸出與解讀限制見 [FORMAL_F1_F2.md](FORMAL_F1_F2.md)。

## 9. 常見失敗與最小診斷

| 現象 | 先查什麼 | 下一個最小實驗 |
|---|---|---|
| Build 找不到 Dockerfile | 壓縮檔根目錄是否有 `Dockerfile` | 只重新執行 packaging script 並列出 archive |
| Ray client/server 版本錯誤 | client 是否真為 2.58.0 | 新建乾淨 venv，只安裝 Ray 2.58.0 |
| Job 一直 pending | Project/個人 quota、queue 是否有 8 張同型 GPU | 將 smoke 暫改 1 worker，先驗證基本排程 |
| 只有少於 8 個 Node IDs | worker 是否一 GPU 一 pod、autoscaler 上限 | 保持 8 actors 同時存活，請管理員查 autoscaler |
| worker import 失敗 | head/worker 是否真的用了同一 image digest | 只跑一個 GPU actor 並輸出 package versions |
| shared result 不見 | 路徑是否所有 workers/head 共掛載且可寫 | 用單一 Ray task 寫一個小文字檔 |
| NIXL 連不到另一 worker | pod 網路、port、advertised IP 是否可達 | 只傳一個最小 KV block，不跑完整模型 |
| GPU OOM | 每 worker 可見 GPU、model path、parallel config | 單一 actor 只載入模型，不送 request |

## 10. 官方平台文件

- [Workspaces](https://nthu-cscc.github.io/CSCC_AI_Platform_Documentation/zh/user-guide/workspaces/)
- [Ray Jobs](https://nthu-cscc.github.io/CSCC_AI_Platform_Documentation/zh/user-guide/ray/)
- [Projects 與 Build](https://nthu-cscc.github.io/CSCC_AI_Platform_Documentation/zh/user-guide/projects/)
