# Environment Fix Change Record

Generated: 2026-06-07, Asia/Taipei

This records the current environment-related repository changes compared with the version before the `environment-fix` commit.

## Compare Range

- Base: `9f50241` (`fix: release pipeline (#309)`)
- Current: `73d5940` (`environment-fix`)
- Status before creating this note: clean working tree

## Changed Files

### `Dockerfile`

- Qualified Docker Hub image names with `docker.io/`.
- This avoids Podman short-name resolution failures such as:

```text
short-name "nvidia/cuda:12.1.1-devel-ubuntu20.04" did not resolve to an alias
```

Changed image references:

```text
nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
-> docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
-> docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
```

### `docker-compose.yml`

- Added a root-level compose file.
- Uses `build.context: .` because this file is placed at the repository root.
- Uses `docker.io/serverlessllm/sllm:latest` to avoid Podman short-name resolution.
- Adds resource/thread limits that helped keep `sllm_head` running under rootless Podman:
  - `RAY_NUM_CPUS=4`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
  - `RAYON_NUM_THREADS=1`
  - `TOKENIZERS_PARALLELISM=false`
  - `shm_size: "10gb"`
  - `pids_limit: 8192`
- Keeps the public ports:
  - `6379:6379`
  - `8343:8343`
- Keeps worker GPU assignment to GPU `0`.

## Runtime Notes

These changes fixed the earlier setup issues:

- `Dockerfile not found in /work/containers`
- Podman short-name image resolution errors
- `sllm_head` exiting from thread/resource pressure

Current remaining issue observed later:

- The RTX 5070 Ti is `sm_120`, but the running worker environment had `torch 2.7.0+cu126`, whose CUDA arch list only went up to `sm_90`.
- The chat completion request hung because the backend hit:

```text
CUDA error: no kernel image is available for execution on the device
```

That remaining issue requires a newer CUDA/PyTorch image or wheel with `sm_120` support, such as a CUDA 12.8+ compatible PyTorch build.

## Follow-Up `sm_120` Fix

The repository was updated to target RTX 50-series / Blackwell GPUs such as the RTX 5070 Ti:

- `Dockerfile`
  - Changed CUDA build base from CUDA 12.1.1 / Ubuntu 20.04 to CUDA 12.8.1 / Ubuntu 22.04.
  - Changed runtime base from `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` to `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`.
  - Changed the build Python from 3.10 to 3.11 so the locally built `sllm_store` extension wheel matches the PyTorch 2.9.0 runtime image, whose base Python is 3.11.
  - Added `12.0` to `TORCH_CUDA_ARCH_LIST`.
  - Uses lightweight Python virtual environments with `--system-site-packages` for head and worker. This lets both venvs reuse the runtime image's base `torch 2.9.0+cu128` instead of installing duplicate CUDA PyTorch stacks.
  - Installs worker `torchvision==0.24.0` and `torchaudio==2.9.0` from the CUDA 12.8 wheel index.
  - Skips the PyPI `serverless-llm-store` dependency during the early head requirements install, then installs the locally built `sllm_store` wheel later. This avoids pulling the old PyPI package that forced `torch 2.7.0/cu126`.
  - Adds `PIP_NO_CACHE_DIR=1` in both build and runtime stages to reduce Podman image/layer bloat.
  - Adds build-time checks based on `torch._C._cuda_getArchFlags()` and fails if the compiled PyTorch arch flags do not contain `sm_120`.
- `entrypoint.sh`
  - Activates `/opt/venvs/head` or `/opt/venvs/worker` instead of conda envs.
- `requirements-worker.txt`
  - Pinned `torch==2.9.0`, `torchvision==0.24.0`, and `torchaudio==2.9.0` to match `vllm==0.11.2`.
- `sllm_store/CMakeLists.txt`
  - Added `12.0` to `CUDA_SUPPORTED_ARCHS`.
- `sllm_store/requirements.txt`
- `sllm_store/pyproject.toml`
  - Raised the local `sllm_store` torch requirement from `torch>=2.7.0` to `torch>=2.9.0`.
- `sllm_store/requirements-build.txt`
  - Changed build-time torch from `2.9.1` to `2.9.0` to align with runtime and vLLM.
- `.github/workflows/publish.yml`
- `.github/workflows/test_sllm_store.yaml`
  - Added `12.0` to `TORCH_CUDA_ARCH_LIST`.

### Validation Notes

- A full image build completed successfully once with the CUDA 12.8 / PyTorch 2.9.0 changes.
- The worker build-time arch check printed:

```text
2.9.0+cu128 12.8 ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

- The final verification after installing local `sllm_store` and `serverless-llm` wheels also kept worker PyTorch at:

```text
2.9.0+cu128 12.8 ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

- The same successful build also applied the local vLLM patch successfully.
- A later final rebuild with duplicated head and worker CUDA PyTorch envs failed at:

```text
write /opt/conda/envs/worker/lib/python3.10/site-packages/vllm/_C.abi3.so: no space left on device
```

- To fix the image size problem, the current Dockerfile now uses Python 3.11 venvs with `--system-site-packages` so head and worker share the base image's `torch 2.9.0+cu128`.
- The venv-based final image rebuild succeeded and produced:

```text
Successfully tagged docker.io/serverlessllm/sllm:latest
9b3fd7e3d54ee8ccf809f491a34b35c85f1eaeff046e55e02dfe829e74f01302
```

- During the first venv-based rebuild, installing the local `serverless-llm` wheel into the worker environment downgraded vLLM's runtime dependencies:

```text
vllm 0.11.2 requires openai>=1.99.1, but you have openai 1.52.0
vllm 0.11.2 requires pydantic>=2.12.0, but you have pydantic 2.11.5
```

- The worker local `serverless-llm` wheel install was changed to `pip install --no-deps /app/dist/*.whl`. This keeps vLLM's dependency set intact while still installing the local ServerlessLLM package.
- The final Dockerfile validation now checks both `sm_120` and the worker vLLM dependency versions. The successful rebuild printed:

```text
2.9.0+cu128 12.8 ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
worker deps 0.11.2 2.41.0 2.13.4 1.2.1
```

- The vLLM patch step also completed successfully after the dependency fix:

```text
Patch applied successfully.
```

## Runtime Validation After Recreate

- After the dependency and vLLM compatibility fixes, the image was rebuilt again:

```text
Successfully tagged docker.io/serverlessllm/sllm:latest
beede9e250ac2e96775040e1378ef145d28883ab0ec6397ed947d0396c45b2a7
```

- A `vllm==0.11.2` API compatibility issue was found during the first API test:

```text
ImportError: cannot import name 'Counter' from 'vllm.utils'
```

- Even `--backend transformers` hit this because `sllm.backends.__init__` imports `vllm_backend` unconditionally. `sllm/backends/vllm_backend.py` now falls back to a local `itertools.count`-based `Counter` when `vllm.utils.Counter` is unavailable.
- The compose stack was recreated with:

```text
MODEL_FOLDER=/work/containers/cpy/ServerlessLLM-Spotserve/model docker compose up -d --force-recreate
```

- Container status after recreate:

```text
sllm_head      Up, ports 6379 and 8343 published
sllm_worker_0  Up
```

- Worker GPU validation inside the container:

```text
2.9.0+cu128 12.8
NVIDIA GeForce RTX 5070 Ti
(12, 0)
tensor(..., device='cuda:0')
```

- Worker import validation:

```text
['DummyBackend', 'VllmBackend', 'TransformersBackend']
```

- Model deploy validation:

```text
[SUCCESS] Model 'Qwen/Qwen3-0.6B' deployed successfully.
```

- Chat completions API validation succeeded against `http://127.0.0.1:8343/v1/chat/completions` with `max_tokens: 32`. The response returned a normal `chat.completion` JSON object for `Qwen/Qwen3-0.6B`.
- The runtime deploy created a local `model/` directory under the repository root because `MODEL_FOLDER` was set to `/work/containers/cpy/ServerlessLLM-Spotserve/model`. This is a downloaded model/cache artifact, not a source-code change.

## Full Diff

```diff
commit 73d59407d6f33671da62203ea2e5ea8f961de91d
Author: Chen Pin-yun <hahalearningwithme@gmail.com>
Date:   Sun Jun 7 01:29:03 2026 +0800

    environment-fix

diff --git a/Dockerfile b/Dockerfile
index 91edaf0..dc53203 100644
--- a/Dockerfile
+++ b/Dockerfile
@@ -20,7 +20,7 @@
 ARG CUDA_VERSION=12.1.1
 #################### BASE BUILD IMAGE ####################
 # prepare basic build environment
-FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder
+FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder
 ARG CUDA_VERSION=12.1.1
 ARG PYTHON_VERSION=3.10
 ARG TARGETPLATFORM
@@ -76,7 +76,7 @@ COPY README.md /app/
 RUN conda run -n build python setup.py bdist_wheel
 
 # Stage 2: Runner with conda environments
-FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
+FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
 
 # Set non-interactive installation
 ENV DEBIAN_FRONTEND=noninteractive \
diff --git a/docker-compose.yml b/docker-compose.yml
new file mode 100644
index 0000000..ae51778
--- /dev/null
+++ b/docker-compose.yml
@@ -0,0 +1,63 @@
+services:
+  # Head Node
+  sllm_head:
+    build:
+      context: .
+      dockerfile: Dockerfile
+    image: docker.io/serverlessllm/sllm:latest
+    container_name: sllm_head
+    environment:
+      - MODEL_FOLDER=${MODEL_FOLDER}
+      - MODE=HEAD
+      - RAY_NUM_CPUS=4
+      - OMP_NUM_THREADS=1
+      - MKL_NUM_THREADS=1
+      - OPENBLAS_NUM_THREADS=1
+      - NUMEXPR_NUM_THREADS=1
+      - RAYON_NUM_THREADS=1
+      - TOKENIZERS_PARALLELISM=false
+    ports:
+      - "6379:6379"    # Redis port
+      - "8343:8343"    # ServerlessLLM port
+    shm_size: "10gb"
+    pids_limit: 8192
+    networks:
+      - sllm_network
+    command: []
+
+  # Worker Node 0
+  sllm_worker_0:
+    build:
+      context: .
+      dockerfile: Dockerfile
+    image: docker.io/serverlessllm/sllm:latest
+    container_name: sllm_worker_0
+    deploy:
+      resources:
+        reservations:
+          devices:
+            - driver: nvidia
+              capabilities: ["gpu"]
+              device_ids: ["0"] # Assigns GPU 0 to the worker
+    environment:
+      - WORKER_ID=0
+      - STORAGE_PATH=/models
+      - MODE=WORKER
+      - OMP_NUM_THREADS=1
+      - MKL_NUM_THREADS=1
+      - OPENBLAS_NUM_THREADS=1
+      - NUMEXPR_NUM_THREADS=1
+      - RAYON_NUM_THREADS=1
+      - TOKENIZERS_PARALLELISM=false
+    shm_size: "10gb"
+    pids_limit: 8192
+    networks:
+      - sllm_network
+    volumes:
+      - ${MODEL_FOLDER}:/models
+    command: ["--mem-pool-size", "4GB", "--registration-required", "true"] # Customize the memory pool size here
+
+networks:
+  sllm_network:
+    driver: bridge
+    name: sllm
```
