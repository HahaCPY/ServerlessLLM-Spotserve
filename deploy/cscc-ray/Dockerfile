# syntax=docker/dockerfile:1.7

ARG RAY_BASE_IMAGE=docker.io/rayproject/ray:2.58.0-py311-gpu
ARG PYTORCH_BUILD_IMAGE=docker.io/pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

FROM ${PYTORCH_BUILD_IMAGE} AS wheel-builder

ARG RAY_VERSION=2.58.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build patch pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN python -m pip install --upgrade pip setuptools wheel


COPY sllm_store/requirements-build.txt /src/sllm_store/requirements-build.txt
RUN python -m pip install -r /src/sllm_store/requirements-build.txt
COPY sllm_store/cmake /src/sllm_store/cmake
COPY sllm_store/csrc /src/sllm_store/csrc
COPY sllm_store/sllm_store /src/sllm_store/sllm_store
COPY sllm_store/proto /src/sllm_store/proto
COPY sllm_store/CMakeLists.txt \
     sllm_store/MANIFEST.in \
     sllm_store/README.md \
     sllm_store/pyproject.toml \
     sllm_store/requirements.txt \
     sllm_store/setup.py \
     /src/sllm_store/
RUN cd /src/sllm_store && python setup.py bdist_wheel

COPY deploy/cscc-ray/requirements-serverless.txt /src/requirements.txt
COPY deploy/cscc-ray/requirements-worker.txt /src/requirements-worker.txt
COPY pyproject.toml setup.py py.typed README.md /src/
COPY sllm/backends /src/sllm/backends
COPY sllm/cli /src/sllm/cli
COPY sllm/ft_backends /src/sllm/ft_backends
COPY sllm/routers /src/sllm/routers
COPY sllm/schedulers /src/sllm/schedulers
COPY sllm/spot /src/sllm/spot
COPY sllm/*.py /src/sllm/
RUN python setup.py bdist_wheel

FROM ${RAY_BASE_IMAGE} AS runtime

ARG RAY_VERSION=2.58.0
ARG PYTORCH_VERSION=2.9.0
ARG TORCHVISION_VERSION=0.24.0
ARG TORCHAUDIO_VERSION=2.9.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

LABEL org.opencontainers.image.title="ServerlessLLM SpotServe for CSCC Ray" \
      org.opencontainers.image.description="Ray 2.58.0 + PyTorch/vLLM/NIXL + SpotServe F1/F2 runtime" \
      org.opencontainers.image.version="ray-${RAY_VERSION}-torch-${PYTORCH_VERSION}-vllm-0.11.2"

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    RAY_DEDUP_LOGS=0 \
    SPOTSERVE_EXPECTED_RAY_VERSION=2.58.0 \
    SPOTSERVE_EXPECTED_TORCH_VERSION=2.9.0 \
    SPOTSERVE_EXPECTED_VLLM_VERSION=0.11.2

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates git libaio1 libibverbs1 libnuma1 librdmacm1 \
        patch procps wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/cscc-ray/requirements-runtime.txt /tmp/requirements-runtime.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" && \
    python -m pip install -r /tmp/requirements-runtime.txt && \
    python -m pip install "ray[default]==${RAY_VERSION}"

COPY --from=wheel-builder /src/sllm_store/dist/*.whl /tmp/wheels/
COPY --from=wheel-builder /src/dist/*.whl /tmp/wheels/
RUN python -m pip install --no-deps /tmp/wheels/*.whl

COPY sllm_store/vllm_patch /app/vllm_patch
RUN /app/vllm_patch/patch.sh && /app/vllm_patch/check_patch.sh


COPY benchmarks/spotserve/run_benchmark.py /app/benchmarks/spotserve/run_benchmark.py
COPY benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
     /app/benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json
COPY benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu_trace_plan.json \
     /app/benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu_trace_plan.json
COPY scripts/analyze_spotserve_benchmark.py \
     scripts/plot_spotserve_benchmark.py \
     scripts/run_k8s_moe_f1_f2.py \
     /app/scripts/
COPY deploy/cscc-ray/verify_image.py \
     deploy/cscc-ray/smoke_ray_image.py \
     deploy/cscc-ray/run_formal_f1_f2.py \
     /app/deploy/cscc-ray/

RUN python -m pip check && \
    python /app/deploy/cscc-ray/verify_image.py --build-time && \
    mkdir -p /app/results && chown -R ray:users /app

USER ray
WORKDIR /app
