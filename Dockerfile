# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

# Adapted from https://github.com/vllm-project/vllm/blob/23c1b10a4c8cd77c5b13afa9242d67ffd055296b/Dockerfile
ARG CUDA_VERSION=12.8.1
ARG PYTORCH_VERSION=2.9.0
ARG TORCHVISION_VERSION=0.24.0
ARG TORCHAUDIO_VERSION=2.9.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder
ARG CUDA_VERSION
ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX_URL
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p ${CONDA_DIR} \
    && rm /tmp/mf.sh
ENV PATH=${CONDA_DIR}/bin:$PATH
SHELL ["/bin/bash", "-lc"]
RUN conda create -y -n build python=${PYTHON_VERSION} pip setuptools wheel \
    && conda run -n build python -V \
    && conda run -n build pip -V
# Set the working directory
WORKDIR /app

# Build checkpoint store
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
COPY sllm_store/requirements-build.txt /app/sllm_store/requirements-build.txt
RUN cd sllm_store && \
  conda run -n build python -m pip install --index-url ${PYTORCH_INDEX_URL} torch==${PYTORCH_VERSION} && \
  conda run -n build python -m pip install -r requirements-build.txt && \
  conda run -n build python -m pip install setuptools wheel

COPY sllm_store/cmake /app/sllm_store/cmake
COPY sllm_store/CMakeLists.txt /app/sllm_store/CMakeLists.txt
COPY sllm_store/csrc /app/sllm_store/csrc
COPY sllm_store/sllm_store /app/sllm_store/sllm_store
COPY sllm_store/setup.py /app/sllm_store/setup.py
COPY sllm_store/pyproject.toml /app/sllm_store/pyproject.toml
COPY sllm_store/MANIFEST.in /app/sllm_store/MANIFEST.in
COPY sllm_store/requirements.txt /app/sllm_store/requirements.txt
COPY sllm_store/README.md /app/sllm_store/README.md
COPY sllm_store/proto/storage.proto /app/sllm_store/proto/storage.proto
RUN cd sllm_store && conda run -n build python setup.py bdist_wheel

COPY requirements.txt requirements-worker.txt /app/
COPY pyproject.toml setup.py py.typed /app/
COPY sllm/backends /app/sllm/backends
COPY sllm/ft_backends /app/sllm/ft_backends
COPY sllm/cli /app/sllm/cli
COPY sllm/routers /app/sllm/routers
COPY sllm/schedulers /app/sllm/schedulers
COPY sllm/spot /app/sllm/spot
COPY sllm/*.py /app/sllm/
COPY README.md /app/
RUN conda run -n build python setup.py bdist_wheel

# Stage 2: Runner with lightweight virtual environments
FROM docker.io/pytorch/pytorch:${PYTORCH_VERSION}-cuda12.8-cudnn9-devel
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG PYTORCH_INDEX_URL

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# The PyTorch runtime image already provides torch 2.9.0+cu128 in base.
# Use venvs that can see base site-packages so head/worker do not duplicate
# the full CUDA PyTorch stack.
RUN python -m venv --system-site-packages /opt/venvs/head && \
    python -m venv --system-site-packages /opt/venvs/worker

RUN /opt/venvs/head/bin/python -m pip install -U pip && \
    /opt/venvs/worker/bin/python -m pip install -U pip

RUN /opt/venvs/worker/bin/python -m pip install --index-url ${PYTORCH_INDEX_URL} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION}

# Copy requirements files
COPY requirements.txt /app/

RUN grep -v -E '^serverless-llm-store([<>= ].*)?$' /app/requirements.txt > /tmp/requirements-head.txt && \
    /opt/venvs/head/bin/python -m pip install -r /tmp/requirements-head.txt
COPY requirements-worker.txt /app/

RUN /opt/venvs/worker/bin/python -m pip install -r /app/requirements-worker.txt
RUN /opt/venvs/worker/bin/python -c "import torch; flags = torch._C._cuda_getArchFlags().split(); print(torch.__version__, torch.version.cuda, flags); assert 'sm_120' in flags"

# Copy vllm patch for worker
COPY sllm_store/vllm_patch /app/vllm_patch

# Copy the built wheels from the builder
COPY --from=builder /app/sllm_store/dist /app/sllm_store/dist
COPY --from=builder /app/dist /app/dist

# Install packages in head environment
RUN /opt/venvs/head/bin/python -m pip install /app/sllm_store/dist/*.whl && \
    /opt/venvs/head/bin/python -m pip install /app/dist/*.whl

# Install packages in worker environment
RUN /opt/venvs/worker/bin/python -m pip install /app/sllm_store/dist/*.whl && \
    /opt/venvs/worker/bin/python -m pip install --no-deps /app/dist/*.whl
RUN /opt/venvs/head/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)" && \
    /opt/venvs/worker/bin/python -c "import torch; from importlib.metadata import version; from packaging.version import Version; flags = torch._C._cuda_getArchFlags().split(); print(torch.__version__, torch.version.cuda, flags); print('worker deps', version('vllm'), version('openai'), version('pydantic'), version('starlette')); assert 'sm_120' in flags; assert Version(version('openai')) >= Version('1.99.1'); assert Version(version('pydantic')) >= Version('2.12.0'); assert Version(version('starlette')) >= Version('1.0.0')"

# Apply vLLM patch in worker environment
RUN bash -c "source /opt/venvs/worker/bin/activate && cd /app && ./vllm_patch/patch.sh"

# Copy the entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint directly to the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
