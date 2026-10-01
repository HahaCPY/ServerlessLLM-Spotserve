#!/usr/bin/env bash

# Start one control-plane head or one GPU worker on a physical host.  This
# script never builds an image; every host must use the same prebuilt digest.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy_spotserve_cross_host.sh head \
    --node-ip IP [--image IMAGE] [--repo-root PATH]

  deploy_spotserve_cross_host.sh worker \
    --head-address HEAD_IP:6379 --node-ip IP --worker-id ID \
    --physical-host-id HOST_ID --model-folder PATH \
    [--gpu-device DEVICE] [--image IMAGE]

  deploy_spotserve_cross_host.sh stop --container NAME

Use one CPU/control host for `head` and two distinct physical GPU hosts for
workers 0 and 1.  Each worker should expose exactly one GPU to this cluster.
EOF
}

ROLE="${1:-}"
if [[ -z "$ROLE" ]]; then
  usage
  exit 2
fi
shift

IMAGE="docker.io/serverlessllm/sllm:latest"
CONTAINER=""
NODE_IP=""
HEAD_ADDRESS=""
WORKER_ID=""
PHYSICAL_HOST_ID=""
MODEL_FOLDER=""
GPU_DEVICE="0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-/tmp/sllm-hf-cache-rootless}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --node-ip) NODE_IP="$2"; shift 2 ;;
    --head-address) HEAD_ADDRESS="$2"; shift 2 ;;
    --worker-id) WORKER_ID="$2"; shift 2 ;;
    --physical-host-id) PHYSICAL_HOST_ID="$2"; shift 2 ;;
    --model-folder) MODEL_FOLDER="$2"; shift 2 ;;
    --gpu-device) GPU_DEVICE="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$HF_CACHE_DIR"

case "$ROLE" in
  head)
    [[ -n "$NODE_IP" ]] || { echo "--node-ip is required" >&2; exit 2; }
    [[ -d "$REPO_ROOT" ]] || { echo "Repo not found: $REPO_ROOT" >&2; exit 2; }
    CONTAINER="${CONTAINER:-sllm_cross_host_head}"
    mkdir -p /tmp/sllm-cross-host-ray/head
    exec podman run --replace --name "$CONTAINER" --detach \
      --network host --shm-size 10g --pids-limit 32768 \
      -e MODE=HEAD \
      -e RAY_NODE_IP="$NODE_IP" \
      -e RAY_PORT=6379 \
      -e RAY_ADDRESS=127.0.0.1:6379 \
      -e RAY_NUM_CPUS=4 \
      -e RAY_TEMP_DIR=/raytmp/head \
      -e RAY_START_EXTRA_ARGS="--node-manager-port=6700 --object-manager-port=6701 --min-worker-port=10002 --max-worker-port=10100 --dashboard-host=0.0.0.0" \
      -e HF_HOME=/hf-cache \
      -e HF_HUB_DISABLE_XET=1 \
      -v "$REPO_ROOT:/workspace/ServerlessLLM-Spotserve:ro" \
      -v /tmp/sllm-cross-host-ray/head:/raytmp/head \
      -v "$HF_CACHE_DIR:/hf-cache" \
      "$IMAGE"
    ;;
  worker)
    [[ -n "$NODE_IP" ]] || { echo "--node-ip is required" >&2; exit 2; }
    [[ -n "$HEAD_ADDRESS" ]] || { echo "--head-address is required" >&2; exit 2; }
    [[ -n "$WORKER_ID" ]] || { echo "--worker-id is required" >&2; exit 2; }
    [[ -n "$PHYSICAL_HOST_ID" ]] || {
      echo "--physical-host-id is required" >&2
      exit 2
    }
    [[ -d "$MODEL_FOLDER" ]] || {
      echo "Model folder not found: $MODEL_FOLDER" >&2
      exit 2
    }
    CONTAINER="${CONTAINER:-sllm_cross_host_worker_${WORKER_ID}}"
    RAY_DIR="/tmp/sllm-cross-host-ray/worker-${WORKER_ID}"
    mkdir -p "$RAY_DIR"
    exec podman run --replace --name "$CONTAINER" --detach \
      --network host --shm-size 10g --pids-limit 32768 \
      --group-add keep-groups \
      --device "nvidia.com/gpu=${GPU_DEVICE}" \
      -e MODE=WORKER \
      -e WORKER_ID="$WORKER_ID" \
      -e SPOTSERVE_PHYSICAL_HOST_ID="$PHYSICAL_HOST_ID" \
      -e RAY_HEAD_ADDRESS="$HEAD_ADDRESS" \
      -e RAY_NODE_IP="$NODE_IP" \
      -e RAY_TEMP_DIR=/raytmp/worker \
      -e RAY_START_EXTRA_ARGS="--node-manager-port=6700 --object-manager-port=6701 --min-worker-port=10002 --max-worker-port=10100" \
      -e STORAGE_PATH=/models \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      -e VLLM_SPOTSERVE_EXPERT_REMAP=1 \
      -e VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP=1 \
      -e VLLM_SPOTSERVE_A2A_TRACE=1 \
      -e HF_HOME=/hf-cache \
      -e HF_HUB_DISABLE_XET=1 \
      -v "$MODEL_FOLDER:/models:ro" \
      -v "$RAY_DIR:/raytmp/worker" \
      -v "$HF_CACHE_DIR:/hf-cache" \
      "$IMAGE" \
      --mem-pool-size 4GB --registration-required true
    ;;
  stop)
    [[ -n "$CONTAINER" ]] || { echo "--container is required" >&2; exit 2; }
    exec podman stop "$CONTAINER"
    ;;
  *)
    echo "Role must be head, worker, or stop" >&2
    usage
    exit 2
    ;;
esac
