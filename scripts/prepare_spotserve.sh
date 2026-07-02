#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_spotserve.sh [options]

Build/recreate the SpotServe head container, copy benchmark artifacts, and
deploy the selected SpotServe policy models.

Options:
  --skip-build       Do not run "docker compose build sllm_head"
  --skip-recreate    Do not recreate the sllm_head container
  --skip-deploy      Copy artifacts but do not deploy models
  --deploy-set SET   Models to deploy: standard, correctness, vllm-dense,
                     vllm-moe, vllm-blackbox, or all.
                     Default: standard. "all" can require substantial GPU capacity.
  -h, --help         Show this help

Environment overrides:
  SPOTSERVE_CONTAINER        Container name. Default: sllm_head
  SPOTSERVE_COMPOSE_SERVICE  Compose service name. Default: sllm_head
  SPOTSERVE_COMPOSE_SERVICES Space-separated compose services to build/recreate
  SPOTSERVE_COMPOSE_BUILD_SERVICES
                            Space-separated compose services to build.
                            Default: sllm_head because head/worker share one image.
  SPOTSERVE_HOST_TMPDIR     Host temp dir for prepare logs.
                            Default: <repo>/.spotserve-tmp
  SPOTSERVE_WORKDIR          Container workdir. Default: /tmp/spotserve-work
  SPOTSERVE_HEALTH_URL       Container API health URL. Default: http://127.0.0.1:8343/health
  SPOTSERVE_WORKER_CONTAINER Worker container used by vllm-dense. Default: sllm_worker_0
  SPOTSERVE_VLLM_DENSE_MODEL_PATH
                            Optional container-local HF snapshot path to verify.
  SPOTSERVE_VLLM_MOE_MODEL  MoE HF model id or container-local path.
                            Default: Qwen/Qwen1.5-MoE-A2.7B
  SPOTSERVE_VLLM_MOE_LOAD_FORMAT
                            vLLM load_format for MoE direct load. Default: auto
  SPOTSERVE_VLLM_MOE_TP     MoE tensor_parallel_size override. Default: 1
  SPOTSERVE_VLLM_MOE_MODEL_PATH
                            Optional container-local MoE snapshot path to verify.
  MODEL_FOLDER               Host model directory mounted as /models. Default: <repo>/model
EOF
}

SKIP_BUILD=0
SKIP_RECREATE=0
SKIP_DEPLOY=0
DEPLOY_SET="${SPOTSERVE_DEPLOY_SET:-standard}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-recreate)
      SKIP_RECREATE=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --deploy-set)
      DEPLOY_SET="${2:-}"
      if [[ "$DEPLOY_SET" != "standard" && "$DEPLOY_SET" != "correctness" && "$DEPLOY_SET" != "vllm-dense" && "$DEPLOY_SET" != "vllm-moe" && "$DEPLOY_SET" != "vllm-blackbox" && "$DEPLOY_SET" != "all" ]]; then
        echo "--deploy-set must be one of: standard, correctness, vllm-dense, vllm-moe, vllm-blackbox, all" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${SPOTSERVE_CONTAINER:-sllm_head}"
COMPOSE_SERVICE="${SPOTSERVE_COMPOSE_SERVICE:-sllm_head}"
WORKER_CONTAINER="${SPOTSERVE_WORKER_CONTAINER:-sllm_worker_0}"
HOST_TMPDIR="${SPOTSERVE_HOST_TMPDIR:-${ROOT_DIR}/.spotserve-tmp}"
WORKDIR_IN_CONTAINER="${SPOTSERVE_WORKDIR:-/tmp/spotserve-work}"
HEALTH_URL="${SPOTSERVE_HEALTH_URL:-http://127.0.0.1:8343/health}"
MODELS_URL="${SPOTSERVE_MODELS_URL:-http://127.0.0.1:8343/v1/models}"
VLLM_DENSE_MODEL_PATH="${SPOTSERVE_VLLM_DENSE_MODEL_PATH:-}"
VLLM_MOE_MODEL="${SPOTSERVE_VLLM_MOE_MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
VLLM_MOE_LOAD_FORMAT="${SPOTSERVE_VLLM_MOE_LOAD_FORMAT:-auto}"
VLLM_MOE_TP="${SPOTSERVE_VLLM_MOE_TP:-1}"
VLLM_MOE_MODEL_PATH="${SPOTSERVE_VLLM_MOE_MODEL_PATH:-}"
HEAD_PYTHON="/opt/venvs/head/bin/python"
HEAD_SLLM="/opt/venvs/head/bin/sllm"
HEAD_RAY="/opt/venvs/head/bin/ray"

MODEL_FOLDER="${MODEL_FOLDER:-${ROOT_DIR}/model}"
export MODEL_FOLDER

if [[ -n "${SPOTSERVE_COMPOSE_SERVICES:-}" ]]; then
  read -r -a COMPOSE_SERVICES <<<"$SPOTSERVE_COMPOSE_SERVICES"
elif [[ "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  COMPOSE_SERVICES=("$COMPOSE_SERVICE" "sllm_worker_0")
else
  COMPOSE_SERVICES=("$COMPOSE_SERVICE")
fi

if [[ -n "${SPOTSERVE_COMPOSE_BUILD_SERVICES:-}" ]]; then
  read -r -a COMPOSE_BUILD_SERVICES <<<"$SPOTSERVE_COMPOSE_BUILD_SERVICES"
else
  COMPOSE_BUILD_SERVICES=("$COMPOSE_SERVICE")
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

cd "$ROOT_DIR"
mkdir -p "$HOST_TMPDIR"
RAY_STATUS_LOG="${HOST_TMPDIR}/spotserve-ray-status.log"
VLLM_RESOURCES_LOG="${HOST_TMPDIR}/spotserve-vllm-resources.log"
HEALTH_LOG="${HOST_TMPDIR}/spotserve-health.log"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  log "Building ${COMPOSE_BUILD_SERVICES[*]}"
  docker compose build "${COMPOSE_BUILD_SERVICES[@]}"
fi

if [[ "$SKIP_RECREATE" -eq 0 ]]; then
  log "Recreating ${COMPOSE_SERVICES[*]}"
  docker compose up -d --force-recreate "${COMPOSE_SERVICES[@]}"
fi

log "Waiting for Ray in ${CONTAINER}"
for attempt in $(seq 1 60); do
  if podman exec "$CONTAINER" "$HEAD_RAY" status >"$RAY_STATUS_LOG" 2>&1 &&
      grep -q "Active:" "$RAY_STATUS_LOG" &&
      ! grep -q "No cluster status" "$RAY_STATUS_LOG"; then
    cat "$RAY_STATUS_LOG"
    break
  fi
  if [[ "$attempt" -eq 60 ]]; then
    cat "$RAY_STATUS_LOG" >&2 || true
    echo "Ray did not become ready in time" >&2
    exit 1
  fi
  sleep 2
done

if [[ "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  log "Checking vLLM worker resources"
  if [[ -n "$VLLM_DENSE_MODEL_PATH" ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${VLLM_DENSE_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM dense model is not visible in ${WORKER_CONTAINER}.
Expected: ${VLLM_DENSE_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_VLLM_DENSE_MODEL_PATH only when you have a container-local
Hugging Face snapshot path with config/tokenizer/weight files.
EOF
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ -n "$VLLM_MOE_MODEL_PATH" ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${VLLM_MOE_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM MoE model is not visible in ${WORKER_CONTAINER}.
Expected: ${VLLM_MOE_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_VLLM_MOE_MODEL_PATH only when you have a container-local
Hugging Face snapshot path with config/tokenizer/weight files.
EOF
        exit 1
      fi
      sleep 2
    done
  fi

  for attempt in $(seq 1 90); do
    if podman exec -i "$CONTAINER" "$HEAD_PYTHON" - >"$VLLM_RESOURCES_LOG" 2>&1 <<'PY'
import ray
import sys

ray.init(address="auto", namespace="sllm", ignore_reinit_error=True)
resources = ray.cluster_resources()
print("Ray cluster resources:", resources)
if resources.get("worker_node", 0) <= 0:
    print("No Ray worker_node resource is registered.", file=sys.stderr)
    sys.exit(1)
if resources.get("GPU", 0) <= 0:
    print("No Ray GPU resource is registered.", file=sys.stderr)
    sys.exit(1)
PY
    then
      cat "$VLLM_RESOURCES_LOG"
      break
    fi
    if [[ "$attempt" -eq 90 ]]; then
      cat "$VLLM_RESOURCES_LOG" >&2 || true
      cat >&2 <<'EOF'
Ray has no GPU worker available for vLLM deploy.
Start sllm_worker_0 on a node where the NVIDIA driver is available, then rerun:
  scripts/prepare_spotserve.sh --deploy-set vllm-dense
or:
  scripts/prepare_spotserve.sh --deploy-set vllm-moe
EOF
      exit 1
    fi
    sleep 2
  done
fi

log "Waiting for SLLM HTTP API at ${HEALTH_URL}"
for attempt in $(seq 1 90); do
  if podman exec -i "$CONTAINER" "$HEAD_PYTHON" - "$HEALTH_URL" "$MODELS_URL" >"$HEALTH_LOG" 2>&1 <<'PY'
import sys
from urllib import request

for url in sys.argv[1:]:
    with request.urlopen(url, timeout=2) as response:
        body = response.read().decode("utf-8")
        if response.status != 200:
            raise SystemExit(f"{url}: unexpected HTTP status {response.status}")
        print(f"{url}: {body}")
PY
  then
    cat "$HEALTH_LOG"
    break
  fi
  if [[ "$attempt" -eq 90 ]]; then
    cat "$HEALTH_LOG" >&2 || true
    echo "SLLM HTTP API did not become ready in time" >&2
    exit 1
  fi
  sleep 2
done

log "Preparing ${WORKDIR_IN_CONTAINER}"
podman exec "$CONTAINER" mkdir -p \
  "${WORKDIR_IN_CONTAINER}/benchmarks/spotserve" \
  "${WORKDIR_IN_CONTAINER}/examples/spotserve" \
  "${WORKDIR_IN_CONTAINER}/scripts" \
  "${WORKDIR_IN_CONTAINER}/results"

log "Copying benchmark artifacts"
podman cp benchmarks/spotserve/. "${CONTAINER}:${WORKDIR_IN_CONTAINER}/benchmarks/spotserve"
podman cp examples/spotserve/. "${CONTAINER}:${WORKDIR_IN_CONTAINER}/examples/spotserve"
podman cp scripts/. "${CONTAINER}:${WORKDIR_IN_CONTAINER}/scripts"

if [[ "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  log "Applying vLLM MoE config overrides"
  podman exec -i "$CONTAINER" "$HEAD_PYTHON" - \
    "$WORKDIR_IN_CONTAINER" \
    "$VLLM_MOE_MODEL" \
    "$VLLM_MOE_LOAD_FORMAT" \
    "$VLLM_MOE_TP" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
moe_model = sys.argv[2]
load_format = sys.argv[3]
tensor_parallel_size = int(sys.argv[4])

for path in sorted((workdir / "examples" / "spotserve").glob("config-vllm-moe-*.json")):
    config = json.loads(path.read_text(encoding="utf-8"))
    config["num_gpus"] = tensor_parallel_size
    backend_config = config.setdefault("backend_config", {})
    backend_config["pretrained_model_name_or_path"] = moe_model
    backend_config["load_format"] = load_format
    backend_config["tensor_parallel_size"] = tensor_parallel_size
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path}: model={moe_model}, load_format={load_format}, tp={tensor_parallel_size}")
PY
fi

if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
  log "Deploying SpotServe policies (${DEPLOY_SET})"
  STANDARD_CONFIGS=(
    "examples/spotserve/config-dummy-none.json"
    "examples/spotserve/config-dummy-naive-retry.json"
    "examples/spotserve/config-dummy-token-replay.json"
  )
  CORRECTNESS_CONFIGS=(
    "examples/spotserve/config-dummy-correctness-none.json"
    "examples/spotserve/config-dummy-correctness-naive-retry.json"
    "examples/spotserve/config-dummy-correctness-token-replay.json"
  )
  VLLM_DENSE_CONFIGS=(
    "examples/spotserve/config-vllm-dense-baseline.json"
    "examples/spotserve/config-vllm-dense-none.json"
    "examples/spotserve/config-vllm-dense-naive-retry.json"
    "examples/spotserve/config-vllm-dense-token-replay.json"
  )
  VLLM_MOE_CONFIGS=(
    "examples/spotserve/config-vllm-moe-baseline.json"
    "examples/spotserve/config-vllm-moe-none.json"
    "examples/spotserve/config-vllm-moe-naive-retry.json"
    "examples/spotserve/config-vllm-moe-token-replay.json"
  )
  DEPLOY_CONFIGS=()
  if [[ "$DEPLOY_SET" == "standard" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${STANDARD_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "correctness" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${CORRECTNESS_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${VLLM_DENSE_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${VLLM_MOE_CONFIGS[@]}")
  fi

  podman exec "$CONTAINER" bash -lc "
    set -euo pipefail
    cd '$WORKDIR_IN_CONTAINER'
    for config in ${DEPLOY_CONFIGS[*]}
    do
      for attempt in \$(seq 1 30); do
        if '$HEAD_SLLM' deploy --config \"\$config\"; then
          break
        fi
        if [[ \"\$attempt\" -eq 30 ]]; then
          echo \"Failed to deploy \$config after \$attempt attempts\" >&2
          exit 1
        fi
        sleep 2
      done
    done
  "

  log "Current model status"
  podman exec "$CONTAINER" bash -lc "'$HEAD_SLLM' status"
fi

log "SpotServe environment is ready"
if [[ "$DEPLOY_SET" == "standard" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the long benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_long.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 30 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "correctness" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the recovery-correctness benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 30 \\
  --skip-trace
'
EOF
fi

if [[ "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the vLLM dense black-box benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 120 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the vLLM MoE black-box benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 180 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the dense vs MoE black-box benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 180 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi
