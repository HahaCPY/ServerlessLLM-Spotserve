#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_spotserve.sh [options]

Build/recreate the SpotServe head container, copy benchmark artifacts, and
deploy the dummy policy models.

Options:
  --skip-build       Do not run "docker compose build sllm_head"
  --skip-recreate    Do not recreate the sllm_head container
  --skip-deploy      Copy artifacts but do not deploy dummy models
  -h, --help         Show this help

Environment overrides:
  SPOTSERVE_CONTAINER        Container name. Default: sllm_head
  SPOTSERVE_COMPOSE_SERVICE  Compose service name. Default: sllm_head
  SPOTSERVE_WORKDIR          Container workdir. Default: /tmp/spotserve-work
  SPOTSERVE_HEALTH_URL       Container API health URL. Default: http://127.0.0.1:8343/health
EOF
}

SKIP_BUILD=0
SKIP_RECREATE=0
SKIP_DEPLOY=0

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
WORKDIR_IN_CONTAINER="${SPOTSERVE_WORKDIR:-/tmp/spotserve-work}"
HEALTH_URL="${SPOTSERVE_HEALTH_URL:-http://127.0.0.1:8343/health}"
MODELS_URL="${SPOTSERVE_MODELS_URL:-http://127.0.0.1:8343/v1/models}"
HEAD_PYTHON="/opt/venvs/head/bin/python"
HEAD_SLLM="/opt/venvs/head/bin/sllm"
HEAD_RAY="/opt/venvs/head/bin/ray"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

cd "$ROOT_DIR"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  log "Building ${COMPOSE_SERVICE}"
  docker compose build "$COMPOSE_SERVICE"
fi

if [[ "$SKIP_RECREATE" -eq 0 ]]; then
  log "Recreating ${COMPOSE_SERVICE}"
  docker compose up -d --force-recreate "$COMPOSE_SERVICE"
fi

log "Waiting for Ray in ${CONTAINER}"
for attempt in $(seq 1 60); do
  if podman exec "$CONTAINER" "$HEAD_RAY" status >/tmp/spotserve-ray-status.log 2>&1 &&
      grep -q "Active:" /tmp/spotserve-ray-status.log &&
      ! grep -q "No cluster status" /tmp/spotserve-ray-status.log; then
    cat /tmp/spotserve-ray-status.log
    break
  fi
  if [[ "$attempt" -eq 60 ]]; then
    cat /tmp/spotserve-ray-status.log >&2 || true
    echo "Ray did not become ready in time" >&2
    exit 1
  fi
  sleep 2
done

log "Waiting for SLLM HTTP API at ${HEALTH_URL}"
for attempt in $(seq 1 90); do
  if podman exec "$CONTAINER" "$HEAD_PYTHON" - "$HEALTH_URL" "$MODELS_URL" >/tmp/spotserve-health.log 2>&1 <<'PY'
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
    cat /tmp/spotserve-health.log
    break
  fi
  if [[ "$attempt" -eq 90 ]]; then
    cat /tmp/spotserve-health.log >&2 || true
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

if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
  log "Deploying dummy SpotServe policies"
  podman exec "$CONTAINER" bash -lc "
    set -euo pipefail
    cd '$WORKDIR_IN_CONTAINER'
    for config in \
      examples/spotserve/config-dummy-none.json \
      examples/spotserve/config-dummy-naive-retry.json \
      examples/spotserve/config-dummy-token-replay.json
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
