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
  --build-only       Build the image, run cleanup, and exit
  --no-cleanup       Do not prune stale build artifacts after build/recreate
  --cleanup-only     Prune stale build artifacts and exit
  --deploy-set SET   Models to deploy: standard, correctness,
                     reparallelization, reparallelization-performance,
                     reparallelization-multi-worker-performance,
                     context-migration-performance,
                     stateful-recovery-performance,
                     spotserve-core-performance,
                     vllm-dense, vllm-moe,
                     vllm-blackbox, or all.
                     Default: standard. "all" can require substantial GPU capacity.
  -h, --help         Show this help

Environment overrides:
  SPOTSERVE_CONTAINER        Container name. Default: sllm_head
  SPOTSERVE_COMPOSE_SERVICE  Compose service name. Default: sllm_head
  SPOTSERVE_COMPOSE_SERVICES Space-separated compose services to build/recreate
  SPOTSERVE_COMPOSE_BUILD_SERVICES
                            Space-separated compose services to build.
                            Default: sllm_head because head/worker share one image.
  SPOTSERVE_BUILD_NO_CACHE   Set to 1 to run docker compose build --no-cache.
                            Useful when validating vLLM patch layer changes.
  SPOTSERVE_HOST_TMPDIR     Host temp dir for prepare logs.
                            Default: <repo>/.spotserve-tmp
  SPOTSERVE_WORKDIR          Container workdir. Default: /tmp/spotserve-work
  SPOTSERVE_HEALTH_URL       Container API health URL. Default: http://127.0.0.1:8343/health
  SPOTSERVE_WORKER_CONTAINER Worker container used by vLLM deploy sets. Default: sllm_worker_0
  SPOTSERVE_EXPECTED_WORKER_NODES
                            Minimum Ray worker_node resources required before
                            deploying. Default: 2 for
                            reparallelization-multi-worker-performance,
                            otherwise 1.
  SPOTSERVE_REPARALLELIZATION_MULTI_WORKER
                            When set to 1, include sllm_worker_1 in the
                            compose services for V6 deploy sets.
  SPOTSERVE_HF_CACHE_DIR     Host HF cache dir mounted as /hf-cache.
                            Default: /tmp/sllm-hf-cache-rootless
  SPOTSERVE_CLEANUP_BUILD_ARTIFACTS
                            Prune dangling images/build cache after build.
                            Default: 1
  SPOTSERVE_CLEANUP_UNTAGGED_IMAGES
                            Remove unused <none>:<none> images after cleanup.
                            Default: same as SPOTSERVE_CLEANUP_BUILD_ARTIFACTS.
  SPOTSERVE_CLEANUP_STOPPED_CONTAINERS
                            Remove stopped sllm_head/sllm_worker_* containers.
                            Default: 1
  SPOTSERVE_SYNC_SOURCE     Copy local sllm/ into running containers and restart
                            before readiness checks. Default: 1 with
                            --skip-build, otherwise 0.
  SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION
                            When set to 1, require patched vLLM MoE routing
                            hooks during vLLM deploy checks. Default: 0.
  SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS
                            When set to 1, require patched vLLM expert
                            placement apply/verify hooks. Default: 0.
  SPOTSERVE_REPARALLELIZATION_MODEL_PATH
                            vLLM model path/id used by V6 reparallelization.
                            Default: /models/vllm/vllm-dense-baseline
  SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT
                            vLLM load_format used by V6 reparallelization.
                            Default: serverless_llm
  SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH
                            vLLM model path/id used by V7 context migration.
                            Default: SPOTSERVE_REPARALLELIZATION_MODEL_PATH
  SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT
                            vLLM load_format used by V7 context migration.
                            Default: SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT
  SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH
                            vLLM model path/id used by V8 stateful recovery.
                            Default: SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH
  SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT
                            vLLM load_format used by V8 stateful recovery.
                            Default: SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT
  SPOTSERVE_CORE_MODEL_PATH
                            vLLM model path/id used by the combined
                            V7/V8/V9 core benchmark.
                            Default: SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH
  SPOTSERVE_CORE_LOAD_FORMAT
                            vLLM load_format used by the combined
                            V7/V8/V9 core benchmark.
                            Default: SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT
  SPOTSERVE_VLLM_DENSE_MODEL_PATH
                            Optional container-local HF snapshot path to verify.
  SPOTSERVE_VLLM_MOE_MODEL  MoE HF model id or container-local path.
                            Default: /models/Qwen1.5-MoE-A2.7B
  SPOTSERVE_VLLM_MOE_LOAD_FORMAT
                            vLLM load_format for MoE direct load. Default: auto
  SPOTSERVE_VLLM_MOE_TP     MoE tensor_parallel_size override. Default: 1
  SPOTSERVE_VLLM_MOE_MODEL_PATH
                            Optional container-local MoE snapshot path to verify.
  SPOTSERVE_DEFAULT_MODEL_FOLDER
                            Host model dir used when MODEL_FOLDER is unset.
                            Default: /work/spotserve-models if present,
                            otherwise <repo>/model
  MODEL_FOLDER               Host model directory mounted as /models.
                            Default: SPOTSERVE_DEFAULT_MODEL_FOLDER
EOF
}

SKIP_BUILD=0
SKIP_RECREATE=0
SKIP_DEPLOY=0
BUILD_ONLY=0
CLEANUP_ONLY=0
CLEANUP_BUILD_ARTIFACTS="${SPOTSERVE_CLEANUP_BUILD_ARTIFACTS:-1}"
CLEANUP_UNTAGGED_IMAGES="${SPOTSERVE_CLEANUP_UNTAGGED_IMAGES:-$CLEANUP_BUILD_ARTIFACTS}"
CLEANUP_STOPPED_CONTAINERS="${SPOTSERVE_CLEANUP_STOPPED_CONTAINERS:-1}"
CLEANUP_RAN=0
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
    --build-only)
      BUILD_ONLY=1
      SKIP_RECREATE=1
      SKIP_DEPLOY=1
      shift
      ;;
    --no-cleanup)
      CLEANUP_BUILD_ARTIFACTS=0
      CLEANUP_STOPPED_CONTAINERS=0
      shift
      ;;
    --cleanup-only)
      CLEANUP_ONLY=1
      shift
      ;;
    --deploy-set)
      DEPLOY_SET="${2:-}"
      if [[ "$DEPLOY_SET" != "standard" && "$DEPLOY_SET" != "correctness" && "$DEPLOY_SET" != "reparallelization" && "$DEPLOY_SET" != "reparallelization-performance" && "$DEPLOY_SET" != "reparallelization-multi-worker-performance" && "$DEPLOY_SET" != "context-migration-performance" && "$DEPLOY_SET" != "stateful-recovery-performance" && "$DEPLOY_SET" != "spotserve-core-performance" && "$DEPLOY_SET" != "vllm-dense" && "$DEPLOY_SET" != "vllm-moe" && "$DEPLOY_SET" != "vllm-blackbox" && "$DEPLOY_SET" != "all" ]]; then
        echo "--deploy-set must be one of: standard, correctness, reparallelization, reparallelization-performance, reparallelization-multi-worker-performance, context-migration-performance, stateful-recovery-performance, spotserve-core-performance, vllm-dense, vllm-moe, vllm-blackbox, all" >&2
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
REPARALLELIZATION_MULTI_WORKER="${SPOTSERVE_REPARALLELIZATION_MULTI_WORKER:-0}"
if [[ "$DEPLOY_SET" == "reparallelization-multi-worker-performance" ]]; then
  EXPECTED_WORKER_NODES="${SPOTSERVE_EXPECTED_WORKER_NODES:-2}"
else
  EXPECTED_WORKER_NODES="${SPOTSERVE_EXPECTED_WORKER_NODES:-1}"
fi
HOST_TMPDIR="${SPOTSERVE_HOST_TMPDIR:-${ROOT_DIR}/.spotserve-tmp}"
HF_CACHE_DIR="${SPOTSERVE_HF_CACHE_DIR:-/tmp/sllm-hf-cache-rootless}"
WORKDIR_IN_CONTAINER="${SPOTSERVE_WORKDIR:-/tmp/spotserve-work}"
HEALTH_URL="${SPOTSERVE_HEALTH_URL:-http://127.0.0.1:8343/health}"
MODELS_URL="${SPOTSERVE_MODELS_URL:-http://127.0.0.1:8343/v1/models}"
REPARALLELIZATION_MODEL_PATH="${SPOTSERVE_REPARALLELIZATION_MODEL_PATH:-/models/vllm/vllm-dense-baseline}"
REPARALLELIZATION_LOAD_FORMAT="${SPOTSERVE_REPARALLELIZATION_LOAD_FORMAT:-serverless_llm}"
CONTEXT_MIGRATION_MODEL_PATH="${SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH:-$REPARALLELIZATION_MODEL_PATH}"
CONTEXT_MIGRATION_LOAD_FORMAT="${SPOTSERVE_CONTEXT_MIGRATION_LOAD_FORMAT:-$REPARALLELIZATION_LOAD_FORMAT}"
STATEFUL_RECOVERY_MODEL_PATH="${SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH:-$CONTEXT_MIGRATION_MODEL_PATH}"
STATEFUL_RECOVERY_LOAD_FORMAT="${SPOTSERVE_STATEFUL_RECOVERY_LOAD_FORMAT:-$CONTEXT_MIGRATION_LOAD_FORMAT}"
CORE_MODEL_PATH="${SPOTSERVE_CORE_MODEL_PATH:-$STATEFUL_RECOVERY_MODEL_PATH}"
CORE_LOAD_FORMAT="${SPOTSERVE_CORE_LOAD_FORMAT:-$STATEFUL_RECOVERY_LOAD_FORMAT}"
VLLM_DENSE_MODEL_PATH="${SPOTSERVE_VLLM_DENSE_MODEL_PATH:-}"
VLLM_MOE_MODEL="${SPOTSERVE_VLLM_MOE_MODEL:-/models/Qwen1.5-MoE-A2.7B}"
VLLM_MOE_LOAD_FORMAT="${SPOTSERVE_VLLM_MOE_LOAD_FORMAT:-auto}"
VLLM_MOE_TP="${SPOTSERVE_VLLM_MOE_TP:-1}"
VLLM_MOE_MODEL_PATH="${SPOTSERVE_VLLM_MOE_MODEL_PATH:-}"
HEAD_PYTHON="/opt/venvs/head/bin/python"
HEAD_SLLM="/opt/venvs/head/bin/sllm"
HEAD_RAY="/opt/venvs/head/bin/ray"
WORKER_PYTHON="/opt/venvs/worker/bin/python"
BUILD_NO_CACHE="${SPOTSERVE_BUILD_NO_CACHE:-0}"
SYNC_SOURCE="${SPOTSERVE_SYNC_SOURCE:-}"

MODEL_FOLDER_WAS_SET=0
if [[ -n "${MODEL_FOLDER+x}" ]]; then
  MODEL_FOLDER_WAS_SET=1
fi

DEFAULT_MODEL_FOLDER="${SPOTSERVE_DEFAULT_MODEL_FOLDER:-/work/spotserve-models}"
if [[ ! -d "$DEFAULT_MODEL_FOLDER" ]]; then
  DEFAULT_MODEL_FOLDER="${ROOT_DIR}/model"
fi
MODEL_FOLDER="${MODEL_FOLDER:-$DEFAULT_MODEL_FOLDER}"
maybe_use_repo_model_folder() {
  local container_model_path="$1"
  if [[ "$MODEL_FOLDER_WAS_SET" -eq 1 || "$container_model_path" != /models/* ]]; then
    return
  fi
  local relative_model_path="${container_model_path#/models/}"
  if [[ -f "${MODEL_FOLDER}/${relative_model_path}/config.json" ]]; then
    return
  fi
  if [[ -f "${ROOT_DIR}/model/${relative_model_path}/config.json" ]]; then
    MODEL_FOLDER="${ROOT_DIR}/model"
  fi
}
if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "reparallelization-multi-worker-performance" ]]; then
  maybe_use_repo_model_folder "$REPARALLELIZATION_MODEL_PATH"
elif [[ "$DEPLOY_SET" == "context-migration-performance" ]]; then
  maybe_use_repo_model_folder "$CONTEXT_MIGRATION_MODEL_PATH"
elif [[ "$DEPLOY_SET" == "stateful-recovery-performance" ]]; then
  maybe_use_repo_model_folder "$STATEFUL_RECOVERY_MODEL_PATH"
elif [[ "$DEPLOY_SET" == "spotserve-core-performance" ]]; then
  maybe_use_repo_model_folder "$CORE_MODEL_PATH"
fi
export MODEL_FOLDER
export HF_CACHE_DIR

if [[ -z "$SYNC_SOURCE" ]]; then
  if [[ "$SKIP_BUILD" -eq 1 ]]; then
    SYNC_SOURCE=1
  else
    SYNC_SOURCE=0
  fi
fi

if [[ -n "${SPOTSERVE_COMPOSE_SERVICES:-}" ]]; then
  read -r -a COMPOSE_SERVICES <<<"$SPOTSERVE_COMPOSE_SERVICES"
elif [[ "$DEPLOY_SET" == "reparallelization-multi-worker-performance" ]]; then
  COMPOSE_SERVICES=("$COMPOSE_SERVICE" "sllm_worker_0" "sllm_worker_1")
elif [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  COMPOSE_SERVICES=("$COMPOSE_SERVICE" "sllm_worker_0")
  if [[ "$REPARALLELIZATION_MULTI_WORKER" == "1" && ( "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "all" ) ]]; then
    COMPOSE_SERVICES+=("sllm_worker_1")
  fi
else
  COMPOSE_SERVICES=("$COMPOSE_SERVICE")
fi
if printf '%s\n' "${COMPOSE_SERVICES[@]}" | grep -qx "sllm_worker_1"; then
  MULTI_WORKER_COMPOSE_FILE="examples/spotserve/docker-compose.multi-worker.yml"
  export SLLM_WORKER_0_GPU_DEVICE="${SLLM_WORKER_0_GPU_DEVICE:-0}"
  export SLLM_WORKER_0_VISIBLE_DEVICES="${SLLM_WORKER_0_VISIBLE_DEVICES:-0}"
  export SLLM_WORKER_1_GPU_DEVICE="${SLLM_WORKER_1_GPU_DEVICE:-1}"
  export SLLM_WORKER_1_VISIBLE_DEVICES="${SLLM_WORKER_1_VISIBLE_DEVICES:-1}"
  if [[ -z "${COMPOSE_FILE:-}" ]]; then
    export COMPOSE_FILE="docker-compose.yml:${MULTI_WORKER_COMPOSE_FILE}"
  elif [[ ":${COMPOSE_FILE}:" != *":${MULTI_WORKER_COMPOSE_FILE}:"* ]]; then
    export COMPOSE_FILE="${COMPOSE_FILE}:${MULTI_WORKER_COMPOSE_FILE}"
  fi
fi

if [[ -n "${SPOTSERVE_COMPOSE_BUILD_SERVICES:-}" ]]; then
  read -r -a COMPOSE_BUILD_SERVICES <<<"$SPOTSERVE_COMPOSE_BUILD_SERVICES"
else
  COMPOSE_BUILD_SERVICES=("$COMPOSE_SERVICE")
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

cleanup_build_artifacts() {
  if [[ "$CLEANUP_BUILD_ARTIFACTS" -ne 1 && "$CLEANUP_UNTAGGED_IMAGES" -ne 1 && "$CLEANUP_STOPPED_CONTAINERS" -ne 1 ]]; then
    return
  fi
  CLEANUP_RAN=1
  log "Cleaning stale SpotServe build artifacts"
  if [[ "$CLEANUP_BUILD_ARTIFACTS" -eq 1 ]]; then
    podman image prune -f || true
    if podman builder prune --help >/dev/null 2>&1; then
      podman builder prune -f || true
    fi
  fi
  if [[ "$CLEANUP_UNTAGGED_IMAGES" -eq 1 ]]; then
    local image_id
    while read -r image_id; do
      if [[ -z "${image_id:-}" ]]; then
        continue
      fi
      podman rmi "$image_id" >/dev/null 2>&1 || true
    done < <(
      podman images -a \
        --format "{{.Repository}} {{.Tag}} {{.ID}}" 2>/dev/null |
        awk '$1 == "<none>" && $2 == "<none>" {print $3}' |
        sort -u
    )
  fi
  if [[ "$CLEANUP_STOPPED_CONTAINERS" -eq 1 ]]; then
    local status id name
    for status in created exited dead; do
      while read -r id name; do
        if [[ -z "${id:-}" || -z "${name:-}" ]]; then
          continue
        fi
        case "$name" in
          "$CONTAINER"|"$WORKER_CONTAINER"|sllm_head|sllm_worker_*)
            podman rm -f "$id" || true
            ;;
        esac
      done < <(podman ps -a --filter "status=${status}" --format "{{.ID}} {{.Names}}" 2>/dev/null || true)
    done
  fi
}

sync_source_into_containers() {
  if [[ "$SYNC_SOURCE" != "1" ]]; then
    return
  fi
  log "Syncing local sllm package into running containers"
  podman cp sllm/. "${CONTAINER}:/opt/venvs/head/lib/python3.11/site-packages/sllm"
  restart_containers=("$CONTAINER")
  for service in "${COMPOSE_SERVICES[@]}"; do
    case "$service" in
      sllm_worker_*)
        if podman ps -a --format "{{.Names}}" | grep -qx "$service"; then
          podman cp sllm/. "${service}:/opt/venvs/worker/lib/python3.11/site-packages/sllm"
          restart_containers+=("$service")
        fi
        ;;
    esac
  done
  podman restart "${restart_containers[@]}"
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT
  if [[ "$CLEANUP_RAN" -ne 1 ]]; then
    cleanup_build_artifacts
  fi
  exit "$status"
}

cd "$ROOT_DIR"
mkdir -p "$HOST_TMPDIR"
mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/modules"
chmod -R a+rwX "$HF_CACHE_DIR" 2>/dev/null || true
RAY_STATUS_LOG="${HOST_TMPDIR}/spotserve-ray-status.log"
VLLM_RESOURCES_LOG="${HOST_TMPDIR}/spotserve-vllm-resources.log"
VLLM_RUNTIME_LOG="${HOST_TMPDIR}/spotserve-vllm-runtime.log"
HEALTH_LOG="${HOST_TMPDIR}/spotserve-health.log"

if [[ "$CLEANUP_ONLY" -eq 1 ]]; then
  cleanup_build_artifacts
  exit 0
fi

trap cleanup_on_exit EXIT

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  log "Building ${COMPOSE_BUILD_SERVICES[*]}"
  BUILD_ARGS=()
  if [[ "$BUILD_NO_CACHE" == "1" ]]; then
    BUILD_ARGS+=(--no-cache)
  fi
  docker compose build "${BUILD_ARGS[@]}" "${COMPOSE_BUILD_SERVICES[@]}"
  cleanup_build_artifacts
fi

if [[ "$BUILD_ONLY" -eq 1 ]]; then
  log "Build-only mode complete"
  exit 0
fi

if [[ "$SKIP_RECREATE" -eq 0 ]]; then
  log "Recreating ${COMPOSE_SERVICES[*]}"
  if ! docker compose up -d --force-recreate --remove-orphans --no-build "${COMPOSE_SERVICES[@]}"; then
    log "Retrying recreate without --remove-orphans"
    docker compose up -d --force-recreate --no-build "${COMPOSE_SERVICES[@]}"
  fi
  cleanup_build_artifacts
fi

sync_source_into_containers

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

if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "reparallelization-multi-worker-performance" || "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
  log "Checking vLLM worker resources"
  podman exec "$WORKER_CONTAINER" bash -lc \
    "mkdir -p /hf-cache/hub /hf-cache/modules && chmod -R a+rwX /hf-cache && touch /hf-cache/modules/.spotserve-write-test"
  if ! podman exec -i "$WORKER_CONTAINER" "$WORKER_PYTHON" - \
      "$DEPLOY_SET" \
      "${SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION:-0}" \
      "${SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS:-0}" \
      >"$VLLM_RUNTIME_LOG" 2>&1 <<'PY'
import inspect
import sys
from importlib.metadata import PackageNotFoundError, version

from vllm import AsyncLLMEngine

deploy_set = sys.argv[1]
require_moe_route_instrumentation = sys.argv[2] == "1"
require_expert_placement_hooks = sys.argv[3] == "1"
required_hooks = ()
if deploy_set == "context-migration-performance":
    required_hooks = (
        "get_request_kv_metadata",
        "get_all_request_kv_metadata",
    )
elif deploy_set not in ("reparallelization", "reparallelization-performance", "reparallelization-multi-worker-performance"):
    from nixl._api import nixl_agent

    required_hooks = (
        "get_request_kv_metadata",
        "get_all_request_kv_metadata",
        "export_inference_state",
        "restore_inference_state",
        "supports_state_restore",
    )
if require_moe_route_instrumentation:
    required_hooks = (
        *required_hooks,
        "get_request_moe_metadata",
        "get_moe_runtime_metadata",
    )
if require_expert_placement_hooks:
    required_hooks = (
        *required_hooks,
        "apply_expert_placement_plan",
        "verify_expert_placement_plan",
    )
missing = [name for name in required_hooks if not hasattr(AsyncLLMEngine, name)]
try:
    nixl_version = version("nixl")
except PackageNotFoundError:
    nixl_version = "not installed"
print("vLLM version:", version("vllm"))
print("NIXL version:", nixl_version)
print("AsyncLLMEngine module:", inspect.getfile(AsyncLLMEngine))
if required_hooks:
    print("runtime metadata hooks:", {name: name not in missing for name in required_hooks})
else:
    print("runtime metadata hooks: not required for reparallelization deploy set")
if missing:
    raise SystemExit(f"Missing patched vLLM runtime hooks: {missing}")
PY
  then
    cat "$VLLM_RUNTIME_LOG" >&2 || true
    exit 1
  fi
  cat "$VLLM_RUNTIME_LOG"
  if [[ "${SPOTSERVE_REQUIRE_MOE_ROUTE_INSTRUMENTATION:-0}" == "1" ||
        "${SPOTSERVE_REQUIRE_EXPERT_PLACEMENT_RUNTIME_HOOKS:-0}" == "1" ]]; then
    VLLM_PATH="$(
      podman exec "$WORKER_CONTAINER" "$WORKER_PYTHON" -c \
        'import os, vllm; print(os.path.dirname(os.path.abspath(vllm.__file__)))'
    )"
    MISSING_MOE_MARKERS=()
    check_moe_marker() {
      local name="$1"
      local path="$2"
      local marker="$3"
      if ! podman exec "$WORKER_CONTAINER" test -f "$path" ||
          ! podman exec "$WORKER_CONTAINER" grep -q "$marker" "$path"; then
        MISSING_MOE_MARKERS+=("$name")
      fi
    }
    check_moe_marker \
      "vllm.spotserve_moe" \
      "$VLLM_PATH/spotserve_moe.py" \
      "def record_moe_routing"
    check_moe_marker \
      "fused_moe_modular_method.record_moe_routing" \
      "$VLLM_PATH/model_executor/layers/fused_moe/fused_moe_modular_method.py" \
      "record_moe_routing("
    check_moe_marker \
      "unquantized_fused_moe_method.record_moe_routing" \
      "$VLLM_PATH/model_executor/layers/fused_moe/unquantized_fused_moe_method.py" \
      "record_moe_routing("
    check_moe_marker \
      "gpu_model_runner.moe_request_context" \
      "$VLLM_PATH/v1/worker/gpu_model_runner.py" \
      "moe_request_context(req_ids, num_scheduled_tokens_np)"
    check_moe_marker \
      "worker_base.get_request_moe_metadata" \
      "$VLLM_PATH/v1/worker/worker_base.py" \
      "def get_request_moe_metadata"
    check_moe_marker \
      "async_llm.apply_expert_placement_plan" \
      "$VLLM_PATH/v1/engine/async_llm.py" \
      "def apply_expert_placement_plan"
    check_moe_marker \
      "async_llm.verify_expert_placement_plan" \
      "$VLLM_PATH/v1/engine/async_llm.py" \
      "def verify_expert_placement_plan"
    check_moe_marker \
      "worker_base.apply_expert_placement_plan" \
      "$VLLM_PATH/v1/worker/worker_base.py" \
      "def apply_expert_placement_plan"
    check_moe_marker \
      "worker_base.verify_expert_placement_plan" \
      "$VLLM_PATH/v1/worker/worker_base.py" \
      "def verify_expert_placement_plan"
    if ! podman exec "$WORKER_CONTAINER" "$WORKER_PYTHON" -m py_compile \
        "$VLLM_PATH/spotserve_moe.py"; then
      MISSING_MOE_MARKERS+=("vllm.spotserve_moe.py_compile")
    fi
    printf 'MoE/placement runtime patch markers: missing=%s\n' \
      "${MISSING_MOE_MARKERS[*]:-none}"
    if [[ "${#MISSING_MOE_MARKERS[@]}" -gt 0 ]]; then
      echo "Missing patched vLLM MoE/placement markers: ${MISSING_MOE_MARKERS[*]}" >&2
      exit 1
    fi
  fi
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
  if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "reparallelization-multi-worker-performance" || "$DEPLOY_SET" == "all" ]] &&
      [[ "$REPARALLELIZATION_MODEL_PATH" == /* ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${REPARALLELIZATION_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM reparallelization model is not visible in ${WORKER_CONTAINER}.
Expected: ${REPARALLELIZATION_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_REPARALLELIZATION_MODEL_PATH to a container-local model path
or Hugging Face model id that the worker can load.
EOF
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "all" ]] &&
      [[ "$CONTEXT_MIGRATION_MODEL_PATH" == /* ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${CONTEXT_MIGRATION_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM context-migration model is not visible in ${WORKER_CONTAINER}.
Expected: ${CONTEXT_MIGRATION_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_CONTEXT_MIGRATION_MODEL_PATH to a container-local model path
or Hugging Face model id that the worker can load.
EOF
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "all" ]] &&
      [[ "$STATEFUL_RECOVERY_MODEL_PATH" == /* ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${STATEFUL_RECOVERY_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM stateful-recovery model is not visible in ${WORKER_CONTAINER}.
Expected: ${STATEFUL_RECOVERY_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_STATEFUL_RECOVERY_MODEL_PATH to a container-local model path
or Hugging Face model id that the worker can load.
EOF
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "all" ]] &&
      [[ "$CORE_MODEL_PATH" == /* ]]; then
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${CORE_MODEL_PATH}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM SpotServe core model is not visible in ${WORKER_CONTAINER}.
Expected: ${CORE_MODEL_PATH}/config.json
Host MODEL_FOLDER is: ${MODEL_FOLDER}
Set SPOTSERVE_CORE_MODEL_PATH to a container-local model path
or Hugging Face model id that the worker can load.
EOF
        exit 1
      fi
      sleep 2
    done
  fi
  if [[ "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "all" ]] &&
      [[ -n "$VLLM_MOE_MODEL_PATH" || "$VLLM_MOE_MODEL" == /* ]]; then
    VLLM_MOE_MODEL_TO_CHECK="${VLLM_MOE_MODEL_PATH:-$VLLM_MOE_MODEL}"
    for attempt in $(seq 1 90); do
      if podman exec "$WORKER_CONTAINER" test -f "${VLLM_MOE_MODEL_TO_CHECK}/config.json"; then
        break
      fi
      if [[ "$attempt" -eq 90 ]]; then
        cat >&2 <<EOF
vLLM MoE model is not visible in ${WORKER_CONTAINER}.
Expected: ${VLLM_MOE_MODEL_TO_CHECK}/config.json
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
    if podman exec -i "$CONTAINER" "$HEAD_PYTHON" - "$EXPECTED_WORKER_NODES" >"$VLLM_RESOURCES_LOG" 2>&1 <<'PY'
import ray
import sys

expected_worker_nodes = int(sys.argv[1])
ray.init(address="auto", namespace="sllm", ignore_reinit_error=True)
resources = ray.cluster_resources()
print("Ray cluster resources:", resources)
worker_nodes = resources.get("worker_node", 0)
if worker_nodes < expected_worker_nodes:
    print(
        f"Expected at least {expected_worker_nodes} Ray worker_node "
        f"resources, found {worker_nodes}.",
        file=sys.stderr,
    )
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
Ray does not have the expected GPU worker capacity for vLLM deploy.
Start sllm_worker_0 on a node where the NVIDIA driver is available, then rerun:
  scripts/prepare_spotserve.sh --deploy-set reparallelization
For the multi-worker V6 benchmark, start at least two workers or rerun:
  scripts/prepare_spotserve.sh --deploy-set reparallelization-multi-worker-performance
or:
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

if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "reparallelization-multi-worker-performance" || "$DEPLOY_SET" == "all" ]]; then
  log "Applying vLLM reparallelization config override"
  podman exec -i "$CONTAINER" "$HEAD_PYTHON" - \
    "$WORKDIR_IN_CONTAINER" \
    "$REPARALLELIZATION_MODEL_PATH" \
    "$REPARALLELIZATION_LOAD_FORMAT" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
model_path = sys.argv[2]
load_format = sys.argv[3]
for relative_path in (
    "examples/spotserve/config-vllm-reparallelization-baseline-performance.json",
    "examples/spotserve/config-vllm-reparallelization-applied-performance.json",
    "examples/spotserve/config-vllm-reparallelization-applied-multi-worker-performance.json",
    "examples/spotserve/config-vllm-reparallelization-baseline-multi-worker-performance.json",
    "examples/spotserve/config-vllm-reparallelization-baseline-gpu-smoke.json",
    "examples/spotserve/config-vllm-reparallelization-gpu-smoke.json",
):
    path = workdir / relative_path
    config = json.loads(path.read_text(encoding="utf-8"))
    backend_config = config.setdefault("backend_config", {})
    backend_config["pretrained_model_name_or_path"] = model_path
    backend_config["load_format"] = load_format
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path}: model={model_path} load_format={load_format}")
PY
fi

if [[ "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "all" ]]; then
  log "Applying vLLM context-migration config override"
  podman exec -i "$CONTAINER" "$HEAD_PYTHON" - \
    "$WORKDIR_IN_CONTAINER" \
    "$CONTEXT_MIGRATION_MODEL_PATH" \
    "$CONTEXT_MIGRATION_LOAD_FORMAT" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
model_path = sys.argv[2]
load_format = sys.argv[3]
for relative_path in (
    "examples/spotserve/config-vllm-context-migration-applied-performance.json",
    "examples/spotserve/config-vllm-context-migration-disabled-performance.json",
):
    path = workdir / relative_path
    config = json.loads(path.read_text(encoding="utf-8"))
    backend_config = config.setdefault("backend_config", {})
    backend_config["pretrained_model_name_or_path"] = model_path
    backend_config["load_format"] = load_format
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path}: model={model_path} load_format={load_format}")
PY
fi

if [[ "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "all" ]]; then
  log "Applying vLLM stateful-recovery config override"
  podman exec -i "$CONTAINER" "$HEAD_PYTHON" - \
    "$WORKDIR_IN_CONTAINER" \
    "$STATEFUL_RECOVERY_MODEL_PATH" \
    "$STATEFUL_RECOVERY_LOAD_FORMAT" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
model_path = sys.argv[2]
load_format = sys.argv[3]
for relative_path in (
    "examples/spotserve/config-vllm-stateful-recovery-token-replay-performance.json",
    "examples/spotserve/config-vllm-stateful-recovery-applied-performance.json",
):
    path = workdir / relative_path
    config = json.loads(path.read_text(encoding="utf-8"))
    backend_config = config.setdefault("backend_config", {})
    backend_config["pretrained_model_name_or_path"] = model_path
    backend_config["load_format"] = load_format
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path}: model={model_path} load_format={load_format}")
PY
fi

if [[ "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "all" ]]; then
  log "Applying vLLM SpotServe core config override"
  podman exec -i "$CONTAINER" "$HEAD_PYTHON" - \
    "$WORKDIR_IN_CONTAINER" \
    "$CORE_MODEL_PATH" \
    "$CORE_LOAD_FORMAT" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
model_path = sys.argv[2]
load_format = sys.argv[3]
for relative_path in (
    "examples/spotserve/config-vllm-spotserve-core-baseline-performance.json",
    "examples/spotserve/config-vllm-spotserve-core-applied-performance.json",
):
    path = workdir / relative_path
    config = json.loads(path.read_text(encoding="utf-8"))
    backend_config = config.setdefault("backend_config", {})
    backend_config["pretrained_model_name_or_path"] = model_path
    backend_config["load_format"] = load_format
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"patched {path}: model={model_path} load_format={load_format}")
PY
fi

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
    "examples/spotserve/config-dummy-correctness-stateful-recovery.json"
  )
  REPARALLELIZATION_CONFIGS=(
    "examples/spotserve/config-vllm-reparallelization-gpu-smoke.json"
  )
  REPARALLELIZATION_PERFORMANCE_CONFIGS=(
    "examples/spotserve/config-vllm-reparallelization-baseline-performance.json"
    "examples/spotserve/config-vllm-reparallelization-applied-performance.json"
  )
  CONTEXT_MIGRATION_PERFORMANCE_CONFIGS=(
    "examples/spotserve/config-vllm-context-migration-disabled-performance.json"
    "examples/spotserve/config-vllm-context-migration-applied-performance.json"
  )
  STATEFUL_RECOVERY_PERFORMANCE_CONFIGS=(
    "examples/spotserve/config-vllm-stateful-recovery-token-replay-performance.json"
    "examples/spotserve/config-vllm-stateful-recovery-applied-performance.json"
  )
  SPOTSERVE_CORE_PERFORMANCE_CONFIGS=(
    "examples/spotserve/config-vllm-spotserve-core-baseline-performance.json"
    "examples/spotserve/config-vllm-spotserve-core-applied-performance.json"
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
    "examples/spotserve/config-vllm-moe-stateful-nixl.json"
  )
  DEPLOY_CONFIGS=()
  if [[ "$DEPLOY_SET" == "standard" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${STANDARD_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "correctness" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${CORRECTNESS_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${REPARALLELIZATION_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "all" ]]; then
    log "Reparallelization performance configs will be deployed one run at a time by the benchmark runner"
  fi
  if [[ "$DEPLOY_SET" == "reparallelization-multi-worker-performance" || "$DEPLOY_SET" == "all" ]]; then
    log "Reparallelization multi-worker performance configs will be deployed one run at a time by the benchmark runner"
  fi
  if [[ "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${CONTEXT_MIGRATION_PERFORMANCE_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "all" ]]; then
    log "Stateful-recovery performance configs will be deployed one run at a time by the benchmark runner"
  fi
  if [[ "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "all" ]]; then
    log "SpotServe core performance configs will be deployed one run at a time by the benchmark runner"
  fi
  if [[ "$DEPLOY_SET" == "vllm-dense" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${VLLM_DENSE_CONFIGS[@]}")
  fi
  if [[ "$DEPLOY_SET" == "vllm-moe" || "$DEPLOY_SET" == "vllm-blackbox" || "$DEPLOY_SET" == "all" ]]; then
    DEPLOY_CONFIGS+=("${VLLM_MOE_CONFIGS[@]}")
  fi

  if [[ "${#DEPLOY_CONFIGS[@]}" -gt 0 ]]; then
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
  else
    log "No models predeployed for ${DEPLOY_SET}"
  fi

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

if [[ "$DEPLOY_SET" == "reparallelization" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the dynamic-reparallelization benchmark with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_reparallelization.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 120 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "reparallelization-performance" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the dynamic-reparallelization performance comparison with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_performance.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 180 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "reparallelization-multi-worker-performance" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the dynamic-reparallelization multi-worker performance comparison with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_reparallelization_multi_worker_performance.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 240 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "context-migration-performance" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the context-migration performance comparison with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_context_migration_performance.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 240 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "stateful-recovery-performance" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the stateful-recovery performance comparison with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_stateful_recovery_performance.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 240 \\
  --ray-address auto \\
  --ray-namespace sllm
'
EOF
fi

if [[ "$DEPLOY_SET" == "spotserve-core-performance" || "$DEPLOY_SET" == "all" ]]; then
  cat <<EOF

Run the combined SpotServe core performance comparison with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_performance.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 300 \\
  --ray-address auto \\
  --ray-namespace sllm
'

Run the multi-trace SpotServe core performance sweep with:

podman exec ${CONTAINER} bash -lc '
cd ${WORKDIR_IN_CONTAINER} &&
${HEAD_PYTHON} benchmarks/spotserve/run_benchmark.py \\
  --config benchmarks/spotserve/benchmark_matrix_spotserve_core_trace_sweep.yaml \\
  --endpoint http://127.0.0.1:8343/v1/chat/completions \\
  --request-timeout 300 \\
  --ray-address auto \\
  --ray-namespace sllm
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
