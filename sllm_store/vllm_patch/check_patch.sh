#!/bin/bash
# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PATCH_FILES=(
    "$SCRIPT_DIR/sllm_load.patch"
    "$SCRIPT_DIR/runtime_kv_metadata.patch"
    "$SCRIPT_DIR/runtime_kv_restore.patch"
)

VLLM_PATH_OUTPUT=$(python -c "import vllm; import os; print(os.path.dirname(os.path.abspath(vllm.__file__)))" 2>/dev/null)
VLLM_PATH=$(echo "$VLLM_PATH_OUTPUT" | tail -n 1)

# Sanity check the path
echo "Detected VLLM_PATH: '$VLLM_PATH'"
if [ ! -d "$VLLM_PATH" ]; then
    echo "Error: Detected VLLM_PATH is not a valid directory: '$VLLM_PATH'"
    echo "Full output from python command was:"
    echo "$VLLM_PATH_OUTPUT"
    exit 1
fi

for PATCH_FILE in "${PATCH_FILES[@]}"; do
    if [ ! -f "$PATCH_FILE" ]; then
        echo "File does not exist: $PATCH_FILE"
        exit 1
    fi
    if patch -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE" > /dev/null 2>&1; then
        echo "$(basename "$PATCH_FILE") is not applied"
    elif patch -R -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE" > /dev/null 2>&1; then
        echo "$(basename "$PATCH_FILE") has been applied"
    elif [[ "$(basename "$PATCH_FILE")" == "runtime_kv_metadata.patch" ]] &&
        grep -q "def get_request_kv_metadata" "$VLLM_PATH/v1/engine/async_llm.py" &&
        grep -q "def get_all_request_kv_metadata" "$VLLM_PATH/v1/engine/async_llm.py" &&
        grep -q "def supports_state_restore" "$VLLM_PATH/v1/engine/async_llm.py"; then
        # runtime_kv_restore.patch intentionally touches the same async/core
        # hunks, so a strict reverse dry-run of metadata can fail even though
        # both patches are present.  Check the exported runtime markers in
        # that overlap case instead of reporting a false incompatibility.
        echo "$(basename "$PATCH_FILE") has been applied (overlap markers)"
    else
        echo "$(basename "$PATCH_FILE") is incompatible with the installed vLLM"
        exit 1
    fi
done

# Guard against the deployment regression this smoke check is intended to
# catch: old revisions assigned a field that is absent on deployed Request
# objects, and accidentally advertised cross-node restore support.
if grep -qE '^\+.*request\.kv_transfer_params\s*=' \
    "$SCRIPT_DIR/runtime_kv_restore.patch"; then
    echo "runtime_kv_restore.patch writes Request.kv_transfer_params directly" >&2
    exit 1
fi
if ! grep -q '"can_restore_cross_node": False' \
    "$SCRIPT_DIR/runtime_kv_restore.patch"; then
    echo "runtime_kv_restore.patch must keep cross-node restore disabled" >&2
    exit 1
fi
