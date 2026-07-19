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

for PATCH_FILE in "${PATCH_FILES[@]}"; do
    if [ ! -f "$PATCH_FILE" ]; then
        echo "File does not exist: $PATCH_FILE"
        exit 1
    fi
done

# Get the vLLM installation path
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
    if patch -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE"; then
        echo "Applying $(basename "$PATCH_FILE")..."
        patch -p2 -d "$VLLM_PATH" < "$PATCH_FILE"
        echo "Patch applied successfully."
    elif patch -R -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE" > /dev/null 2>&1; then
        echo "$(basename "$PATCH_FILE") is already applied. Skipping..."
    else
        echo "$(basename "$PATCH_FILE") is incompatible with the installed vLLM." >&2
        exit 1
    fi
done
