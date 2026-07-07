# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
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
from typing import Any, Dict, Mapping


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 0)


def _int_mapping(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _non_negative_int(raw_value)
        for key, raw_value in value.items()
    }


def _token_count(runtime_metadata: Mapping[str, Any]) -> int:
    if "num_tokens" in runtime_metadata:
        return _non_negative_int(runtime_metadata.get("num_tokens"))

    tokens = runtime_metadata.get("tokens")
    if isinstance(tokens, (list, tuple)):
        return len(tokens)

    prompt_tokens = runtime_metadata.get("prompt_tokens")
    output_tokens = runtime_metadata.get("output_tokens")
    if isinstance(prompt_tokens, (list, tuple)) or isinstance(
        output_tokens, (list, tuple)
    ):
        return len(prompt_tokens or []) + len(output_tokens or [])

    return 0


def get_vllm_context_metadata(
    model_name: str,
    instance_id: str,
    node_id: str,
    runtime_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "request_id": runtime_metadata.get("request_id"),
        "instance_id": instance_id,
        "node_id": node_id,
        "model_name": model_name,
        "backend": "vllm",
        "num_tokens": _token_count(runtime_metadata),
        "context_blocks": _non_negative_int(
            runtime_metadata.get("context_blocks")
        ),
        "reusable_tokens_by_target": _int_mapping(
            runtime_metadata.get("reusable_tokens_by_target")
        ),
        "reusable_blocks_by_target": _int_mapping(
            runtime_metadata.get("reusable_blocks_by_target")
        ),
        "supports_state_export": bool(
            runtime_metadata.get("supports_state_export", False)
        ),
        "supports_state_restore": bool(
            runtime_metadata.get("supports_state_restore", False)
        ),
        "metadata": dict(runtime_metadata.get("metadata", {}) or {}),
    }
