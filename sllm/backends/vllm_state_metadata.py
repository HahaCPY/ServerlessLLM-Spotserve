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
from typing import Any, Dict, List, Mapping, Optional

from sllm.backends.vllm_context_metadata import context_block_count_from_runtime


def _token_list(value: Any) -> List[int]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [int(token) for token in value[0]]
    if isinstance(value, (list, tuple)):
        return [int(token) for token in value]
    return []


def _tokens_from_runtime(runtime_metadata: Mapping[str, Any]) -> List[int]:
    tokens = _token_list(runtime_metadata.get("tokens"))
    if tokens:
        return tokens
    prompt_tokens = _token_list(runtime_metadata.get("prompt_tokens"))
    output_tokens = _token_list(runtime_metadata.get("output_tokens"))
    return prompt_tokens + output_tokens


def _runtime_value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _runtime_or_kv_value(runtime_metadata: Mapping[str, Any], key: str) -> Any:
    value = _runtime_value(runtime_metadata, key)
    if value is not None:
        return value
    kv_transfer_params = _runtime_value(runtime_metadata, "kv_transfer_params")
    if kv_transfer_params is None:
        return None
    return _runtime_value(kv_transfer_params, key)


def _runtime_list(runtime_metadata: Mapping[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        value = _runtime_or_kv_value(runtime_metadata, key)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def _runtime_dict(runtime_metadata: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = _runtime_or_kv_value(runtime_metadata, key)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def get_vllm_inference_state(
    model_name: str,
    request_data: Optional[Mapping[str, Any]] = None,
    current_output: Optional[List[List[int]]] = None,
    completed_tokens: Optional[int] = None,
    instance_id: str = "",
    node_id: str = "",
    runtime_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    request_data = request_data or {}
    runtime_metadata = runtime_metadata or {}
    tokens = _token_list(current_output)
    if not tokens:
        tokens = _tokens_from_runtime(runtime_metadata)

    completed = (
        int(completed_tokens)
        if completed_tokens is not None
        else int(runtime_metadata.get("completed_tokens", len(tokens)) or 0)
    )

    prompt_tokens = _token_list(runtime_metadata.get("prompt_tokens"))
    output_tokens = _token_list(runtime_metadata.get("output_tokens"))

    return {
        "request_id": (
            request_data.get("request_id")
            or runtime_metadata.get("request_id")
        ),
        "instance_id": instance_id,
        "node_id": node_id,
        "backend": "vllm",
        "model_name": model_name,
        "tokens": tokens,
        "completed_tokens": max(0, completed),
        "state_kind": "token_snapshot",
        "supports_restore": False,
        "metadata": {
            "prompt_token_count": len(prompt_tokens),
            "generated_token_count": len(output_tokens),
            "kv_block_count": context_block_count_from_runtime(
                runtime_metadata
            ),
            "block_ids": _runtime_list(
                runtime_metadata,
                "block_ids",
                "kv_block_ids",
            ),
            "block_table": _runtime_dict(runtime_metadata, "block_table"),
            "cache_engine": "vllm",
            "can_restore_same_node": False,
            "can_restore_cross_node": False,
            "reason": "vllm_kv_restore_not_available",
        },
    }
