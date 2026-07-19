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
from typing import Any, Dict, Mapping, Optional


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


def _runtime_value(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


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


def _sequence_len(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _block_table_count(value: Any) -> int:
    if isinstance(value, Mapping):
        total = 0
        for child in value.values():
            child_count = _block_table_count(child)
            if child_count > 0:
                total += child_count
            elif child is not None:
                total += 1
        return total
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _explicit_block_count(payload: Any) -> int:
    for key in ("context_blocks", "kv_block_count", "num_kv_blocks"):
        value = _runtime_value(payload, key)
        if value is not None:
            parsed = _non_negative_int(value)
            if parsed > 0:
                return parsed

    for key in ("block_ids", "kv_block_ids"):
        count = _sequence_len(_runtime_value(payload, key))
        if count > 0:
            return count

    count = _block_table_count(_runtime_value(payload, "block_table"))
    if count > 0:
        return count

    return 0


def context_block_count_from_runtime(
    runtime_metadata: Mapping[str, Any],
) -> int:
    direct_count = _explicit_block_count(runtime_metadata)
    if direct_count > 0:
        return direct_count

    kv_transfer_params: Optional[Any] = runtime_metadata.get(
        "kv_transfer_params"
    )
    if kv_transfer_params is None:
        return 0
    return _explicit_block_count(kv_transfer_params)


def get_vllm_context_metadata(
    model_name: str,
    instance_id: str,
    node_id: str,
    runtime_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = dict(runtime_metadata.get("metadata", {}) or {})
    for key in (
        "prompt_tokens",
        "output_tokens",
        "completed_tokens",
        "request_status",
        "sequence_id",
        "sequence_group_id",
        "sequence_ids",
        "kv_block_count",
        "allocated_kv_block_count",
        "block_ids",
        "kv_block_ids_by_group",
        "raw_block_ids_by_group",
        "null_block_mask_by_group",
        "kv_block_count_by_group",
        "block_table",
        "cache_block_size",
        "cache_dtype",
        "configured_cache_dtype",
        "cache_layout",
        "cache_groups",
        "cache_engine",
        "engine_id",
        "source_hostname",
        "source_device_ids",
        "worker_kv_metadata",
        "model_revision",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "kv_connector",
        "runtime_epoch",
        "vllm_version",
        "cache_config_fingerprint",
    ):
        if key in runtime_metadata:
            metadata[key] = runtime_metadata[key]
    return {
        "request_id": runtime_metadata.get("request_id"),
        "instance_id": instance_id,
        "node_id": node_id,
        "model_name": model_name,
        "backend": "vllm",
        "num_tokens": _token_count(runtime_metadata),
        "tokens": list(runtime_metadata.get("tokens", []) or []),
        "context_blocks": context_block_count_from_runtime(runtime_metadata),
        "cache_block_size": runtime_metadata.get("cache_block_size"),
        "cache_dtype": runtime_metadata.get("cache_dtype"),
        "cache_layout": runtime_metadata.get("cache_layout"),
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
        "metadata": metadata,
    }
