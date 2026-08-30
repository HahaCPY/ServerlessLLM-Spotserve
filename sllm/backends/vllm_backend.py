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
import asyncio
import gc
import inspect
import json
import logging
import os
import time
import uuid
import zlib
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

import torch
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
    PromptType,
    RequestOutput,
    SamplingParams,
)
try:
    from vllm.sampling_params import RequestOutputKind
except ImportError:
    RequestOutputKind = None
from vllm.inputs import TokensPrompt
try:
    from vllm.utils import Counter
except ImportError:
    from itertools import count

    class Counter:
        def __init__(self, start: int = 0):
            self._counter = count(start)

        def __next__(self) -> int:
            return next(self._counter)

from sllm.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)
from sllm.backends.vllm_context_metadata import get_vllm_context_metadata
from sllm.backends.vllm_runtime_metadata import get_vllm_runtime_metadata
from sllm.backends.vllm_state_metadata import get_vllm_inference_state

logger = logging.getLogger("ray")


def process_output(output: RequestOutput, model_name: str) -> Dict[str, Any]:
    choices: List[Dict[str, Any]] = [
        {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": result.text,
            },
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }
        for idx, result in enumerate(output.outputs)
    ]

    api_response = {
        "id": output.request_id,
        "object": "chat.completion",
        "created": (
            int(time.time())
            if output.metrics is None
            else output.metrics.arrival_time
        ),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": sum(
                len(result.token_ids) for result in output.outputs
            ),
            "total_tokens": len(output.prompt_token_ids)
            + sum(len(result.token_ids) for result in output.outputs),
        },
    }
    return api_response


def process_embedding_output(
    outputs: List[EmbeddingRequestOutput], model_name: str
) -> Dict[str, Any]:
    valid_outputs = [output for output in outputs if output is not None]
    query_tokens = sum(len(output.prompt_token_ids) for output in valid_outputs)
    api_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": output.outputs.embedding,
            }
            for i, output in enumerate(outputs)
        ],
        "model": model_name,
        "usage": {
            "query_tokens": query_tokens,
            "total_tokens": query_tokens,
        },
    }
    return api_response


def runtime_metadata_from_request_output(
    result: RequestOutput,
) -> Dict[str, Any]:
    prompt_tokens = list(result.prompt_token_ids or [])
    output_tokens = (
        list(result.outputs[0].token_ids or []) if result.outputs else []
    )
    tokens = prompt_tokens + output_tokens
    kv_transfer_params = getattr(result, "kv_transfer_params", None)
    num_cached_tokens = getattr(result, "num_cached_tokens", None)

    metadata: Dict[str, Any] = {
        "prompt_token_count": len(prompt_tokens),
        "generated_token_count": len(output_tokens),
        "kv_transfer_params_available": kv_transfer_params is not None,
    }
    if num_cached_tokens is not None:
        metadata["num_cached_tokens"] = int(num_cached_tokens)

    runtime_metadata: Dict[str, Any] = {
        "request_id": result.request_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tokens": tokens,
        "metadata": metadata,
    }
    if kv_transfer_params is not None:
        runtime_metadata["kv_transfer_params"] = kv_transfer_params
    return runtime_metadata


def _as_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _expert_key_from_parts(parts: Sequence[Any]) -> str:
    cleaned = [str(part) for part in parts if str(part) not in {"", "None"}]
    if not cleaned:
        return ""
    if len(cleaned) >= 2 and all(part.isdigit() for part in cleaned[-2:]):
        return f"layer:{cleaned[-2]}/expert:{cleaned[-1]}"
    if len(cleaned) == 1:
        return cleaned[0]
    return "/".join(cleaned)


def _normalize_expert_route_histogram(
    value: Any,
    request_id: Optional[str] = None,
) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    if request_id and isinstance(value.get(request_id), Mapping):
        value = value[request_id]

    histogram: Dict[str, int] = {}

    def visit(node: Any, parts: Sequence[Any]) -> None:
        parsed = _as_non_negative_int(node)
        if parsed is not None:
            key = _expert_key_from_parts(parts)
            if key and parsed > 0:
                histogram[key] = histogram.get(key, 0) + parsed
            return
        if isinstance(node, Mapping):
            if "layer_id" in node and "expert_id" in node:
                routed_tokens = _as_non_negative_int(
                    node.get("routed_tokens", node.get("tokens", 0))
                )
                if routed_tokens:
                    key = _expert_key_from_parts(
                        (node["layer_id"], node["expert_id"])
                    )
                    histogram[key] = histogram.get(key, 0) + routed_tokens
                return
            for key, child in node.items():
                visit(child, (*parts, key))
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, (*parts, index))

    visit(value, ())
    return histogram


def _normalize_per_request_expert_route_histogram(
    value: Any,
) -> Dict[str, Dict[str, int]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, Dict[str, int]] = {}
    for request_id, histogram_payload in value.items():
        histogram = _normalize_expert_route_histogram(histogram_payload)
        if histogram:
            result[str(request_id)] = histogram
    return result


def _merge_int_histogram(
    target: Dict[str, int],
    source: Mapping[str, int],
) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + max(0, int(value or 0))


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class LLMEngineStatusDict:
    def __init__(self):
        self.status_dict: Dict[str, Union[RequestOutput, str]] = {}
        self.lock = asyncio.Lock()

    async def update_status(
        self, request_id: str, request_output: Union[RequestOutput, str]
    ):
        async with self.lock:
            self.status_dict[request_id] = request_output

    async def delete_request(self, request_id: str):
        async with self.lock:
            self.status_dict.pop(request_id, None)

    async def return_all_results(self) -> List[Union[RequestOutput, str]]:
        async with self.lock:
            return list(self.status_dict.values())

    async def return_all_request_ids(self) -> List[str]:
        async with self.lock:
            return list(self.status_dict.keys())

    async def request_count(self) -> int:
        async with self.lock:
            return len(self.status_dict)


# Note the GPU resource will be decided when the backend is created
class VllmBackend(SllmBackend):
    # This class implements every method in vllm.entrypoints.openai.api_server
    # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
    # except that we use ray.remote instead of @app and we also add a few new methods:
    # - stop: stops every ongoing request and then stops the backend
    # - get_current_tokens: returns a list of all ongoing request tokens
    # - resume_kv_cache: resumes the key-value cache for the given requests
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.model_name = model
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.request_trace = LLMEngineStatusDict()
        self.pending_kv_restores: Dict[str, int] = {}
        self.request_expert_route_histograms: Dict[str, Dict[str, int]] = (
            _normalize_per_request_expert_route_histogram(
                backend_config.get("per_request_expert_route_histogram")
            )
        )
        self.request_expert_route_histogram_sources: Dict[str, str] = {}
        configured_histogram_source = str(
            backend_config.get(
                "moe_route_histogram_source",
                "instrumentation"
                if self.request_expert_route_histograms
                else "unavailable",
            )
            or "unavailable"
        )
        for request_id in self.request_expert_route_histograms:
            self.request_expert_route_histogram_sources[request_id] = (
                configured_histogram_source
            )
        self.request_expert_route_histogram_kinds: Dict[str, str] = {
            request_id: str(
                backend_config.get(
                    "moe_route_histogram_kind",
                    "request_instrumentation",
                )
                or "request_instrumentation"
            )
            for request_id in self.request_expert_route_histograms
        }
        self.global_expert_hotness: Dict[str, int] = (
            _normalize_expert_route_histogram(
                backend_config.get("global_expert_hotness")
            )
        )
        self.recent_window_expert_hotness: Dict[str, int] = (
            _normalize_expert_route_histogram(
                backend_config.get("recent_window_expert_hotness")
            )
        )
        self._model_config_cache: Optional[Dict[str, Any]] = None
        self._forced_failures_seen = set()
        self.abort_reasons: Dict[str, str] = {}
        # Test-only pacing lets a live-migration smoke keep a real GPU request
        # active while a replacement engine is loading.  It is disabled by
        # default and never changes production behavior.
        try:
            self.test_token_delay_s = max(
                0.0,
                float(
                    os.getenv(
                        "SPOTSERVE_TEST_TOKEN_DELAY_S",
                        backend_config.get("test_token_delay_s", 0.0),
                    )
                ),
            )
        except (TypeError, ValueError):
            self.test_token_delay_s = 0.0
        # if trace_debug is True, request trace will not be deleted after completion
        self.trace_debug = backend_config.get("trace_debug", False)
        self.enforce_eager = backend_config.get("enforce_eager", False)
        self.enable_prefix_caching = backend_config.get(
            "enable_prefix_caching", True
        )
        self.task = backend_config.get("task", "auto")

        async_engine_fields = {f.name for f in fields(AsyncEngineArgs)}
        filtered_engine_config = {
            k: v for k, v in backend_config.items() if k in async_engine_fields
        }

        load_format = backend_config.get("load_format")
        torch_dtype = backend_config.get("torch_dtype")
        if torch_dtype is not None:
            filtered_engine_config["dtype"] = torch_dtype

        if load_format is not None:
            filtered_engine_config["load_format"] = load_format
            filtered_engine_config["model"] = backend_config.get(
                "pretrained_model_name_or_path"
            )
        else:
            storage_path = os.getenv(
                "STORAGE_PATH", os.path.expanduser("~/models")
            )
            model_path = os.path.join(storage_path, "vllm", model)
            filtered_engine_config["model"] = model_path
            filtered_engine_config["load_format"] = "serverless_llm"

        # NOTE: Automatic enable prefix cachinging
        filtered_engine_config["enforce_eager"] = self.enforce_eager
        filtered_engine_config["enable_prefix_caching"] = (
            self.enable_prefix_caching
        )
        # ``task`` is not present in every patched/upstream vLLM release.
        # Keep the backend compatible with runtimes whose AsyncEngineArgs
        # predates the task field instead of making actor creation fail.
        if "task" in async_engine_fields:
            filtered_engine_config["task"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)
        self._async_engine_fields = async_engine_fields

        self.engine = None
        self.model_load_time_s = 0.0

    def _config_value_for_runtime(
        self,
        key: str,
        instance_id: str = "",
        node_id: str = "",
    ) -> Any:
        by_instance = self.backend_config.get(f"{key}_by_instance")
        if isinstance(by_instance, Mapping) and instance_id in by_instance:
            return by_instance[instance_id]
        by_node = self.backend_config.get(f"{key}_by_node")
        if isinstance(by_node, Mapping) and node_id in by_node:
            return by_node[node_id]
        return self.backend_config.get(key)

    def _load_model_config(self) -> Dict[str, Any]:
        if getattr(self, "_model_config_cache", None) is not None:
            return self._model_config_cache
        self._model_config_cache = {}
        model_path = self.backend_config.get("pretrained_model_name_or_path")
        if not isinstance(model_path, str) or not model_path.startswith("/"):
            return self._model_config_cache
        config_path = Path(model_path) / "config.json"
        try:
            if config_path.is_file():
                self._model_config_cache = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
        except Exception:
            logger.debug(
                "Unable to read model config for MoE instrumentation",
                exc_info=True,
            )
        return self._model_config_cache

    @staticmethod
    def _first_positive_config_int(
        config: Mapping[str, Any],
        *keys: str,
    ) -> int:
        for key in keys:
            value = _as_non_negative_int(config.get(key))
            if value and value > 0:
                return value
        return 0

    def _instrumented_expert_placement_snapshot(
        self,
        instance_id: str = "",
        node_id: str = "",
        effective_ep_size: int = 1,
    ) -> Optional[Dict[str, Any]]:
        configured = self._config_value_for_runtime(
            "expert_placement_snapshot",
            instance_id=instance_id,
            node_id=node_id,
        )
        if isinstance(configured, Mapping) and configured:
            return dict(configured)

        infer_enabled = bool(
            self.backend_config.get(
                "infer_moe_expert_placement_from_model_config",
                True,
            )
        )
        if not infer_enabled:
            return None

        model_config = self._load_model_config()
        num_layers = self._first_positive_config_int(
            model_config,
            "num_hidden_layers",
            "n_layer",
            "num_layers",
        )
        num_experts = self._first_positive_config_int(
            model_config,
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "moe_num_experts",
        )
        if num_layers <= 0 or num_experts <= 0:
            return None

        ep_size = max(1, int(effective_ep_size or 1))
        placement: Dict[str, Any] = {}
        for layer_id in range(num_layers):
            for expert_id in range(num_experts):
                rank_id = f"ep-rank-{expert_id % ep_size}"
                placement[f"layer:{layer_id}/expert:{expert_id}"] = {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "rank_id": rank_id,
                    "node_id": node_id,
                    "gpu_id": str(expert_id % ep_size),
                    "placement_source": "derived_from_model_config",
                }
        return placement

    def _record_request_expert_route_histogram(
        self,
        request_id: str,
        histogram: Mapping[str, int],
        source: str,
        kind: str = "request_instrumentation",
    ) -> None:
        normalized: Dict[str, int] = {}
        for key, value in histogram.items():
            parsed = _as_non_negative_int(value)
            if parsed and parsed > 0:
                normalized[str(key)] = parsed
        if not normalized:
            return
        request_key = str(request_id)
        if not hasattr(self, "request_expert_route_histograms"):
            self.request_expert_route_histograms = {}
        if not hasattr(self, "request_expert_route_histogram_sources"):
            self.request_expert_route_histogram_sources = {}
        if not hasattr(self, "request_expert_route_histogram_kinds"):
            self.request_expert_route_histogram_kinds = {}
        if not hasattr(self, "global_expert_hotness"):
            self.global_expert_hotness = {}
        self.request_expert_route_histograms[request_key] = normalized
        self.request_expert_route_histogram_sources[request_key] = (
            str(source or "instrumentation")
        )
        self.request_expert_route_histogram_kinds[request_key] = (
            str(kind or "request_instrumentation")
        )
        _merge_int_histogram(self.global_expert_hotness, normalized)

    def _clear_request_expert_route_histogram(self, request_id: str) -> None:
        request_key = str(request_id)
        getattr(self, "request_expert_route_histograms", {}).pop(
            request_key, None
        )
        getattr(self, "request_expert_route_histogram_sources", {}).pop(
            request_key, None
        )
        getattr(self, "request_expert_route_histogram_kinds", {}).pop(
            request_key, None
        )

    def _pop_request_expert_route_histogram(
        self,
        request_data: Dict[str, Any],
        request_id: str,
    ) -> None:
        raw_histogram = request_data.pop(
            "_spotserve_per_request_expert_route_histogram", None
        )
        if raw_histogram is None:
            raw_histogram = request_data.pop(
                "per_request_expert_route_histogram", None
            )
        source = request_data.pop(
            "_spotserve_moe_route_histogram_source",
            "request_instrumentation",
        )
        kind = request_data.pop(
            "_spotserve_moe_route_histogram_kind",
            "request_instrumentation",
        )
        histogram = _normalize_expert_route_histogram(
            raw_histogram,
            request_id=request_id,
        )
        if histogram:
            self._record_request_expert_route_histogram(
                request_id=request_id,
                histogram=histogram,
                source=str(source or "request_instrumentation"),
                kind=str(kind or "request_instrumentation"),
            )

    def _request_expert_route_metadata(
        self,
        request_id: str,
        runtime_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        request_key = str(request_id)
        runtime_metadata = runtime_metadata or {}
        runtime_histogram = _normalize_expert_route_histogram(
            runtime_metadata.get("per_request_expert_route_histogram"),
            request_id=request_key,
        )
        if runtime_histogram:
            return {
                "per_request_expert_route_histogram": {
                    request_key: runtime_histogram
                },
                "moe_route_histogram_available": True,
                "moe_route_histogram_source": str(
                    runtime_metadata.get("moe_route_histogram_source")
                    or "runtime"
                ),
                "moe_route_histogram_kind": str(
                    runtime_metadata.get("moe_route_histogram_kind")
                    or (
                        "runtime_observed_topk"
                        if runtime_metadata.get("moe_route_histogram_source")
                        == "vllm_runtime_topk"
                        else "runtime"
                    )
                ),
            }

        histogram = getattr(
            self, "request_expert_route_histograms", {}
        ).get(request_key)
        if not histogram:
            return {}
        return {
            "per_request_expert_route_histogram": {
                request_key: dict(histogram)
            },
            "moe_route_histogram_available": True,
            "moe_route_histogram_source": (
                getattr(
                    self,
                    "request_expert_route_histogram_sources",
                    {},
                ).get(
                    request_key,
                    "instrumentation",
                )
            ),
            "moe_route_histogram_kind": (
                getattr(
                    self,
                    "request_expert_route_histogram_kinds",
                    {},
                ).get(
                    request_key,
                    "request_instrumentation",
                )
            ),
        }

    def _merge_request_expert_route_metadata(
        self,
        runtime_metadata: Mapping[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        merged = dict(runtime_metadata)
        merged.update(
            self._request_expert_route_metadata(
                request_id=request_id,
                runtime_metadata=merged,
            )
        )
        return merged

    def _engine_parallel_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        engine_args = getattr(self, "engine_args", None)
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "enable_expert_parallel",
        ):
            value = getattr(
                engine_args,
                field_name,
                self.backend_config.get(field_name),
            )
            if value is not None:
                metadata[field_name] = value
        enable_ep = bool(metadata.get("enable_expert_parallel", False))
        tensor_parallel_size = max(
            1, int(metadata.get("tensor_parallel_size", 1) or 1)
        )
        data_parallel_size = max(
            1, int(metadata.get("data_parallel_size", 1) or 1)
        )
        effective_ep_size = (
            tensor_parallel_size * data_parallel_size if enable_ep else 1
        )
        metadata["expert_parallel_enabled"] = enable_ep
        planned_ep_size = self.backend_config.get(
            "planned_effective_expert_parallel_size",
            self.backend_config.get(
                "planned_expert_parallel_size",
                self.backend_config.get("expert_parallel_size", 1),
            )
        )
        try:
            planned_ep_size = max(1, int(planned_ep_size or 1))
        except (TypeError, ValueError):
            planned_ep_size = 1
        metadata["planned_effective_expert_parallel_size"] = planned_ep_size
        metadata["planned_expert_parallel_size"] = metadata[
            "planned_effective_expert_parallel_size"
        ]
        metadata["effective_expert_parallel_size"] = effective_ep_size
        metadata["runtime_effective_expert_parallel_size"] = effective_ep_size
        metadata["derived_effective_expert_parallel_size"] = effective_ep_size
        metadata["expert_parallel_size"] = effective_ep_size
        metadata["expert_parallel_size_verified"] = True
        metadata["expert_parallel_size_source"] = (
            "derived_from_tp_dp" if enable_ep else "disabled"
        )
        metadata["vllm_data_parallel_size"] = data_parallel_size
        metadata["sllm_replica_count"] = max(
            1, int(self.backend_config.get("replica_count", 1) or 1)
        )
        metadata["expert_physical_replication_factor"] = max(
            1,
            int(
                self.backend_config.get(
                    "expert_physical_replication_factor", 1
                )
                or 1
            ),
        )
        placement_snapshot = self._instrumented_expert_placement_snapshot(
            instance_id=instance_id,
            node_id=node_id,
            effective_ep_size=effective_ep_size,
        )
        configured_placement = self._config_value_for_runtime(
            "expert_placement_snapshot",
            instance_id=instance_id,
            node_id=node_id,
        )
        default_placement_source = "unavailable"
        if isinstance(configured_placement, Mapping) and configured_placement:
            default_placement_source = "instrumentation"
        elif placement_snapshot:
            default_placement_source = "derived_from_model_config"
        metadata["expert_placement_available"] = bool(placement_snapshot)
        metadata["placement_epoch"] = (
            _as_non_negative_int(
                self.backend_config.get("placement_epoch", 0)
            )
            or 0
        )
        metadata["placement_version"] = metadata["placement_epoch"]
        metadata["placement_source"] = (
            str(
                self.backend_config.get(
                    "placement_source",
                    default_placement_source,
                )
            )
            or "unavailable"
        )
        if placement_snapshot:
            metadata["expert_placement_snapshot"] = placement_snapshot
            placement_blob = json.dumps(
                placement_snapshot,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            metadata["expert_placement_fingerprint"] = format(
                zlib.crc32(placement_blob) & 0xFFFFFFFF,
                "08x",
            )
        metadata["expert_placement_epoch"] = metadata["placement_epoch"]
        request_histograms = getattr(
            self, "request_expert_route_histograms", {}
        )
        route_histogram_available = bool(request_histograms)
        request_histogram_kinds = getattr(
            self, "request_expert_route_histogram_kinds", {}
        )
        metadata["moe_route_histogram_available"] = route_histogram_available
        metadata["moe_route_histogram_source"] = (
            ",".join(
                sorted(
                    {
                        source
                        for source in (
                            getattr(
                                self,
                                "request_expert_route_histogram_sources",
                                {},
                            )
                        ).values()
                        if source
                    }
                )
            )
            if route_histogram_available
            else "unavailable"
        )
        metadata["moe_route_histogram_kind"] = (
            ",".join(
                sorted(
                    {
                        kind
                        for kind in request_histogram_kinds.values()
                        if kind
                    }
                )
            )
            if route_histogram_available
            else "unavailable"
        )
        if route_histogram_available:
            metadata["per_request_expert_route_histogram"] = {
                request_id: dict(histogram)
                for request_id, histogram in request_histograms.items()
            }
        if getattr(self, "global_expert_hotness", {}):
            metadata["global_expert_hotness"] = dict(
                self.global_expert_hotness
            )
        if getattr(self, "recent_window_expert_hotness", {}):
            metadata["recent_window_expert_hotness"] = dict(
                self.recent_window_expert_hotness
            )
        metadata["parallel_plan_mismatch"] = (
            metadata["planned_effective_expert_parallel_size"] != effective_ep_size
        )
        return metadata

    def _runtime_hook(self, *names: str):
        """Find an optional state-transfer hook exposed by the vLLM runtime.

        Upstream vLLM does not currently have one stable public API for this.
        Keeping the discovery here lets patched engines/connectors expose the
        small CPY contract without depending on scheduler internals.
        """
        owners = [self.engine]
        for attribute in ("engine_core", "engine", "_engine"):
            owner = getattr(self.engine, attribute, None)
            if owner is not None and owner not in owners:
                owners.append(owner)
        for owner in owners:
            for name in names:
                hook = getattr(owner, name, None)
                if callable(hook):
                    return hook
        return None

    async def _call_runtime_hook(self, names, **kwargs):
        hook = self._runtime_hook(*names)
        if hook is None:
            return None
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            # Some RPC/proxy callables do not expose an inspectable signature.
            call_kwargs = kwargs
        else:
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in signature.parameters.values()
            )
            call_kwargs = (
                kwargs
                if accepts_kwargs
                else {
                    k: v
                    for k, v in kwargs.items()
                    if k in signature.parameters
                }
            )
        return await _maybe_await(hook(**call_kwargs))

    async def _request_runtime_moe_metadata(
        self,
        request_id: str,
    ) -> Dict[str, Any]:
        extra = await self._call_runtime_hook(
            (
                "get_request_moe_metadata",
                "get_request_expert_route_metadata",
                "get_request_expert_routing_metadata",
            ),
            request_id=str(request_id),
        )
        return dict(extra) if isinstance(extra, dict) else {}

    async def _runtime_moe_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ) -> Dict[str, Any]:
        extra = await self._call_runtime_hook(
            (
                "get_moe_runtime_metadata",
                "get_expert_placement_metadata",
                "get_moe_placement_metadata",
            ),
            instance_id=instance_id,
            node_id=node_id,
        )
        return dict(extra) if isinstance(extra, dict) else {}

    async def _request_runtime_metadata(
        self, result: RequestOutput
    ) -> Dict[str, Any]:
        metadata = runtime_metadata_from_request_output(result)
        extra = await self._call_runtime_hook(
            ("get_request_kv_metadata", "get_kv_cache_metadata"),
            request_id=result.request_id,
        )
        if isinstance(extra, dict):
            metadata.update(extra)
        metadata.update(
            await self._request_runtime_moe_metadata(result.request_id)
        )
        metadata = self._merge_request_expert_route_metadata(
            metadata,
            str(result.request_id),
        )
        restore_supported = await self.supports_state_restore()
        metadata["supports_state_export"] = restore_supported
        metadata["supports_state_restore"] = restore_supported
        return metadata

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            started_at = time.monotonic()
            self._configure_spotserve_nixl_side_channel_port()
            self._configure_spotserve_moe_route_tracing()
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.model_load_time_s = time.monotonic() - started_at
            self.status = BackendStatus.RUNNING

    def _spotserve_runtime_identity(self) -> str:
        try:
            import ray

            context = ray.get_runtime_context()
            actor_id = context.get_actor_id()
            actor_name = ""
            get_actor_name = getattr(context, "get_actor_name", None)
            if callable(get_actor_name):
                actor_name = str(get_actor_name() or "")
            actor_id_text = (
                actor_id.hex()
                if callable(getattr(actor_id, "hex", None))
                else str(actor_id)
            )
            return f"{actor_name}:{actor_id_text}"
        except Exception:
            return f"{self.model_name}:{id(self)}"

    def _derive_spotserve_nixl_side_channel_port(
        self, identity: str
    ) -> int:
        exact_port = self.backend_config.get("nixl_side_channel_port")
        if exact_port is not None:
            return int(exact_port)
        base_port = int(
            self.backend_config.get(
                "nixl_side_channel_base_port",
                os.getenv(
                    "SPOTSERVE_NIXL_SIDE_CHANNEL_BASE_PORT",
                    os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600"),
                ),
            )
        )
        port_span = max(
            1,
            int(
                self.backend_config.get(
                    "nixl_side_channel_port_span",
                    os.getenv("SPOTSERVE_NIXL_SIDE_CHANNEL_PORT_SPAN", "20000"),
                )
            ),
        )
        return base_port + (zlib.crc32(identity.encode("utf-8")) % port_span)

    def _configure_spotserve_nixl_side_channel_port(self) -> None:
        kv_transfer_config = self.backend_config.get("kv_transfer_config") or {}
        if kv_transfer_config.get("kv_connector") != "NixlConnector":
            return
        port = self._derive_spotserve_nixl_side_channel_port(
            self._spotserve_runtime_identity()
        )
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)
        logger.info("Configured vLLM NIXL side-channel port: %s", port)

    def _configure_spotserve_moe_route_tracing(self) -> None:
        configured = self.backend_config.get(
            "enable_moe_route_instrumentation",
            self.backend_config.get("spotserve_moe_route_tracing"),
        )
        if configured is None:
            return
        enabled = _as_bool(configured)
        os.environ["VLLM_SPOTSERVE_MOE_TRACE"] = "1" if enabled else "0"
        logger.info("Configured vLLM MoE route tracing: %s", enabled)

    def _pop_spotserve_request_controls(
        self,
        request_data: Dict[str, Any],
        request_id: str,
        skip_forced_failure: bool = False,
    ) -> Optional[Dict[str, Any]]:
        failure_mode = request_data.pop("force_failure", None)
        alternate_failure_mode = request_data.pop(
            "force_backend_failure", None
        )
        failure_mode = failure_mode or alternate_failure_mode

        fail_after_tokens = request_data.pop("force_fail_after_tokens", None)
        alternate_fail_after_tokens = request_data.pop(
            "force_preempt_after_tokens", None
        )
        if fail_after_tokens is None:
            fail_after_tokens = alternate_fail_after_tokens

        force_once = bool(request_data.pop("force_fail_once", True))
        no_current_tokens = bool(
            request_data.pop("force_no_current_tokens", False)
        )
        request_data.pop("_completed_tokens", None)

        if not failure_mode or fail_after_tokens is None:
            return None
        if skip_forced_failure:
            return None

        failure_mode = str(failure_mode).lower()
        failure_key = f"{self.model_name}:{request_id}:{failure_mode}"
        if force_once and failure_key in self._forced_failures_seen:
            return None

        return {
            "failure_mode": failure_mode,
            "fail_after_tokens": int(fail_after_tokens),
            "failure_key": failure_key,
            "force_once": force_once,
            "no_current_tokens": no_current_tokens,
        }

    def _forced_failure_ready(
        self,
        forced_failure: Optional[Dict[str, Any]],
        output: RequestOutput,
    ) -> bool:
        if not forced_failure or not output.outputs:
            return False
        output_tokens = list(output.outputs[0].token_ids or [])
        return len(output_tokens) >= int(forced_failure["fail_after_tokens"])

    async def _forced_failure_result(
        self,
        request_data: Dict[str, Any],
        forced_failure: Dict[str, Any],
        output: RequestOutput,
    ) -> Dict[str, Any]:
        if forced_failure.get("force_once", True):
            self._forced_failures_seen.add(str(forced_failure["failure_key"]))

        output_tokens = (
            list(output.outputs[0].token_ids or []) if output.outputs else []
        )
        tokens = list(output.prompt_token_ids or []) + output_tokens
        current_output = [] if forced_failure["no_current_tokens"] else [tokens]
        completed_tokens = len(output_tokens)
        state_snapshot = await self.export_inference_state(
            request_data=request_data,
            current_output=current_output,
            completed_tokens=completed_tokens,
        )
        try:
            await self.engine.abort(output.request_id)
        except Exception:
            logger.debug(
                "Could not abort forced-preempted vLLM request",
                exc_info=True,
            )

        failure_mode = str(forced_failure["failure_mode"])
        if failure_mode in {"preempt", "preempted", "preemption"}:
            self._clear_request_expert_route_histogram(output.request_id)
            return {
                "error": (
                    "Forced vLLM backend preemption after "
                    f"{completed_tokens} tokens"
                ),
                "preempted": True,
                "current_output": current_output,
                "completed_tokens": completed_tokens,
                "_spotserve_inference_state": state_snapshot,
            }

        raise RuntimeError(
            "Forced vLLM backend failure after "
            f"{completed_tokens} tokens"
        )

    async def generate(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if request_data is None:
            return {"error": "Request data is missing"}

        model_name: str = request_data.pop("model", "vllm-model")
        messages: Dict[Dict[str, str], str] = request_data.pop("messages", [])
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # If prompt is not provided, construct it from messages
        inputs: Union[str, TokensPrompt] = request_data.pop(
            "prompt", construct_prompt
        )
        has_input_tokens = request_data.get("input_tokens") is not None
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )
        self._pop_request_expert_route_histogram(request_data, request_id)
        # Test-only observability for deterministic migration experiments.
        # The private request flag is removed before SamplingParams is built,
        # so it cannot change production sampling semantics or leak into the
        # public response unless explicitly requested by a smoke test.
        return_token_ids = bool(
            request_data.pop("_spotserve_return_token_ids", False)
        )
        request_token_delay_s = request_data.pop(
            "_spotserve_token_delay_s", self.test_token_delay_s
        )
        try:
            request_token_delay_s = max(0.0, float(request_token_delay_s))
        except (TypeError, ValueError):
            request_token_delay_s = self.test_token_delay_s
        # A replay/restore request already carries the exported prefix as
        # input_tokens.  Do not apply the synthetic source hold to that retry;
        # it is only there to create a preemption window on the original
        # prompt request.
        if has_input_tokens:
            request_token_delay_s = 0.0
        # Keep migration requests genuinely in-flight.  DELTA output prevents
        # the engine from materializing the full sequence before the planner
        # exports its KV lease, while leaving normal requests unchanged.
        if (
            request_token_delay_s > 0
            and not has_input_tokens
            and RequestOutputKind is not None
        ):
            request_data.setdefault("output_kind", RequestOutputKind.DELTA)
        forced_failure = self._pop_spotserve_request_controls(
            request_data,
            request_id,
            skip_forced_failure=has_input_tokens,
        )
        state_request_data = dict(request_data)
        state_request_data.update(
            {
                "model": model_name,
                "messages": messages,
                "request_id": request_id,
            }
        )
        if isinstance(inputs, str):
            state_request_data["prompt"] = inputs

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            self._clear_request_expert_route_histogram(request_id)
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # TODO stream results

        # Non-stream case.  A V6 replan can abort this generator after the
        # router has exported the request state.  Return a structured marker
        # so the original request coroutine retries on the new deployment
        # instead of surfacing an engine cancellation to the client.
        final_output = None
        try:
            async for response_output in results_generator:
                final_output = response_output
                await self.request_trace.update_status(
                    request_id, response_output
                )
                if request_token_delay_s > 0:
                    # Keep the test-only hold interruptible.  A real abort
                    # must wake the backend quickly so the router can switch
                    # traffic instead of waiting for the whole synthetic
                    # delay during actor shutdown.
                    remaining = request_token_delay_s
                    while remaining > 0 and request_id not in self.abort_reasons:
                        interval = min(0.1, remaining)
                        await asyncio.sleep(interval)
                        remaining -= interval
                if self._forced_failure_ready(forced_failure, response_output):
                    return await self._forced_failure_result(
                        state_request_data, forced_failure, response_output
                    )
        except BaseException:
            reason = self.abort_reasons.pop(request_id, None)
            if reason is not None:
                return await self._reparallelization_abort_result(
                    request_id, reason
                )
            self._clear_request_expert_route_histogram(request_id)
            raise

        reason = self.abort_reasons.pop(request_id, None)
        if reason is not None:
            return await self._reparallelization_abort_result(
                request_id, reason, final_output
            )
        if final_output is None:
            raise RuntimeError("vLLM returned no final output")

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)
            self._clear_request_expert_route_histogram(request_id)

        response = process_output(final_output, model_name)
        expected_blocks = self.pending_kv_restores.pop(request_id, 0)
        if expected_blocks:
            cached_tokens = int(
                getattr(final_output, "num_cached_tokens", 0) or 0
            )
            response["_spotserve_kv_restore"] = {
                "restored": cached_tokens > 0,
                "restored_blocks": expected_blocks if cached_tokens > 0 else 0,
                "cached_tokens": cached_tokens,
                "reason": (
                    "nixl_kv_attach_completed"
                    if cached_tokens > 0
                    else "nixl_kv_attach_empty"
                ),
            }
        if return_token_ids:
            response["_spotserve_token_ids"] = list(
                final_output.outputs[0].token_ids or []
            )
            response["_spotserve_prompt_token_ids"] = list(
                final_output.prompt_token_ids or []
            )
        return response

    async def _reparallelization_abort_result(
        self,
        request_id: str,
        reason: str,
        final_output: Optional[RequestOutput] = None,
    ) -> Dict[str, Any]:
        """Build a retry marker after V6 exported and aborted a request."""
        if final_output is None:
            results = await self.request_trace.return_all_results()
            for result in results:
                if isinstance(result, RequestOutput) and result.request_id == request_id:
                    final_output = result
                    break

        current_output: List[List[int]] = []
        if final_output is not None:
            prompt_tokens = list(final_output.prompt_token_ids or [])
            generated_tokens = list(
                final_output.outputs[0].token_ids or []
            ) if final_output.outputs else []
            tokens = prompt_tokens + generated_tokens
            if tokens:
                current_output = [tokens]

        await self.request_trace.delete_request(request_id)
        self._clear_request_expert_route_histogram(request_id)
        return {
            "preempted": True,
            "_spotserve_reparallelization": reason == "reparallelization",
            "request_id": request_id,
            "current_output": current_output,
            "completed_tokens": len(current_output[0]) if current_output else 0,
            "reason": reason,
        }

    async def abort_request(
        self, request_id: str, reason: str = "preempted"
    ) -> Dict[str, Any]:
        """Abort one live request for a controlled deployment transition."""
        request_id = str(request_id)
        if self.engine is None:
            return {"aborted": False, "reason": "engine_not_initialized"}
        self.abort_reasons[request_id] = str(reason)
        try:
            await self.engine.abort(request_id)
        except Exception as exc:
            self.abort_reasons.pop(request_id, None)
            logger.info("Could not abort request %s: %s", request_id, exc)
            return {"aborted": False, "reason": str(exc)}
        return {"aborted": True, "request_id": request_id, "reason": reason}

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort all requests
        requests = await self.request_trace.return_all_request_ids()
        tasks = [self.engine.abort(request_id) for request_id in requests]
        await asyncio.gather(*tasks)
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        deadline = time.monotonic() + max(
            5.0,
            float(self.backend_config.get("stop_timeout_s", 30.0) or 30.0),
        )
        while (
            await self.request_trace.request_count() > 0
            and time.monotonic() < deadline
        ):
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
        if await self.request_trace.request_count() > 0:
            logger.warning(
                "Stopping vLLM backend with unfinished requests after timeout"
            )
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]
        tokens: List[List[int]] = []
        for result in ongoing_results:
            if not result.outputs:
                continue
            prompt_tokens = list(result.prompt_token_ids or [])
            output_tokens = list(result.outputs[0].token_ids or [])
            if prompt_tokens or output_tokens:
                tokens.append(prompt_tokens + output_tokens)
        return tokens

    async def get_context_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ) -> List[Dict[str, Any]]:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        all_metadata_hook = self._runtime_hook(
            "get_all_request_kv_metadata"
        )
        if all_metadata_hook is not None:
            try:
                runtime_snapshots = await _maybe_await(all_metadata_hook())
            except Exception:
                logger.exception("Could not query active vLLM KV metadata")
            else:
                if isinstance(runtime_snapshots, list):
                    restore_supported = await self.supports_state_restore()
                    contexts = []
                    for snapshot in runtime_snapshots:
                        if not isinstance(snapshot, dict):
                            continue
                        if not snapshot.get("found", True):
                            continue
                        request_id = str(snapshot.get("request_id", ""))
                        snapshot.update(
                            await self._request_runtime_moe_metadata(
                                request_id
                            )
                        )
                        snapshot = self._merge_request_expert_route_metadata(
                            snapshot,
                            request_id,
                        )
                        snapshot["supports_state_export"] = restore_supported
                        snapshot["supports_state_restore"] = restore_supported
                        contexts.append(
                            get_vllm_context_metadata(
                                model_name=self.model_name,
                                instance_id=instance_id or self.model_name,
                                node_id=node_id,
                                runtime_metadata=snapshot,
                            )
                        )
                    return contexts

        results = await self.request_trace.return_all_results()
        ongoing_results: List[RequestOutput] = [
            result for result in results if isinstance(result, RequestOutput)
        ]

        metadata: List[Dict[str, Any]] = []
        for result in ongoing_results:
            if not result.outputs:
                continue
            metadata.append(
                get_vllm_context_metadata(
                    model_name=self.model_name,
                    instance_id=instance_id or self.model_name,
                    node_id=node_id,
                    runtime_metadata=await self._request_runtime_metadata(result),
                )
            )
        return metadata

    async def get_request_kv_metadata(
        self,
        request_id: str,
        instance_id: str = "",
        node_id: str = "",
    ) -> Dict[str, Any]:
        """Return one live request's connector metadata.

        The vLLM frontend keeps an external request ID to internal sequence
        ID map.  Querying that map directly is important during a replan:
        ``get_all_request_kv_metadata`` can legitimately return an empty
        list while another output is being coalesced, even though this
        request still owns live KV blocks.
        """
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"request_id": request_id, "found": False}

        live = await self._call_runtime_hook(
            ("get_request_kv_metadata", "get_kv_cache_metadata"),
            request_id=str(request_id),
        )
        if isinstance(live, dict):
            payload = dict(live)
            payload.setdefault("request_id", str(request_id))
            if not payload.get("found", True):
                return payload
            payload.update(
                await self._request_runtime_moe_metadata(str(request_id))
            )
            payload = self._merge_request_expert_route_metadata(
                payload,
                str(request_id),
            )
            return get_vllm_context_metadata(
                model_name=self.model_name,
                instance_id=instance_id or self.model_name,
                node_id=node_id,
                runtime_metadata={
                    **payload,
                    "supports_state_export": await self.supports_state_restore(),
                    "supports_state_restore": await self.supports_state_restore(),
                },
            )

        # Fallback for runtimes exposing only the request trace.  This path is
        # still useful for token replay, but it is deliberately conservative
        # about KV capability fields.
        results = await self.request_trace.return_all_results()
        for result in results:
            if not isinstance(result, RequestOutput):
                continue
            if str(result.request_id) != str(request_id):
                continue
            return get_vllm_context_metadata(
                model_name=self.model_name,
                instance_id=instance_id or self.model_name,
                node_id=node_id,
                runtime_metadata=await self._request_runtime_metadata(result),
            )
        return {"request_id": str(request_id), "found": False}

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> Dict[str, Any]:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {
                    "action": "prefix_warmup",
                    "warmed": False,
                    "reason": "engine_not_running",
                    "true_kv_block_transfer": False,
                }
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)
        return {
            "action": "prefix_warmup",
            "operation_kind": "prefix_warmup",
            "warmed": True,
            "warmed_sequences": len(request_datas),
            "warmed_tokens": sum(len(tokens) for tokens in request_datas),
            "true_kv_block_transfer": False,
            "reason": "resume_kv_cache_token_replay",
        }

    async def supports_state_restore(self) -> bool:
        if self.engine is None:
            return False
        advertised = self._runtime_hook("supports_state_restore")
        if advertised is not None:
            try:
                if not bool(await _maybe_await(advertised())):
                    return False
            except Exception:
                logger.exception("vLLM state-restore capability probe failed")
                return False
        return (
            self._runtime_hook("export_inference_state", "export_kv_cache_state")
            is not None
            and self._runtime_hook(
                "restore_inference_state", "restore_kv_cache_state"
            )
            is not None
        )

    async def get_state_restore_diagnostics(self) -> Dict[str, Any]:
        """Expose the live engine capability probe for smoke-test diagnosis."""
        advertised = self._runtime_hook("supports_state_restore")
        exporter = self._runtime_hook(
            "export_inference_state", "export_kv_cache_state"
        )
        restorer = self._runtime_hook(
            "restore_inference_state", "restore_kv_cache_state"
        )
        advertised_value = None
        advertised_error = None
        if advertised is not None:
            try:
                advertised_value = bool(await _maybe_await(advertised()))
            except Exception as exc:
                advertised_error = repr(exc)
        return {
            "supports_state_restore": await self.supports_state_restore(),
            "advertised_support": advertised_value,
            "advertised_error": advertised_error,
            "export_hook": getattr(exporter, "__qualname__", repr(exporter))
            if exporter is not None
            else None,
            "restore_hook": getattr(restorer, "__qualname__", repr(restorer))
            if restorer is not None
            else None,
        }

    async def export_inference_state(
        self,
        request_data: Optional[Dict[str, Any]] = None,
        current_output: Optional[List[List[int]]] = None,
        completed_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        request_data = request_data or {}
        request_id = request_data.get("request_id")
        runtime_metadata: Dict[str, Any] = {}
        results = await self.request_trace.return_all_results()
        for result in results:
            if not isinstance(result, RequestOutput):
                continue
            if request_id is not None and result.request_id != request_id:
                continue
            runtime_metadata = await self._request_runtime_metadata(result)
            break
        if request_id is not None and not runtime_metadata:
            live_metadata = await self._call_runtime_hook(
                ("get_request_kv_metadata", "get_kv_cache_metadata"),
                request_id=request_id,
            )
            if isinstance(live_metadata, dict) and live_metadata.get(
                "found", True
            ):
                live_metadata.update(
                    await self._request_runtime_moe_metadata(str(request_id))
                )
                runtime_metadata = self._merge_request_expert_route_metadata(
                    dict(live_metadata),
                    str(request_id),
                )
        if current_output is None:
            if runtime_metadata.get("tokens"):
                current_output = [list(runtime_metadata["tokens"])]
            else:
                current_output = await self.get_current_tokens()
        fallback = get_vllm_inference_state(
            model_name=self.model_name,
            request_data=request_data or {},
            current_output=current_output,
            completed_tokens=completed_tokens,
            instance_id=str(request_data.get("instance_id", "")),
            node_id=str(request_data.get("node_id", "")),
            runtime_metadata=runtime_metadata,
        )
        if not await self.supports_state_restore():
            return fallback

        export_error = None
        try:
            exported = await self._call_runtime_hook(
                ("export_inference_state", "export_kv_cache_state"),
                request_id=request_id or runtime_metadata.get("request_id"),
                request_data=request_data,
                runtime_metadata=runtime_metadata,
            )
        except Exception as exc:
            logger.exception("vLLM KV state export failed")
            exported = None
            export_error = repr(exc)
        if (
            not isinstance(exported, dict)
            or exported.get("supports_restore") is not True
            or exported.get("state_kind") != "vllm_kv_snapshot"
            or not isinstance(exported.get("runtime_state"), dict)
            or not exported["runtime_state"]
        ):
            fallback["metadata"]["reason"] = "vllm_kv_export_failed"
            if isinstance(exported, dict) and exported.get("error"):
                fallback["metadata"]["export_error"] = exported["error"]
            if isinstance(exported, dict) and exported.get("reason"):
                fallback["metadata"]["export_reason"] = exported["reason"]
            elif export_error is not None:
                fallback["metadata"]["export_error"] = export_error
            return fallback

        state = dict(fallback)
        state.update(exported)
        state["backend"] = "vllm"
        state["model_name"] = self.model_name
        state["state_kind"] = "vllm_kv_snapshot"
        state["supports_restore"] = True
        state_metadata = dict(fallback["metadata"])
        state_metadata.update(exported.get("metadata", {}) or {})
        state_metadata.pop("reason", None)
        state_metadata.setdefault("cache_engine", "vllm")
        state_metadata.setdefault("can_restore_same_node", False)
        state_metadata.setdefault("can_restore_cross_node", False)
        parallel_metadata = self._engine_parallel_metadata()
        config_metadata = {
            "tensor_parallel_size": parallel_metadata.get(
                "tensor_parallel_size"
            ),
            "pipeline_parallel_size": parallel_metadata.get(
                "pipeline_parallel_size"
            ),
            "data_parallel_size": parallel_metadata.get(
                "data_parallel_size"
            ),
            "vllm_data_parallel_size": parallel_metadata.get(
                "vllm_data_parallel_size"
            ),
            "replica_count": parallel_metadata.get("sllm_replica_count"),
            "sllm_replica_count": parallel_metadata.get("sllm_replica_count"),
            "expert_parallel_enabled": parallel_metadata.get(
                "expert_parallel_enabled"
            ),
            "planned_expert_parallel_size": parallel_metadata.get(
                "planned_expert_parallel_size"
            ),
            "planned_effective_expert_parallel_size": parallel_metadata.get(
                "planned_effective_expert_parallel_size"
            ),
            "effective_expert_parallel_size": parallel_metadata.get(
                "effective_expert_parallel_size"
            ),
            "runtime_effective_expert_parallel_size": (
                parallel_metadata.get("runtime_effective_expert_parallel_size")
            ),
            "derived_effective_expert_parallel_size": (
                parallel_metadata.get("derived_effective_expert_parallel_size")
            ),
            "expert_parallel_size_verified": parallel_metadata.get(
                "expert_parallel_size_verified"
            ),
            "expert_parallel_size_source": parallel_metadata.get(
                "expert_parallel_size_source"
            ),
            "parallel_plan_mismatch": parallel_metadata.get(
                "parallel_plan_mismatch"
            ),
            "expert_physical_replication_factor": parallel_metadata.get(
                "expert_physical_replication_factor"
            ),
            "expert_placement_available": parallel_metadata.get(
                "expert_placement_available"
            ),
            "placement_epoch": parallel_metadata.get("placement_epoch"),
            "placement_version": parallel_metadata.get("placement_version"),
            "expert_placement_epoch": parallel_metadata.get(
                "expert_placement_epoch"
            ),
            "placement_source": parallel_metadata.get("placement_source"),
            "expert_placement_fingerprint": parallel_metadata.get(
                "expert_placement_fingerprint"
            ),
            "moe_route_histogram_available": parallel_metadata.get(
                "moe_route_histogram_available"
            ),
            "moe_route_histogram_source": parallel_metadata.get(
                "moe_route_histogram_source"
            ),
            "moe_route_histogram_kind": parallel_metadata.get(
                "moe_route_histogram_kind"
            ),
            "per_request_expert_route_histogram": parallel_metadata.get(
                "per_request_expert_route_histogram"
            ),
            "global_expert_hotness": parallel_metadata.get(
                "global_expert_hotness"
            ),
            "recent_window_expert_hotness": parallel_metadata.get(
                "recent_window_expert_hotness"
            ),
            "cache_block_size": self.backend_config.get("block_size"),
            "cache_dtype": self.backend_config.get("kv_cache_dtype"),
            "cache_layout": self.backend_config.get("kv_cache_layout"),
            "state_restore_requires_ep_layout": self.backend_config.get(
                "state_restore_requires_ep_layout"
            ),
        }
        config_metadata["expert_parallel_size"] = parallel_metadata.get(
            "expert_parallel_size"
        )
        for key, value in config_metadata.items():
            if value is not None:
                state_metadata.setdefault(key, value)
        state["metadata"] = state_metadata
        return state

    async def restore_inference_state(
        self,
        state: Dict[str, Any],
        request_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state_kind = state.get("state_kind", "token_snapshot")
        if (
            state_kind != "vllm_kv_snapshot"
            or not state.get("supports_restore")
            or not isinstance(state.get("runtime_state"), dict)
            or not state["runtime_state"]
            or not await self.supports_state_restore()
        ):
            return {
                "restored": False,
                "reason": "vllm_kv_restore_not_available",
                "state_kind": state_kind,
            }
        if (
            state.get("backend") != "vllm"
            or state.get("model_name") != self.model_name
        ):
            return {
                "restored": False,
                "reason": "incompatible_model",
                "state_kind": state_kind,
            }

        request_data = request_data or {}
        metadata = state.get("metadata", {}) or {}
        source_node = state.get("node_id") or metadata.get("source_node_id")
        target_node = request_data.get("node_id") or request_data.get(
            "target_node_id"
        )
        if not source_node or not target_node:
            return {
                "restored": False,
                "reason": "unknown_restore_scope",
                "state_kind": state_kind,
            }
        cross_node = bool(
            source_node and target_node and source_node != target_node
        )
        if cross_node and not metadata.get("can_restore_cross_node", False):
            return {
                "restored": False,
                "reason": "cross_node_restore_unsupported",
                "state_kind": state_kind,
            }
        if not cross_node and not metadata.get("can_restore_same_node", False):
            return {
                "restored": False,
                "reason": "same_node_restore_unsupported",
                "state_kind": state_kind,
            }

        compatibility_keys = {
            "model_revision": "revision",
            "tensor_parallel_size": "tensor_parallel_size",
            "pipeline_parallel_size": "pipeline_parallel_size",
            "cache_block_size": "block_size",
            "cache_dtype": "kv_cache_dtype",
            "cache_layout": "kv_cache_layout",
        }
        for state_key, config_key in compatibility_keys.items():
            source_value = metadata.get(state_key)
            target_value = self.backend_config.get(config_key)
            if (
                source_value is not None
                and target_value is not None
                and str(source_value) != str(target_value)
            ):
                return {
                    "restored": False,
                    "reason": "incompatible_cache_config",
                    "state_kind": state_kind,
                }

        ep_required = _as_bool(
            metadata.get(
                "state_restore_requires_ep_layout",
                self.backend_config.get("state_restore_requires_ep_layout"),
            ),
            False,
        )
        if ep_required:
            target_parallel_metadata = self._engine_parallel_metadata(
                instance_id=str(request_data.get("target_instance_id", "")),
                node_id=str(target_node or ""),
            )
            for state_key in (
                "effective_expert_parallel_size",
                "expert_parallel_enabled",
                "expert_placement_fingerprint",
            ):
                source_value = metadata.get(state_key)
                target_value = target_parallel_metadata.get(state_key)
                if (
                    source_value is not None
                    and target_value is not None
                    and str(source_value) != str(target_value)
                ):
                    return {
                        "restored": False,
                        "reason": "incompatible_ep_layout",
                        "state_kind": state_kind,
                    }

        try:
            restored = await self._call_runtime_hook(
                ("restore_inference_state", "restore_kv_cache_state"),
                state=state,
                request_id=request_data.get("request_id")
                or state.get("request_id"),
                request_data=request_data,
            )
        except Exception:
            logger.exception("vLLM KV state restore failed")
            restored = None
        if not isinstance(restored, dict):
            restored = {"restored": bool(restored)}
        if restored.get("staged"):
            request_id = request_data.get("request_id") or state.get("request_id")
            expected_blocks = int(restored.get("expected_blocks", 0) or 0)
            if not request_id or expected_blocks <= 0:
                return {
                    "restored": False,
                    "reason": "vllm_kv_restore_staging_failed",
                    "state_kind": state_kind,
                }
            self.pending_kv_restores[str(request_id)] = expected_blocks
            return {
                "restored": False,
                "staged": True,
                "expected_blocks": expected_blocks,
                "state_kind": state_kind,
            }
        if restored.get("state_kind", state_kind) != state_kind:
            return {
                "restored": False,
                "reason": "vllm_kv_restore_state_kind_mismatch",
                "state_kind": state_kind,
            }
        restored["state_kind"] = state_kind
        if restored.get("restored"):
            restored_blocks = restored.get(
                "restored_blocks", metadata.get("kv_block_count", 0)
            )
            try:
                restored_blocks = int(restored_blocks or 0)
            except (TypeError, ValueError):
                restored_blocks = 0
            if restored_blocks <= 0:
                return {
                    "restored": False,
                    "reason": "vllm_kv_restore_empty",
                    "state_kind": state_kind,
                }
            restored.setdefault(
                "recovered_tokens", state.get("completed_tokens", 0)
            )
            restored["restored_blocks"] = restored_blocks
            restored.setdefault(
                "restore_scope", "cross_node" if cross_node else "same_node"
            )
        else:
            restored.setdefault("reason", "vllm_kv_restore_failed")
        return restored

    async def get_runtime_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ) -> Dict[str, Any]:
        gpu_metadata: Dict[str, Any] = {}
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                gpu_metadata = {
                    "free_gpu_memory_gb": free_bytes / (1024**3),
                    "total_gpu_memory_gb": total_bytes / (1024**3),
                    "total_gpu": torch.cuda.device_count(),
                    "free_gpu": torch.cuda.device_count(),
                }
            except Exception:
                logger.debug("Unable to query CUDA runtime metadata", exc_info=True)
        runtime_moe_metadata = await self._runtime_moe_metadata(
            instance_id=instance_id,
            node_id=node_id,
        )
        return get_vllm_runtime_metadata(
            model_name=self.model_name,
            backend_config=self.backend_config,
            instance_id=instance_id,
            node_id=node_id,
            runtime_metadata={
                "load_time_s": self.model_load_time_s,
                **gpu_metadata,
                **self._engine_parallel_metadata(
                    instance_id=instance_id,
                    node_id=node_id,
                ),
                **runtime_moe_metadata,
            },
        )

    async def encode(self, request_data: Dict[str, Any]):
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}

        assert self.engine is not None

        if not request_data:
            return {"error": "Request data is missing"}

        request_counter: Counter = Counter()
        pooling_params: PoolingParams = PoolingParams()
        model_name = request_data.get("model", "vllm-model")
        query = request_data.get("input", [])

        if not query:
            return {"error": "No inputs provided"}

        inputs = cast(Union[PromptType, Sequence[PromptType]], query)

        async def process_input(input_data) -> List[EmbeddingRequestOutput]:
            request_id = str(next(request_counter))
            res = self.engine.encode(input_data, pooling_params, request_id)
            return [result async for result in res]

        raw_outputs = await asyncio.gather(
            *[process_input(input_data) for input_data in inputs],
            return_exceptions=True,
        )

        valid_outputs = []
        for output in raw_outputs:
            if isinstance(output, Exception):
                logger.error(f"Error encountered: {output}")
            else:
                valid_outputs.extend(output)

        if not valid_outputs:
            return {"error": "All inputs failed"}

        return process_embedding_output(valid_outputs, model_name)
