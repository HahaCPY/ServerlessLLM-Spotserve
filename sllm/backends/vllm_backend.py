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
import logging
import os
import time
import uuid
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence, Union, cast

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
            del self.status_dict[request_id]

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
        filtered_engine_config["task"] = self.task

        logger.info(
            f"Creating new VLLM engine with config: {filtered_engine_config}"
        )

        self.engine_args = AsyncEngineArgs(**filtered_engine_config)

        self.engine = None
        self.model_load_time_s = 0.0

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
        restore_supported = await self.supports_state_restore()
        metadata["supports_state_export"] = restore_supported
        metadata["supports_state_restore"] = restore_supported
        return metadata

    async def init_backend(self) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            started_at = time.monotonic()
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.model_load_time_s = time.monotonic() - started_at
            self.status = BackendStatus.RUNNING

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
        if request_data.get("input_tokens") is not None:
            inputs = TokensPrompt(
                prompt_token_ids=request_data.pop("input_tokens"),
            )

        request_id: str = request_data.pop(
            "request_id", f"chatcmpl-{uuid.uuid4()}"
        )

        try:
            sampling_params = SamplingParams(**request_data)
        except Exception as e:
            return {"error": f"Invalid sampling parameters: {e}"}

        results_generator = self.engine.generate(
            inputs, sampling_params, request_id
        )

        # TODO stream results

        # Non-stream case
        final_output = None
        async for response_output in results_generator:
            final_output = response_output
            await self.request_trace.update_status(request_id, response_output)

        assert final_output is not None

        if not self.trace_debug:
            await self.request_trace.delete_request(request_id)

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
        return response

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
        while await self.request_trace.request_count() > 0:
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
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
                    return [
                        get_vllm_context_metadata(
                            model_name=self.model_name,
                            instance_id=instance_id or self.model_name,
                            node_id=node_id,
                            runtime_metadata={
                                **snapshot,
                                "supports_state_export": restore_supported,
                                "supports_state_restore": restore_supported,
                            },
                        )
                        for snapshot in runtime_snapshots
                        if isinstance(snapshot, dict)
                        and snapshot.get("found", True)
                    ]

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

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
        constructed_inputs = [
            {
                "input_tokens": request_data,
                "max_tokens": 1,
            }
            for request_data in request_datas
        ]
        tasks = [self.generate(inputs) for inputs in constructed_inputs]
        await asyncio.gather(*tasks)

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
                runtime_metadata = dict(live_metadata)
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

        try:
            exported = await self._call_runtime_hook(
                ("export_inference_state", "export_kv_cache_state"),
                request_id=request_id or runtime_metadata.get("request_id"),
                request_data=request_data,
                runtime_metadata=runtime_metadata,
            )
        except Exception:
            logger.exception("vLLM KV state export failed")
            exported = None
        if (
            not isinstance(exported, dict)
            or exported.get("supports_restore") is not True
            or exported.get("state_kind") != "vllm_kv_snapshot"
            or not isinstance(exported.get("runtime_state"), dict)
            or not exported["runtime_state"]
        ):
            fallback["metadata"]["reason"] = "vllm_kv_export_failed"
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
        config_metadata = {
            "tensor_parallel_size": self.backend_config.get(
                "tensor_parallel_size"
            ),
            "pipeline_parallel_size": self.backend_config.get(
                "pipeline_parallel_size"
            ),
            "cache_block_size": self.backend_config.get("block_size"),
            "cache_dtype": self.backend_config.get("kv_cache_dtype"),
            "cache_layout": self.backend_config.get("kv_cache_layout"),
        }
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
            "expert_parallel_enabled": "enable_expert_parallel",
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
        return get_vllm_runtime_metadata(
            model_name=self.model_name,
            backend_config=self.backend_config,
            instance_id=instance_id,
            node_id=node_id,
            runtime_metadata={
                "load_time_s": self.model_load_time_s,
                **gpu_metadata,
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
