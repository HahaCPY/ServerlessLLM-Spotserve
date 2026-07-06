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
import time
import uuid
from typing import Any, Dict, List, Optional

from sllm.backends.backend_utils import SllmBackend
from sllm.logger import init_logger


class DummyBackend(SllmBackend):
    _forced_failures_seen = set()

    def __init__(
        self,
        model_name: str = "dummy-model",
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(model_name, dict) and backend_config is None:
            backend_config = model_name
            model_name = "dummy-model"
        self.model_name = model_name
        self.backend_config = backend_config
        self.current_tokens: List[List[int]] = []
        self.restored_states: Dict[str, Dict[str, Any]] = {}

    def init_backend(self) -> None:
        # sleep to simulate model latency
        sleep_time = 5
        self.log(
            f"Sleeping for {sleep_time} seconds to simulate model init time."
        )
        time.sleep(sleep_time)

    def log(self, msg):
        logger = init_logger(__name__)
        logger.info(msg)

    def _normalize_input_tokens(self, request_data: Dict[str, Any]) -> List[int]:
        input_tokens = request_data.get("input_tokens") or []
        if not input_tokens:
            return []
        if isinstance(input_tokens, list) and input_tokens:
            if isinstance(input_tokens[0], list):
                return [int(token) for token in input_tokens[0]]
            return [int(token) for token in input_tokens]
        return []

    def _request_id(self, request_data: Dict[str, Any]) -> str:
        return str(
            request_data.get("request_id") or f"anonymous-{id(request_data)}"
        )

    def _tokens_from_state(self, state: Dict[str, Any]) -> List[int]:
        tokens = state.get("tokens") or []
        if tokens and isinstance(tokens[0], list):
            return [int(token) for token in tokens[0]]
        return [int(token) for token in tokens]

    def _apply_restored_state(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "input_tokens" in request_data:
            return request_data

        request_id = self._request_id(request_data)
        restored_state = self.restored_states.pop(request_id, None)
        if not restored_state:
            return request_data

        tokens = self._tokens_from_state(restored_state)
        if not tokens:
            return request_data

        restored_request = dict(request_data)
        restored_request["input_tokens"] = tokens
        completed_tokens = int(
            restored_state.get("completed_tokens", len(tokens)) or 0
        )
        if completed_tokens and "max_tokens" in restored_request:
            restored_request["max_tokens"] = max(
                1,
                int(restored_request["max_tokens"]) - completed_tokens,
            )
        return restored_request

    def _forced_failure_key(self, request_data: Dict[str, Any]) -> str:
        request_id = self._request_id(request_data)
        failure_mode = request_data.get("force_failure") or request_data.get(
            "force_backend_failure"
        )
        return f"{self.model_name}:{request_id}:{failure_mode}"

    def _should_force_failure(
        self, request_data: Dict[str, Any], generated_tokens: List[int]
    ) -> bool:
        failure_mode = request_data.get("force_failure") or request_data.get(
            "force_backend_failure"
        )
        if not failure_mode:
            return False

        fail_after_tokens = request_data.get("force_fail_after_tokens")
        if fail_after_tokens is None:
            fail_after_tokens = request_data.get("force_preempt_after_tokens")
        if fail_after_tokens is None:
            return False

        failure_key = self._forced_failure_key(request_data)
        if request_data.get("force_fail_once", True):
            if failure_key in self._forced_failures_seen:
                return False
            if len(generated_tokens) >= int(fail_after_tokens):
                self._forced_failures_seen.add(failure_key)
                return True
            return False

        return len(generated_tokens) >= int(fail_after_tokens)

    def _forced_failure_result(
        self, request_data: Dict[str, Any], generated_tokens: List[int]
    ):
        failure_mode = (
            request_data.get("force_failure")
            or request_data.get("force_backend_failure")
            or ""
        ).lower()
        if request_data.get("force_no_current_tokens", False):
            self.current_tokens = []

        if failure_mode in {"preempt", "preempted", "preemption"}:
            current_output = [] if not self.current_tokens else self.current_tokens
            return {
                "error": (
                    "Forced dummy backend preemption after "
                    f"{len(generated_tokens)} tokens"
                ),
                "preempted": True,
                "current_output": current_output,
                "completed_tokens": len(generated_tokens),
            }

        raise RuntimeError(
            "Forced dummy backend failure after "
            f"{len(generated_tokens)} tokens"
        )

    async def generate(self, request_data):
        request_data = self._apply_restored_state(request_data)
        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        # temperature = request_data.get("temperature", 0.7)
        max_tokens = int(request_data.get("max_tokens", 10))
        token_latency = float(request_data.get("token_latency", 0.1))
        input_tokens = self._normalize_input_tokens(request_data)
        generated_tokens = list(input_tokens)
        self.current_tokens = [generated_tokens.copy()] if generated_tokens else []

        # Combine messages to form the prompt
        prompt = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )

        # Simulate token-by-token latency so recovery tests can interrupt
        # generation after a known partial output.
        self.log(
            f"Sleeping for {max_tokens * token_latency} seconds to simulate model response time."
        )
        for i in range(max_tokens):
            await asyncio.sleep(token_latency)
            next_token = len(generated_tokens) + 1
            generated_tokens.append(next_token)
            self.current_tokens = [generated_tokens.copy()]
            if self._should_force_failure(request_data, generated_tokens):
                return self._forced_failure_result(
                    request_data, generated_tokens
                )

        # Dummy response content
        response_content = (
            f"Debug model received prompt: {prompt}; "
            f"generated_tokens={generated_tokens}"
        )

        # Simulate token counts for the response
        prompt_tokens = len(prompt.split())
        completion_tokens = len(generated_tokens)
        total_tokens = prompt_tokens + completion_tokens

        # Generate response compatible with OpenAI's API
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response

    async def resume_generate(self, request_data, current_output):
        if current_output and "input_tokens" not in request_data:
            request_data = dict(request_data)
            request_data["input_tokens"] = current_output[0]
        return await self.generate(request_data)

    async def encode(self, request_data):
        model_name = request_data.get("model", "dummy-model")
        input_data = request_data.get("input", "")
        if isinstance(input_data, str):
            inputs = [input_data]
        else:
            inputs = list(input_data)

        data = []
        for index, item in enumerate(inputs):
            token_count = len(str(item).split())
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [float(token_count), 0.0, 1.0],
                }
            )

        return {
            "object": "list",
            "model": model_name,
            "data": data,
            "usage": {
                "prompt_tokens": sum(len(str(item).split()) for item in inputs),
                "total_tokens": sum(len(str(item).split()) for item in inputs),
            },
        }

    async def shutdown(self):
        pass

    async def stop(self):
        pass

    async def get_current_tokens(self):
        return self.current_tokens

    async def resume_kv_cache(self, request_datas):
        self.current_tokens = request_datas or []
        return True

    async def supports_state_restore(self):
        return bool(self.backend_config.get("supports_state_restore", True))

    async def export_inference_state(
        self,
        request_data: Optional[Dict[str, Any]] = None,
        current_output: Optional[List[List[int]]] = None,
        completed_tokens: Optional[int] = None,
    ):
        request_data = request_data or {}
        token_sequences = current_output or self.current_tokens or []
        tokens = token_sequences[0] if token_sequences else []
        completed = (
            int(completed_tokens)
            if completed_tokens is not None
            else len(tokens)
        )
        return {
            "request_id": request_data.get("request_id"),
            "backend": "dummy",
            "model_name": self.model_name,
            "tokens": [int(token) for token in tokens],
            "completed_tokens": max(0, completed),
            "state_kind": "dummy_token_state",
            "supports_restore": await self.supports_state_restore(),
            "metadata": {
                "token_count": len(tokens),
                "source": "dummy_backend",
            },
        }

    async def restore_inference_state(
        self,
        state: Dict[str, Any],
        request_data: Optional[Dict[str, Any]] = None,
    ):
        if not await self.supports_state_restore():
            return {
                "restored": False,
                "reason": "dummy_state_restore_disabled",
            }

        request_data = request_data or {}
        tokens = self._tokens_from_state(state)
        if not tokens:
            return {"restored": False, "reason": "empty_state"}

        request_id = state.get("request_id") or request_data.get("request_id")
        if not request_id:
            request_id = self._request_id(request_data)
        self.restored_states[str(request_id)] = dict(state)
        self.current_tokens = [tokens]
        return {
            "restored": True,
            "request_id": request_id,
            "tokens": tokens,
            "completed_tokens": int(
                state.get("completed_tokens", len(tokens)) or 0
            ),
            "state_kind": state.get("state_kind", "dummy_token_state"),
        }
