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

    def _forced_failure_key(self, request_data: Dict[str, Any]) -> str:
        request_id = request_data.get("request_id") or f"anonymous-{id(request_data)}"
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
