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
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class BackendStatus(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    STOPPING = auto()
    DELETING = auto()


class SllmBackend(ABC):
    @abstractmethod
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    @abstractmethod
    async def init_backend(self) -> None:
        pass

    @abstractmethod
    async def encode(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def generate(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def get_current_tokens(self) -> List[List[int]]:
        pass

    @abstractmethod
    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        pass

    async def get_context_metadata(
        self,
        instance_id: str = "",
        node_id: str = "",
    ) -> List[Dict[str, Any]]:
        return []

    async def supports_state_restore(self) -> bool:
        return False

    async def export_inference_state(
        self,
        request_data: Optional[Dict[str, Any]] = None,
        current_output: Optional[List[List[int]]] = None,
        completed_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "request_id": (
                request_data.get("request_id") if request_data else None
            ),
            "tokens": [],
            "completed_tokens": completed_tokens or 0,
            "supports_restore": False,
            "state_kind": "unsupported",
            "metadata": {"reason": "backend_state_restore_unsupported"},
        }

    async def restore_inference_state(
        self,
        state: Dict[str, Any],
        request_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "restored": False,
            "reason": "backend_state_restore_unsupported",
        }
