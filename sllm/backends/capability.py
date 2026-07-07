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
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from sllm.spot.reparallelization import ParallelPlan


@dataclass(frozen=True)
class BackendCapability:
    backend: str
    model_name: str
    supports_tp: bool
    supports_dp: bool
    supports_ep: bool
    supports_state_export: bool
    supports_state_restore: bool
    max_num_gpus: int
    supported_configs: List[ParallelPlan]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "supports_tp": self.supports_tp,
            "supports_dp": self.supports_dp,
            "supports_ep": self.supports_ep,
            "supports_state_export": self.supports_state_export,
            "supports_state_restore": self.supports_state_restore,
            "max_num_gpus": self.max_num_gpus,
            "supported_configs": [
                config.to_dict() for config in self.supported_configs
            ],
        }


def get_backend_capability(
    model_config: Mapping[str, Any],
) -> Optional[BackendCapability]:
    backend = model_config.get("backend")
    if backend == "vllm":
        from sllm.backends.vllm_capability import get_vllm_capability

        return get_vllm_capability(model_config)
    return None
