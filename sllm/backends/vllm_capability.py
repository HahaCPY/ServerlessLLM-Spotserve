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
from typing import Any, Mapping, Optional

from sllm.backends.capability import BackendCapability
from sllm.spot.reparallelization import ParallelPlan


VLLM_MOE_SUPPORTED_SHAPES = (
    {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
    },
    {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "expert_parallel_size": 2,
    },
    {
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
    },
    {
        "tensor_parallel_size": 2,
        "data_parallel_size": 2,
        "expert_parallel_size": 1,
    },
    {
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "expert_parallel_size": 2,
    },
    {
        "tensor_parallel_size": 2,
        "data_parallel_size": 2,
        "expert_parallel_size": 2,
    },
)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 1)


def _is_moe_config(model_name: str, backend_config: Mapping[str, Any]) -> bool:
    model_id = str(
        backend_config.get("pretrained_model_name_or_path")
        or backend_config.get("model")
        or model_name
    )
    return "moe" in model_name.lower() or "moe" in model_id.lower()


def _make_plan(
    model_name: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    reason: str,
    num_gpus: Optional[int] = None,
) -> ParallelPlan:
    if num_gpus is None:
        num_gpus = tensor_parallel_size * data_parallel_size
    return ParallelPlan(
        model_name=model_name,
        backend="vllm",
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=1,
        expert_parallel_size=expert_parallel_size,
        num_replicas=data_parallel_size,
        num_gpus=num_gpus,
        reason=reason,
    )


def get_vllm_capability(
    model_config: Mapping[str, Any],
) -> BackendCapability:
    model_name = str(model_config["model"])
    backend_config = model_config.get("backend_config", {}) or {}
    configured_num_gpus = _positive_int(model_config.get("num_gpus"), 1)
    tensor_parallel_size = _positive_int(
        backend_config.get("tensor_parallel_size"), configured_num_gpus
    )

    if _is_moe_config(model_name, backend_config):
        supported_configs = [
            _make_plan(
                model_name=model_name,
                tensor_parallel_size=shape["tensor_parallel_size"],
                data_parallel_size=shape["data_parallel_size"],
                expert_parallel_size=shape["expert_parallel_size"],
                reason="verified_vllm_moe_config",
            )
            for shape in VLLM_MOE_SUPPORTED_SHAPES
        ]
    else:
        supported_configs = [
            _make_plan(
                model_name=model_name,
                tensor_parallel_size=tensor_parallel_size,
                data_parallel_size=1,
                expert_parallel_size=1,
                reason="current_vllm_config",
                num_gpus=max(configured_num_gpus, tensor_parallel_size),
            )
        ]

    max_num_gpus = max(
        configured_num_gpus,
        *(config.num_gpus for config in supported_configs),
    )
    supports_ep = any(
        config.expert_parallel_size > 1 for config in supported_configs
    )

    return BackendCapability(
        backend="vllm",
        model_name=model_name,
        supports_tp=True,
        supports_dp=True,
        supports_ep=supports_ep,
        supports_state_export=False,
        supports_state_restore=False,
        max_num_gpus=max_num_gpus,
        supported_configs=supported_configs,
    )
