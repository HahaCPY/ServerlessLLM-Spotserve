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


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 1)


def _non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 0.0)


def get_vllm_model_resource_profile(
    model_name: str,
    backend_config: Optional[Mapping[str, Any]] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    backend_config = backend_config or {}
    runtime_metadata = runtime_metadata or {}

    tensor_parallel_size = _positive_int(
        backend_config.get("tensor_parallel_size"), 1
    )
    pipeline_parallel_size = _positive_int(
        backend_config.get("pipeline_parallel_size"), 1
    )
    data_parallel_size = _positive_int(
        backend_config.get("data_parallel_size"), 1
    )
    num_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    return {
        "model_name": model_name,
        "backend": "vllm",
        "num_gpus": num_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "expert_parallel_enabled": bool(
            backend_config.get("enable_expert_parallel", False)
        ),
        "estimated_load_time_s": _non_negative_float(
            runtime_metadata.get("estimated_load_time_s"),
            _non_negative_float(runtime_metadata.get("load_time_s"), 0.0),
        ),
        "gpu_memory_required_gb": _non_negative_float(
            runtime_metadata.get("gpu_memory_required_gb"), 0.0
        ),
        "gpu_memory_utilization": _non_negative_float(
            backend_config.get("gpu_memory_utilization"), 0.0
        ),
        "max_model_len": _positive_int(
            backend_config.get("max_model_len"), 1
        ),
        "max_num_seqs": _positive_int(
            backend_config.get("max_num_seqs"), 1
        ),
    }


def get_vllm_runtime_metadata(
    model_name: str,
    backend_config: Optional[Mapping[str, Any]] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    instance_id: str = "",
    node_id: str = "",
) -> Dict[str, Any]:
    runtime_metadata = runtime_metadata or {}
    profile = get_vllm_model_resource_profile(
        model_name=model_name,
        backend_config=backend_config,
        runtime_metadata=runtime_metadata,
    )
    return {
        "instance_id": instance_id,
        "node_id": node_id,
        "backend": "vllm",
        "model_name": model_name,
        "model_resource_profile": profile,
        "loading_cost": profile["estimated_load_time_s"],
        "free_gpu": int(runtime_metadata.get("free_gpu", 0) or 0),
        "total_gpu": int(runtime_metadata.get("total_gpu", 0) or 0),
        "free_gpu_memory_gb": _non_negative_float(
            runtime_metadata.get("free_gpu_memory_gb"), 0.0
        ),
        "total_gpu_memory_gb": _non_negative_float(
            runtime_metadata.get("total_gpu_memory_gb"), 0.0
        ),
        "spot_risk": _non_negative_float(
            runtime_metadata.get("spot_risk"), 0.0
        ),
        "remaining_lifetime_s": _non_negative_float(
            runtime_metadata.get("remaining_lifetime_s"), 0.0
        ),
        "metadata": dict(runtime_metadata.get("metadata", {}) or {}),
    }
