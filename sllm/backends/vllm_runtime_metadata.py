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


def _optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 1)


def _optional_non_negative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 0.0)


def _to_bool(value: Any, default: bool = False) -> bool:
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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _has_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def get_vllm_model_resource_profile(
    model_name: str,
    backend_config: Optional[Mapping[str, Any]] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    backend_config = backend_config or {}
    runtime_metadata = runtime_metadata or {}

    tensor_parallel_size = _positive_int(
        _first_present(
            runtime_metadata.get("tensor_parallel_size"),
            backend_config.get("tensor_parallel_size"),
        ),
        1,
    )
    pipeline_parallel_size = _positive_int(
        _first_present(
            runtime_metadata.get("pipeline_parallel_size"),
            backend_config.get("pipeline_parallel_size"),
        ),
        1,
    )
    data_parallel_size = _positive_int(
        _first_present(
            runtime_metadata.get("data_parallel_size"),
            runtime_metadata.get("vllm_data_parallel_size"),
            backend_config.get("data_parallel_size"),
            backend_config.get("vllm_data_parallel_size"),
        ),
        1,
    )
    sllm_replica_count = _positive_int(
        _first_present(
            runtime_metadata.get("sllm_replica_count"),
            runtime_metadata.get("replica_count"),
            backend_config.get("sllm_replica_count"),
            backend_config.get("replica_count"),
        ),
        1,
    )
    num_gpus = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    planned_expert_parallel_size = _positive_int(
        _first_present(
            backend_config.get("planned_effective_expert_parallel_size"),
            backend_config.get("planned_expert_parallel_size"),
            backend_config.get("expert_parallel_size"),
        ),
        1,
    )
    runtime_expert_parallel_size = _optional_positive_int(
        _first_present(
            runtime_metadata.get("effective_expert_parallel_size"),
            runtime_metadata.get("expert_parallel_size"),
            runtime_metadata.get("runtime_expert_parallel_size"),
        )
    )
    runtime_expert_parallel_size_provided = (
        runtime_expert_parallel_size is not None
    )
    expert_parallel_enabled = _to_bool(
        _first_present(
            runtime_metadata.get("expert_parallel_enabled"),
            backend_config.get("enable_expert_parallel"),
        ),
        default=planned_expert_parallel_size > 1
        or bool(runtime_expert_parallel_size and runtime_expert_parallel_size > 1),
    )
    expert_parallel_size_verified = _to_bool(
        runtime_metadata.get("expert_parallel_size_verified"),
        default=runtime_expert_parallel_size is not None,
    )
    if runtime_expert_parallel_size and runtime_expert_parallel_size > 1:
        expert_parallel_enabled = True
    derived_expert_parallel_size = (
        tensor_parallel_size * data_parallel_size
        if expert_parallel_enabled
        else 1
    )
    effective_expert_parallel_size = (
        runtime_expert_parallel_size or derived_expert_parallel_size
    )
    runtime_expert_parallel_size = effective_expert_parallel_size
    expert_parallel_size_verified = True
    if runtime_metadata.get("expert_parallel_size_source"):
        expert_parallel_size_source = str(
            runtime_metadata["expert_parallel_size_source"]
        )
    elif runtime_expert_parallel_size_provided:
        expert_parallel_size_source = "runtime_metadata"
    elif expert_parallel_enabled:
        expert_parallel_size_source = "derived_from_tp_dp"
    else:
        expert_parallel_size_source = "disabled"
    expert_physical_replication_factor = _positive_int(
        _first_present(
            runtime_metadata.get("expert_physical_replication_factor"),
            backend_config.get("expert_physical_replication_factor"),
        ),
        1,
    )
    expert_placement_snapshot = _first_present(
        runtime_metadata.get("expert_placement_snapshot"),
        runtime_metadata.get("expert_placement"),
        backend_config.get("expert_placement_snapshot"),
    )
    placement_available = _has_payload(expert_placement_snapshot)
    placement_epoch = _optional_non_negative_int(
        _first_present(
            runtime_metadata.get("placement_epoch"),
            runtime_metadata.get("placement_version"),
            backend_config.get("placement_epoch"),
        )
    )
    placement_source = str(
        _first_present(
            runtime_metadata.get("placement_source"),
            backend_config.get("placement_source"),
            "runtime" if placement_available else "unavailable",
        )
    )
    route_histogram = runtime_metadata.get("per_request_expert_route_histogram")
    route_histogram_payload_available = _has_payload(route_histogram)
    route_histogram_available = _to_bool(
        runtime_metadata.get("moe_route_histogram_available"),
        default=route_histogram_payload_available,
    ) and route_histogram_payload_available
    route_histogram_source = str(
        _first_present(
            runtime_metadata.get("moe_route_histogram_source"),
            runtime_metadata.get("route_histogram_source"),
            "runtime_or_instrumentation"
            if route_histogram_available
            else "unavailable",
        )
    )

    profile = {
        "model_name": model_name,
        "backend": "vllm",
        "num_gpus": num_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "vllm_data_parallel_size": data_parallel_size,
        "replica_count": sllm_replica_count,
        "sllm_replica_count": sllm_replica_count,
        "expert_parallel_enabled": expert_parallel_enabled,
        "planned_effective_expert_parallel_size": (
            planned_expert_parallel_size
        ),
        "planned_expert_parallel_size": planned_expert_parallel_size,
        "effective_expert_parallel_size": effective_expert_parallel_size,
        "runtime_effective_expert_parallel_size": (
            effective_expert_parallel_size
        ),
        "derived_effective_expert_parallel_size": derived_expert_parallel_size,
        "expert_parallel_size_verified": expert_parallel_size_verified,
        "expert_parallel_size_source": expert_parallel_size_source,
        "expert_physical_replication_factor": (
            expert_physical_replication_factor
        ),
        "expert_placement_available": placement_available,
        "placement_epoch": placement_epoch,
        "placement_version": placement_epoch,
        "placement_source": placement_source,
        "moe_route_histogram_available": route_histogram_available,
        "moe_route_histogram_source": route_histogram_source,
        "parallel_plan_mismatch": (
            planned_expert_parallel_size != effective_expert_parallel_size
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
    profile["expert_parallel_size"] = runtime_expert_parallel_size
    if placement_available:
        profile["expert_placement_snapshot"] = expert_placement_snapshot
    for source_key, target_key in (
        ("global_expert_hotness", "global_expert_hotness"),
        ("recent_window_expert_hotness", "recent_window_expert_hotness"),
        (
            "per_request_expert_route_histogram",
            "per_request_expert_route_histogram",
        ),
    ):
        if _has_payload(runtime_metadata.get(source_key)):
            profile[target_key] = runtime_metadata[source_key]
    return profile


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
    result = {
        "instance_id": instance_id,
        "node_id": node_id,
        "backend": "vllm",
        "model_name": model_name,
        "model_resource_profile": profile,
        "loading_cost": profile["estimated_load_time_s"],
        "tensor_parallel_size": profile["tensor_parallel_size"],
        "pipeline_parallel_size": profile["pipeline_parallel_size"],
        "data_parallel_size": profile["data_parallel_size"],
        "vllm_data_parallel_size": profile["vllm_data_parallel_size"],
        "replica_count": profile["replica_count"],
        "sllm_replica_count": profile["sllm_replica_count"],
        "expert_parallel_enabled": profile["expert_parallel_enabled"],
        "effective_expert_parallel_size": (
            profile["effective_expert_parallel_size"]
        ),
        "expert_parallel_size": profile["expert_parallel_size"],
        "expert_parallel_size_source": (
            profile["expert_parallel_size_source"]
        ),
        "expert_placement_available": (
            profile["expert_placement_available"]
        ),
        "placement_epoch": profile["placement_epoch"],
        "placement_source": profile["placement_source"],
        "moe_route_histogram_available": (
            profile["moe_route_histogram_available"]
        ),
        "moe_route_histogram_source": (
            profile["moe_route_histogram_source"]
        ),
        "free_gpu": int(runtime_metadata.get("free_gpu", 0) or 0),
        "total_gpu": int(runtime_metadata.get("total_gpu", 0) or 0),
        "free_gpu_memory_gb": _non_negative_float(
            runtime_metadata.get("free_gpu_memory_gb"), 0.0
        ),
        "total_gpu_memory_gb": _non_negative_float(
            runtime_metadata.get("total_gpu_memory_gb"), 0.0
        ),
        "metadata": dict(runtime_metadata.get("metadata", {}) or {}),
    }
    if runtime_metadata.get("spot_risk") is not None:
        result["spot_risk"] = _non_negative_float(
            runtime_metadata.get("spot_risk"), 0.0
        )
    if runtime_metadata.get("risk_score") is not None:
        result["risk_score"] = _non_negative_float(
            runtime_metadata.get("risk_score"), 0.0
        )
    if runtime_metadata.get("preemption_risk") is not None:
        result["preemption_risk"] = _non_negative_float(
            runtime_metadata.get("preemption_risk"), 0.0
        )
    if runtime_metadata.get("remaining_lifetime_s") is not None:
        result["remaining_lifetime_s"] = _non_negative_float(
            runtime_metadata.get("remaining_lifetime_s"), 0.0
        )
    if runtime_metadata.get("expected_remaining_lifetime_s") is not None:
        result["expected_remaining_lifetime_s"] = _non_negative_float(
            runtime_metadata.get("expected_remaining_lifetime_s"), 0.0
        )
    return result
