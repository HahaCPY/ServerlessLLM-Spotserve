"""Experimental, quiescent in-place remap for unquantized vLLM FusedMoE.

Only a fixed EP group with one physical copy of each expert is supported.
All EP workers must enter ``remap_expert_weights`` through collective_rpc.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

_TRUTHY = {"1", "true", "yes", "on"}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EP_RANK_RE = re.compile(r"^(?:replica:0/)?ep-rank:(\d+)$")
_LAST_REMAP: dict[str, Any] = {}
_ALL_TO_ALL_LOCK = threading.Lock()
_ALL_TO_ALL_COUNTERS: dict[str, Any] = {
    "dispatch_calls": 0,
    "combine_calls": 0,
    "dispatch_input_bytes": 0,
    "dispatch_output_bytes": 0,
    "combine_input_bytes": 0,
    "combine_output_bytes": 0,
    "internode_calls": 0,
    "sparse_remote_rows": 0,
    "sparse_local_rows": 0,
    "backends": {},
}


def _tensor_payload_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_payload_bytes(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    return 0


def all_to_all_counters_enabled() -> bool:
    return os.environ.get(
        "VLLM_SPOTSERVE_A2A_TRACE", ""
    ).strip().lower() in _TRUTHY


def record_all_to_all_collective(
    phase: str,
    inputs: Any,
    outputs: Any,
    manager: Any,
) -> None:
    """Record payload crossing vLLM's real EP communicator boundary.

    These are observed tensor payload bytes at the dispatch/combine API, not
    an estimate from router probabilities and not physical wire bytes.
    """
    if not all_to_all_counters_enabled():
        return
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"dispatch", "combine"}:
        raise ValueError(f"unsupported_all_to_all_phase:{phase}")
    backend = type(manager).__name__ if manager is not None else "unknown"
    with _ALL_TO_ALL_LOCK:
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_calls"] += 1
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_input_bytes"] += (
            _tensor_payload_bytes(inputs)
        )
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_output_bytes"] += (
            _tensor_payload_bytes(outputs)
        )
        if bool(getattr(manager, "internode", False)):
            _ALL_TO_ALL_COUNTERS["internode_calls"] += 1
        backends = _ALL_TO_ALL_COUNTERS["backends"]
        backends[backend] = int(backends.get(backend, 0)) + 1


def record_sparse_all_to_all_transfer(
    phase: str,
    *,
    sent_bytes: int,
    received_bytes: int,
    remote_rows: int,
    local_rows: int,
) -> None:
    """Record payload actually exchanged by the sparse token backend."""
    if not all_to_all_counters_enabled():
        return
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"dispatch", "combine"}:
        raise ValueError(f"unsupported_all_to_all_phase:{phase}")
    with _ALL_TO_ALL_LOCK:
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_calls"] += 1
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_input_bytes"] += int(
            sent_bytes
        )
        _ALL_TO_ALL_COUNTERS[f"{normalized_phase}_output_bytes"] += int(
            received_bytes
        )
        _ALL_TO_ALL_COUNTERS["sparse_remote_rows"] += int(remote_rows)
        _ALL_TO_ALL_COUNTERS["sparse_local_rows"] += int(local_rows)
        backends = _ALL_TO_ALL_COUNTERS["backends"]
        backend = "SpotServeSparseAllToAll"
        backends[backend] = int(backends.get(backend, 0)) + 1


def get_all_to_all_counters() -> dict[str, Any]:
    with _ALL_TO_ALL_LOCK:
        counters = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in _ALL_TO_ALL_COUNTERS.items()
        }
    counters["enabled"] = all_to_all_counters_enabled()
    counters["collective_calls"] = int(counters["dispatch_calls"]) + int(
        counters["combine_calls"]
    )
    counters["observed_input_bytes"] = int(
        counters["dispatch_input_bytes"]
    ) + int(counters["combine_input_bytes"])
    counters["observed_output_bytes"] = int(
        counters["dispatch_output_bytes"]
    ) + int(counters["combine_output_bytes"])
    counters["available"] = bool(
        counters["enabled"] and counters["collective_calls"] > 0
    )
    counters["measurement_kind"] = (
        "runtime_sparse_transfer_payload"
        if counters["backends"].get("SpotServeSparseAllToAll")
        else "runtime_collective_tensor_payload"
    )
    return counters


def reset_all_to_all_counters() -> None:
    with _ALL_TO_ALL_LOCK:
        for key in tuple(_ALL_TO_ALL_COUNTERS):
            _ALL_TO_ALL_COUNTERS[key] = {} if key == "backends" else 0


def aggregate_dp_engine_hook_results(
    engine_results: list[Any],
    success_key: str,
) -> dict[str, Any]:
    """Combine placement-hook results returned by every DP EngineCore."""
    normalized = [result for result in engine_results if isinstance(result, Mapping)]
    worker_results: list[dict[str, Any]] = []
    for result in normalized:
        rows = result.get("worker_results")
        if isinstance(rows, list):
            worker_results.extend(row for row in rows if isinstance(row, dict))
        else:
            worker_results.append(dict(result))
    succeeded = bool(normalized) and len(normalized) == len(engine_results) and all(
        bool(result.get(success_key, result.get("success", False)))
        for result in normalized
    )
    reason = ""
    if not succeeded:
        reason = next(
            (
                str(result.get("reason"))
                for result in normalized
                if not result.get(success_key, result.get("success", False))
                and result.get("reason")
            ),
            "dp_engine_placement_hook_failed",
        )
    contract_seen_count = sum(
        bool(row.get("contract_seen_by_runtime")) for row in worker_results
    )
    return {
        success_key: succeeded,
        "success": succeeded,
        "reason": reason,
        "dp_engine_coordinated": True,
        "dp_engine_count": len(engine_results),
        "dp_engine_success_count": sum(
            bool(result.get(success_key, result.get("success", False)))
            for result in normalized
        ),
        "worker_count": len(worker_results),
        "worker_success_count": sum(
            bool(row.get(success_key, row.get("success", False)))
            for row in worker_results
        ),
        "contract_seen_count": contract_seen_count,
        "contract_seen_by_runtime": bool(contract_seen_count),
        "contract_seen_by_all_workers": bool(
            worker_results and contract_seen_count == len(worker_results)
        ),
        "contract_seen_worker_count": sum(
            int(row.get("contract_seen_worker_count", 0) or 0)
            for row in worker_results
        ),
        "contract_seen_worker_total": sum(
            int(row.get("contract_seen_worker_total", 0) or 0)
            for row in worker_results
        ),
        "physical_weight_migration": any(
            bool(row.get("physical_weight_migration"))
            for row in worker_results
        ),
        "moved_local_expert_shards": sum(
            int(row.get("moved_local_expert_shards", 0) or 0)
            for row in worker_results
        ),
        "moved_local_weight_bytes": sum(
            int(row.get("moved_local_weight_bytes", 0) or 0)
            for row in worker_results
        ),
        "physical_host_ids_observed": bool(
            worker_results
            and all(row.get("physical_host_ids_observed") for row in worker_results)
        ),
        "cross_node_weight_migration": any(
            bool(row.get("cross_node_weight_migration"))
            for row in worker_results
        ),
        "cross_node_moved_local_expert_shards": sum(
            int(row.get("cross_node_moved_local_expert_shards", 0) or 0)
            for row in worker_results
        ),
        "cross_node_moved_local_weight_bytes": sum(
            int(row.get("cross_node_moved_local_weight_bytes", 0) or 0)
            for row in worker_results
        ),
        "remap_duration_ms": max(
            (
                float(row.get("remap_duration_ms", 0.0) or 0.0)
                for row in worker_results
            ),
            default=0.0,
        ),
        "runtime_verified_placement": bool(
            worker_results
            and all(row.get("runtime_verified_placement") for row in worker_results)
        ),
        "verification_levels": ",".join(
            sorted(
                {
                    str(row.get("verification_level"))
                    for row in worker_results
                    if row.get("verification_level")
                }
            )
        ),
        "can_verify_physical_placement": any(
            bool(row.get("can_verify_physical_placement"))
            for row in worker_results
        ),
        "can_remap_live_ep_rank": any(
            bool(row.get("can_remap_live_ep_rank")) for row in worker_results
        ),
        "can_measure_all_to_all": any(
            bool(row.get("can_measure_all_to_all")) for row in worker_results
        ),
        "capability_reasons": ",".join(
            sorted(
                {
                    str(row.get("capability_reason"))
                    for row in worker_results
                    if row.get("capability_reason")
                }
            )
        ),
        "worker_results": worker_results,
        "active_requests_at_barrier": any(
            bool(result.get("active_requests_at_barrier"))
            for result in normalized
        ),
        "step_boundary_barrier": bool(normalized) and all(
            bool(result.get("step_boundary_barrier", True))
            for result in normalized
        ),
    }


def aggregate_dp_engine_moe_metadata(
    engine_results: list[Any],
) -> dict[str, Any]:
    """Merge runtime placement and A2A counters from all DP EngineCores."""
    global_hotness: dict[str, int] = {}
    recent_hotness: dict[str, int] = {}
    placement_shards: dict[str, list[dict[str, Any]]] = {}
    worker_snapshots: dict[str, dict[str, Any]] = {}
    a2a_worker_snapshots: list[dict[str, Any]] = []
    tracing_enabled = False
    for dp_rank, result in enumerate(engine_results):
        if not isinstance(result, Mapping):
            continue
        tracing_enabled = tracing_enabled or bool(
            result.get("moe_route_tracing_enabled")
        )
        for target, field in (
            (global_hotness, "global_expert_hotness"),
            (recent_hotness, "recent_window_expert_hotness"),
        ):
            values = result.get(field)
            if isinstance(values, Mapping):
                for key, value in values.items():
                    target[str(key)] = target.get(str(key), 0) + int(value or 0)
        shards = result.get("runtime_expert_placement_shards")
        if isinstance(shards, Mapping):
            for expert_key, rows in shards.items():
                if not isinstance(rows, list):
                    continue
                placement_shards.setdefault(str(expert_key), []).extend(
                    dict(row) for row in rows if isinstance(row, Mapping)
                )
        snapshots = result.get("runtime_expert_placement_worker_snapshots")
        if isinstance(snapshots, Mapping):
            for worker_key, snapshot in snapshots.items():
                if isinstance(snapshot, Mapping):
                    worker_snapshots[f"dp:{dp_rank}/worker:{worker_key}"] = dict(
                        snapshot
                    )
        rows = result.get("all_to_all_worker_snapshots")
        if isinstance(rows, list):
            a2a_worker_snapshots.extend(
                dict(row) for row in rows if isinstance(row, Mapping)
            )

    def sum_field(field: str) -> int:
        return sum(
            int(result.get(field, 0) or 0)
            for result in engine_results
            if isinstance(result, Mapping)
        )

    available = bool(global_hotness)
    return {
        "moe_route_tracing_enabled": tracing_enabled,
        "moe_route_histogram_available": available,
        "moe_route_histogram_source": (
            "vllm_runtime_topk" if available else "unavailable"
        ),
        "moe_route_histogram_kind": (
            "runtime_observed_topk" if available else "unavailable"
        ),
        "global_expert_hotness": global_hotness,
        "recent_window_expert_hotness": recent_hotness,
        "runtime_expert_placement_available": bool(placement_shards),
        "runtime_expert_placement_worker_count": len(worker_snapshots),
        "runtime_expert_placement_shard_count": sum(
            len(rows) for rows in placement_shards.values()
        ),
        "runtime_expert_placement_shards": placement_shards,
        "runtime_expert_placement_worker_snapshots": worker_snapshots,
        "all_to_all_counters_available": any(
            bool(result.get("all_to_all_counters_available"))
            for result in engine_results
            if isinstance(result, Mapping)
        ),
        "all_to_all_collective_calls": sum_field("all_to_all_collective_calls"),
        "all_to_all_observed_input_bytes": sum_field(
            "all_to_all_observed_input_bytes"
        ),
        "all_to_all_observed_output_bytes": sum_field(
            "all_to_all_observed_output_bytes"
        ),
        "all_to_all_internode_calls": sum_field("all_to_all_internode_calls"),
        "all_to_all_measurement_kind": ",".join(
            sorted(
                {
                    str(result.get("all_to_all_measurement_kind"))
                    for result in engine_results
                    if isinstance(result, Mapping)
                    and result.get("all_to_all_measurement_kind")
                }
            )
        ) or "unavailable",
        "all_to_all_worker_snapshots": a2a_worker_snapshots,
        "dp_engine_coordinated": True,
        "dp_engine_count": len(engine_results),
    }


@dataclass
class _Layer:
    name: str
    index: int
    module: Any
    old_local_ids: list[int]
    new_local_ids: list[int]
    weights: list[torch.Tensor]


def _target_ep_rank(value: Any, ep_size: int) -> int:
    match = _EP_RANK_RE.fullmatch(str(value))
    if match is None or int(match.group(1)) >= ep_size:
        raise ValueError(f"unsupported_target_rank:{value}")
    return int(match.group(1))


def _layers_for_plan(model: Any, plan: Mapping[str, Any]) -> tuple[list[_Layer], Any]:
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
        FusedMoEModularMethod,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    if os.environ.get("VLLM_SPOTSERVE_EXPERT_REMAP", "").lower() not in _TRUTHY:
        raise ValueError("live_expert_remap_not_enabled")
    if plan.get("live_expert_remap") is not True:
        raise ValueError("live_expert_remap_not_requested")
    if model is None:
        raise ValueError("model_unavailable")
    if int(plan.get("sllm_replica_count", 1)) != 1:
        raise ValueError("multiple_replicas_not_supported")
    if int(plan.get("expert_physical_replication_factor", 1)) != 1:
        raise ValueError("expert_replication_not_supported")
    rank_map = plan.get("expert_to_target_rank")
    if not isinstance(rank_map, Mapping) or not rank_map:
        raise ValueError("expert_to_target_rank_required")
    if not plan.get("placement_fingerprint"):
        raise ValueError("placement_fingerprint_required")

    coordinator = get_ep_group()
    group = coordinator.device_group
    ep_size = group.size()
    ep_rank = group.rank()
    if ep_size < 2:
        raise ValueError("at_least_two_ep_ranks_required")
    target_count = plan.get("target_rank_count")
    if target_count is not None and int(target_count) != ep_size:
        raise ValueError("changing_ep_size_not_supported")
    target_parallel = plan.get("target_parallel_plan") or {}
    if not isinstance(target_parallel, Mapping):
        raise ValueError("target_parallel_plan_invalid")
    if target_parallel.get("enable_expert_parallel") is False:
        raise ValueError("target_disables_expert_parallel")

    layers: list[_Layer] = []
    all_keys: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        match = _LAYER_RE.search(name)
        if match is None:
            raise ValueError(f"moe_layer_id_unavailable:{name}")
        layer_id = int(match.group(1))
        quant_method = module.quant_method
        if isinstance(quant_method, FusedMoEModularMethod):
            quant_method = quant_method.old_quant_method
        if not isinstance(quant_method, UnquantizedFusedMoEMethod):
            raise ValueError("quantized_expert_remap_not_supported")
        if (not module.use_ep or module.enable_eplb
                or module.num_fused_shared_experts
                or module.rocm_aiter_fmoe_enabled):
            raise ValueError("unsupported_moe_layout")
        if module.moe_parallel_config.all2all_backend not in {
            "naive",
            "allgather_reducescatter",
            "spotserve_sparse",
        }:
            raise ValueError("all_to_all_backend_does_not_support_custom_map")
        if module.ep_rank != ep_rank or module.ep_size != ep_size:
            raise ValueError("ep_group_mismatch")
        runtime_parallel = module.vllm_config.parallel_config
        # Each vLLM DP rank owns an independent EngineCore.  The current
        # ServerlessLLM hook enters collectives through one EngineCore only,
        # so a DP-spanning remap would leave the other ranks outside the
        # transfer collective and deadlock.  Keep DP2 available for traffic
        # instrumentation, but fail closed for physical remap until a
        # coordinator can barrier every DP EngineCore simultaneously.
        runtime_dp_size = int(runtime_parallel.data_parallel_size)
        coordinated_dp_engines = int(plan.get("dp_engine_count", 0) or 0)
        if runtime_dp_size != 1 and not (
            plan.get("dp_engine_coordinated") is True
            and coordinated_dp_engines == runtime_dp_size
        ):
            raise ValueError("data_parallel_expert_remap_not_coordinated")
        for field in ("tensor_parallel_size", "data_parallel_size"):
            target_value = target_parallel.get(field)
            if target_value is not None and int(target_value) != int(
                getattr(runtime_parallel, field)
            ):
                raise ValueError(f"changing_{field}_not_supported")
        local_count = int(module.local_num_experts)
        expert_count = int(module.global_num_experts)
        if local_count < 1 or expert_count != ep_size * local_count:
            raise ValueError("uneven_or_empty_ep_partition_not_supported")
        expert_map = module.expert_map
        if expert_map is None or expert_map.numel() != expert_count:
            raise ValueError("expert_map_unavailable")
        old_map = [int(v) for v in expert_map.detach().cpu().tolist()]
        old_local = [0] * local_count
        seen_slots: set[int] = set()
        for expert_id, slot in enumerate(old_map):
            if slot < 0:
                continue
            if slot >= local_count or slot in seen_slots:
                raise ValueError("invalid_local_expert_map")
            seen_slots.add(slot)
            old_local[slot] = expert_id
        if len(seen_slots) != local_count:
            raise ValueError("incomplete_local_expert_map")
        new_local: list[int] = []
        for expert_id in range(expert_count):
            key = f"layer:{layer_id}/expert:{expert_id}"
            all_keys.add(key)
            if key not in rank_map:
                raise ValueError(f"missing_target_expert:{key}")
            if _target_ep_rank(rank_map[key], ep_size) == ep_rank:
                new_local.append(expert_id)
        if len(new_local) != local_count:
            raise ValueError(f"local_expert_count_change:{name}")
        if isinstance(module.quant_method, FusedMoEModularMethod):
            weights = [module.w13_weight, module.w2_weight]
        else:
            weights = list(module.get_expert_weights())
        if len(weights) != 2 or any(
            weight.shape[0] != local_count or not weight.is_contiguous()
            for weight in weights
        ):
            raise ValueError("unsupported_expert_weight_layout")
        layers.append(_Layer(name, layer_id, module, old_local, new_local, weights))
    if not layers:
        raise ValueError("no_fused_moe_layers")
    if set(rank_map) != all_keys:
        raise ValueError("unexpected_target_experts")
    layers.sort(key=lambda layer: layer.index)
    if len({layer.index for layer in layers}) != len(layers):
        raise ValueError("duplicate_moe_layer_ids")
    shapes = [tuple(weight.shape) for weight in layers[0].weights]
    dtypes = [weight.dtype for weight in layers[0].weights]
    if any(
        [tuple(w.shape) for w in layer.weights] != shapes
        or [w.dtype for w in layer.weights] != dtypes
        for layer in layers[1:]
    ):
        raise ValueError("mixed_expert_weight_layout_not_supported")
    return layers, group


def prepare_expert_remap(model: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        layers, group = _layers_for_plan(model, plan)
    except Exception as exc:
        return {"ready": False, "reason": str(exc)}
    return {
        "ready": True,
        "reason": "fixed_ep_unquantized_layout_supported",
        "ep_rank": group.rank(),
        "ep_size": group.size(),
        "layer_count": len(layers),
        "local_experts_per_layer": len(layers[0].old_local_ids),
    }


def _weight_digests(layers: list[_Layer], *, new: bool) -> dict[str, str]:
    digests: dict[str, str] = {}
    for layer in layers:
        ids = layer.new_local_ids if new else layer.old_local_ids
        for slot, expert_id in enumerate(ids):
            digest = hashlib.sha256()
            for weight in layer.weights:
                raw = weight[slot].detach().contiguous().view(torch.uint8)
                digest.update(raw.cpu().numpy().tobytes())
            digests[f"layer:{layer.index}/expert:{expert_id}"] = digest.hexdigest()
    return digests


def _gather_digests(local: dict[str, str], group: Any) -> dict[str, str]:
    gathered: list[Any] = [None] * group.size()
    dist.all_gather_object(gathered, local, group=group)
    merged: dict[str, str] = {}
    for rank_digests in gathered:
        for key, value in rank_digests.items():
            if key in merged:
                raise RuntimeError(f"duplicate_resident_expert:{key}")
            merged[key] = value
    return merged


def _cross_node_movement(
    layers: list[_Layer], old_indices: list[list[int]], host_ids: list[str],
    local_rank: int,
) -> tuple[int, int]:
    if not host_ids or not all(host_ids):
        return 0, 0
    moved = 0
    moved_bytes = 0
    for layer, old_row in zip(layers, old_indices):
        local_count = len(layer.old_local_ids)
        weight_bytes = sum(
            int(weight[0].numel() * weight.element_size())
            for weight in layer.weights
        )
        for expert_id in layer.new_local_ids:
            source_rank = old_row.index(expert_id) // local_count
            if host_ids[source_rank] != host_ids[local_rank]:
                moved += 1
                moved_bytes += weight_bytes
    return moved, moved_bytes


@torch.no_grad()
def remap_expert_weights(model: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Move weights on all EP ranks, then atomically change local expert maps.

    The caller must have quiesced the engine and preflighted every rank.
    A failed collective is fatal; it must not be retried on a live engine.
    """
    from vllm.distributed.eplb.rebalance_execute import (
        rearrange_expert_weights_inplace,
    )

    started_at = time.perf_counter()
    _LAST_REMAP.clear()
    layers, group = _layers_for_plan(model, plan)
    device = layers[0].weights[0].device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    old_local = torch.tensor(
        [layer.old_local_ids for layer in layers], device=device, dtype=torch.int64
    )
    gathered = [torch.empty_like(old_local) for _ in range(group.size())]
    dist.all_gather(gathered, old_local, group=group)
    old_indices = torch.cat(gathered, dim=1)
    host_ids: list[str] = [""] * group.size()
    dist.all_gather_object(
        host_ids,
        os.environ.get("SPOTSERVE_PHYSICAL_HOST_ID", "").strip(),
        group=group,
    )
    host_ids = [str(host_id or "").strip() for host_id in host_ids]
    if plan.get("require_cross_node") is True and (
        not all(host_ids) or len(set(host_ids)) < 2
    ):
        raise ValueError("cross_node_requires_distinct_physical_host_ids")
    new_indices = torch.tensor(
        [
            [
                expert_id
                for rank in range(group.size())
                for expert_id in range(layer.module.global_num_experts)
                if _target_ep_rank(
                    plan["expert_to_target_rank"][
                        f"layer:{layer.index}/expert:{expert_id}"
                    ],
                    group.size(),
                ) == rank
            ]
            for layer in layers
        ],
        device=device,
        dtype=torch.int64,
    )
    for row, layer in zip(old_indices.tolist(), layers):
        if sorted(row) != list(range(layer.module.global_num_experts)):
            raise RuntimeError("current_expert_coverage_invalid")
    cross_node_moved, cross_node_bytes = _cross_node_movement(
        layers, old_indices.tolist(), host_ids, group.rank()
    )
    if plan.get("require_cross_node") is True:
        global_cross_node_moved = torch.tensor(
            [cross_node_moved], device=device, dtype=torch.int64
        )
        dist.all_reduce(global_cross_node_moved, group=group)
        if not int(global_cross_node_moved.item()):
            raise ValueError("plan_does_not_move_experts_across_physical_hosts")
    before = _gather_digests(_weight_digests(layers, new=False), group)
    if len(before) != len(plan["expert_to_target_rank"]):
        raise RuntimeError("current_weight_coverage_invalid")
    moved_by_layer = [
        len(set(layer.new_local_ids) - set(layer.old_local_ids))
        for layer in layers
    ]
    moved = sum(moved_by_layer)
    local_bytes = sum(
        int(weight[0].numel() * weight.element_size()) * moved_count
        for layer, moved_count in zip(layers, moved_by_layer)
        for weight in layer.weights
    )
    if not torch.equal(old_indices, new_indices):
        rearrange_expert_weights_inplace(
            old_indices,
            new_indices,
            [layer.weights for layer in layers],
            group,
        )
    after = _gather_digests(_weight_digests(layers, new=True), group)
    if before != after:
        raise RuntimeError("expert_weight_digest_mismatch_after_transfer")
    for layer in layers:
        updated = torch.full_like(layer.module.expert_map, -1)
        for slot, expert_id in enumerate(layer.new_local_ids):
            updated[expert_id] = slot
        layer.module.expert_map.copy_(updated)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    all_to_all = get_all_to_all_counters()
    result = {
        "applied": True,
        "success": True,
        "reason": "expert_weights_transferred_and_verified",
        "hook_kind": "spotserve_quiescent_ep_remap",
        "physical_weight_migration": bool(moved),
        "runtime_verified_placement": True,
        "verification_level": "weights_and_expert_map_verified",
        "can_verify_physical_placement": True,
        "can_remap_live_ep_rank": False,
        # Instrumentation being enabled is not proof that this parallel shape
        # traversed an EP collective.  Report measurement support only after
        # at least one real dispatch/combine call has been observed.
        "can_measure_all_to_all": bool(all_to_all["available"]),
        "moved_local_expert_shards": moved,
        "moved_local_weight_bytes": local_bytes,
        "physical_host_ids_observed": bool(all(host_ids)),
        "cross_node_weight_migration": bool(cross_node_moved),
        "cross_node_moved_local_expert_shards": cross_node_moved,
        "cross_node_moved_local_weight_bytes": cross_node_bytes,
        "remap_duration_ms": (time.perf_counter() - started_at) * 1000.0,
        "placement_fingerprint": str(plan.get("placement_fingerprint") or ""),
        "contract_seen_by_runtime": True,
        "contract_seen_worker_count": 1,
        "contract_seen_worker_total": 1,
        "worker_rank": group.rank(),
    }
    result.update({
        f"all_to_all_{key}": value for key, value in all_to_all.items()
    })
    _LAST_REMAP.clear()
    _LAST_REMAP.update(result)
    return result


def last_remap_status(plan: Mapping[str, Any]) -> dict[str, Any]:
    if (
        plan.get("live_expert_remap") is True
        and _LAST_REMAP.get("applied")
        and _LAST_REMAP.get("placement_fingerprint")
        == str(plan.get("placement_fingerprint") or "")
    ):
        result = dict(_LAST_REMAP)
        result.update({
            f"all_to_all_{key}": value
            for key, value in get_all_to_all_counters().items()
        })
        result["can_measure_all_to_all"] = bool(
            result.get("all_to_all_available", False)
        )
        return result
    return {}
