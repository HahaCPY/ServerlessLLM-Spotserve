"""Experimental, quiescent in-place remap for unquantized vLLM FusedMoE.

Only a fixed EP group with one physical copy of each expert is supported.
All EP workers must enter ``remap_expert_weights`` through collective_rpc.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

_TRUTHY = {"1", "true", "yes", "on"}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EP_RANK_RE = re.compile(r"^(?:replica:0/)?ep-rank:(\d+)$")
_LAST_REMAP: dict[str, Any] = {}


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
        if not isinstance(module.quant_method, UnquantizedFusedMoEMethod):
            raise ValueError("quantized_expert_remap_not_supported")
        if (not module.use_ep or module.enable_eplb
                or module.num_fused_shared_experts
                or module.rocm_aiter_fmoe_enabled):
            raise ValueError("unsupported_moe_layout")
        if module.moe_parallel_config.all2all_backend not in {
            "naive",
            "allgather_reducescatter",
        }:
            raise ValueError("all_to_all_backend_does_not_support_custom_map")
        if module.ep_rank != ep_rank or module.ep_size != ep_size:
            raise ValueError("ep_group_mismatch")
        runtime_parallel = module.vllm_config.parallel_config
        if int(runtime_parallel.data_parallel_size) != 1:
            raise ValueError("data_parallel_expert_remap_not_supported")
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
        "can_measure_all_to_all": False,
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
        return dict(_LAST_REMAP)
    return {}
