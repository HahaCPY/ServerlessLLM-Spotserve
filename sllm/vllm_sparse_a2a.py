"""Routing-aware variable-size All-to-All for unquantized vLLM MoE.

This backend is intentionally narrow. It supports the TP1, DP/EP execution
shape used by the SpotServe MoE experiments and fails closed for unsupported
quantization or replicated-expert layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class _DispatchState:
    original_tokens: int
    local_tokens: torch.Tensor
    remote_tokens: torch.Tensor
    send_counts: list[int]
    recv_counts: list[int]


def _payload_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


class SpotServeSparsePrepareAndFinalize:
    """Pure PyTorch sparse token dispatch for vLLM's modular MoE kernel."""

    def __init__(self, dp_size: int, ep_size: int):
        from vllm.distributed import get_dp_group

        if dp_size <= 1 or ep_size != dp_size:
            raise ValueError(
                "spotserve_sparse_requires_tp1_and_ep_size_equal_dp_size"
            )
        self.group = get_dp_group()
        self.rank = int(self.group.rank_in_group)
        self.world_size = int(self.group.world_size)
        if self.world_size != dp_size:
            raise ValueError("spotserve_sparse_dp_group_size_mismatch")
        self._owner_cache_key: tuple[int, int] | None = None
        self._owner_cache: torch.Tensor | None = None
        self._state: _DispatchState | None = None

    def post_init_setup(self, fused_experts: Any) -> None:
        """Satisfy the modular MoE lifecycle hook used after kernel setup."""
        del fused_experts

    @property
    def activation_format(self):
        from vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEActivationFormat,
        )

        return FusedMoEActivationFormat.Standard

    def supports_async(self) -> bool:
        return False

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.world_size

    def output_is_reduced(self) -> bool:
        return True

    def _group(self):
        return self.group.device_group

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self._group())
        return recv_counts

    def _exchange_rows(
        self,
        tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> torch.Tensor:
        output = torch.empty(
            (sum(recv_counts), *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self._group(),
        )
        return output

    def _expert_owners(self, expert_map: torch.Tensor) -> torch.Tensor:
        version = int(getattr(expert_map, "_version", 0))
        cache_key = (int(expert_map.data_ptr()), version)
        if self._owner_cache_key == cache_key and self._owner_cache is not None:
            return self._owner_cache

        gathered = [torch.empty_like(expert_map) for _ in range(self.world_size)]
        dist.all_gather(gathered, expert_map, group=self._group())
        ownership = torch.stack([row >= 0 for row in gathered])
        coverage = ownership.sum(dim=0)
        if not bool(torch.all(coverage == 1).item()):
            raise ValueError(
                "spotserve_sparse_requires_exactly_one_owner_per_expert"
            )
        owners = ownership.to(torch.int64).argmax(dim=0)
        self._owner_cache_key = cache_key
        self._owner_cache = owners
        return owners

    @staticmethod
    def _require_unquantized(quant_config: Any) -> None:
        if getattr(quant_config, "quant_dtype", None) is not None:
            raise ValueError("spotserve_sparse_quantization_not_supported")

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: Any,
    ):
        self._require_unquantized(quant_config)
        if expert_map is None or expert_map.numel() != num_experts:
            raise ValueError("spotserve_sparse_requires_global_expert_map")
        if self._state is not None:
            raise RuntimeError("spotserve_sparse_overlapping_dispatch_not_supported")
        if apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise ValueError(
                    "spotserve_sparse_weight_on_input_requires_topk_one"
                )
            a1 = a1 * topk_weights.to(a1.dtype)

        owners = self._expert_owners(expert_map)
        routed_owners = owners[topk_ids.to(torch.int64)]
        destinations = torch.zeros(
            (a1.shape[0], self.world_size),
            dtype=torch.bool,
            device=a1.device,
        )
        destinations.scatter_(1, routed_owners, True)

        local_tokens = torch.nonzero(
            destinations[:, self.rank], as_tuple=False
        ).flatten()
        remote_pairs = torch.nonzero(destinations, as_tuple=False)
        remote_pairs = remote_pairs[remote_pairs[:, 1] != self.rank]
        if remote_pairs.numel():
            order = torch.argsort(remote_pairs[:, 1], stable=True)
            remote_pairs = remote_pairs[order]
            remote_tokens = remote_pairs[:, 0].to(torch.int64)
            remote_destinations = remote_pairs[:, 1].to(torch.int64)
        else:
            remote_tokens = torch.empty(0, dtype=torch.int64, device=a1.device)
            remote_destinations = remote_tokens

        send_counts_tensor = torch.bincount(
            remote_destinations, minlength=self.world_size
        ).to(dtype=torch.int64, device=a1.device)
        recv_counts_tensor = self._exchange_counts(send_counts_tensor)
        send_counts = [int(value) for value in send_counts_tensor.cpu().tolist()]
        recv_counts = [int(value) for value in recv_counts_tensor.cpu().tolist()]

        sent_hidden = a1[remote_tokens]
        sent_ids = topk_ids[remote_tokens].to(torch.int64)
        sent_weights = topk_weights[remote_tokens]
        recv_hidden = self._exchange_rows(sent_hidden, send_counts, recv_counts)
        recv_ids = self._exchange_rows(sent_ids, send_counts, recv_counts)
        recv_weights = self._exchange_rows(
            sent_weights, send_counts, recv_counts
        )
        recv_tokens = self._exchange_rows(
            remote_tokens, send_counts, recv_counts
        )

        dispatched_hidden = torch.cat((a1[local_tokens], recv_hidden), dim=0)
        dispatched_ids = torch.cat(
            (topk_ids[local_tokens].to(torch.int64), recv_ids), dim=0
        )
        dispatched_weights = torch.cat(
            (topk_weights[local_tokens], recv_weights), dim=0
        )
        is_local = expert_map[dispatched_ids] >= 0
        nonlocal_ids = torch.nonzero(expert_map < 0, as_tuple=False).flatten()
        if nonlocal_ids.numel() == 0:
            raise ValueError("spotserve_sparse_requires_nonlocal_expert_sentinel")
        dispatched_ids = torch.where(
            is_local, dispatched_ids, nonlocal_ids[0].to(dispatched_ids.dtype)
        )

        self._state = _DispatchState(
            original_tokens=int(a1.shape[0]),
            local_tokens=local_tokens.to(torch.int64),
            remote_tokens=remote_tokens,
            send_counts=send_counts,
            recv_counts=recv_counts,
        )
        from sllm.vllm_expert_remap import record_sparse_all_to_all_transfer

        sent_bytes = sum(
            _payload_bytes(tensor)
            for tensor in (sent_hidden, sent_ids, sent_weights, remote_tokens)
        )
        recv_bytes = sum(
            _payload_bytes(tensor)
            for tensor in (recv_hidden, recv_ids, recv_weights, recv_tokens)
        )
        record_sparse_all_to_all_transfer(
            "dispatch",
            sent_bytes=sent_bytes,
            received_bytes=recv_bytes,
            remote_rows=int(remote_tokens.numel()),
            local_rows=int(local_tokens.numel()),
        )
        return dispatched_hidden, None, None, dispatched_ids, dispatched_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: Any,
    ) -> None:
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceContiguous,
            TopKWeightAndReduceDelegate,
        )

        state = self._state
        if state is None:
            raise RuntimeError("spotserve_sparse_finalize_without_dispatch")
        self._state = None
        if fused_expert_output.numel():
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            contributions = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            contributions = torch.empty(
                (0, output.shape[1]), dtype=output.dtype, device=output.device
            )

        local_count = int(state.local_tokens.numel())
        local_results = contributions[:local_count]
        remote_results = contributions[local_count:]
        returned_results = self._exchange_rows(
            remote_results, state.recv_counts, state.send_counts
        )
        output.zero_()
        if local_count:
            output.index_add_(0, state.local_tokens, local_results)
        if state.remote_tokens.numel():
            output.index_add_(0, state.remote_tokens, returned_results)

        from sllm.vllm_expert_remap import record_sparse_all_to_all_transfer

        record_sparse_all_to_all_transfer(
            "combine",
            sent_bytes=_payload_bytes(remote_results),
            received_bytes=_payload_bytes(returned_results),
            remote_rows=int(remote_results.shape[0]),
            local_rows=local_count,
        )
