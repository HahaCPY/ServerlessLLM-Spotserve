"""Two-GPU smoke for the vLLM primitive used by quiescent expert remap."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sllm.vllm_expert_remap import remap_expert_weights
from vllm.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
)


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        device = torch.device("cuda", rank)
        old = torch.tensor([[0, 1, 2, 3]], device=device)
        new = torch.tensor([[0, 2, 1, 3]], device=device)
        local_ids = [0, 1] if rank == 0 else [2, 3]
        w13 = torch.stack(
            [torch.full((4,), float(expert), device=device) for expert in local_ids]
        )
        w2 = torch.stack(
            [torch.full((2,), float(expert + 10), device=device)
             for expert in local_ids]
        )
        rearrange_expert_weights_inplace(
            old, new, [[w13, w2]], dist.group.WORLD
        )
        expected = [0, 2] if rank == 0 else [1, 3]
        for slot, expert in enumerate(expected):
            assert torch.all(w13[slot] == float(expert))
            assert torch.all(w2[slot] == float(expert + 10))
        print(f"rank={rank} verified_experts={expected}", flush=True)

        class FakeQuant:
            pass

        class FakeMoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_num_experts = 4
                self.local_num_experts = 2
                self.expert_map = torch.full((4,), -1, device=device, dtype=torch.int32)
                for slot, expert in enumerate(local_ids):
                    self.expert_map[expert] = slot
                self.quant_method = FakeQuant()
                self.use_ep = True
                self.enable_eplb = False
                self.num_fused_shared_experts = 0
                self.rocm_aiter_fmoe_enabled = False
                self.ep_rank = rank
                self.ep_size = 2
                self.vllm_config = SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        tensor_parallel_size=2, data_parallel_size=1
                    )
                )
                self.moe_parallel_config = SimpleNamespace(
                    all2all_backend="allgather_reducescatter"
                )
                self.w13_weight = torch.nn.Parameter(torch.stack([
                    torch.full((4,), float(expert), device=device)
                    for expert in local_ids
                ]))
                self.w2_weight = torch.nn.Parameter(torch.stack([
                    torch.full((2,), float(expert + 10), device=device)
                    for expert in local_ids
                ]))

            def get_expert_weights(self):
                return [self.w13_weight.view(2, -1), self.w2_weight.view(2, -1)]

        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.layers[0].experts = FakeMoE()
        plan = {
            "live_expert_remap": True,
            "placement_fingerprint": "two-gpu-synthetic-remap",
            "target_rank_count": 2,
            "target_parallel_plan": {
                "enable_expert_parallel": True,
                "tensor_parallel_size": 2,
                "data_parallel_size": 1,
            },
            "expert_to_target_rank": {
                f"layer:0/expert:{expert}": f"replica:0/ep-rank:{target}"
                for expert, target in enumerate((0, 1, 0, 1))
            },
        }
        os.environ["VLLM_SPOTSERVE_EXPERT_REMAP"] = "1"
        with patch(
            "vllm.model_executor.layers.fused_moe.layer.FusedMoE", FakeMoE
        ), patch(
            "vllm.model_executor.layers.fused_moe."
            "unquantized_fused_moe_method.UnquantizedFusedMoEMethod", FakeQuant
        ), patch(
            "vllm.distributed.parallel_state.get_ep_group",
            return_value=SimpleNamespace(device_group=dist.group.WORLD),
        ):
            result = remap_expert_weights(model, plan)
        assert result["applied"] and result["physical_weight_migration"]
        assert result["moved_local_expert_shards"] == 1
        expert_map = model.layers[0].experts.expert_map.tolist()
        assert expert_map == ([0, -1, 1, -1] if rank == 0 else [-1, 0, -1, 1])
        for slot, expert in enumerate(expected):
            assert torch.all(model.layers[0].experts.w13_weight[slot] == expert)
            assert torch.all(model.layers[0].experts.w2_weight[slot] == expert + 10)
        print(f"rank={rank} remap_digest_and_map=passed", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("Two visible GPUs are required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
    print("physical_expert_tensor_transfer=passed")
    print("spotserve_quiescent_ep_remap=passed")


if __name__ == "__main__":
    main()
