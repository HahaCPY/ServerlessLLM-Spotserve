"""Read-only rank evidence for a native-MoE Elastic EP feasibility probe."""

import os
import re

import torch

from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.worker.gpu_worker import Worker


class ElasticEPPhase1Worker(Worker):
    def get_elastic_ep_phase1_snapshot(self):
        model = self.model_runner.model
        layers = []
        for name, module in model.named_modules():
            if not (
                hasattr(module, "w13_weight")
                and hasattr(module, "w2_weight")
                and hasattr(module, "moe_config")
            ):
                continue
            match = re.search(r"layers\.(\d+)", name)
            if match is None:
                continue
            mapping = module.expert_map
            total_experts = int(module.moe_config.num_experts)
            if mapping is None:
                expert_ids = list(range(total_experts))
            else:
                expert_ids = [
                    expert_id
                    for expert_id, local_id in enumerate(mapping.cpu().tolist())
                    if local_id >= 0
                ]
            layers.append(
                {
                    "layer": int(match.group(1)),
                    "name": name,
                    "total_experts": total_experts,
                    "expert_ids": expert_ids,
                    "expert_map": None if mapping is None else mapping.cpu().tolist(),
                    "w13_shape": list(module.w13_weight.shape),
                    "w2_shape": list(module.w2_weight.shape),
                    "w13_data_ptr": int(module.w13_weight.data_ptr()),
                    "w2_data_ptr": int(module.w2_weight.data_ptr()),
                    "w13_device": str(module.w13_weight.device),
                    "w2_device": str(module.w2_weight.device),
                }
            )
        ep_group = get_ep_group()
        dp_group = get_dp_group()
        return {
            "pid": os.getpid(),
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "data_parallel_rank": int(self.parallel_config.data_parallel_rank),
            "data_parallel_size": int(self.parallel_config.data_parallel_size),
            "gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
            "model_class": type(model).__name__,
            "mixture_of_experts_interface": bool(is_mixture_of_experts(model)),
            "num_moe_layers": int(getattr(model, "num_moe_layers", 0)),
            "num_logical_experts": int(getattr(model, "num_logical_experts", 0)),
            "expert_weights_layer_count": len(getattr(model, "expert_weights", [])),
            "model_object_id": id(model),
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "ep_group": {
                "object_id": id(ep_group),
                "device_group_id": id(ep_group.device_group),
                "world_size": int(ep_group.world_size),
                "ranks": list(ep_group.ranks),
            },
            "dp_group": {
                "object_id": id(dp_group),
                "device_group_id": id(dp_group.device_group),
                "world_size": int(dp_group.world_size),
                "ranks": list(dp_group.ranks),
            },
            "layers": sorted(layers, key=lambda row: row["layer"]),
        }
