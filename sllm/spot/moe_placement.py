from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ExpertShard:
    layer_id: int
    expert_id: int
    rank_id: str
    node_id: str
    gpu_id: str
    physical_expert_id: Optional[int] = None
    weight_size_bytes: int = 0
    weight_resident: bool = True
    routed_tokens: int = 0
    recent_execution_count: int = 0
    load_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "physical_expert_id": self.physical_expert_id,
            "rank_id": self.rank_id,
            "node_id": self.node_id,
            "gpu_id": self.gpu_id,
            "weight_size_bytes": self.weight_size_bytes,
            "weight_resident": self.weight_resident,
            "routed_tokens": self.routed_tokens,
            "recent_execution_count": self.recent_execution_count,
            "load_score": self.load_score,
        }


@dataclass(frozen=True)
class ExpertPlacementState:
    model_name: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    vllm_data_parallel_size: int
    sllm_replica_count: int
    expert_parallel_enabled: bool
    effective_expert_parallel_size: int
    expert_parallel_size_source: str
    expert_physical_replication_factor: int = 1
    placement_epoch: int = 0
    placement_source: str = "unavailable"
    shards: tuple[ExpertShard, ...] = field(default_factory=tuple)

    @property
    def placement_available(self) -> bool:
        return bool(self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "vllm_data_parallel_size": self.vllm_data_parallel_size,
            "sllm_replica_count": self.sllm_replica_count,
            "expert_parallel_enabled": self.expert_parallel_enabled,
            "effective_expert_parallel_size": (
                self.effective_expert_parallel_size
            ),
            "expert_parallel_size_source": self.expert_parallel_size_source,
            "expert_physical_replication_factor": (
                self.expert_physical_replication_factor
            ),
            "expert_placement_available": self.placement_available,
            "placement_epoch": self.placement_epoch,
            "placement_source": self.placement_source,
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class ExpertPlacementPlan:
    model_name: str
    target_parallel_plan: Mapping[str, Any]
    expert_to_target_rank: Mapping[str, str] = field(default_factory=dict)
    placement_epoch: int = 0
    moved_expert_count: int = 0
    moved_weight_bytes: int = 0
    estimated_dispatch_cost: float = 0.0
    estimated_load_balance_penalty: float = 0.0
    reason: str = "metadata_only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_parallel_plan": dict(self.target_parallel_plan),
            "expert_to_target_rank": dict(self.expert_to_target_rank),
            "placement_epoch": self.placement_epoch,
            "moved_expert_count": self.moved_expert_count,
            "moved_weight_bytes": self.moved_weight_bytes,
            "estimated_dispatch_cost": self.estimated_dispatch_cost,
            "estimated_load_balance_penalty": (
                self.estimated_load_balance_penalty
            ),
            "reason": self.reason,
        }
