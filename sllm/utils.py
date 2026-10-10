# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
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
import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import ray


def _managed_ray_fallback_enabled() -> bool:
    return os.getenv("SLLM_ALLOW_RAY_NODE_ID_WORKERS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _alive_ray_nodes():
    return [node for node in ray.nodes() if node.get("Alive", False)]


def get_worker_nodes():
    ray_nodes = _alive_ray_nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        assert ray_node_id is not None, "NodeID not found"
        resources = node.get("Resources", {})
        assert resources != {}, "Resources not found"
        node_address = node.get("NodeManagerAddress", None)
        assert (
            node_address is not None and node_address != ""
        ), "NodeManagerAddress not found"
        if resources.get("control_node", 0) > 0:
            continue  # Skip the control node

        for key, value in resources.items():
            if key.startswith("worker_id_"):
                node_id = key.split("_")[-1]
                worker_node_info[node_id] = {
                    "ray_node_id": ray_node_id,
                    "address": node_address,
                    "free_gpu": resources.get("GPU", 0),
                    "total_gpu": resources.get("GPU", 0),
                }

    if worker_node_info or not _managed_ray_fallback_enabled():
        return worker_node_info

    # CSCC Managed Ray owns the raylets and does not expose a way for a job to
    # attach worker_id_* custom resources to those existing nodes.  In this
    # explicit opt-in mode, a one-GPU Ray node is the worker identity.  The
    # full NodeID is stable for the session and NodeAffinity is used by every
    # placement call below, so this does not silently weaken placement.
    for node in sorted(
        ray_nodes,
        key=lambda row: (
            str(row.get("NodeManagerAddress") or ""),
            str(row.get("NodeID") or ""),
        ),
    ):
        resources = node.get("Resources", {})
        gpu_count = int(float(resources.get("GPU", 0) or 0))
        if gpu_count <= 0:
            continue
        if gpu_count != 1:
            raise RuntimeError(
                "managed_ray_node_id_fallback_requires_one_gpu_per_node"
            )
        node_id = str(node.get("NodeID") or "")
        address = str(node.get("NodeManagerAddress") or "")
        if not node_id or not address:
            raise RuntimeError("managed_ray_worker_identity_is_incomplete")
        worker_node_info[node_id] = {
            "ray_node_id": node_id,
            "address": address,
            "free_gpu": float(resources.get("GPU", 0) or 0),
            "total_gpu": float(resources.get("GPU", 0) or 0),
            "identity_source": "ray_node_id",
        }

    return worker_node_info


def worker_placement_options(
    worker_id: str, resource_fraction: float = 0.1
) -> Dict[str, Any]:
    """Return strict placement options for an SLLM logical worker."""
    worker_id = str(worker_id)
    resource_key = f"worker_id_{worker_id}"
    for node in _alive_ray_nodes():
        resources = node.get("Resources", {})
        if float(resources.get(resource_key, 0) or 0) > 0:
            return {
                "resources": {
                    "worker_node": resource_fraction,
                    resource_key: resource_fraction,
                }
            }
    if not _managed_ray_fallback_enabled():
        return {
            "resources": {
                "worker_node": resource_fraction,
                resource_key: resource_fraction,
            }
        }
    workers = get_worker_nodes()
    if worker_id not in workers:
        raise ValueError(f"unknown_managed_ray_worker:{worker_id}")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=str(workers[worker_id]["ray_node_id"]), soft=False
        )
    }


def control_plane_placement_options() -> Dict[str, Any]:
    """Keep control actors on the Ray head when custom resources are absent."""
    for node in _alive_ray_nodes():
        if float(node.get("Resources", {}).get("control_node", 0) or 0) > 0:
            return {"resources": {"control_node": 0.1}}
    if not _managed_ray_fallback_enabled():
        return {"resources": {"control_node": 0.1}}
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    current_node_id = str(ray.get_runtime_context().get_node_id())
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=current_node_id, soft=False
        )
    }


class InstanceState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    PREEMPTING = "preempting"
    DEAD = "dead"


class NodeState(str, Enum):
    READY = "ready"
    PREEMPTING = "preempting"
    DEAD = "dead"


@dataclass
class InstanceStatus:
    instance_id: str
    node_id: str
    num_gpu: int
    concurrency: int

    model_name: Optional[str] = None
    state: Optional[str] = None
    num_current_tokens: Optional[int] = None
    resuming_latency: Optional[float] = None
    member_node_ids: List[str] = field(default_factory=list)


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int
    num_gpu: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    concurrency: int = 0
    state: InstanceState = InstanceState.STARTING

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Ray worker IDs, not Kubernetes host names. node_id is only the primary
    # actor location; every participating rank belongs to this instance.
    member_node_ids: List[str] = field(default_factory=list)
    node_allocations: Dict[str, int] = field(default_factory=dict)
    preemption_deadline_time_s: Optional[float] = None

    def worker_node_ids(self) -> List[str]:
        return list(dict.fromkeys(
            [str(node) for node in self.member_node_ids]
            + ([str(self.node_id)] if self.node_id is not None else [])
        ))

    def _can_accept_request_locked(self, num_requests: int = 1) -> bool:
        if num_requests <= 0:
            return self.concurrency + num_requests >= 0
        if not self.ready:
            return False
        if self.state in {
            InstanceState.STARTING,
            InstanceState.DRAINING,
            InstanceState.PREEMPTING,
            InstanceState.DEAD,
        }:
            return False
        return self.concurrency + num_requests <= self.max_queue_length

    async def can_accept_request(self, num_requests: int = 1) -> bool:
        async with self.lock:
            return self._can_accept_request_locked(num_requests)

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if not self._can_accept_request_locked(num_requests):
                return False
            self.concurrency += num_requests
            return True

    async def check_request_queue(self):
        return await self.can_accept_request(1)

    async def mark_ready(self, node_id: Optional[str] = None):
        async with self.lock:
            if node_id is not None:
                self.node_id = node_id
            if self.state in {
                InstanceState.DRAINING,
                InstanceState.PREEMPTING,
                InstanceState.DEAD,
            }:
                return False
            self.ready = True
            self.state = InstanceState.READY
            return True

    async def mark_draining(self):
        async with self.lock:
            self.ready = False
            self.state = InstanceState.DRAINING

    async def mark_ready_after_drain(self, num_gpu: Optional[int] = None):
        """Reopen admission after a successful or safely aborted drain."""
        async with self.lock:
            if self.state != InstanceState.DRAINING:
                return False
            if num_gpu is not None:
                self.num_gpu = num_gpu
            self.ready = True
            self.state = InstanceState.READY
            return True

    async def mark_preempting(self):
        async with self.lock:
            self.ready = False
            self.state = InstanceState.PREEMPTING

    async def mark_dead(self):
        async with self.lock:
            self.ready = False
            self.state = InstanceState.DEAD

    async def mark_recovered(self):
        async with self.lock:
            if self.state != InstanceState.PREEMPTING:
                return False
            self.ready = True
            self.state = InstanceState.READY
            return True

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.num_gpu,
                self.concurrency,
                state=self.state.value,
                member_node_ids=self.worker_node_ids(),
            )
