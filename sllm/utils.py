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
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import ray


def get_worker_nodes():
    ray_nodes = ray.nodes()
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

    return worker_node_info


class InstanceState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
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

    async def mark_preempting(self):
        async with self.lock:
            self.ready = False
            self.state = InstanceState.PREEMPTING

    async def mark_dead(self):
        async with self.lock:
            self.ready = False
            self.state = InstanceState.DEAD

    async def get_status(self):
        async with self.lock:
            return InstanceStatus(
                self.instance_id,
                self.node_id,
                self.num_gpu,
                self.concurrency,
                state=self.state.value,
            )
