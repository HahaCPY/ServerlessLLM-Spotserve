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
import copy
import inspect
import time
from typing import Any, Dict, List, Mapping, Optional

import ray
from sllm.logger import init_logger
from sllm.spot.metrics import (
    JsonlMetricsWriter,
    make_risk_aware_scheduling_event,
)
from sllm.spot.risk_metadata_provider import (
    build_risk_metadata_provider,
    normalize_risk_metadata,
)
from sllm.spot.risk_aware_scheduling import (
    node_risk_score,
    plan_risk_aware_scheduling,
)
from sllm.utils import NodeState, get_worker_nodes

from .scheduler_utils import SllmScheduler

logger = init_logger(__name__)


class FcfsScheduler(SllmScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = dict(scheduler_config or {})
        self.latest_scheduling_decision = None
        self.risk_metadata_provider = None
        self.enable_spot_risk_aware = False
        self.enable_backend_runtime_metadata = False
        self.backend_actor_namespaces = []
        self.runtime_metadata_timeout_s = 1.0
        self.metrics_writer = None
        self._configure_features_from_scheduler_config()

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues = {}

        self.metadata_lock = asyncio.Lock()
        self.worker_nodes = {}
        self.model_instance = {}
        self.model_instance_resources = {}

        self.loop = asyncio.get_running_loop()

        self.running_lock = asyncio.Lock()
        self.running = False

    def _configure_features_from_scheduler_config(self):
        # Provider output is collected for every node when available, even if
        # risk-aware ordering is disabled. The ranking policy can therefore
        # consume the same provenance-bearing metadata without changing the
        # conservative fallback behavior.
        self.risk_metadata_provider = build_risk_metadata_provider(
            self.scheduler_config
        )
        self.enable_spot_risk_aware = bool(
            self.scheduler_config.get("enable_spot_risk_aware", False)
        )
        self.enable_backend_runtime_metadata = bool(
            self.scheduler_config.get("enable_backend_runtime_metadata", False)
        )
        configured_namespaces = self.scheduler_config.get(
            "backend_actor_namespaces", [None, "models"]
        )
        if isinstance(configured_namespaces, str):
            configured_namespaces = [configured_namespaces]
        self.backend_actor_namespaces = []
        for namespace in configured_namespaces:
            normalized_namespace = namespace or None
            if normalized_namespace not in self.backend_actor_namespaces:
                self.backend_actor_namespaces.append(normalized_namespace)
        self.runtime_metadata_timeout_s = max(
            0.001,
            float(
                self.scheduler_config.get("runtime_metadata_timeout_s", 1.0)
                or 1.0
            ),
        )
        metrics_path = self.scheduler_config.get("metrics_path")
        self.metrics_writer = (
            JsonlMetricsWriter(metrics_path) if metrics_path else None
        )

    def _ensure_node_metadata(
        self,
        node_info: Mapping,
        node_id: Optional[str] = None,
    ) -> dict:
        updated_node_info = dict(node_info)
        updated_node_info.setdefault("state", NodeState.READY.value)
        configured_node_id = str(
            node_id if node_id is not None else updated_node_info.get("node_id", "")
        )
        if configured_node_id:
            updated_node_info.update(
                self._configured_node_risk(configured_node_id)
            )
        return updated_node_info

    def _node_is_ready(self, node_info: Mapping) -> bool:
        return node_info.get("state", NodeState.READY.value) == NodeState.READY.value

    def _emit_metric(self, event: dict):
        if self.metrics_writer is None:
            return
        try:
            self.metrics_writer.emit(event)
        except Exception as e:
            logger.error(f"Failed to emit scheduler metric: {e}")

    def _configured_node_risk(self, node_id: str) -> dict:
        node_risk = self.scheduler_config.get("node_risk", {}) or {}
        return dict(node_risk.get(str(node_id), {}) or {})

    def _preserve_spot_metadata(
        self,
        updated_node_info: dict,
        previous_node_info: Mapping,
        node_id: str,
    ) -> dict:
        for key in (
            "spot_risk",
            "risk_score",
            "preemption_risk",
            "remaining_lifetime_s",
            "expected_remaining_lifetime_s",
            "loading_cost",
            "model_loading_cost",
            "load_cost",
            "risk_metadata_source",
            "risk_provider",
            "risk_observed_at",
            "risk_confidence",
            "provider",
            "region",
            "instance_type",
        ):
            if key in previous_node_info:
                updated_node_info[key] = previous_node_info[key]
        updated_node_info.update(self._configured_node_risk(str(node_id)))
        return updated_node_info

    @staticmethod
    def _optional_float_value(
        payload: Mapping[str, Any], key: str
    ) -> Optional[float]:
        try:
            value = payload.get(key)
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _apply_scheduler_capacity(
        self,
        updated_node_info: dict,
        allocated_gpu: float,
    ) -> dict:
        live_total_gpu = self._optional_float_value(
            updated_node_info, "total_gpu"
        )
        live_free_gpu = self._optional_float_value(
            updated_node_info, "free_gpu"
        )
        if live_total_gpu is None and live_free_gpu is None:
            return updated_node_info
        total_gpu = live_total_gpu if live_total_gpu is not None else live_free_gpu
        if total_gpu is None:
            return updated_node_info
        updated_node_info["total_gpu"] = total_gpu
        updated_node_info["free_gpu"] = max(
            0.0,
            min(
                total_gpu,
                (live_free_gpu if live_free_gpu is not None else total_gpu),
                total_gpu - allocated_gpu,
            ),
        )
        return updated_node_info

    def _apply_ray_available_capacity(
        self,
        updated_node_info: dict,
        node_id: str,
        available_resources: Mapping[str, Any],
    ) -> dict:
        """Conservatively reconcile scheduler state with live Ray resources.

        ``ray.nodes()`` reports static node resources, not currently available
        capacity.  The per-worker custom resource is consumed together with a
        vLLM backend actor, so on single-GPU worker containers it is a reliable
        guard against assigning a second actor to a GPU already held by a stale
        or pending Ray actor that the scheduler does not own.
        """
        total_gpu = self._optional_float_value(updated_node_info, "total_gpu")
        if total_gpu is None or total_gpu > 1.0:
            return updated_node_info

        worker_resource_key = f"worker_id_{node_id}"
        try:
            available_worker_resource = float(
                available_resources.get(worker_resource_key, 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return updated_node_info

        updated_node_info["ray_available_worker_resource"] = (
            available_worker_resource
        )
        if available_worker_resource < 0.999:
            updated_node_info["free_gpu"] = 0.0
        return updated_node_info

    def _allocated_gpu_by_node_unlocked(self) -> Dict[str, float]:
        allocated: Dict[str, float] = {}
        for model_name, instances in self.model_instance.items():
            resource_entries = self.model_instance_resources.get(
                model_name, {}
            )
            for instance_id, node_id in instances.items():
                resource_entry = resource_entries.get(instance_id, {})
                node_allocations = resource_entry.get("node_allocations")
                if isinstance(node_allocations, Mapping) and node_allocations:
                    for allocated_node_id, allocated_gpus in (
                        node_allocations.items()
                    ):
                        try:
                            parsed_gpus = max(0.0, float(allocated_gpus))
                        except (TypeError, ValueError):
                            continue
                        allocated[str(allocated_node_id)] = (
                            allocated.get(str(allocated_node_id), 0.0)
                            + parsed_gpus
                        )
                    continue
                try:
                    num_gpus = float(resource_entry.get("num_gpus", 1.0))
                except (TypeError, ValueError):
                    num_gpus = 1.0
                allocated[str(node_id)] = (
                    allocated.get(str(node_id), 0.0) + max(0.0, num_gpus)
                )
        return allocated

    @staticmethod
    def _distributed_gpu_allocation(
        worker_nodes: Mapping[str, Mapping[str, Any]],
        target_node_ids: List[str],
        requested_gpus: int,
        require_all_target_nodes: bool,
    ) -> Dict[str, int]:
        if requested_gpus <= 0:
            raise ValueError("distributed allocation requires at least one GPU")
        normalized_targets = list(
            dict.fromkeys(str(node) for node in target_node_ids)
        )
        if not normalized_targets:
            raise ValueError("distributed allocation requires target_node_ids")
        if require_all_target_nodes and requested_gpus < len(normalized_targets):
            raise ValueError(
                "distributed allocation has fewer GPUs than target nodes"
            )

        capacities: Dict[str, int] = {}
        for node_id in normalized_targets:
            node = worker_nodes.get(node_id)
            if node is None:
                raise RuntimeError(f"distributed_target_node_not_found:{node_id}")
            if node.get("state", NodeState.READY.value) != NodeState.READY.value:
                raise RuntimeError(f"distributed_target_node_not_ready:{node_id}")
            capacities[node_id] = max(
                0, int(float(node.get("free_gpu", 0) or 0))
            )

        allocations = {node_id: 0 for node_id in normalized_targets}
        remaining = requested_gpus
        if require_all_target_nodes:
            unavailable = [
                node_id for node_id, capacity in capacities.items() if capacity < 1
            ]
            if unavailable:
                raise RuntimeError(
                    "distributed_target_nodes_have_no_gpu:"
                    + ",".join(unavailable)
                )
            for node_id in normalized_targets:
                allocations[node_id] = 1
                remaining -= 1

        for node_id in sorted(
            normalized_targets,
            key=lambda value: (
                -(capacities[value] - allocations[value]),
                value,
            ),
        ):
            if remaining <= 0:
                break
            available = capacities[node_id] - allocations[node_id]
            assigned = min(available, remaining)
            allocations[node_id] += assigned
            remaining -= assigned

        if remaining:
            raise RuntimeError(
                "insufficient_distributed_gpu_capacity:"
                f"requested={requested_gpus},available={sum(capacities.values())},"
                f"targets={normalized_targets}"
            )
        return {
            node_id: count for node_id, count in allocations.items() if count > 0
        }

    async def _collect_provider_risk_metadata(
        self, worker_nodes: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect provider signals without making them mandatory."""
        collected: Dict[str, Dict[str, Any]] = {}
        for node_id, node_info in worker_nodes.items():
            try:
                payload = self.risk_metadata_provider.collect(
                    str(node_id), dict(node_info)
                )
                if inspect.isawaitable(payload):
                    payload = await asyncio.wait_for(
                        payload, timeout=self.runtime_metadata_timeout_s
                    )
            except Exception as exc:
                logger.info(
                    "Risk metadata provider failed for node %s: %s",
                    node_id,
                    exc,
                )
                continue
            if not isinstance(payload, Mapping):
                continue
            normalized = dict(payload)
            # Ray's live GPU accounting is authoritative for capacity.  A
            # provider may expose capacity too, but must not replace a
            # non-zero value reported by the worker heartbeat.
            for key in ("free_gpu", "total_gpu"):
                if key in node_info and int(node_info.get(key, 0) or 0) > 0:
                    normalized.pop(key, None)
            collected[str(node_id)] = normalized
        return collected

    def _ordered_candidate_nodes(
        self,
        model_name: str,
        num_gpus: int,
        worker_nodes: Mapping,
    ):
        if not self.enable_spot_risk_aware:
            self.latest_scheduling_decision = self._health_only_decision(
                model_name,
                num_gpus,
                worker_nodes,
            )
            self._emit_metric(
                make_risk_aware_scheduling_event(
                    model=model_name,
                    policy="health_only",
                    decision=self.latest_scheduling_decision,
                )
            )
            return list(worker_nodes.items())

        decision = plan_risk_aware_scheduling(
            model_name=model_name,
            worker_nodes=worker_nodes,
            requested_gpus=num_gpus,
            scheduler_config=self.scheduler_config,
        )
        self.latest_scheduling_decision = decision.to_dict()
        self._emit_metric(
            make_risk_aware_scheduling_event(
                model=model_name,
                policy="risk_aware",
                decision=self.latest_scheduling_decision,
            )
        )
        self.risk_metadata_provider = build_risk_metadata_provider(
            self.scheduler_config
        )
        ranked = [
            (candidate.node_id, worker_nodes[candidate.node_id])
            for candidate in decision.candidates
        ]
        logger.info(
            f"Risk-aware scheduling decision for {model_name}: "
            f"{self.latest_scheduling_decision}"
        )
        return ranked

    def _health_only_decision(
        self,
        model_name: str,
        num_gpus: int,
        worker_nodes: Mapping,
    ) -> Dict[str, Any]:
        selected_node_id = None
        candidates = []
        for node_id, node_info in worker_nodes.items():
            candidate = node_risk_score(
                str(node_id),
                node_info,
                num_gpus,
                scheduler_config=self.scheduler_config,
            ).to_dict()
            candidates.append(candidate)
            if selected_node_id is not None:
                continue
            if not self._node_is_ready(node_info):
                continue
            if int(node_info.get("free_gpu", 0) or 0) >= num_gpus:
                selected_node_id = str(node_id)
        return {
            "action": "allocate" if selected_node_id is not None else "no_capacity",
            "model_name": model_name,
            "requested_gpus": num_gpus,
            "selected_node_id": selected_node_id,
            "candidates": candidates,
            "reason": "health_only",
        }

    async def mark_node_preempting(self, node_id: str):
        return await self._mark_node_state(node_id, NodeState.PREEMPTING)

    async def mark_node_recovered(self, node_id: str):
        return await self._mark_node_state(node_id, NodeState.READY)

    async def mark_node_dead(self, node_id: str):
        return await self._mark_node_state(node_id, NodeState.DEAD)

    async def add_worker_node(
        self, node_id: str, node_info: Optional[Mapping[str, Any]] = None
    ):
        """Register capacity discovered while the service is running."""
        async with self.metadata_lock:
            previous = dict(self.worker_nodes.get(str(node_id), {}))
            updated = {
                **previous,
                **dict(node_info or {}),
                "state": NodeState.READY.value,
            }
            updated.setdefault("ray_node_id", None)
            updated.setdefault("address", None)
            updated.setdefault("free_gpu", updated.get("total_gpu", 0))
            updated.setdefault("total_gpu", updated.get("free_gpu", 0))
            self.worker_nodes[str(node_id)] = self._ensure_node_metadata(
                updated, node_id=str(node_id)
            )
            return {
                "node_id": str(node_id),
                "node": dict(self.worker_nodes[str(node_id)]),
            }

    async def remove_worker_node(self, node_id: str):
        """Keep a removed node in metadata as DEAD for conservative replans."""
        return await self._mark_node_state(node_id, NodeState.DEAD)

    async def update_node_risk(self, node_id: str, risk_metadata: Mapping):
        async with self.metadata_lock:
            if node_id not in self.worker_nodes:
                self.worker_nodes[node_id] = {
                    "ray_node_id": None,
                    "address": None,
                    "free_gpu": 0,
                    "total_gpu": 0,
                    "state": NodeState.READY.value,
                }
            self.worker_nodes[node_id].update(dict(risk_metadata))
            logger.info(
                f"Updated scheduler risk metadata for node {node_id}: "
                f"{risk_metadata}"
            )
            return {
                "node_id": node_id,
                "risk_metadata": dict(risk_metadata),
            }

    async def update_scheduler_config(
        self,
        scheduler_config: Mapping,
        replace: bool = False,
    ):
        async with self.metadata_lock:
            if replace:
                self.scheduler_config = dict(scheduler_config or {})
            else:
                self.scheduler_config.update(dict(scheduler_config or {}))
            self._configure_features_from_scheduler_config()
            return {
                "enable_spot_risk_aware": self.enable_spot_risk_aware,
                "enable_backend_runtime_metadata": (
                    self.enable_backend_runtime_metadata
                ),
                "metrics_path": self.scheduler_config.get("metrics_path", ""),
            }

    async def _call_actor_method(
        self,
        actor: Any,
        method_name: str,
        **kwargs,
    ):
        method = getattr(actor, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            result = remote(**kwargs)
        else:
            result = method(**kwargs)
        if inspect.isawaitable(result):
            return await asyncio.wait_for(
                result,
                timeout=self.runtime_metadata_timeout_s,
            )
        return result

    def _get_backend_actor(self, instance_id: str):
        last_error = None
        for namespace in self.backend_actor_namespaces:
            try:
                if namespace is None:
                    return ray.get_actor(instance_id)
                return ray.get_actor(instance_id, namespace=namespace)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return ray.get_actor(instance_id)

    def _normalize_backend_runtime_metadata(
        self,
        metadata: Mapping[str, Any],
        instance_id: str,
        node_id: str,
    ) -> Dict[str, Any]:
        payload = dict(metadata)
        payload.setdefault("instance_id", instance_id)
        payload.setdefault("node_id", node_id)
        provider = str(
            payload.get("risk_provider")
            or payload.get("provider")
            or payload.get("backend")
            or "backend_runtime"
        )
        normalized = normalize_risk_metadata(
            payload,
            source="backend_runtime",
            provider=provider,
        )
        row = dict(payload)
        row.update(normalized)
        row["instance_id"] = instance_id
        row["node_id"] = node_id
        return row

    async def _collect_backend_runtime_metadata(self) -> List[Dict[str, Any]]:
        if not self.enable_backend_runtime_metadata:
            return []

        async with self.metadata_lock:
            model_instances = copy.deepcopy(self.model_instance)

        metadata_rows: List[Dict[str, Any]] = []
        for instances in model_instances.values():
            for instance_id, node_id in instances.items():
                try:
                    actor = self._get_backend_actor(str(instance_id))
                    metadata = await self._call_actor_method(
                        actor,
                        "get_runtime_metadata",
                        instance_id=str(instance_id),
                        node_id=str(node_id),
                    )
                except Exception as e:
                    logger.info(
                        f"Could not collect runtime metadata from "
                        f"{instance_id}: {e}"
                    )
                    continue
                if not isinstance(metadata, Mapping):
                    continue
                metadata_rows.append(
                    self._normalize_backend_runtime_metadata(
                        metadata,
                        instance_id=str(instance_id),
                        node_id=str(node_id),
                    )
                )
        return metadata_rows

    @staticmethod
    def _positive_int_value(payload: Mapping[str, Any], key: str) -> Optional[int]:
        try:
            value = int(payload.get(key, 0) or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _merge_backend_runtime_metadata(
        self,
        worker_nodes: Mapping,
        runtime_metadata_rows: List[Mapping[str, Any]],
    ) -> dict:
        updated_worker_nodes = copy.deepcopy(worker_nodes)
        rows_by_node: Dict[str, List[Mapping[str, Any]]] = {}
        for row in runtime_metadata_rows:
            node_id = row.get("node_id")
            if node_id is None:
                continue
            rows_by_node.setdefault(str(node_id), []).append(row)

        for node_id, rows in rows_by_node.items():
            if node_id not in updated_worker_nodes:
                continue
            node_info = updated_worker_nodes[node_id]
            node_info["backend_runtime_metadata"] = [
                dict(row) for row in rows
            ]

            resource_profiles = [
                row.get("model_resource_profile")
                for row in rows
                if isinstance(row.get("model_resource_profile"), Mapping)
            ]
            if resource_profiles:
                node_info["model_resource_profiles"] = [
                    dict(profile) for profile in resource_profiles
                ]

            spot_risks = []
            lifetimes = []
            loading_costs = []
            free_gpu_values = []
            total_gpu_values = []
            metadata_sources = []
            metadata_providers = []
            confidence_values = []
            for row in rows:
                try:
                    if row.get("spot_risk") is not None:
                        spot_risks.append(float(row["spot_risk"]))
                    if row.get("remaining_lifetime_s") is not None:
                        lifetimes.append(float(row["remaining_lifetime_s"]))
                    if row.get("loading_cost") is not None:
                        loading_costs.append(float(row["loading_cost"]))
                    if row.get("free_gpu") is not None:
                        free_gpu = int(row["free_gpu"])
                        if free_gpu > 0:
                            free_gpu_values.append(free_gpu)
                    if row.get("total_gpu") is not None:
                        total_gpu = int(row["total_gpu"])
                        if total_gpu > 0:
                            total_gpu_values.append(total_gpu)
                    if row.get("risk_confidence") is not None:
                        confidence_values.append(float(row["risk_confidence"]))
                except (TypeError, ValueError):
                    continue
                if row.get("risk_metadata_source"):
                    metadata_sources.append(str(row["risk_metadata_source"]))
                if row.get("risk_provider"):
                    metadata_providers.append(str(row["risk_provider"]))

            if spot_risks:
                node_info["spot_risk"] = max(spot_risks)
            if lifetimes:
                node_info["remaining_lifetime_s"] = min(lifetimes)
            if loading_costs:
                node_info["loading_cost"] = max(loading_costs)
            if free_gpu_values:
                node_info["backend_reported_free_gpu"] = max(free_gpu_values)
            if total_gpu_values:
                node_info["backend_reported_total_gpu"] = max(total_gpu_values)
                if self._positive_int_value(node_info, "total_gpu") is None:
                    node_info["total_gpu"] = max(total_gpu_values)
            if metadata_sources:
                node_info["risk_metadata_source"] = ",".join(
                    sorted(set(metadata_sources))
                )
            if metadata_providers:
                node_info["risk_provider"] = ",".join(
                    sorted(set(metadata_providers))
                )
            if confidence_values:
                node_info["risk_confidence"] = max(confidence_values)

            node_info.update(self._configured_node_risk(str(node_id)))

        return updated_worker_nodes

    async def _mark_node_state(self, node_id: str, state: NodeState):
        async with self.metadata_lock:
            if node_id not in self.worker_nodes:
                self.worker_nodes[node_id] = {
                    "ray_node_id": None,
                    "address": None,
                    "free_gpu": 0,
                    "total_gpu": 0,
                }
            from_state = self.worker_nodes[node_id].get(
                "state", NodeState.READY.value
            )
            self.worker_nodes[node_id]["state"] = state.value
            logger.info(
                f"Marked scheduler node {node_id} "
                f"{from_state} -> {state.value}"
            )
            return {
                "node_id": node_id,
                "from": from_state,
                "to": state.value,
            }

    async def start(self) -> None:
        async with self.running_lock:
            if self.running:
                logger.error("FCFS scheduler already started")
                return
            self.running = True
        logger.info("Starting FCFS scheduler")
        self.loop_task = self.loop.create_task(self._control_loop())

    async def shutdown(self) -> None:
        async with self.running_lock:
            if not self.running:
                logger.error("FCFS scheduler not running")
                return
            self.running = False
        async with self.queue_lock:
            self.model_loading_queues = {}
        if self.loop_task is not None:
            await self.loop_task

    async def allocate_resource(
        self,
        model_name: str,
        instance_id: str,
        resources: Mapping,
        target_node_id: Optional[str] = None,
    ) -> int:
        logger.info(f"Model {model_name} requested")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (
                    time.time(),
                    instance_id,
                    num_gpus,
                    allocation_result,
                    target_node_id,
                )
            )
        logger.info(
            f"Model {model_name} added to the loading queue"
            + (
                f" for target node {target_node_id}"
                if target_node_id is not None
                else ""
            )
        )
        node_id = await allocation_result
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id
            if model_name not in self.model_instance_resources:
                self.model_instance_resources[model_name] = {}
            self.model_instance_resources[model_name][instance_id] = {
                "node_id": node_id,
                "num_gpus": num_gpus,
            }
        return node_id

    async def allocate_distributed_resource(
        self,
        model_name: str,
        instance_id: str,
        resources: Mapping,
        target_node_ids: List[str],
        require_all_target_nodes: bool = True,
    ) -> Dict[str, Any]:
        """Reserve one logical model instance across several Ray workers."""
        try:
            requested_gpus = int(resources.get("num_gpus", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_gpus must be an integer") from exc

        await self._get_worker_nodes()
        async with self.metadata_lock:
            allocations = self._distributed_gpu_allocation(
                self.worker_nodes,
                target_node_ids,
                requested_gpus,
                require_all_target_nodes,
            )
            for node_id, gpu_count in allocations.items():
                self.worker_nodes[node_id]["free_gpu"] = max(
                    0.0,
                    float(self.worker_nodes[node_id].get("free_gpu", 0) or 0)
                    - gpu_count,
                )
            primary_node_id = next(iter(allocations))
            self.model_instance.setdefault(model_name, {})[instance_id] = (
                primary_node_id
            )
            self.model_instance_resources.setdefault(model_name, {})[
                instance_id
            ] = {
                "node_id": primary_node_id,
                "num_gpus": requested_gpus,
                "node_allocations": dict(allocations),
                "distributed": True,
            }
        logger.info(
            "Distributed allocation for model %s instance %s: %s",
            model_name,
            instance_id,
            allocations,
        )
        return {
            "model_name": model_name,
            "instance_id": instance_id,
            "node_id": primary_node_id,
            "target_node_ids": list(allocations),
            "node_allocations": dict(allocations),
            "num_gpus": requested_gpus,
        }

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        logger.info(f"Deallocating model {model_name} instance {instance_id}")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error(f"Model {model_name} not found")
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error(f"Instance {instance_id} not found")
                return
            node_id = self.model_instance[model_name].pop(instance_id)
            resource_entry = self.model_instance_resources.get(
                model_name, {}
            ).pop(instance_id, {})
            if not self.model_instance[model_name]:
                self.model_instance.pop(model_name, None)
            if not self.model_instance_resources.get(model_name):
                self.model_instance_resources.pop(model_name, None)
            try:
                num_gpus = float(resource_entry.get("num_gpus", num_gpus))
            except (TypeError, ValueError):
                pass
            node_allocations = resource_entry.get("node_allocations")
            if isinstance(node_allocations, Mapping) and node_allocations:
                for allocated_node_id, allocated_gpus in node_allocations.items():
                    node = self.worker_nodes.get(str(allocated_node_id))
                    if node is None:
                        logger.error("Node %s not found", allocated_node_id)
                        continue
                    node["free_gpu"] = min(
                        float(node.get("total_gpu", 0) or 0),
                        float(node.get("free_gpu", 0) or 0)
                        + float(allocated_gpus or 0),
                    )
                logger.info(
                    "Distributed nodes %s deallocated for %s/%s",
                    node_allocations,
                    model_name,
                    instance_id,
                )
                return
            logger.info(f"Node {node_id} deallocated {num_gpus} GPUs")
            if node_id not in self.worker_nodes:
                logger.error(f"Node {node_id} not found")
                return
            self.worker_nodes[node_id]["free_gpu"] += num_gpus
        logger.info(f"Model {model_name} instance {instance_id} deallocated")

    async def resize_resource(
        self,
        model_name: str,
        instance_id: str,
        resources: Mapping,
    ):
        """Atomically resize a model allocation on its current worker node."""
        try:
            requested_gpus = float(resources.get("num_gpus", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_gpus must be numeric") from exc
        if requested_gpus < 0:
            raise ValueError("num_gpus must be non-negative")

        async with self.metadata_lock:
            allocation = self.model_instance_resources.get(
                model_name, {}
            ).get(instance_id)
            if allocation is None:
                raise ValueError(
                    f"Allocation {model_name}/{instance_id} not found"
                )
            node_id = allocation["node_id"]
            node = self.worker_nodes.get(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found")
            current_gpus = float(allocation.get("num_gpus", 0) or 0)
            delta = requested_gpus - current_gpus
            available_gpus = float(node.get("free_gpu", 0) or 0)
            if delta > available_gpus:
                raise RuntimeError(
                    "insufficient_capacity_for_resource_resize: "
                    f"requested_delta={delta}, available={available_gpus}, "
                    f"node={node_id}"
                )
            node["free_gpu"] = available_gpus - delta
            allocation["num_gpus"] = requested_gpus
            logger.info(
                "Resized model %s instance %s on node %s from %s to %s GPUs",
                model_name,
                instance_id,
                node_id,
                current_gpus,
                requested_gpus,
            )
            return {
                "model_name": model_name,
                "instance_id": instance_id,
                "node_id": str(node_id),
                "previous_num_gpus": current_gpus,
                "num_gpus": requested_gpus,
                "delta_num_gpus": delta,
                "free_gpu": node["free_gpu"],
            }

    async def clear_model(self, model_name: str):
        logger.info(f"Clearing scheduler state for model {model_name}")
        cleared_pending = 0
        released_allocations = 0
        async with self.queue_lock:
            pending = self.model_loading_queues.pop(model_name, [])
            for entry in pending:
                try:
                    allocation_result = entry[3]
                except IndexError:
                    continue
                if allocation_result.done():
                    continue
                allocation_result.set_exception(
                    RuntimeError(
                        f"Allocation cancelled for deleted model {model_name}"
                    )
                )
                cleared_pending += 1

        async with self.metadata_lock:
            instances = self.model_instance.pop(model_name, {})
            resource_entries = self.model_instance_resources.pop(
                model_name, {}
            )
            for instance_id, node_id in instances.items():
                resource_entry = resource_entries.get(instance_id, {})
                node_allocations = resource_entry.get("node_allocations")
                if isinstance(node_allocations, Mapping) and node_allocations:
                    for allocated_node_id, allocated_gpus in (
                        node_allocations.items()
                    ):
                        node = self.worker_nodes.get(str(allocated_node_id))
                        if node is None:
                            continue
                        node["free_gpu"] = min(
                            float(node.get("total_gpu", 0) or 0),
                            float(node.get("free_gpu", 0) or 0)
                            + float(allocated_gpus or 0),
                        )
                    released_allocations += 1
                    continue
                try:
                    num_gpus = float(resource_entry.get("num_gpus", 1.0))
                except (TypeError, ValueError):
                    num_gpus = 1.0
                if node_id in self.worker_nodes:
                    self.worker_nodes[node_id]["free_gpu"] = min(
                        float(
                            self.worker_nodes[node_id].get("total_gpu", 0)
                            or 0
                        ),
                        float(
                            self.worker_nodes[node_id].get("free_gpu", 0)
                            or 0
                        )
                        + num_gpus,
                    )
                released_allocations += 1
        return {
            "model_name": model_name,
            "cleared_pending": cleared_pending,
            "released_allocations": released_allocations,
        }

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, (
                        request_time,
                        instance_id,
                        num_gpus,
                        allocation_result,
                        target_node_id,
                    ) in enumerate(loading_queue):
                        loading_requests.append(
                            (
                                model_name,
                                idx,
                                request_time,
                                instance_id,
                                num_gpus,
                                allocation_result,
                                target_node_id,
                            )
                        )
            # logger.info(f"Loading requests: {loading_requests}")
            # first come first serve
            if len(loading_requests) > 0:
                worker_nodes = await self._get_worker_nodes()
                logger.info(f"Worker nodes: {worker_nodes}")
                loading_requests.sort(key=lambda x: x[1])
                for (
                    model_name,
                    idx,
                    request_time,
                    instance_id,
                    num_gpus,
                    allocation_result,
                    target_node_id,
                ) in loading_requests:
                    allocated = False
                    for node_id, node_info in self._ordered_candidate_nodes(
                        model_name,
                        num_gpus,
                        worker_nodes,
                    ):
                        if (
                            target_node_id is not None
                            and str(node_id) != str(target_node_id)
                        ):
                            continue
                        if not self._node_is_ready(node_info):
                            logger.info(
                                f"Skipping node {node_id} in state "
                                f"{node_info.get('state')}"
                            )
                            continue
                        if node_info["free_gpu"] >= num_gpus:
                            async with self.queue_lock:
                                # allocation_result was set
                                if allocation_result.done():
                                    allocated = True
                                    # skip current instance
                                    break
                                try:
                                    self.model_loading_queues[
                                        model_name
                                    ].remove(
                                        (
                                            request_time,
                                            instance_id,
                                            num_gpus,
                                            allocation_result,
                                            target_node_id,
                                        )
                                    )
                                    allocation_result.set_result(node_id)
                                except ValueError:
                                    break
                            allocated = True
                            logger.info(
                                f"Allocated node {node_id} for model {model_name}"
                            )
                            node_info["free_gpu"] -= num_gpus
                            break
                    if not allocated:
                        logger.info(f"No available node for model {model_name}")
                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(1)

    async def _get_worker_nodes(self):
        worker_nodes = get_worker_nodes()
        try:
            available_resources = ray.available_resources()
        except Exception:
            available_resources = {}
        provider_metadata = await self._collect_provider_risk_metadata(
            worker_nodes
        )
        for node_id, metadata in provider_metadata.items():
            if node_id in worker_nodes:
                worker_nodes[node_id].update(metadata)
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
            allocated_gpu_by_node = self._allocated_gpu_by_node_unlocked()
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                updated_worker_nodes[node_id] = self._ensure_node_metadata(
                    copy.deepcopy(node_info),
                    node_id=node_id,
                )
                updated_worker_nodes[node_id] = (
                    self._apply_scheduler_capacity(
                        updated_worker_nodes[node_id],
                        allocated_gpu_by_node.get(str(node_id), 0.0),
                    )
                )
            else:
                current_state = updated_worker_nodes[node_id].get(
                    "state", NodeState.READY.value
                )
                updated_worker_nodes[node_id].update(copy.deepcopy(node_info))
                updated_worker_nodes[node_id] = (
                    self._apply_scheduler_capacity(
                        updated_worker_nodes[node_id],
                        allocated_gpu_by_node.get(str(node_id), 0.0),
                    )
                )
                updated_worker_nodes[node_id]["state"] = current_state
                updated_worker_nodes[node_id] = self._preserve_spot_metadata(
                    updated_worker_nodes[node_id],
                    self.worker_nodes[node_id],
                    node_id,
                )
            updated_worker_nodes[node_id] = self._apply_ray_available_capacity(
                updated_worker_nodes[node_id],
                str(node_id),
                available_resources,
            )

        for node_id, node_info in updated_worker_nodes.items():
            if node_id in worker_nodes:
                continue
            updated_worker_nodes[node_id] = self._apply_scheduler_capacity(
                node_info,
                allocated_gpu_by_node.get(str(node_id), 0.0),
            )

        runtime_metadata_rows = await self._collect_backend_runtime_metadata()
        updated_worker_nodes = self._merge_backend_runtime_metadata(
            updated_worker_nodes,
            runtime_metadata_rows,
        )
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes

        return copy.deepcopy(updated_worker_nodes)

    # TODO: implement a dedicated class to manage worker nodes
    async def _update_worker_nodes(self, worker_nodes) -> None:
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                logger.error(f"Node {node_id} not found")
                continue
            current_state = updated_worker_nodes[node_id].get(
                "state", NodeState.READY.value
            )
            updated_worker_nodes[node_id] = self._ensure_node_metadata(
                copy.deepcopy(node_info),
                node_id=node_id,
            )
            updated_worker_nodes[node_id]["state"] = current_state
            updated_worker_nodes[node_id] = self._preserve_spot_metadata(
                updated_worker_nodes[node_id],
                self.worker_nodes[node_id],
                node_id,
            )
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes
        logger.info(f"Worker nodes updated: {updated_worker_nodes}")
