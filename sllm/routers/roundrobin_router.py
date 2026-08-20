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
import logging
import os
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.spot.metrics import (
    JsonlMetricsWriter,
    make_context_migration_event,
    make_instance_state_event,
    make_replanning_event,
    make_request_event,
    make_state_recovery_event,
)
from sllm.spot.context_migration import (
    ContextMetadata,
    MigrationTarget,
    plan_low_cost_migration,
)
from sllm.spot.recovery_policy import RecoveryPolicy, normalize_policy
from sllm.spot.reparallelization import (
    READY,
    ParallelPlan,
    apply_spot_event_to_worker_nodes,
    plan_dynamic_reparallelization,
)
from sllm.spot.reparallelization_executor import ReparallelizationExecutor
from sllm.spot.vllm_deployment_adapter import (
    VllmDeployment,
    VllmDeploymentAdapter,
)
from sllm.spot.stateful_recovery import (
    InferenceState,
    plan_compatible_state_target,
    plan_stateful_recovery,
)

from ..utils import InstanceHandle, InstanceState
from .router_utils import SllmRouter

logger = init_logger(__name__)


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.router_config = router_config
        self.recovery_policy = normalize_policy(
            router_config.get("recovery_policy", RecoveryPolicy.NONE.value)
        )
        self.max_retries = int(router_config.get("max_retries", 0))
        self.count_preempting_toward_capacity = bool(
            router_config.get("count_preempting_toward_capacity", False)
        )
        self.enable_reparallelization = bool(
            router_config.get("enable_reparallelization", False)
        )
        self.reparallelization_config = dict(
            router_config.get("reparallelization_config", {})
        )
        self.enable_context_migration = bool(
            router_config.get("enable_context_migration", False)
        )
        self.context_migration_config = dict(
            router_config.get("context_migration_config", {})
        )
        self.enable_kv_cache_migration = bool(
            router_config.get(
                "enable_kv_cache_migration",
                self.context_migration_config.get(
                    "enable_kv_cache_migration",
                    self.context_migration_config.get(
                        "execute_kv_cache_migration", False
                    ),
                ),
            )
        )
        # Stateful recovery first asks the planner for an already-ready,
        # cache-compatible target. It never changes TP/PP/EP or creates an
        # engine; those actions remain the explicit re-parallelization path.
        self.enable_stateful_target_planner = bool(
            router_config.get("enable_stateful_target_planner", True)
        )
        synthetic_worker_nodes = self.reparallelization_config.get(
            "synthetic_worker_nodes", {}
        )
        self.reparallelization_worker_nodes = {
            str(node_id): dict(node_info)
            for node_id, node_info in synthetic_worker_nodes.items()
        }
        self.vllm_deployment_adapter: Optional[VllmDeploymentAdapter] = None
        self.reparallelization_executor: Optional[
            ReparallelizationExecutor
        ] = None
        metrics_path = router_config.get("metrics_path") or os.getenv(
            "SLLM_SPOT_METRICS_PATH"
        )
        self.metrics_writer = (
            JsonlMetricsWriter(metrics_path) if metrics_path else None
        )

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.fine_tuning_count = 0
        self.fine_tuning_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.enable_lora = enable_lora
        self.loaded_lora_adapters = lora_adapters
        self.lora_lock = asyncio.Lock()

        self.recovery_tokens_by_instance: Dict[str, List[List[int]]] = {}
        self.recovery_state_by_instance: Dict[str, Dict[str, Any]] = {}
        # Requests currently executing on a backend.  V6 uses this registry
        # to snapshot and abort an in-flight request before switching the
        # active deployment; the original external request_id is retained for
        # the retry on the new worker.
        self.inflight_requests: Dict[str, Dict[str, Any]] = {}
        self.inflight_requests_lock = asyncio.Lock()
        self.auto_scaler = None
        logger.info(f"Created new handler for model {self.model_name}")

    async def _call_backend_method(
        self, backend_instance: Any, method_name: str, **kwargs
    ):
        method = getattr(backend_instance, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await remote(**kwargs)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _stop_backend(self, backend_instance: Any, method_name: str):
        await self._call_backend_method(backend_instance, method_name)
        if hasattr(backend_instance, "_ray_actor_id"):
            ray.kill(backend_instance)

    async def _track_inflight_request(
        self,
        request_id: str,
        request_data: dict,
        action: str,
        instance: InstanceHandle,
    ):
        async with self.inflight_requests_lock:
            entry = self.inflight_requests.get(request_id)
            if entry is None:
                entry = {
                    "request_id": request_id,
                    "request_data": copy.deepcopy(request_data),
                    "action": action,
                    "instance": instance,
                    "instance_id": instance.instance_id,
                    "migration_state": None,
                    "migration_requested": False,
                }
                self.inflight_requests[request_id] = entry
            else:
                entry["instance"] = instance
                entry["instance_id"] = instance.instance_id
        return entry

    async def _untrack_inflight_request(self, request_id: str):
        async with self.inflight_requests_lock:
            self.inflight_requests.pop(request_id, None)

    async def _prepare_reparallelization_requests(
        self, deployment: VllmDeployment
    ) -> Dict[str, Any]:
        """Snapshot and interrupt requests before V6 switches deployments.

        The backend owns the actual engine abort.  Once the abort response is
        returned, the request coroutine receives a migration marker and
        retries on the new ready deployment using the captured V8 state (or a
        token replay fallback).  Requests for which the backend has no abort
        hook are left running so the normal drain path remains safe.
        """
        old_instance_ids = set(deployment.instances)
        async with self.inflight_requests_lock:
            entries = [
                entry
                for entry in self.inflight_requests.values()
                if entry.get("instance_id") in old_instance_ids
            ]

        summary = {
            "attempted": len(entries),
            "state_exported": 0,
            "abort_requested": 0,
            "migratable": 0,
            "unsupported": 0,
            "request_ids": sorted(
                str(entry.get("request_id")) for entry in entries
            ),
        }
        for entry in entries:
            instance = entry.get("instance")
            if instance is None or instance.backend_instance is None:
                summary["unsupported"] += 1
                continue

            # Export before abort so the connector still sees the live request
            # and its block table/lease metadata.
            logger.info(
                "Exporting live inference state for %s before re-parallelization",
                entry["request_id"],
            )
            try:
                # A connector utility must not stall the whole deployment
                # switch.  If a runtime cannot answer while its output loop
                # is coalescing, fall back to the token snapshot path and
                # still issue the controlled abort.
                state = await asyncio.wait_for(
                    self._capture_inference_state(
                        instance,
                        request_data=entry.get("request_data", {}),
                    ),
                    timeout=float(
                        self.reparallelization_config.get(
                            "state_export_timeout_s", 30.0
                        )
                    ),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out exporting live state for %s; using token fallback",
                    entry["request_id"],
                )
                state = await self._capture_inference_state(
                    instance,
                    request_data=entry.get("request_data", {}),
                    current_output=await self._capture_current_tokens(instance),
                )
            entry["migration_state"] = state
            if state is not None:
                summary["state_exported"] += 1

            try:
                abort_result = await self._call_backend_method(
                    instance.backend_instance,
                    "abort_request",
                    request_id=entry["request_id"],
                    reason="reparallelization",
                )
            except (AttributeError, NotImplementedError):
                summary["unsupported"] += 1
                continue
            except Exception:
                logger.exception(
                    "Could not abort request %s for re-parallelization",
                    entry["request_id"],
                )
                summary["unsupported"] += 1
                continue

            if not isinstance(abort_result, dict) or not abort_result.get(
                "aborted", False
            ):
                summary["unsupported"] += 1
                continue
            logger.info(
                "Aborted live request %s for re-parallelization",
                entry["request_id"],
            )
            entry["migration_requested"] = True
            summary["abort_requested"] += 1
            summary["migratable"] += 1
        return summary

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config
            self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
            self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    def _ensure_vllm_reparallelization_adapter(self) -> bool:
        """Wire V6 planning to real vLLM actors when enabled.

        Dummy and transformer routers continue to use the planner as a
        decision-only path.  vLLM gets the concrete Ray actor deployment
        adapter, so a selected ``ParallelPlan`` is executable rather than only
        a metric artifact.
        """
        if self.backend != "vllm" or not self.enable_reparallelization:
            return False
        if self.vllm_deployment_adapter is not None:
            return True
        scheduler = getattr(self, "model_loading_scheduler", None)
        if scheduler is None:
            logger.warning(
                "Cannot enable vLLM re-parallelization before scheduler start"
            )
            return False
        self.vllm_deployment_adapter = VllmDeploymentAdapter(
            model_name=self.model_name,
            backend_config=self.backend_config,
            resource_requirements=self.resource_requirements,
            scheduler=scheduler,
            traffic_switcher=self._switch_vllm_deployment,
            request_migrator=self._prepare_reparallelization_requests,
            max_queue_length=max(
                1, int(self.auto_scaling_config.get("target", 1) or 1)
            ),
            drain_timeout_s=float(
                self.reparallelization_config.get("drain_timeout_s", 30.0)
                or 30.0
            ),
            migration_completion_waiter=self._wait_for_reparallelization_requests,
        )
        self.reparallelization_executor = ReparallelizationExecutor(
            self.vllm_deployment_adapter.create_workers,
            self.vllm_deployment_adapter.ready_workers,
            self.vllm_deployment_adapter.switch_workers,
            self.vllm_deployment_adapter.drain_workers,
            self.vllm_deployment_adapter.stop_workers,
            stop_current_before_create=bool(
                self.reparallelization_config.get(
                    "allow_stop_before_recreate", False
                )
            ),
            wait_for_migration=self.vllm_deployment_adapter.wait_for_migration_completion,
            migrate_before_create=bool(
                self.reparallelization_config.get(
                    "migrate_before_create", False
                )
            ),
        )
        return True

    async def _wait_for_reparallelization_requests(
        self, deployment: VllmDeployment, timeout_s: float
    ) -> None:
        """Wait for migrated request retries before closing source NIXL."""
        adapter = self.vllm_deployment_adapter
        summary = getattr(adapter, "last_request_migration", None) or {}
        request_ids = {str(item) for item in summary.get("request_ids", [])}
        if not request_ids:
            return
        deadline = asyncio.get_running_loop().time() + max(0.1, timeout_s)
        while asyncio.get_running_loop().time() < deadline:
            async with self.inflight_requests_lock:
                active = request_ids.intersection(self.inflight_requests)
            if not active:
                return
            await asyncio.sleep(0.1)
        logger.warning(
            "Timed out waiting for %d migrated requests before source stop",
            len(active),
        )

    async def _switch_vllm_deployment(
        self, deployment: VllmDeployment, plan: ParallelPlan
    ):
        """Atomically make the newly initialised actors serve traffic."""
        if deployment.plan != plan:
            raise ValueError("vLLM deployment plan mismatch during switch")
        async with self.instance_management_lock:
            self.ready_inference_instances = dict(deployment.instances)
            self.backend_config = dict(deployment.backend_config)
            self.resource_requirements = dict(deployment.resource_requirements)
            self.vllm_deployment_adapter.backend_config = dict(
                deployment.backend_config
            )
            self.vllm_deployment_adapter.resource_requirements = dict(
                deployment.resource_requirements
            )
        return {
            "status": "switched",
            "instance_ids": sorted(deployment.instances),
            "parallel_plan": plan.to_dict(),
        }

    def _vllm_active_deployment(self) -> Optional[VllmDeployment]:
        if self.vllm_deployment_adapter is None:
            return None
        return self.vllm_deployment_adapter.snapshot(
            self.ready_inference_instances
        )

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        lora_adapters: Optional[Dict[str, str]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        if lora_adapters is not None:
            async with self.lora_lock:
                self.loaded_lora_adapters = lora_adapters

        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    async def get_instance_states(self):
        async with self.instance_management_lock:
            pools = {
                "starting": self.starting_inference_instances,
                "ready": self.ready_inference_instances,
                "deleting": self.deleting_inference_instances,
            }
            states = {}
            for pool_name, pool in pools.items():
                for instance_id, instance in pool.items():
                    status = await instance.get_status()
                    states[instance_id] = {
                        "pool": pool_name,
                        "node_id": status.node_id,
                        "state": status.state,
                        "concurrency": status.concurrency,
                    }
            return states

    def _matches_spot_target(
        self,
        instance: InstanceHandle,
        node_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> bool:
        if node_id is None and instance_id is None:
            return False
        if instance_id is not None and instance.instance_id != instance_id:
            return False
        if node_id is not None and instance.node_id != node_id:
            return False
        return True

    async def _matching_inference_instances(
        self,
        node_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        async with self.instance_management_lock:
            pools = [
                self.starting_inference_instances,
                self.ready_inference_instances,
                self.deleting_inference_instances,
            ]
            matches = []
            for pool in pools:
                for instance in pool.values():
                    if self._matches_spot_target(
                        instance, node_id=node_id, instance_id=instance_id
                    ):
                        matches.append(instance)
            return matches

    def _emit_metric(self, event: Dict):
        if self.metrics_writer is None:
            return
        try:
            self.metrics_writer.emit(event)
        except Exception as e:
            logger.error(f"Failed to emit metric: {e}")

    def _spot_target_node_ids(
        self,
        node_id: Optional[str],
        matches: List[InstanceHandle],
    ) -> List[str]:
        if node_id is not None:
            return [str(node_id)]

        target_node_ids = []
        seen = set()
        for instance in matches:
            if instance.node_id is None:
                continue
            target_node_id = str(instance.node_id)
            if target_node_id in seen:
                continue
            seen.add(target_node_id)
            target_node_ids.append(target_node_id)
        return target_node_ids

    async def _snapshot_reparallelization_worker_nodes(self):
        if self.reparallelization_worker_nodes:
            return {
                node_id: dict(node_info)
                for node_id, node_info in (
                    self.reparallelization_worker_nodes.items()
                )
            }

        # The router's active instances only describe nodes already serving
        # this model.  Ask the real scheduler for the complete worker set so a
        # replan can place the replacement on an idle vLLM worker node.
        scheduler = getattr(self, "model_loading_scheduler", None)
        scheduler_snapshot = getattr(scheduler, "_get_worker_nodes", None)
        if scheduler_snapshot is not None:
            try:
                remote = getattr(scheduler_snapshot, "remote", None)
                worker_nodes = (
                    await remote()
                    if remote is not None
                    else await scheduler_snapshot()
                )
                if isinstance(worker_nodes, dict) and worker_nodes:
                    return {
                        str(node_id): dict(node_info)
                        for node_id, node_info in worker_nodes.items()
                    }
            except Exception:
                logger.info(
                    "Could not query scheduler worker nodes for replan",
                    exc_info=True,
                )

        async with self.instance_management_lock:
            instances = (
                list(self.starting_inference_instances.values())
                + list(self.ready_inference_instances.values())
                + list(self.deleting_inference_instances.values())
            )

        worker_nodes = {}
        for instance in instances:
            if instance.node_id is None:
                continue
            node_id = str(instance.node_id)
            node_info = worker_nodes.setdefault(
                node_id,
                {
                    "ray_node_id": None,
                    "address": node_id,
                    "free_gpu": 0,
                    "total_gpu": 0,
                    "state": READY,
                },
            )
            node_info["total_gpu"] += int(instance.num_gpu or 0)
            if instance.state == InstanceState.PREEMPTING:
                node_info["state"] = InstanceState.PREEMPTING.value
            elif instance.state == InstanceState.DEAD:
                node_info["state"] = InstanceState.DEAD.value
            elif node_info["state"] == READY:
                node_info["state"] = instance.state.value
        return worker_nodes

    async def _replan_after_spot_event(
        self,
        event: str,
        node_id: Optional[str],
        instance_id: Optional[str],
        matches: List[InstanceHandle],
        worker_node_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        if not self.enable_reparallelization:
            return None

        worker_nodes = await self._snapshot_reparallelization_worker_nodes()
        normalized_updates = {
            str(update_node_id): dict(update)
            for update_node_id, update in (worker_node_updates or {}).items()
        }
        for update_node_id, update in normalized_updates.items():
            current = dict(worker_nodes.get(update_node_id, {}))
            current.update(dict(update))
            worker_nodes[update_node_id] = current
        target_node_ids = self._spot_target_node_ids(node_id, matches)
        for target_node_id in target_node_ids:
            worker_nodes = apply_spot_event_to_worker_nodes(
                worker_nodes,
                event,
                target_node_id,
                node_info=normalized_updates.get(target_node_id),
            )
        if event == "add" and node_id is not None and not target_node_ids:
            worker_nodes = apply_spot_event_to_worker_nodes(
                worker_nodes,
                event,
                node_id,
                node_info=normalized_updates.get(node_id),
            )
        self.reparallelization_worker_nodes = worker_nodes

        model_config = {
            "model": self.model_name,
            "backend": self.backend,
            "num_gpus": self.resource_requirements.get("num_gpus", 1),
            # Capability negotiation needs the concrete model/runtime
            # configuration.  Without this field the planner falls back to
            # the runtime-neutral candidate generator, which intentionally
            # uses runtime DP=1 and EP disabled, so it cannot select an
            # expert-parallel shape for a MoE model.
            "backend_config": dict(self.backend_config),
        }
        decision = plan_dynamic_reparallelization(
            model_name=self.model_name,
            worker_nodes=worker_nodes,
            model_config=model_config,
            planner_config=self.reparallelization_config,
            event=event,
            node_id=node_id,
            instance_id=instance_id,
            backend=self.backend,
        )
        selected_plan = decision.get("parallel_plan")
        active_deployment = self._vllm_active_deployment()
        if (
            decision.get("action") == "reparallelize"
            and selected_plan is not None
            and active_deployment is not None
            and self._parallel_plans_match(
                active_deployment.plan,
                ParallelPlan.from_dict(selected_plan),
            )
        ):
            # A capacity event does not automatically imply a deployment
            # change.  Keep the current engine and in-flight requests alive
            # when the newly selected plan is identical.
            decision["action"] = "unchanged"
            decision["execution"] = {
                "status": "unchanged",
                "parallel_plan": selected_plan,
                "reason": "planner_selected_existing_plan",
            }
        if decision.get("action") == "reparallelize":
            if self._ensure_vllm_reparallelization_adapter():
                try:
                    plan = ParallelPlan.from_dict(decision["parallel_plan"])
                    self.reparallelization_executor.current = (
                        self._vllm_active_deployment()
                    )
                    deployment = await self.reparallelization_executor.apply(
                        plan
                    )
                    decision["execution"] = {
                        "status": "applied",
                        "instance_ids": sorted(deployment.instances),
                        "parallel_plan": plan.to_dict(),
                        "request_migration": getattr(
                            self.vllm_deployment_adapter,
                            "last_request_migration",
                            None,
                        ),
                    }
                except Exception as exc:
                    logger.exception(
                        "Failed applying vLLM re-parallelization plan"
                    )
                    decision["execution"] = {
                        "status": "failed",
                        "reason": str(exc),
                    }
            else:
                decision["execution"] = {
                    "status": "decision_only",
                    "reason": "vllm_deployment_adapter_unavailable",
                }
        self._emit_metric(
            make_replanning_event(
                model=self.model_name,
                event=event,
                node_id=node_id,
                instance_id=instance_id,
                decision=decision,
            )
        )
        logger.info(
            f"Reparallelization decision for {self.model_name}: {decision}"
        )
        return decision

    @staticmethod
    def _parallel_plans_match(
        left: ParallelPlan, right: ParallelPlan
    ) -> bool:
        """Compare deployment shape and placement, ignoring replan reason."""
        return (
            left.model_name == right.model_name
            and left.backend == right.backend
            and left.tensor_parallel_size == right.tensor_parallel_size
            and left.pipeline_parallel_size == right.pipeline_parallel_size
            and left.data_parallel_size == right.data_parallel_size
            and left.enable_expert_parallel == right.enable_expert_parallel
            and left.effective_expert_parallel_size
            == right.effective_expert_parallel_size
            and left.replica_count == right.replica_count
            and left.num_gpus == right.num_gpus
            and sorted(left.target_nodes) == sorted(right.target_nodes)
        )

    async def _set_instance_state(
        self, instance: InstanceHandle, state: InstanceState, reason: str
    ):
        from_state = instance.state.value
        if state == InstanceState.PREEMPTING:
            await instance.mark_preempting()
        elif state == InstanceState.DRAINING:
            await instance.mark_draining()
        elif state == InstanceState.DEAD:
            await instance.mark_dead()
        elif state == InstanceState.READY:
            await instance.mark_ready()
        else:
            async with instance.lock:
                instance.state = state

        self._emit_metric(
            make_instance_state_event(
                model=self.model_name,
                instance_id=instance.instance_id,
                node_id=instance.node_id,
                from_state=from_state,
                to_state=instance.state.value,
                reason=reason,
            )
        )

    async def _capture_current_tokens(
        self, instance: InstanceHandle
    ) -> List[List[int]]:
        if instance.backend_instance is None:
            return []
        try:
            tokens = await self._call_backend_method(
                instance.backend_instance, "get_current_tokens"
            )
        except Exception as e:
            logger.info(
                f"Could not capture current tokens from "
                f"{instance.instance_id}: {e}"
            )
            return []
        if not tokens:
            return []
        self.recovery_tokens_by_instance[instance.instance_id] = tokens
        return tokens

    async def _capture_inference_state(
        self,
        instance: InstanceHandle,
        request_data: Optional[dict] = None,
        current_output: Optional[List[List[int]]] = None,
        completed_tokens: Optional[int] = None,
        exported_state: Optional[dict] = None,
    ) -> Optional[InferenceState]:
        if instance.backend_instance is None and exported_state is None:
            return None

        payload = exported_state
        if payload is None:
            try:
                payload = await self._call_backend_method(
                    instance.backend_instance,
                    "export_inference_state",
                    request_data=request_data or {},
                    current_output=current_output,
                    completed_tokens=completed_tokens,
                )
            except Exception as e:
                logger.info(
                    f"Could not export inference state from "
                    f"{instance.instance_id}: {e}"
                )
                tokens = current_output or await self._capture_current_tokens(
                    instance
                )
                if not tokens:
                    return None
                payload = {
                    "request_id": (
                        request_data.get("request_id")
                        if request_data
                        else None
                    ),
                    "tokens": tokens[0],
                    "completed_tokens": completed_tokens or len(tokens[0]),
                    "state_kind": "token_snapshot",
                    "supports_restore": False,
                }

        if not isinstance(payload, dict):
            return None

        payload = dict(payload)
        if not payload.get("instance_id"):
            payload["instance_id"] = instance.instance_id
        if not payload.get("node_id"):
            payload["node_id"] = instance.node_id or ""
        payload.setdefault("backend", self.backend)
        payload.setdefault("model_name", self.model_name)

        state = InferenceState.from_dict(payload)
        if not state.tokens and current_output:
            state = InferenceState.from_tokens(
                tokens=current_output[0],
                request_id=(
                    request_data.get("request_id") if request_data else None
                ),
                instance_id=instance.instance_id,
                node_id=instance.node_id or "",
                backend=self.backend,
                model_name=self.model_name,
                completed_tokens=completed_tokens,
                state_kind="token_snapshot",
                supports_restore=False,
            )
        if not state.tokens:
            return None

        self.recovery_state_by_instance[instance.instance_id] = state.to_dict()
        self.recovery_tokens_by_instance[instance.instance_id] = [
            list(state.tokens)
        ]
        return state

    async def _capture_context_metadata(
        self,
        instance: InstanceHandle,
    ) -> List[ContextMetadata]:
        if instance.backend_instance is None:
            return []

        try:
            rows = await self._call_backend_method(
                instance.backend_instance,
                "get_context_metadata",
                instance_id=instance.instance_id,
                node_id=instance.node_id or "",
            )
        except Exception as e:
            logger.info(
                f"Could not capture context metadata from "
                f"{instance.instance_id}: {e}"
            )
            return []

        if isinstance(rows, dict):
            rows = rows.get("contexts", [])
        if not rows:
            return []

        contexts = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            payload.setdefault("instance_id", instance.instance_id)
            payload.setdefault("node_id", instance.node_id or "")
            try:
                contexts.append(ContextMetadata.from_dict(payload))
            except Exception as e:
                logger.info(
                    f"Skipping invalid context metadata from "
                    f"{instance.instance_id}: {e}"
                )
        return contexts

    async def _context_migration_sources(
        self, matches: List[InstanceHandle]
    ) -> List[ContextMetadata]:
        sources: List[ContextMetadata] = []
        for instance in matches:
            sources.extend(await self._capture_context_metadata(instance))
        return sources

    @staticmethod
    def _context_metadata_value(
        context: ContextMetadata, key: str
    ) -> Any:
        if key == "cache_block_size":
            return context.cache_block_size or None
        if key == "cache_dtype":
            return context.cache_dtype or None
        if key == "cache_layout":
            return context.cache_layout or None
        return (context.metadata or {}).get(key)

    @classmethod
    def _context_cache_compatible(
        cls,
        source: ContextMetadata,
        target: ContextMetadata,
    ) -> bool:
        if source.cache_block_size <= 0 or target.cache_block_size <= 0:
            return False
        if source.cache_block_size != target.cache_block_size:
            return False
        if source.context_blocks <= 0 or target.context_blocks <= 0:
            return False

        for key in ("cache_dtype", "cache_layout"):
            source_value = cls._context_metadata_value(source, key)
            target_value = cls._context_metadata_value(target, key)
            if source_value and target_value and source_value != target_value:
                return False

        compatibility_keys = (
            "cache_config_fingerprint",
            "model_revision",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "cache_engine",
            "kv_connector",
            "configured_cache_dtype",
        )
        for key in compatibility_keys:
            source_value = cls._context_metadata_value(source, key)
            target_value = cls._context_metadata_value(target, key)
            if source_value is None or target_value is None:
                continue
            if str(source_value) != str(target_value):
                return False

        source_groups = cls._context_metadata_value(source, "cache_groups")
        target_groups = cls._context_metadata_value(target, "cache_groups")
        if source_groups and target_groups and source_groups != target_groups:
            return False
        return True

    @classmethod
    def _target_specific_reuse(
        cls,
        source: ContextMetadata,
        target_context: ContextMetadata,
    ) -> Tuple[int, int]:
        if not source.tokens or not target_context.tokens:
            return 0, 0
        if not cls._context_cache_compatible(source, target_context):
            return 0, 0

        common_tokens = 0
        for source_token, target_token in zip(
            source.tokens, target_context.tokens
        ):
            if source_token != target_token:
                break
            common_tokens += 1
        aligned_tokens = (
            common_tokens // source.cache_block_size
        ) * source.cache_block_size
        reusable_blocks = min(
            source.context_blocks,
            target_context.context_blocks,
            aligned_tokens // source.cache_block_size,
        )
        if reusable_blocks <= 0:
            return 0, 0
        source_token_count = source.num_tokens or len(source.tokens)
        target_token_count = target_context.num_tokens or len(
            target_context.tokens
        )
        reusable_tokens = min(
            source_token_count,
            target_token_count,
            reusable_blocks * source.cache_block_size,
        )
        return reusable_tokens, reusable_blocks

    async def _populate_target_reuse_maps(
        self,
        sources: List[ContextMetadata],
        targets: List[MigrationTarget],
    ) -> List[ContextMetadata]:
        """Annotate reuse only when a target exposes the same cached prefix.

        A target-specific reuse value is evidence-based: both runtimes must
        expose tokens and compatible cache geometry, run on the same node, and
        share a non-empty token prefix aligned to the cache block size.
        """
        async with self.instance_management_lock:
            target_instances = dict(self.ready_inference_instances)

        target_contexts: Dict[str, List[ContextMetadata]] = {}
        for target in targets:
            instance = target_instances.get(target.instance_id)
            if instance is None or instance.backend_instance is None:
                continue
            try:
                rows = await self._call_backend_method(
                    instance.backend_instance,
                    "get_context_metadata",
                    instance_id=target.instance_id,
                    node_id=target.node_id,
                )
            except Exception:
                logger.info(
                    "Could not query target KV metadata from %s",
                    target.instance_id,
                    exc_info=True,
                )
                continue
            if isinstance(rows, dict):
                rows = rows.get("contexts", [])
            target_contexts[target.instance_id] = [
                ContextMetadata.from_dict(row)
                for row in (rows or [])
                if isinstance(row, dict)
            ]

        annotated: List[ContextMetadata] = []
        for source in sources:
            reusable_tokens: Dict[str, int] = dict(
                source.reusable_tokens_by_target
            )
            reusable_blocks: Dict[str, int] = dict(
                source.reusable_blocks_by_target
            )
            for target in targets:
                if source.node_id != target.node_id:
                    continue
                for target_context in target_contexts.get(target.instance_id, []):
                    tokens, blocks = self._target_specific_reuse(
                        source, target_context
                    )
                    if tokens > 0 and blocks > 0:
                        reusable_tokens[target.instance_id] = tokens
                        reusable_blocks[target.instance_id] = blocks
                        break
            annotated.append(
                replace(
                    source,
                    reusable_tokens_by_target=reusable_tokens,
                    reusable_blocks_by_target=reusable_blocks,
                )
            )
        return annotated

    def _context_migration_target_capacity(
        self,
        instance: InstanceHandle,
    ) -> int:
        configured_capacity = self.context_migration_config.get(
            "target_capacity"
        )
        if configured_capacity is not None:
            return max(0, int(configured_capacity))
        return max(0, int(instance.max_queue_length) - int(instance.concurrency))

    def _context_migration_target_warmup_cost(
        self,
        instance: InstanceHandle,
    ) -> float:
        warmup_by_instance = self.context_migration_config.get(
            "warmup_cost_by_instance", {}
        ) or {}
        if instance.instance_id in warmup_by_instance:
            return max(0.0, float(warmup_by_instance[instance.instance_id]))

        warmup_by_node = self.context_migration_config.get(
            "warmup_cost_by_node", {}
        ) or {}
        if (
            instance.node_id is not None
            and str(instance.node_id) in warmup_by_node
        ):
            return max(0.0, float(warmup_by_node[str(instance.node_id)]))

        return max(
            0.0,
            float(
                self.context_migration_config.get(
                    "default_target_warmup_cost",
                    self.context_migration_config.get("target_warmup_cost", 0.0),
                )
                or 0.0
            ),
        )

    def _context_migration_planner_config(self) -> Dict[str, Any]:
        configured = self.context_migration_config.get(
            "planner_config", self.context_migration_config
        )
        planner_config = dict(configured or {})
        if self.context_migration_config.get(
            "require_target_runtime_reuse", True
        ):
            planner_config.setdefault("same_node_token_reuse_ratio", 0.0)
            planner_config.setdefault("same_node_block_reuse_ratio", 0.0)
            planner_config.setdefault("cross_node_token_reuse_ratio", 0.0)
            planner_config.setdefault("cross_node_block_reuse_ratio", 0.0)
        return planner_config

    async def _context_migration_targets(
        self,
        source_instance_ids: set[str],
    ) -> List[MigrationTarget]:
        async with self.instance_management_lock:
            instances = list(self.ready_inference_instances.values())

        targets = []
        for instance in instances:
            if instance.instance_id in source_instance_ids:
                continue
            if instance.state != InstanceState.READY or not instance.ready:
                continue
            capacity = self._context_migration_target_capacity(instance)
            if capacity <= 0:
                continue
            targets.append(
                MigrationTarget(
                    instance_id=instance.instance_id,
                    node_id=str(instance.node_id or ""),
                    capacity=capacity,
                    warmup_cost=self._context_migration_target_warmup_cost(
                        instance
                    ),
                )
            )
        return targets

    async def _execute_prefix_warmup(
        self,
        decision: Dict[str, Any],
        source_instances: List[InstanceHandle],
    ):
        if not self.enable_kv_cache_migration:
            return None
        if decision.get("action") != "migrate":
            return {
                "action": "skipped",
                "operation_kind": "prefix_warmup",
                "true_kv_block_transfer": False,
                "reason": decision.get("action", "no_migration_plan"),
            }

        source_by_id = {
            instance.instance_id: instance for instance in source_instances
        }
        async with self.instance_management_lock:
            target_by_id = dict(self.ready_inference_instances)

        attempted = 0
        succeeded = 0
        skipped = []
        failures = []
        total_tokens = 0
        for plan in decision.get("plans", []):
            source_id = plan.get("old_instance_id")
            target_id = plan.get("new_instance_id")
            source = source_by_id.get(source_id)
            target = target_by_id.get(target_id)
            if source is None or target is None:
                skipped.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": "missing_source_or_target",
                    }
                )
                continue
            if target.backend_instance is None:
                skipped.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": "target_backend_unavailable",
                    }
                )
                continue

            token_batches = self.recovery_tokens_by_instance.get(
                source.instance_id
            )
            if not token_batches:
                token_batches = await self._capture_current_tokens(source)
            if not token_batches:
                skipped.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": "no_source_tokens",
                    }
                )
                continue

            attempted += 1
            total_tokens += sum(len(tokens) for tokens in token_batches)
            try:
                result = await self._call_backend_method(
                    target.backend_instance,
                    "resume_kv_cache",
                    request_datas=token_batches,
                )
            except Exception as e:
                failures.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": str(e),
                    }
                )
                continue
            if result is False:
                failures.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": "backend_returned_false",
                    }
                )
                continue
            succeeded += 1

        return {
            "action": "prefix_warmup",
            "legacy_action": "resume_kv_cache",
            "operation_kind": "prefix_warmup",
            "state_kind": "token_replay_prefix_warmup",
            "true_kv_block_transfer": False,
            "attempted": attempted,
            "succeeded": succeeded,
            "skipped": skipped,
            "failures": failures,
            "warmed_tokens": total_tokens,
            "total_tokens": total_tokens,
            "reason": "resume_kv_cache_token_replay",
        }

    async def _plan_context_migration_after_spot_event(
        self,
        event: str,
        matches: List[InstanceHandle],
    ):
        if not self.enable_context_migration:
            return None

        sources = await self._context_migration_sources(matches)
        source_instance_ids = {instance.instance_id for instance in matches}
        targets = await self._context_migration_targets(source_instance_ids)
        planner_config = self._context_migration_planner_config()
        sources = await self._populate_target_reuse_maps(sources, targets)
        decision = plan_low_cost_migration(
            sources=sources,
            targets=targets,
            planner_config=planner_config,
        ).to_dict()
        prefix_warmup = await self._execute_prefix_warmup(
            decision,
            matches,
        )
        if prefix_warmup is not None:
            decision["prefix_warmup"] = prefix_warmup
            decision["kv_cache_migration"] = {
                **prefix_warmup,
                "deprecated_alias": True,
            }

        self._emit_metric(
            make_context_migration_event(
                model=self.model_name,
                decision=decision,
                reason=f"{event}_spot_event",
            )
        )
        logger.info(
            f"Context migration decision for {self.model_name}: {decision}"
        )
        return decision

    # Router 裡面真正把 instance 標記成 PREEMPTING
    # 把某些 worker 標記成 PREEMPTING，讓 router 停止送新的 request 給它們。
    async def handle_preemption(
        self,
        node_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        matches = await self._matching_inference_instances(
            node_id=node_id, instance_id=instance_id
        )
        marked_instances = []
        for instance in matches:
            await self._set_instance_state(
                instance, InstanceState.PREEMPTING, reason="trace_event"
            )
            await self._capture_current_tokens(instance)
            marked_instances.append(instance.instance_id)
            logger.info(
                f"Marked instance {instance.instance_id} as preempting "
                f"for model {self.model_name}"
            )
        replanning = await self._replan_after_spot_event(
            event="preempt",
            node_id=node_id,
            instance_id=instance_id,
            matches=matches,
        )
        context_migration = await self._plan_context_migration_after_spot_event(
            event="preempt",
            matches=matches,
        )
        return {
            "model": self.model_name,
            "event": "preempt",
            "instances": marked_instances,
            "reparallelization": replanning,
            "context_migration": context_migration,
        }

    async def mark_instance_preempting(self, instance_id: str):
        return await self.handle_preemption(instance_id=instance_id)

    async def handle_recover(
        self,
        node_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        matches = await self._matching_inference_instances(
            node_id=node_id, instance_id=instance_id
        )
        recovered_instances = []
        for instance in matches:
            from_state = instance.state.value
            recovered = await instance.mark_recovered()
            if not recovered:
                logger.info(
                    f"Skipping recover for instance {instance.instance_id} "
                    f"in state {from_state}"
                )
                continue
            self._emit_metric(
                make_instance_state_event(
                    model=self.model_name,
                    instance_id=instance.instance_id,
                    node_id=instance.node_id,
                    from_state=from_state,
                    to_state=instance.state.value,
                    reason="trace_recover",
                )
            )
            recovered_instances.append(instance.instance_id)
            logger.info(
                f"Recovered instance {instance.instance_id} "
                f"for model {self.model_name}"
            )
        replanning = await self._replan_after_spot_event(
            event="recover",
            node_id=node_id,
            instance_id=instance_id,
            matches=matches,
        )
        return {
            "model": self.model_name,
            "event": "recover",
            "instances": recovered_instances,
            "reparallelization": replanning,
        }

    async def mark_instance_recovered(self, instance_id: str):
        return await self.handle_recover(instance_id=instance_id)

    async def handle_add(
        self,
        node_id: str,
        node_info: Optional[Mapping[str, Any]] = None,
    ):
        """Handle a newly available worker while requests are running.

        Adding capacity can change the selected TP/DP/EP plan.  The planner
        is therefore rerun immediately; if the selected shape or placement
        changes, the deployment executor creates the new target and migrates
        tracked requests before switching traffic.
        """
        normalized_node_id = str(node_id)
        updates = {
            normalized_node_id: {"state": READY, **dict(node_info or {})}
        }
        replanning = await self._replan_after_spot_event(
            event="add",
            node_id=normalized_node_id,
            instance_id=None,
            matches=[],
            worker_node_updates=updates,
        )
        return {
            "model": self.model_name,
            "event": "add",
            "node_id": normalized_node_id,
            "reparallelization": replanning,
        }

    async def handle_remove(self, node_id: str):
        """Remove a worker node and migrate any live requests from it."""
        matches = await self._matching_inference_instances(node_id=node_id)
        marked_instances = []
        for instance in matches:
            await self._capture_current_tokens(instance)
            await self._set_instance_state(
                instance, InstanceState.DEAD, reason="trace_remove"
            )
            marked_instances.append(instance.instance_id)
        replanning = await self._replan_after_spot_event(
            event="remove",
            node_id=str(node_id),
            instance_id=None,
            matches=matches,
        )
        context_migration = await self._plan_context_migration_after_spot_event(
            event="remove",
            matches=matches,
        )
        return {
            "model": self.model_name,
            "event": "remove",
            "instances": marked_instances,
            "reparallelization": replanning,
            "context_migration": context_migration,
        }

    async def handle_dead(
        self,
        node_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        matches = await self._matching_inference_instances(
            node_id=node_id, instance_id=instance_id
        )
        marked_instances = []
        for instance in matches:
            await self._capture_current_tokens(instance)
            await self._set_instance_state(
                instance, InstanceState.DEAD, reason="trace_dead"
            )
            marked_instances.append(instance.instance_id)
            logger.info(
                f"Marked instance {instance.instance_id} as dead "
                f"for model {self.model_name}"
            )
        replanning = await self._replan_after_spot_event(
            event="dead",
            node_id=node_id,
            instance_id=instance_id,
            matches=matches,
        )
        context_migration = await self._plan_context_migration_after_spot_event(
            event="dead",
            matches=matches,
        )
        return {
            "model": self.model_name,
            "event": "dead",
            "instances": marked_instances,
            "reparallelization": replanning,
            "context_migration": context_migration,
        }

    async def mark_instance_dead(self, instance_id: str):
        return await self.handle_dead(instance_id=instance_id)

    async def drain_instance(self, instance_id: str):
        matches = await self._matching_inference_instances(
            instance_id=instance_id
        )
        drained_instances = []
        for instance in matches:
            await self._set_instance_state(
                instance, InstanceState.DRAINING, reason="manual_drain"
            )
            drained_instances.append(instance.instance_id)
            logger.info(
                f"Marked instance {instance.instance_id} as draining "
                f"for model {self.model_name}"
            )
        return {
            "model": self.model_name,
            "event": "drain",
            "instances": drained_instances,
        }

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def _allocate_instance_for_request(self) -> Tuple[str, InstanceHandle]:
        instance_allocation = self.loop.create_future()
        await self.request_queue.put(instance_allocation)
        logger.info(f"Enqueued request for model {self.model_name}")

        instance_id = await instance_allocation
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                raise RuntimeError("Instance not found")
            return instance_id, self.ready_inference_instances[instance_id]

    async def _ensure_lora_adapter(
        self, instance: InstanceHandle, request_data: dict
    ):
        if not self.enable_lora or "lora_adapter_name" not in request_data:
            return
        lora_adapter_name = request_data["lora_adapter_name"]
        if lora_adapter_name not in self.loaded_lora_adapters:
            logger.error(f"Lora adapter {lora_adapter_name} not found")
            raise ValueError(f"Lora adapter {lora_adapter_name} not found")
        await self._call_backend_method(
            instance.backend_instance,
            "load_lora_adapter",
            lora_name=lora_adapter_name,
            lora_path=self.loaded_lora_adapters[lora_adapter_name],
        )

    def _build_token_replay_request(
        self, request_data: dict, replay_tokens: Optional[List[List[int]]]
    ) -> dict:
        replay_request = copy.deepcopy(request_data)
        completed_tokens = replay_request.pop("_completed_tokens", None)
        if not replay_tokens:
            return replay_request

        first_sequence = replay_tokens[0]
        replay_request["input_tokens"] = first_sequence
        if completed_tokens is not None and "max_tokens" in replay_request:
            replay_request["max_tokens"] = max(
                1, int(replay_request["max_tokens"]) - int(completed_tokens)
            )
        return replay_request

    def _build_stateful_restore_request(
        self, request_data: dict, state: InferenceState
    ) -> dict:
        restore_request = copy.deepcopy(request_data)
        # Internal migration bookkeeping is consumed by the router and must
        # never be forwarded as a vLLM SamplingParams field.
        restore_request.pop("_completed_tokens", None)
        if state.tokens:
            restore_request["input_tokens"] = list(state.tokens)
        completed_tokens = state.completed_tokens
        if completed_tokens and "max_tokens" in restore_request:
            restore_request["max_tokens"] = max(
                1,
                int(restore_request["max_tokens"]) - int(completed_tokens),
            )
        return restore_request

    async def _backend_supports_state_restore(
        self, instance: InstanceHandle
    ) -> bool:
        if instance.backend_instance is None:
            return False
        try:
            return bool(
                await self._call_backend_method(
                    instance.backend_instance,
                    "supports_state_restore",
                )
            )
        except Exception as e:
            logger.info(
                f"Could not check state restore support on "
                f"{instance.instance_id}: {e}"
            )
            return False

    async def _plan_stateful_recovery_target(
        self,
        state: InferenceState,
        source_instance_id: str,
    ) -> Tuple[Optional[Tuple[str, InstanceHandle]], Dict[str, Any]]:
        """Select and reserve a compatible READY target before retry.

        This is the executor boundary for the first planner-integrated NIXL
        path.  The planner only selects an existing target; it never changes
        the parallel shape or starts an engine.  If no target is provably
        usable, normal allocation is left to the caller and stateful recovery
        can fall back to token replay.
        """
        if not self.enable_stateful_target_planner:
            return None, {
                "action": "fallback_token_replay",
                "target_instance_id": None,
                "reason": "stateful_target_planner_disabled",
                "candidates": [],
            }

        async with self.instance_management_lock:
            instances = list(self.ready_inference_instances.values())

        candidates: List[Dict[str, Any]] = []
        instance_by_id: Dict[str, InstanceHandle] = {}
        for instance in instances:
            if instance.instance_id == source_instance_id:
                continue
            if not await instance.can_accept_request():
                continue
            if instance.backend_instance is None:
                continue
            supports_restore = await self._backend_supports_state_restore(
                instance
            )
            if not supports_restore:
                continue

            candidate: Dict[str, Any] = {
                "instance_id": instance.instance_id,
                "node_id": instance.node_id or "",
                "ready": (
                    instance.ready and instance.state == InstanceState.READY
                ),
                "supports_state_restore": True,
                "concurrency": instance.concurrency,
                "backend": state.backend,
                "model_name": state.model_name or self.model_name,
            }
            get_runtime_metadata = getattr(
                instance.backend_instance, "get_runtime_metadata", None
            )
            if get_runtime_metadata is None:
                # A vLLM target without a runtime probe cannot prove that its
                # TP/PP/EP and KV layout match the exported state.
                continue
            try:
                runtime_metadata = await self._call_backend_method(
                    instance.backend_instance,
                    "get_runtime_metadata",
                    instance_id=instance.instance_id,
                    node_id=instance.node_id or "",
                )
                if not isinstance(runtime_metadata, dict):
                    continue
                candidate.update(runtime_metadata)
            except Exception:
                logger.info(
                    "Could not read runtime metadata from recovery target %s",
                    instance.instance_id,
                    exc_info=True,
                )
                continue
            candidates.append(candidate)
            instance_by_id[instance.instance_id] = instance

        decision = plan_compatible_state_target(
            state=state,
            candidates=candidates,
            source_instance_id=source_instance_id,
        )
        target_id = decision.get("target_instance_id")
        if not target_id:
            return None, decision
        target = instance_by_id.get(str(target_id))
        if target is None or not await target.add_requests(1):
            decision = {
                **decision,
                "action": "fallback_token_replay",
                "target_instance_id": None,
                "reason": "target_became_busy_before_reservation",
            }
            return None, decision
        return (target.instance_id, target), decision

    async def _restore_inference_state(
        self,
        instance: InstanceHandle,
        state: InferenceState,
        request_data: dict,
    ) -> dict:
        if instance.backend_instance is None:
            return {"restored": False}
        try:
            restore_request_data = dict(request_data)
            restore_request_data["target_instance_id"] = instance.instance_id
            restore_request_data["target_node_id"] = instance.node_id or ""
            result = await self._call_backend_method(
                instance.backend_instance,
                "restore_inference_state",
                state=state.to_dict(),
                request_data=restore_request_data,
            )
        except Exception as e:
            logger.info(
                f"State restore failed on {instance.instance_id}: {e}"
            )
            return {"restored": False}
        return result if isinstance(result, dict) else {"restored": bool(result)}

    async def _prepare_stateful_recovery(
        self,
        request_id: str,
        source_instance_id: str,
        target_instance: InstanceHandle,
        state: Optional[InferenceState],
        request_data: dict,
        target_selection: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[
        Optional[InferenceState],
        Optional[List[List[int]]],
        Dict[str, int],
        bool,
    ]:
        counters = {
            "state_restore_attempts": 0,
            "state_restore_successes": 0,
            "state_restored_tokens": 0,
        }
        if state is None:
            return None, None, counters, False

        planner_allows_restore = (
            target_selection is None
            or target_selection.get("action") == "restore_state"
        )
        restore_supported = planner_allows_restore and await (
            self._backend_supports_state_restore(target_instance)
        )
        decision = plan_stateful_recovery(
            request_id=request_id,
            source_instance_id=source_instance_id,
            target_instance_id=target_instance.instance_id,
            state=state,
            restore_supported=restore_supported,
            reason=str(
                (target_selection or {}).get(
                    "reason", "stateful_recovery"
                )
            ),
        )
        self._emit_metric(
            make_state_recovery_event(
                model=self.model_name,
                request_id=request_id,
                decision=decision.to_dict(),
                source_instance_id=source_instance_id,
                target_instance_id=target_instance.instance_id,
            )
        )

        if decision.action == "restore_state":
            counters["state_restore_attempts"] = 1
            restore_result = await self._restore_inference_state(
                target_instance, state, request_data
            )
            if restore_result.get("restored"):
                counters["state_restore_successes"] = 1
                counters["state_restored_tokens"] = decision.recovered_tokens
                return state, None, counters, False
            if restore_result.get("staged"):
                return state, None, counters, False

        if state.tokens:
            return None, [list(state.tokens)], counters, True
        return None, None, counters, True

    async def _counts_toward_capacity(self, instance: InstanceHandle) -> bool:
        async with instance.lock:
            capacity_states = {
                InstanceState.STARTING,
                InstanceState.READY,
                InstanceState.BUSY,
            }
            if self.count_preempting_toward_capacity:
                capacity_states.add(InstanceState.PREEMPTING)
            return instance.state in capacity_states

    async def _call_backend(
        self,
        instance: InstanceHandle,
        request_data: dict,
        action: str,
        replay_tokens: Optional[List[List[int]]] = None,
        state_snapshot: Optional[InferenceState] = None,
    ):
        await self._ensure_lora_adapter(instance, request_data)
        if state_snapshot is not None:
            backend_request = self._build_stateful_restore_request(
                request_data, state_snapshot
            )
        else:
            backend_request = self._build_token_replay_request(
                request_data, replay_tokens
            )

        if (
            action == "generate"
            and replay_tokens
            and self.backend == "transformers"
        ):
            try:
                return await self._call_backend_method(
                    instance.backend_instance,
                    "resume_generate",
                    request_data=backend_request,
                    current_output=replay_tokens,
                )
            except Exception as e:
                logger.info(
                    f"Backend resume_generate failed on "
                    f"{instance.instance_id}: {e}; falling back to generate"
                )

        # NOTE: `.remote(request_data)` does not work, don't know why.
        # Looks like a known issue:
        # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
        if action == "generate":
            return await self._call_backend_method(
                instance.backend_instance,
                "generate",
                request_data=backend_request,
            )
        if action == "encode":
            return await self._call_backend_method(
                instance.backend_instance,
                "encode",
                request_data=backend_request,
            )
        return {"error": "Invalid action"}

    async def _release_instance_request(self, instance: InstanceHandle):
        released = await instance.add_requests(-1)
        if not released:
            logger.error(
                f"Failed to release request slot on {instance.instance_id}"
            )

    def _result_is_preempted(self, result) -> bool:
        if not isinstance(result, dict):
            return False
        preempted = result.get("preempted", False)
        return preempted is True or preempted == "True"

    def _result_is_error(self, result) -> bool:
        return isinstance(result, dict) and "error" in result

    def _should_retry(self, attempt: int) -> bool:
        if self.recovery_policy == RecoveryPolicy.NONE:
            return False
        return attempt < self.max_retries

    def _tokens_from_preempted_result(self, result) -> List[List[int]]:
        if not isinstance(result, dict):
            return []
        current_output = result.get("current_output")
        if not current_output:
            return []
        return current_output

    def _count_recovered_tokens(
        self,
        replay_tokens: Optional[List[List[int]]],
        completed_tokens: Optional[int] = None,
    ) -> int:
        if completed_tokens is not None:
            return max(0, int(completed_tokens))
        if not replay_tokens:
            return 0
        return sum(len(sequence) for sequence in replay_tokens)

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        request_id = request_data.get("request_id", f"req-{uuid.uuid4()}")
        # Make the router's external ID the connector/backend ID as well.  It
        # is the key used by export/abort/restore during a live replan.
        request_data.setdefault("request_id", request_id)
        request_start = time.time()
        attempts = 0
        failed_attempts = 0
        assigned_instances = []
        recovered_tokens = 0
        recovery_fallback = False
        replay_tokens = None
        recovery_state: Optional[InferenceState] = None
        recovery_state_source_instance_id = ""
        state_restore_attempts = 0
        state_restore_successes = 0
        state_restore_fallback = False
        state_restored_tokens = 0
        force_state_recovery = False
        force_retry_budget = 0

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

        try:
            while True:
                instance = None
                target_selection: Optional[Dict[str, Any]] = None
                attempts += 1
                try:
                    if (
                        recovery_state is not None
                        and self.enable_stateful_target_planner
                        and (
                            self.recovery_policy
                            == RecoveryPolicy.STATEFUL_RECOVERY
                            or force_state_recovery
                        )
                    ):
                        planned_target, target_selection = (
                            await self._plan_stateful_recovery_target(
                                state=recovery_state,
                                source_instance_id=(
                                    recovery_state_source_instance_id
                                ),
                            )
                        )
                        if planned_target is not None:
                            instance_id, instance = planned_target
                        else:
                            instance_id, instance = (
                                await self._allocate_instance_for_request()
                            )
                    else:
                        instance_id, instance = (
                            await self._allocate_instance_for_request()
                        )
                    assigned_instances.append(instance_id)
                    inflight = await self._track_inflight_request(
                        request_id,
                        request_data,
                        action,
                        instance,
                    )
                    logger.info(
                        f"{request_data}, type: {type(request_data)}, "
                        f"attempt: {attempts}"
                    )
                    state_snapshot = None
                    if (
                        recovery_state is not None
                        and (
                            self.recovery_policy
                            == RecoveryPolicy.STATEFUL_RECOVERY
                            or force_state_recovery
                        )
                    ):
                        (
                            state_snapshot,
                            replay_tokens,
                            state_counters,
                            used_fallback,
                        ) = await self._prepare_stateful_recovery(
                            request_id=request_id,
                            source_instance_id=(
                                recovery_state_source_instance_id
                            ),
                            target_instance=instance,
                            state=recovery_state,
                            request_data=request_data,
                            target_selection=target_selection,
                        )
                        state_restore_attempts += state_counters[
                            "state_restore_attempts"
                        ]
                        state_restore_successes += state_counters[
                            "state_restore_successes"
                        ]
                        restored_tokens = state_counters[
                            "state_restored_tokens"
                        ]
                        state_restored_tokens += restored_tokens
                        recovered_tokens += restored_tokens
                        if used_fallback:
                            state_restore_fallback = True
                            recovery_fallback = True
                            recovered_tokens += self._count_recovered_tokens(
                                replay_tokens,
                                recovery_state.completed_tokens,
                            )
                        force_state_recovery = False

                    result = await self._call_backend(
                        instance,
                        request_data,
                        action,
                        replay_tokens=replay_tokens,
                        state_snapshot=state_snapshot,
                    )
                    kv_restore = (
                        result.get("_spotserve_kv_restore", {})
                        if isinstance(result, dict)
                        else {}
                    )
                    if state_snapshot is not None and kv_restore:
                        if kv_restore.get("restored"):
                            restored_count = (
                                state_snapshot.completed_tokens
                                or len(state_snapshot.tokens)
                            )
                            state_restore_successes += 1
                            state_restored_tokens += restored_count
                            recovered_tokens += restored_count
                        else:
                            state_restore_fallback = True
                            recovery_fallback = True
                    logger.info("Finished processing request")

                    reparallelized = (
                        isinstance(result, dict)
                        and result.get("_spotserve_reparallelization")
                    )
                    if not self._result_is_preempted(
                        result
                    ) and not self._result_is_error(result):
                        self._emit_metric(
                            make_request_event(
                                request_id=request_id,
                                model=self.model_name,
                                policy=self.recovery_policy.value,
                                success=True,
                                latency_ms=(time.time() - request_start)
                                * 1000,
                                retry_count=attempts - 1,
                                failed_attempts=failed_attempts,
                                recovered_tokens=recovered_tokens,
                                recovery_fallback=recovery_fallback,
                                state_restore_attempts=(
                                    state_restore_attempts
                                ),
                                state_restore_successes=(
                                    state_restore_successes
                                ),
                                state_restore_fallback=(
                                    state_restore_fallback
                                ),
                                state_restored_tokens=state_restored_tokens,
                            )
                        )
                        return result

                    failed_attempts += 1
                    if reparallelized:
                        migration_state = inflight.get("migration_state")
                        recovery_state = migration_state
                        recovery_state_source_instance_id = instance_id
                        force_state_recovery = migration_state is not None
                        force_retry_budget = max(force_retry_budget, 1)
                        replay_tokens = (
                            [list(migration_state.tokens)]
                            if migration_state is not None
                            and migration_state.tokens
                            else None
                        )
                        if migration_state is not None:
                            request_data["_completed_tokens"] = (
                                migration_state.completed_tokens
                            )
                    elif self._result_is_preempted(result):
                        await self._set_instance_state(
                            instance,
                            InstanceState.PREEMPTING,
                            reason="backend_preempted",
                        )
                        replay_tokens = self._tokens_from_preempted_result(
                            result
                        )
                        completed_tokens = result.get("completed_tokens", 0)
                        if (
                            self.recovery_policy
                            == RecoveryPolicy.STATEFUL_RECOVERY
                        ):
                            exported_state = result.get(
                                "_spotserve_inference_state"
                            )
                            recovery_state = (
                                await self._capture_inference_state(
                                    instance,
                                    request_data=request_data,
                                    current_output=replay_tokens,
                                    completed_tokens=completed_tokens,
                                    exported_state=(
                                        exported_state
                                        if isinstance(exported_state, dict)
                                        else None
                                    ),
                                )
                            )
                            recovery_state_source_instance_id = instance_id
                        if completed_tokens:
                            request_data["_completed_tokens"] = (
                                completed_tokens
                            )
                    else:
                        replay_tokens = (
                            self.recovery_tokens_by_instance.get(instance_id)
                        )
                        if (
                            self.recovery_policy
                            == RecoveryPolicy.STATEFUL_RECOVERY
                        ):
                            recovery_state = (
                                await self._capture_inference_state(
                                    instance,
                                    request_data=request_data,
                                    current_output=replay_tokens,
                                )
                            )
                            recovery_state_source_instance_id = instance_id

                    if (
                        self.recovery_policy
                        == RecoveryPolicy.GENERATED_TOKEN_REPLAY
                        and replay_tokens
                    ):
                        recovered_tokens += self._count_recovered_tokens(
                            replay_tokens,
                            result.get("completed_tokens"),
                        )
                    elif (
                        self.recovery_policy
                        == RecoveryPolicy.GENERATED_TOKEN_REPLAY
                    ):
                        recovery_fallback = True

                    should_retry = self._should_retry(
                        attempts - 1
                    ) or force_retry_budget > 0
                    if force_retry_budget > 0:
                        force_retry_budget -= 1
                    if not should_retry:
                        self._emit_metric(
                            make_request_event(
                                request_id=request_id,
                                model=self.model_name,
                                policy=self.recovery_policy.value,
                                success=False,
                                latency_ms=(time.time() - request_start)
                                * 1000,
                                retry_count=attempts - 1,
                                failed_attempts=failed_attempts,
                                recovered_tokens=recovered_tokens,
                                recovery_fallback=recovery_fallback,
                                state_restore_attempts=(
                                    state_restore_attempts
                                ),
                                state_restore_successes=(
                                    state_restore_successes
                                ),
                                state_restore_fallback=(
                                    state_restore_fallback
                                ),
                                state_restored_tokens=state_restored_tokens,
                            )
                        )
                        return result

                    if (
                        self.recovery_policy
                        not in {
                            RecoveryPolicy.GENERATED_TOKEN_REPLAY,
                            RecoveryPolicy.STATEFUL_RECOVERY,
                        }
                    ):
                        replay_tokens = None

                except ValueError as e:
                    self._emit_metric(
                        make_request_event(
                            request_id=request_id,
                            model=self.model_name,
                            policy=self.recovery_policy.value,
                            success=False,
                            latency_ms=(time.time() - request_start) * 1000,
                            retry_count=attempts - 1,
                            failed_attempts=failed_attempts,
                            state_restore_attempts=state_restore_attempts,
                            state_restore_successes=state_restore_successes,
                            state_restore_fallback=state_restore_fallback,
                            state_restored_tokens=state_restored_tokens,
                        )
                    )
                    return {"error": str(e)}
                except Exception as e:
                    failed_attempts += 1
                    logger.error(f"Request attempt failed: {e}")
                    if instance is not None:
                        if (
                            self.recovery_policy
                            == RecoveryPolicy.STATEFUL_RECOVERY
                        ):
                            recovery_state = (
                                await self._capture_inference_state(
                                    instance,
                                    request_data=request_data,
                                )
                            )
                            recovery_state_source_instance_id = instance_id
                            replay_tokens = (
                                [list(recovery_state.tokens)]
                                if recovery_state
                                else []
                            )
                        else:
                            replay_tokens = await self._capture_current_tokens(
                                instance
                            )
                        await self._set_instance_state(
                            instance,
                            InstanceState.DEAD,
                            reason="request_exception",
                        )

                    if (
                        self.recovery_policy
                        == RecoveryPolicy.GENERATED_TOKEN_REPLAY
                        and replay_tokens
                    ):
                        recovered_tokens += self._count_recovered_tokens(
                            replay_tokens
                        )
                    elif (
                        self.recovery_policy
                        == RecoveryPolicy.GENERATED_TOKEN_REPLAY
                    ):
                        recovery_fallback = True

                    if not self._should_retry(attempts - 1):
                        self._emit_metric(
                            make_request_event(
                                request_id=request_id,
                                model=self.model_name,
                                policy=self.recovery_policy.value,
                                success=False,
                                latency_ms=(time.time() - request_start)
                                * 1000,
                                retry_count=attempts - 1,
                                failed_attempts=failed_attempts,
                                recovered_tokens=recovered_tokens,
                                recovery_fallback=recovery_fallback,
                                state_restore_attempts=(
                                    state_restore_attempts
                                ),
                                state_restore_successes=(
                                    state_restore_successes
                                ),
                                state_restore_fallback=(
                                    state_restore_fallback
                                ),
                                state_restored_tokens=state_restored_tokens,
                            )
                        )
                        return {"error": str(e)}

                    if (
                        self.recovery_policy
                        not in {
                            RecoveryPolicy.GENERATED_TOKEN_REPLAY,
                            RecoveryPolicy.STATEFUL_RECOVERY,
                        }
                    ):
                        replay_tokens = None
                finally:
                    if instance is not None:
                        await self._release_instance_request(instance)
        finally:
            await self._untrack_inflight_request(request_id)
            async with self.request_count_lock:
                self.request_count -= 1

    async def fine_tuning(self, request_data: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name}")
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.fine_tuning_count_lock:
            self.fine_tuning_count += 1

        try:
            instance_id = await self._create_ft_instance()
        except Exception as e:
            logger.error(f"Failed to create fine-tuning instance: {str(e)}")
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {"error": f"Failed to create fine-tuning instance: {str(e)}"}

        max_wait_time = 300
        wait_time = 0
        while wait_time < max_wait_time:
            async with self.instance_management_lock:
                if instance_id in self.ready_ft_instances:
                    instance = self.ready_ft_instances[instance_id]
                    break
                elif instance_id not in self.starting_ft_instances:
                    logger.error(
                        f"Fine tuning instance {instance_id} not found in starting or ready instances"
                    )
                    async with self.fine_tuning_count_lock:
                        self.fine_tuning_count -= 1
                    return {"error": "Fine tuning instance not found"}
            await asyncio.sleep(0.1)
            wait_time += 0.1
        else:
            logger.error(
                f"Timeout waiting for fine tuning instance {instance_id} to be ready"
            )
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {
                "error": "Timeout waiting for fine tuning instance to be ready"
            }

        try:
            logger.info(f"Calling fine_tuning method on instance {instance_id}")
            result = await instance.backend_instance.fine_tuning.remote(
                request_data=request_data
            )

            logger.info(f"Finished processing fine-tuning {self.model_name}")
            await instance.add_requests(-1)

            await self._shutdown_instance(instance_id, is_ft=True)

            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1

            return result
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            await instance.add_requests(-1)
            await self._shutdown_instance(instance_id, is_ft=True)
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            logger.info(
                f"Fine-tuning failed and cleaned up for model {self.model_name}"
            )
            return {"error": f"Fine-tuning failed: {str(e)}"}

    async def delete_adapters(self, lora_adapters: List[str]):
        async with self.lora_lock:
            for adapter_name in lora_adapters:
                if adapter_name in self.loaded_lora_adapters:
                    del self.loaded_lora_adapters[adapter_name]
        logger.info(
            f"Deleted LoRA adapters {lora_adapters} on model {self.model_name}"
        )

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
        # stop all inference instances
        # return all unfinished requests
        while not self.request_queue.empty():
            request_data, done_event = await self.request_queue.get()
            done_event.set_result({"error": "Instance cancelled"})

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_inference_instances.keys())
        delete_tasks = [
            self._shutdown_instance(instance_id)
            for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)

        return deleted_instance_id

    async def _load_balancer_loop(self):
        # this is a simple round-robin load balancer
        round_robin_index = 0
        while True:
            instance_allocation = await self.request_queue.get()
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated:
                # 1. get ready instances
                instance_options = None
                while not instance_options:
                    await asyncio.sleep(1)
                    async with self.instance_management_lock:
                        candidate_instances = list(
                            self.ready_inference_instances.items()
                        )
                    instance_options = []
                    for candidate_id, candidate in candidate_instances:
                        if await candidate.can_accept_request():
                            instance_options.append(candidate_id)
                    logger.info(f"{instance_options}")
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    round_robin_index % len(instance_options)
                ]
                round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_inference_instances:
                        continue
                    instance = self.ready_inference_instances[instance_id]
                    # check if the request queue reaches max length
                    allocated = await instance.add_requests(1)
                    if allocated:
                        instance_allocation.set_result(instance_id)
                    else:
                        logger.info(
                            f"Instance {instance_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                capacity_candidates = list(
                    self.starting_inference_instances.values()
                ) + list(self.ready_inference_instances.values())
            num_running_instances = sum(
                [
                    1
                    for instance in capacity_candidates
                    if await self._counts_toward_capacity(instance)
                ]
            )
            logger.info(
                f"{self.model_name}: {num_running_instances} instances,"
                f"need {desired_instances} instances",
            )
            if desired_instances > num_running_instances:
                logger.info("Creating new instance")
                await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                # logger.info("No scaling needed")
                pass
            await asyncio.sleep(self.loop_interval)

    async def _create_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # get max_queue_length from auto_scaling_config
        if self.auto_scaling_config.get("metric", "") == "concurrency":
            max_request_length = self.auto_scaling_config.get("target", 1)
        else:
            max_request_length = 1
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _create_ft_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new FT instance {instance_id} for model {self.model_name}"
        )

        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=1,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_ft_instances[instance_id] = instance
        self.loop.create_task(self._start_ft_instance(instance_id))
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        if self.backend == "dummy":
            startup_node = "control"
            startup_resources = {}
        else:
            # Now ask model loading scheduler to load the model
            logger.info(
                f"Allocating resources for model {self.model_name} on instance {instance_id}"
            )
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            startup_resources = {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            }
        async with instance.lock:
            instance.node_id = startup_node
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": startup_resources,
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        starter_options = {}
        if startup_resources:
            starter_options["resources"] = startup_resources

        if self.backend == "dummy":
            from sllm.backends.dummy_backend import DummyBackend

            instance.backend_instance = DummyBackend(
                self.model_name, self.backend_config
            )
        else:
            instance.backend_instance = await start_instance.options(
                **starter_options
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        await self._call_backend_method(
            instance.backend_instance, "init_backend"
        )
        await instance.mark_ready(node_id=startup_node)

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)
        return instance_id

    async def _start_ft_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_ft_instances:
                logger.error(f"FT Instance {instance_id} not found")
                return
            instance = self.starting_ft_instances[instance_id]

        logger.info(
            f"Allocating FT resources for model {self.model_name} on {instance_id}"
        )
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            logger.debug(
                f"Allocated resources on node {startup_node} for FT instance {instance_id}"
            )
            async with instance.lock:
                instance.node_id = startup_node
        except Exception as e:
            logger.error(
                f"Failed to allocate resources for FT instance {instance_id}: {str(e)}"
            )
            raise

        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                resources=startup_config["resources"]
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to create Ray actor for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        try:
            await instance.backend_instance.init_backend.remote()
        except Exception as e:
            logger.error(
                f"Failed to initialize backend for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        await instance.mark_ready(node_id=startup_node)

        async with self.instance_management_lock:
            self.ready_ft_instances[instance_id] = instance
            self.starting_ft_instances.pop(instance_id)
        logger.info(f"Fine-tuning instance {instance_id} is now ready")
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_inference_instances.popitem()
            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        await instance.mark_draining()
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        await instance.mark_dead()
        await self._stop_backend(instance.backend_instance, "stop")
        if self.backend != "dummy":
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pool = self.ready_ft_instances
            else:
                pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
        await instance.mark_dead()
        await self._stop_backend(instance.backend_instance, "shutdown")
        if self.backend != "dummy":
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )
        return
