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
import hashlib
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
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
    build_candidate_component_costs,
    plan_low_cost_migration,
)
from sllm.spot.recovery_policy import RecoveryPolicy, normalize_policy
from sllm.spot.reparallelization import (
    PREEMPTING,
    READY,
    ParallelPlan,
    apply_spot_event_to_worker_nodes,
    plan_dynamic_reparallelization,
)
from sllm.spot.reparallelization_executor import (
    BREAK_BEFORE_MAKE,
    MAKE_BEFORE_BREAK,
    ReparallelizationExecutor,
)
from sllm.spot.vllm_deployment_adapter import (
    VllmDeployment,
    VllmDeploymentAdapter,
)
from sllm.spot.stateful_recovery import (
    InferenceState,
    plan_compatible_state_target,
    plan_stateful_recovery,
)

from ..utils import InstanceHandle, InstanceState, worker_placement_options
from .router_utils import SllmRouter

logger = init_logger(__name__)


class NativeKVRecoveryError(ValueError):
    """A stateful treatment must not silently become a replay treatment."""


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
        self.require_native_kv_restore = bool(
            router_config.get("require_native_kv_restore", False)
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
        self.emit_context_migration_candidate_costs = bool(
            self.context_migration_config.get(
                "emit_candidate_component_costs",
                self.context_migration_config.get(
                    "emit_candidate_costs", False
                ),
            )
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
        self.stateful_recovery_config = dict(
            router_config.get("stateful_recovery_config", {})
        )
        synthetic_worker_nodes = self.reparallelization_config.get(
            "synthetic_worker_nodes", {}
        )
        self.reparallelization_worker_nodes = {}
        for node_id, node_info in synthetic_worker_nodes.items():
            synthetic_node_info = dict(node_info)
            synthetic_node_info.setdefault("_spotserve_synthetic", True)
            self.reparallelization_worker_nodes[str(node_id)] = (
                synthetic_node_info
            )
        self.vllm_deployment_adapter: Optional[VllmDeploymentAdapter] = None
        self.reparallelization_executor: Optional[
            ReparallelizationExecutor
        ] = None
        self._reparallelization_execution_active = False
        self._reparallelization_lock = asyncio.Lock()
        workload_window_max_requests = int(
            self.reparallelization_config.get(
                "workload_window_max_requests", 128
            )
            or 128
        )
        self.reparallelization_workload_window_s = float(
            self.reparallelization_config.get("workload_window_s", 60.0)
            or 60.0
        )
        self.reparallelization_request_arrivals = deque(
            maxlen=max(workload_window_max_requests, 1)
        )
        self.reparallelization_request_latencies_ms = deque(
            maxlen=max(workload_window_max_requests, 1)
        )
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
        self.failed_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_start_tasks: Dict[str, asyncio.Task] = {}
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ft_start_tasks: Dict[str, asyncio.Task] = {}
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
        self.native_kv_receipts: Dict[str, Dict[str, Any]] = {}
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
        try:
            timeout_s = float(
                self.router_config.get("backend_shutdown_timeout_s", 30.0)
            )
        except (TypeError, ValueError):
            timeout_s = 30.0
        if timeout_s <= 0:
            timeout_s = 30.0
        try:
            await asyncio.wait_for(
                self._call_backend_method(backend_instance, method_name),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting %.3fs for backend actor %s during %s; "
                "killing actor",
                timeout_s,
                backend_instance,
                method_name,
            )
        finally:
            if hasattr(backend_instance, "_ray_actor_id"):
                ray.kill(backend_instance)

    async def _clear_scheduler_model_state(self) -> None:
        if self.backend == "dummy":
            return
        clear_model = getattr(self.model_loading_scheduler, "clear_model", None)
        if clear_model is None:
            return
        try:
            remote = getattr(clear_model, "remote", None)
            if remote is not None:
                await remote(self.model_name)
            else:
                result = clear_model(self.model_name)
                if inspect.isawaitable(result):
                    await result
        except Exception:
            logger.exception(
                "Failed to clear scheduler state for model %s",
                self.model_name,
            )

    async def _cancel_start_tasks(self) -> None:
        task_items = [
            (instance_id, task)
            for instance_id, task in list(self.instance_start_tasks.items())
            if not task.done()
        ]
        task_items.extend(
            (instance_id, task)
            for instance_id, task in list(self.ft_start_tasks.items())
            if not task.done()
        )
        tasks = [task for _, task in task_items]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        try:
            timeout_s = float(
                self.router_config.get("startup_task_cancel_timeout_s", 5.0)
            )
        except (TypeError, ValueError):
            timeout_s = 5.0
        if timeout_s <= 0:
            timeout_s = 5.0
        done, pending = await asyncio.wait(tasks, timeout=timeout_s)
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Startup task failed while shutting down model %s",
                    self.model_name,
                )
        if pending:
            logger.warning(
                "Timed out waiting %.3fs for %d startup tasks while "
                "shutting down model %s",
                timeout_s,
                len(pending),
                self.model_name,
            )
            pending_task_ids = {
                task: instance_id for instance_id, task in task_items
            }
            await asyncio.gather(
                *(
                    self._kill_detached_backend_actor_by_name(
                        pending_task_ids[task]
                    )
                    for task in pending
                ),
                return_exceptions=True,
            )
        self.instance_start_tasks.clear()
        self.ft_start_tasks.clear()

    async def _kill_detached_backend_actor_by_name(
        self, actor_name: str
    ) -> bool:
        try:
            timeout_s = float(
                self.router_config.get(
                    "startup_actor_cleanup_timeout_s", 60.0
                )
            )
        except (TypeError, ValueError):
            timeout_s = 60.0
        try:
            quiet_s = float(
                self.router_config.get("startup_actor_cleanup_quiet_s", 15.0)
            )
        except (TypeError, ValueError):
            quiet_s = 15.0
        try:
            interval_s = float(
                self.router_config.get(
                    "startup_actor_cleanup_interval_s", 0.5
                )
            )
        except (TypeError, ValueError):
            interval_s = 0.5
        timeout_s = max(timeout_s, 0.0)
        quiet_s = max(quiet_s, 0.0)
        interval_s = max(interval_s, 0.05)
        deadline = time.monotonic() + timeout_s
        quiet_deadline = time.monotonic() + quiet_s
        namespaces = (None, "sllm", "models")

        while True:
            actor_handle = None
            for namespace in namespaces:
                try:
                    if namespace is None:
                        actor_handle = ray.get_actor(actor_name)
                    else:
                        actor_handle = ray.get_actor(
                            actor_name, namespace=namespace
                        )
                    break
                except Exception:
                    continue
            if actor_handle is not None:
                try:
                    ray.kill(actor_handle, no_restart=True)
                except TypeError:
                    ray.kill(actor_handle)
                logger.info(
                    "Killed detached backend actor %s while shutting down "
                    "model %s",
                    actor_name,
                    self.model_name,
                )
                return True
            now = time.monotonic()
            if now >= quiet_deadline or now >= deadline:
                logger.info(
                    "No detached backend actor %s appeared while shutting "
                    "down model %s",
                    actor_name,
                    self.model_name,
                )
                return False
            await asyncio.sleep(interval_s)

    def _track_start_task(
        self,
        instance_id: str,
        task: asyncio.Task,
        task_map: Dict[str, asyncio.Task],
    ) -> None:
        task_map[instance_id] = task

        def _forget_task(_task: asyncio.Task) -> None:
            task_map.pop(instance_id, None)

        task.add_done_callback(_forget_task)

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
            "state_export_started_at_s": 0.0,
            "state_export_finished_at_s": 0.0,
            "state_export_duration_ms": 0.0,
            "abort_requested": 0,
            "migratable": 0,
            "unsupported": 0,
            "request_ids": sorted(
                str(entry.get("request_id")) for entry in entries
            ),
        }
        state_export_durations_ms: List[float] = []
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
            export_started_at = time.time()
            export_timeout = float(self.reparallelization_config.get("state_export_timeout_s", 30.0))
            if instance.preemption_deadline_time_s is not None:
                export_timeout = min(export_timeout, max(
                    0.0, instance.preemption_deadline_time_s - export_started_at))
            if summary["state_export_started_at_s"] == 0.0:
                summary["state_export_started_at_s"] = export_started_at
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
                    timeout=export_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out exporting live state for %s; using token fallback",
                    entry["request_id"],
                )
                # Do not invoke the timed-out hook again. With no committed
                # snapshot, restart from the original client input; this is
                # not KV recovery or a successfully migrated token snapshot.
                state = None
            export_finished_at = time.time()
            summary["state_export_finished_at_s"] = export_finished_at
            state_export_durations_ms.append(
                (export_finished_at - export_started_at) * 1000
            )
            entry["migration_state"] = state
            if self.require_native_kv_restore and (
                state is None or state.state_kind != "vllm_kv_snapshot"
                or not state.supports_restore
            ):
                raise NativeKVRecoveryError("native_kv_export_required_before_abort")
            if (self.require_native_kv_restore
                    and self.reparallelization_config.get("require_free_gpu_overlap")
                    and not state.metadata.get("can_restore_cross_node", False)):
                raise NativeKVRecoveryError("cross_node_kv_restore_not_validated_before_abort")
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
        if state_export_durations_ms:
            summary["state_export_duration_ms"] = sum(
                state_export_durations_ms
            )
            summary["state_export_avg_duration_ms"] = (
                summary["state_export_duration_ms"]
                / len(state_export_durations_ms)
            )
            summary["state_export_max_duration_ms"] = max(
                state_export_durations_ms
            )
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
        stop_current_before_create = bool(
            self.reparallelization_config.get(
                "allow_stop_before_recreate", False
            )
        )
        transition_mode = str(
            self.reparallelization_config.get(
                "transition_mode",
                BREAK_BEFORE_MAKE
                if stop_current_before_create
                else MAKE_BEFORE_BREAK,
            )
        )
        self.reparallelization_executor = ReparallelizationExecutor(
            self.vllm_deployment_adapter.create_workers,
            self.vllm_deployment_adapter.ready_workers,
            self.vllm_deployment_adapter.switch_workers,
            self.vllm_deployment_adapter.drain_workers,
            self.vllm_deployment_adapter.stop_workers,
            stop_current_before_create=stop_current_before_create,
            wait_for_migration=self.vllm_deployment_adapter.wait_for_migration_completion,
            migrate_before_create=bool(
                self.reparallelization_config.get(
                    "migrate_before_create", False
                )
            ),
            transition_mode=transition_mode,
            migration_requires_live_source=bool(
                self.reparallelization_config.get(
                    "migration_requires_live_source", False
                )
            ),
        )
        # Validate contradictory transaction settings before a spot event
        # drains or destroys a serving deployment.
        self.reparallelization_executor.resolved_transition_mode()
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
        source_deadlines = [handle.preemption_deadline_time_s
                            for handle in deployment.instances.values()
                            if handle.preemption_deadline_time_s is not None]
        if source_deadlines:
            deadline = min(deadline, asyncio.get_running_loop().time()
                           + max(0.0, min(source_deadlines) - time.time()))
        active = request_ids
        while asyncio.get_running_loop().time() < deadline:
            async with self.inflight_requests_lock:
                active = request_ids.intersection(self.inflight_requests)
            if not active:
                if source_deadlines and time.time() >= min(source_deadlines):
                    raise TimeoutError("source_kv_lease_deadline_passed_before_release")
                if self.require_native_kv_restore:
                    receipts = [self.native_kv_receipts.get(key) for key in request_ids]
                    if not all(receipts):
                        raise NativeKVRecoveryError("native_kv_receipt_missing_before_source_release")
                    self._emit_metric({
                        "type": "native_kv_source_release_authorized",
                        "model": self.model_name, "request_ids": sorted(request_ids),
                        "source_instance_ids": sorted(deployment.instances),
                        "receipt_kind": "completed_native_request",
                        "deadline_time_s": min(source_deadlines) if source_deadlines else None,
                    })
                    for key in request_ids:
                        self.native_kv_receipts.pop(key, None)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"source_kv_lease_not_released_before_deadline:{len(active)}")

    async def _switch_vllm_deployment(
        self, deployment: VllmDeployment, plan: ParallelPlan
    ):
        """Atomically make the newly initialised actors serve traffic."""
        if deployment.plan != plan:
            raise ValueError("vLLM deployment plan mismatch during switch")
        scheduler = getattr(self, "model_loading_scheduler", None)
        snapshot = getattr(scheduler, "_get_worker_nodes", None)
        if snapshot is not None:
            workers = await self._call_backend_method(self.model_loading_scheduler, "_get_worker_nodes")
            for handle in deployment.instances.values():
                for worker in handle.worker_node_ids():
                    if worker not in workers or workers[worker].get("state", READY) != READY:
                        raise RuntimeError(f"target_worker_no_longer_ready:{worker}")
        async with self.instance_management_lock:
            # Source KV leases remain live after the routing switch. Keep the
            # old actors discoverable by worker ID until they are stopped, so
            # a deadline event can revoke them even after target activation.
            self.deleting_inference_instances.update({
                instance_id: handle
                for instance_id, handle in self.ready_inference_instances.items()
                if instance_id not in deployment.instances
            })
            self.ready_inference_instances = dict(deployment.instances)
            # Once the planner commits a replica count, the generic scaler
            # must not recreate the old initial count with the new GPU shape.
            self.auto_scaling_config = {
                **self.auto_scaling_config,
                "min_instances": plan.replica_count,
                "max_instances": plan.replica_count,
            }
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
                "failed": self.failed_inference_instances,
            }
            states = {}
            for pool_name, pool in pools.items():
                for instance_id, instance in pool.items():
                    status = await instance.get_status()
                    states[instance_id] = {
                        "pool": pool_name,
                        "node_id": status.node_id,
                        "member_node_ids": status.member_node_ids,
                        "state": status.state,
                        "concurrency": status.concurrency,
                    }
                    failure_reason = getattr(instance, "failure_reason", None)
                    if failure_reason:
                        states[instance_id]["failure_reason"] = failure_reason
            return states

    async def get_runtime_ep_audit(
        self, expected_ep_size: int
    ) -> Dict[str, Any]:
        """Read back the physical EP ranks reported by every ready backend.

        Configured TP/DP values are deliberately not accepted as evidence.
        A run passes only when the vLLM runtime reports an explicit ``ep_size``
        and ``ep_rank`` for every expected rank on every serving instance.
        """
        expected_ep_size = max(1, int(expected_ep_size))
        async with self.instance_management_lock:
            instances = list(self.ready_inference_instances.values())

        def collect_rank_rows(value: Any) -> List[Mapping[str, Any]]:
            rows: List[Mapping[str, Any]] = []
            if isinstance(value, Mapping):
                if "ep_rank" in value or "ep_size" in value:
                    rows.append(value)
                for nested in value.values():
                    rows.extend(collect_rank_rows(nested))
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    rows.extend(collect_rank_rows(nested))
            return rows

        audits: List[Dict[str, Any]] = []
        for instance in instances:
            audit: Dict[str, Any] = {
                "instance_id": instance.instance_id,
                "node_id": str(instance.node_id or ""),
                "expected_ep_size": expected_ep_size,
                "verified": False,
            }
            if instance.backend_instance is None:
                audit["reason"] = "backend_unavailable"
                audits.append(audit)
                continue
            try:
                runtime_metadata = await self._call_backend_method(
                    instance.backend_instance,
                    "get_runtime_metadata",
                    instance_id=instance.instance_id,
                    node_id=str(instance.node_id or ""),
                )
            except Exception as exc:
                audit.update(
                    reason="runtime_metadata_error", error=repr(exc)
                )
                audits.append(audit)
                continue
            if not isinstance(runtime_metadata, Mapping):
                audit["reason"] = "runtime_metadata_not_mapping"
                audits.append(audit)
                continue

            combined: Dict[str, Any] = {}
            profile = runtime_metadata.get("model_resource_profile")
            if isinstance(profile, Mapping):
                combined.update(profile)
            combined.update(runtime_metadata)
            evidence = {
                "worker_snapshots": combined.get(
                    "runtime_expert_placement_worker_snapshots", {}
                ),
                "placement_shards": combined.get(
                    "runtime_expert_placement_shards", {}
                ),
            }
            rank_rows = collect_rank_rows(evidence)
            observed_sizes = sorted(
                {
                    int(row["ep_size"])
                    for row in rank_rows
                    if row.get("ep_size") is not None
                }
            )
            observed_ranks = sorted(
                {
                    int(row["ep_rank"])
                    for row in rank_rows
                    if row.get("ep_rank") is not None
                }
            )
            worker_count = int(
                combined.get("runtime_expert_placement_worker_count", 0) or 0
            )
            runtime_available = bool(
                combined.get("runtime_expert_placement_available", False)
            )
            expected_ranks = list(range(expected_ep_size))
            verified = bool(
                runtime_available
                and worker_count == expected_ep_size
                and observed_sizes == [expected_ep_size]
                and observed_ranks == expected_ranks
            )
            if not runtime_available:
                reason = "runtime_expert_placement_unavailable"
            elif worker_count != expected_ep_size:
                reason = "runtime_ep_worker_count_mismatch"
            elif observed_sizes != [expected_ep_size]:
                reason = "runtime_ep_size_mismatch_or_missing"
            elif observed_ranks != expected_ranks:
                reason = "runtime_ep_rank_coverage_mismatch"
            else:
                reason = "runtime_ep_rank_and_size_verified"
            audit.update(
                verified=verified,
                reason=reason,
                runtime_worker_count=worker_count,
                observed_ep_sizes=observed_sizes,
                observed_ep_ranks=observed_ranks,
                rank_evidence_rows=len(rank_rows),
            )
            audits.append(audit)

        return {
            "verified": bool(audits)
            and all(bool(audit.get("verified")) for audit in audits),
            "expected_ep_size": expected_ep_size,
            "ready_instance_count": len(instances),
            "audited_instance_count": len(audits),
            "instances": audits,
        }

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
        if node_id is not None and str(node_id) not in instance.worker_node_ids():
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
            for target_node_id in instance.worker_node_ids():
                if target_node_id not in seen:
                    seen.add(target_node_id)
                    target_node_ids.append(target_node_id)
        return target_node_ids

    def _record_reparallelization_request_start(
        self, timestamp: float
    ) -> None:
        if not self.enable_reparallelization:
            return
        self.reparallelization_request_arrivals.append(float(timestamp))

    def _record_reparallelization_request_latency(
        self, latency_ms: float
    ) -> None:
        if not self.enable_reparallelization:
            return
        self.reparallelization_request_latencies_ms.append(
            max(0.0, float(latency_ms))
        )

    def _reparallelization_workload_snapshot(self) -> Dict[str, Any]:
        window_s = max(float(self.reparallelization_workload_window_s), 1.0)
        now = time.time()
        while (
            self.reparallelization_request_arrivals
            and now - self.reparallelization_request_arrivals[0] > window_s
        ):
            self.reparallelization_request_arrivals.popleft()

        arrivals = list(self.reparallelization_request_arrivals)
        if len(arrivals) >= 2:
            elapsed_s = max(arrivals[-1] - arrivals[0], 1e-6)
            arrival_rate_req_s = (len(arrivals) - 1) / elapsed_s
        elif arrivals:
            arrival_rate_req_s = 1.0 / window_s
        else:
            arrival_rate_req_s = 0.0

        latencies = list(self.reparallelization_request_latencies_ms)
        snapshot: Dict[str, Any] = {
            "window_s": window_s,
            "sample_count": len(arrivals),
            "latency_sample_count": len(latencies),
            "arrival_rate_req_s": arrival_rate_req_s,
            "batch_size": int(
                self.reparallelization_config.get("batch_size")
                or self.backend_config.get("max_num_seqs", 1)
                or 1
            ),
        }
        if latencies:
            snapshot["latency_estimate_ms"] = sum(latencies) / len(latencies)
        return snapshot

    def _reparallelization_planner_config(self) -> Dict[str, Any]:
        planner_config = dict(self.reparallelization_config)
        planner_config["runtime_workload"] = (
            self._reparallelization_workload_snapshot()
        )
        return planner_config

    async def _snapshot_reparallelization_worker_nodes(self):
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

        if self.reparallelization_worker_nodes:
            return {
                node_id: dict(node_info)
                for node_id, node_info in self.reparallelization_worker_nodes.items()
            }

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

    async def _reparallelization_runtime_metadata_snapshot(
        self,
        matches: List[InstanceHandle],
    ) -> Dict[str, Any]:
        timeout_s = max(
            0.1,
            float(
                self.reparallelization_config.get(
                    "runtime_metadata_timeout_s", 5.0
                )
                or 5.0
            ),
        )
        seen = set()
        instances = list(matches)
        async with self.instance_management_lock:
            instances.extend(self.ready_inference_instances.values())
            instances.extend(self.starting_inference_instances.values())
        for instance in instances:
            if instance.backend_instance is None:
                continue
            if instance.instance_id in seen:
                continue
            seen.add(instance.instance_id)
            try:
                metadata = await asyncio.wait_for(
                    self._call_backend_method(
                        instance.backend_instance,
                        "get_runtime_metadata",
                        instance_id=instance.instance_id,
                        node_id=instance.node_id or "",
                    ),
                    timeout=timeout_s,
                )
            except Exception:
                logger.info(
                    "Could not collect runtime metadata for replan from %s",
                    instance.instance_id,
                    exc_info=True,
                )
                continue
            if isinstance(metadata, Mapping) and metadata:
                return dict(metadata)
        return {}

    @staticmethod
    def _reparallelization_execution_model_fields(
        decision: Mapping[str, Any],
    ) -> Dict[str, Any]:
        action = str(decision.get("action") or "")
        parallel_plan = decision.get("parallel_plan")
        if not isinstance(parallel_plan, Mapping):
            parallel_plan = {}
        expert_plan = (
            decision.get("expert_placement_plan")
            or parallel_plan.get("expert_placement_plan")
            or {}
        )
        expert_plan_available = bool(
            decision.get(
                "expert_placement_plan_available",
                isinstance(expert_plan, Mapping) and bool(expert_plan),
            )
        )
        live_expert_remap = bool(
            isinstance(expert_plan, Mapping)
            and expert_plan.get("live_expert_remap", False)
        )
        physical_migration_required = bool(
            isinstance(expert_plan, Mapping)
            and expert_plan.get(
                "expert_placement_physical_migration_required",
                live_expert_remap,
            )
        )
        executed = action == "reparallelize"
        fields: Dict[str, Any] = {
            "reparallelization_execution_model": (
                "actor_recreate" if executed else "not_executed"
            ),
            "reparallelization_execution_model_reason": (
                "vllm_actor_recreate"
                if executed
                else f"{action or 'unknown'}_without_actor_recreate"
            ),
            "expert_placement_execution_model": (
                "quiescent_fixed_ep_remap"
                if executed and expert_plan_available and live_expert_remap
                else "expert_aware_actor_recreate"
                if executed and expert_plan_available
                else "no_expert_placement_plan"
                if executed
                else "not_executed"
            ),
            "expert_placement_execution_model_reason": (
                "opt_in_weight_transfer_after_actor_load"
                if executed and expert_plan_available and live_expert_remap
                else "logical_expert_placement_plan_carried_into_recreated_actor"
                if executed and expert_plan_available
                else "planner_did_not_emit_expert_placement_plan"
                if executed
                else f"{action or 'unknown'}_without_actor_recreate"
            ),
            "expert_placement_runtime_contract_mode": (
                "quiescent_fixed_ep_remap"
                if executed and expert_plan_available and live_expert_remap
                else "observe_only_contract"
                if executed and expert_plan_available
                else "unavailable"
            ),
            "expert_placement_live_migration_enabled": False,
            "expert_placement_quiescent_remap_enabled": bool(
                executed and live_expert_remap
            ),
            "expert_placement_physical_migration_required": bool(
                executed and physical_migration_required
            ),
        }
        return fields

    async def _replan_after_spot_event(
        self,
        event: str,
        node_id: Optional[str],
        instance_id: Optional[str],
        matches: List[InstanceHandle],
        worker_node_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
        spot_event_id: Optional[str] = None,
        event_state_marked_at_s: Optional[float] = None,
    ):
        # Notices/deadline revocation occur outside this lock. Only planning
        # and topology commits are serialized; each obtains a fresh snapshot.
        async with self._reparallelization_lock:
            return await self._replan_after_spot_event_serialized(
                event, node_id, instance_id, matches, worker_node_updates,
                spot_event_id, event_state_marked_at_s,
            )

    async def _replan_after_spot_event_serialized(
        self,
        event: str,
        node_id: Optional[str],
        instance_id: Optional[str],
        matches: List[InstanceHandle],
        worker_node_updates: Optional[Mapping[str, Mapping[str, Any]]] = None,
        spot_event_id: Optional[str] = None,
        event_state_marked_at_s: Optional[float] = None,
    ):
        if not self.enable_reparallelization:
            return None

        planner_invocation_id = f"planner-{uuid.uuid4().hex}"
        planner_started_at_s = time.time()
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
        if (
            event == "preempt"
            and bool(
                self.reparallelization_config.get(
                    "allow_preempting_target_recreate", False
                )
            )
            and bool(
                self.reparallelization_config.get(
                    "allow_stop_before_recreate", False
                )
            )
        ):
            # Single-worker benchmarks cannot move to another failure domain.
            # When explicitly requested, model this as stop-before-create on
            # the same worker node so the executable vLLM adapter targets a
            # real scheduler node instead of an unreachable synthetic target.
            for target_node_id in target_node_ids:
                node = worker_nodes.get(str(target_node_id))
                if node is None or node.get("state") != PREEMPTING:
                    continue
                node["state"] = READY
                node["preempting_target_recreate"] = True
                node["preempting_target_recreate_original_state"] = PREEMPTING
        if event == "add" and node_id is not None and not target_node_ids:
            worker_nodes = apply_spot_event_to_worker_nodes(
                worker_nodes,
                event,
                node_id,
                node_info=normalized_updates.get(node_id),
            )
        self.reparallelization_worker_nodes = worker_nodes
        runtime_metadata = (
            await self._reparallelization_runtime_metadata_snapshot(matches)
        )
        planner_backend_config = dict(self.backend_config)
        if runtime_metadata:
            planner_backend_config["runtime_metadata"] = dict(
                runtime_metadata
            )

        planner_config = self._reparallelization_planner_config()
        async with self.inflight_requests_lock:
            active_request_count = len(self.inflight_requests)
        runtime_metadata_fields = sorted(str(key) for key in runtime_metadata)
        planner_input_snapshot = {
            "spot_event_id": spot_event_id,
            "event": event,
            "node_id": node_id,
            "instance_id": instance_id,
            "worker_nodes": worker_nodes,
            "active_request_count": active_request_count,
            "queue_depth": self.request_queue.qsize(),
            "runtime_workload": planner_config.get("runtime_workload", {}),
            "runtime_metadata_fields": runtime_metadata_fields,
            "kv_snapshot_available": any(
                "kv" in field.lower() or "cache" in field.lower()
                for field in runtime_metadata_fields
            ),
            "route_snapshot_available": any(
                "route" in field.lower() or "expert_hotness" in field.lower()
                for field in runtime_metadata_fields
            ),
            "placement_snapshot_available": any(
                "placement" in field.lower()
                for field in runtime_metadata_fields
            ),
        }
        planner_input_snapshot_hash = hashlib.sha256(
            json.dumps(
                planner_input_snapshot,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        model_config = {
            "model": self.model_name,
            "backend": self.backend,
            "num_gpus": self.resource_requirements.get("num_gpus", 1),
            # Capability negotiation needs the concrete model/runtime
            # configuration.  Without this field the planner falls back to
            # the runtime-neutral candidate generator, which intentionally
            # uses runtime DP=1 and EP disabled, so it cannot select an
            # expert-parallel shape for a MoE model.
            "backend_config": planner_backend_config,
            "runtime_metadata": runtime_metadata,
        }
        measured_capability = planner_backend_config.get(
            "spotserve_backend_capability"
        )
        if isinstance(measured_capability, Mapping):
            # The K8s experiment driver only installs this field after every
            # advertised shape has passed an exact-workload GPU profile.  Keep
            # it outside AsyncEngineArgs while allowing the planner to consume
            # the measured latency/load/migration components verbatim.
            model_config["backend_capability"] = dict(measured_capability)
        decision = plan_dynamic_reparallelization(
            model_name=self.model_name,
            worker_nodes=worker_nodes,
            model_config=model_config,
            planner_config=planner_config,
            event=event,
            node_id=node_id,
            instance_id=instance_id,
            backend=self.backend,
        )
        planner_finished_at_s = time.time()
        decision.update({
            "spot_event_id": spot_event_id,
            "preemption_event_id": (
                spot_event_id if event == "preempt" else None
            ),
            "planner_invocation_id": planner_invocation_id,
            "event_state_marked_at_s": event_state_marked_at_s,
            "planner_started_at_s": planner_started_at_s,
            "planner_finished_at_s": planner_finished_at_s,
            "planner_input_snapshot_hash": planner_input_snapshot_hash,
            "planner_input_snapshot": planner_input_snapshot,
        })
        execution_model = self._reparallelization_execution_model_fields(
            decision
        )
        decision.update(execution_model)
        selected_plan = decision.get("parallel_plan")
        adapter_available = False
        if selected_plan is not None and self.backend == "vllm":
            adapter_available = self._ensure_vllm_reparallelization_adapter()
        active_deployment = (
            self._vllm_active_deployment() if adapter_available else None
        )
        selected_parallel_plan = (
            ParallelPlan.from_dict(selected_plan)
            if selected_plan is not None
            else None
        )
        elastic_ep_resize = bool(
            selected_parallel_plan is not None
            and active_deployment is not None
            and len(active_deployment.instances) == 1
            and active_deployment.instances
            and all(
                instance.state == InstanceState.READY
                for instance in active_deployment.instances.values()
            )
            and self.backend_config.get(
                "spotserve_elastic_ep_enabled", False
            )
            and self.backend_config.get(
                "spotserve_vllm_ray_dp_owns_gpus", False
            )
            and self.backend_config.get("data_parallel_backend") == "ray"
            and active_deployment.plan.tensor_parallel_size == 1
            and selected_parallel_plan.tensor_parallel_size == 1
            and active_deployment.plan.pipeline_parallel_size == 1
            and selected_parallel_plan.pipeline_parallel_size == 1
            and active_deployment.plan.replica_count == 1
            and selected_parallel_plan.replica_count == 1
            and active_deployment.plan.enable_expert_parallel
            and selected_parallel_plan.enable_expert_parallel
            and active_deployment.plan.data_parallel_size
            != selected_parallel_plan.data_parallel_size
            and (
                not active_deployment.plan.target_nodes
                or not selected_parallel_plan.target_nodes
                or sorted(active_deployment.plan.target_nodes)
                == sorted(selected_parallel_plan.target_nodes)
            )
        )
        live_in_place_remap = bool(
            selected_parallel_plan is not None
            and active_deployment is not None
            and active_deployment.instances
            and all(
                instance.state == InstanceState.READY
                for instance in active_deployment.instances.values()
            )
            and self._parallel_shapes_match(
                active_deployment.plan,
                selected_parallel_plan,
            )
            and isinstance(
                selected_parallel_plan.expert_placement_plan, Mapping
            )
            and selected_parallel_plan.expert_placement_plan.get(
                "live_expert_remap"
            )
            and not self._parallel_plans_match(
                active_deployment.plan,
                selected_parallel_plan,
            )
        )
        if (
            decision.get("action") == "reparallelize"
            and selected_parallel_plan is not None
            and active_deployment is not None
            and self._parallel_plans_match(
                active_deployment.plan,
                selected_parallel_plan,
            )
        ):
            # A capacity event does not automatically imply a deployment
            # change.  Keep the current engine and in-flight requests alive
            # when the newly selected plan is identical.
            decision["action"] = "unchanged"
            execution_model = self._reparallelization_execution_model_fields(
                decision
            )
            decision.update(execution_model)
            decision["execution"] = {
                "status": "unchanged",
                **execution_model,
                "parallel_plan": selected_plan,
                "reason": "planner_selected_existing_plan",
            }
        if decision.get("action") == "reparallelize":
            if adapter_available or self._ensure_vllm_reparallelization_adapter():
                apply_started_at = time.time()
                self._reparallelization_execution_active = True
                try:
                    plan = selected_parallel_plan or ParallelPlan.from_dict(
                        decision["parallel_plan"]
                    )
                    self.reparallelization_executor.current = (
                        self._vllm_active_deployment()
                    )
                    if elastic_ep_resize:
                        deployment = (
                            await self.vllm_deployment_adapter
                            .resize_workers_elastic_ep(
                                active_deployment,
                                plan,
                            )
                        )
                        await self._switch_vllm_deployment(deployment, plan)
                        execution_model.update({
                            "reparallelization_execution_model": (
                                "in_place_elastic_ep_resize"
                            ),
                            "reparallelization_execution_model_reason": (
                                "vllm_ray_dp_elastic_ep"
                            ),
                            "expert_placement_execution_model": (
                                "quiescent_elastic_ep_resize"
                            ),
                            "expert_placement_execution_model_reason": (
                                "vllm_reinitialized_ep_process_group"
                            ),
                            "expert_placement_runtime_contract_mode": (
                                "quiescent_elastic_ep_resize"
                            ),
                            "source_effective_expert_parallel_size": (
                                active_deployment.plan
                                .effective_expert_parallel_size
                            ),
                            "target_effective_expert_parallel_size": (
                                plan.effective_expert_parallel_size
                            ),
                            "dynamic_ep_resize": True,
                            "in_place_ep_resize": True,
                            "elastic_ep_admission_drained": bool(
                                deployment.backend_config.get(
                                    "elastic_ep_admission_drained", False
                                )
                            ),
                            "elastic_ep_drained_request_count": int(
                                deployment.backend_config.get(
                                    "elastic_ep_drained_request_count", 0
                                )
                                or 0
                            ),
                        })
                    elif live_in_place_remap:
                        deployment = (
                            await self.vllm_deployment_adapter.remap_workers_in_place(
                                active_deployment,
                                plan,
                            )
                        )
                        await self._switch_vllm_deployment(deployment, plan)
                        execution_model = (
                            self._reparallelization_execution_model_fields(
                                decision
                            )
                        )
                        execution_model.update({
                            "reparallelization_execution_model": (
                                "in_place_expert_remap"
                            ),
                            "reparallelization_execution_model_reason": (
                                "fixed_ep_runtime_expert_weight_remap"
                            ),
                            "expert_placement_execution_model": (
                                "active_fixed_ep_remap"
                                if (
                                    isinstance(
                                        plan.expert_placement_plan, Mapping
                                    )
                                    and plan.expert_placement_plan.get(
                                        "allow_active_requests"
                                    )
                                )
                                else "quiescent_fixed_ep_remap"
                            ),
                            "expert_placement_execution_model_reason": (
                                "runtime_apply_on_existing_actor"
                            ),
                            "expert_placement_runtime_contract_mode": (
                                "fixed_ep_runtime_remap"
                            ),
                        })
                    else:
                        deployment = await self.reparallelization_executor.apply(
                            plan
                        )
                        if (
                            active_deployment is not None
                            and active_deployment.plan.effective_expert_parallel_size
                            != plan.effective_expert_parallel_size
                        ):
                            execution_model.update({
                                "reparallelization_execution_model": (
                                    "actor_recreate_ep_resize"
                                ),
                                "reparallelization_execution_model_reason": (
                                    "ep_process_group_size_changed"
                                ),
                                "expert_placement_execution_model": (
                                    "expert_aware_actor_recreate_ep_resize"
                                ),
                                "expert_placement_execution_model_reason": (
                                    "vllm_ep_size_requires_engine_recreate"
                                ),
                                "expert_placement_runtime_contract_mode": (
                                    "actor_recreate_ep_resize"
                                ),
                                "source_effective_expert_parallel_size": (
                                    active_deployment.plan
                                    .effective_expert_parallel_size
                                ),
                                "target_effective_expert_parallel_size": (
                                    plan.effective_expert_parallel_size
                                ),
                                "dynamic_ep_resize": True,
                            })
                    expert_placement_runtime = (
                        await self._deployment_expert_placement_runtime_status(
                            deployment
                        )
                    )
                    decision["execution"] = {
                        "status": "applied",
                        **execution_model,
                        "instance_ids": sorted(deployment.instances),
                        "parallel_plan": plan.to_dict(),
                        "expert_placement_runtime": (
                            expert_placement_runtime
                        ),
                        "request_migration": getattr(
                            self.vllm_deployment_adapter,
                            "last_request_migration",
                            None,
                        ),
                        "transition": getattr(
                            self.reparallelization_executor,
                            "last_transition",
                            None,
                        ),
                        "duration_ms": (
                            time.time() - apply_started_at
                        )
                        * 1000,
                    }
                except Exception as exc:
                    logger.exception(
                        "Failed applying vLLM re-parallelization plan"
                    )
                    decision["execution"] = {
                        "status": "failed",
                        **execution_model,
                        "reason": str(exc),
                        "duration_ms": (
                            time.time() - apply_started_at
                        )
                        * 1000,
                    }
                finally:
                    self._reparallelization_execution_active = False
            else:
                not_executed_model = {
                    "reparallelization_execution_model": "not_executed",
                    "reparallelization_execution_model_reason": (
                        "vllm_deployment_adapter_unavailable"
                    ),
                    "expert_placement_execution_model": "not_executed",
                    "expert_placement_execution_model_reason": (
                        "vllm_deployment_adapter_unavailable"
                    ),
                    "expert_placement_runtime_contract_mode": "unavailable",
                    "expert_placement_live_migration_enabled": False,
                    "expert_placement_physical_migration_required": False,
                }
                decision["execution"] = {
                    "status": "decision_only",
                    **not_executed_model,
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
    def _parallel_plan_placement_fingerprint(plan: ParallelPlan) -> str:
        expert_placement_plan = plan.expert_placement_plan
        if not isinstance(expert_placement_plan, Mapping):
            return ""
        fingerprint = str(
            expert_placement_plan.get("placement_fingerprint") or ""
        )
        if fingerprint:
            return fingerprint
        return "unfingerprinted:" + json.dumps(
            expert_placement_plan,
            sort_keys=True,
            default=str,
        )

    @classmethod
    def _parallel_shapes_match(
        cls,
        left: ParallelPlan, right: ParallelPlan
    ) -> bool:
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

    @classmethod
    def _parallel_plans_match(
        cls,
        left: ParallelPlan, right: ParallelPlan
    ) -> bool:
        """Compare deployment shape and placement, ignoring replan reason."""
        left_placement_fingerprint = cls._parallel_plan_placement_fingerprint(
            left
        )
        right_placement_fingerprint = cls._parallel_plan_placement_fingerprint(
            right
        )
        if left_placement_fingerprint or right_placement_fingerprint:
            if left_placement_fingerprint != right_placement_fingerprint:
                return False
        return cls._parallel_shapes_match(left, right)

    async def _set_instance_state(
        self,
        instance: InstanceHandle,
        state: InstanceState,
        reason: str,
        metadata: Optional[Mapping[str, Any]] = None,
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
                **dict(metadata or {}),
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

    async def _context_migration_target_metadata(
        self,
        instance: InstanceHandle,
    ) -> Dict[str, Any]:
        if instance.backend_instance is None:
            return {}
        try:
            runtime_metadata = await self._call_backend_method(
                instance.backend_instance,
                "get_runtime_metadata",
                instance_id=instance.instance_id,
                node_id=str(instance.node_id or ""),
            )
        except Exception:
            logger.info(
                "Could not query target MoE metadata from %s",
                instance.instance_id,
                exc_info=True,
            )
            return {}
        if not isinstance(runtime_metadata, dict):
            return {}
        profile = runtime_metadata.get("model_resource_profile")
        metadata: Dict[str, Any] = {}
        if isinstance(profile, dict):
            metadata.update(profile)
        metadata.update(runtime_metadata)
        return {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "expert_placement_available",
                "expert_placement_snapshot",
                "runtime_expert_placement_available",
                "runtime_expert_placement_worker_count",
                "runtime_expert_placement_shard_count",
                "runtime_expert_placement_shards",
                "runtime_expert_placement_worker_snapshots",
                "placement_epoch",
                "placement_version",
                "placement_source",
                "expert_placement_epoch",
                "expert_placement_fingerprint",
                "expert_placement_plan_fingerprint",
                "expert_placement_snapshot_fingerprint",
                "expert_placement_contract_available",
                "expert_placement_contract_bound",
                "expert_placement_contract_source",
                "expert_placement_contract_epoch",
                "expert_placement_contract_fingerprint",
                "expert_placement_contract_snapshot_fingerprint",
                "expert_placement_contract_snapshot_match",
                "expert_placement_plan_applied",
                "expert_placement_plan_verified",
                "expert_placement_contract_reason",
                "expert_placement_apply_hook_available",
                "expert_placement_apply_attempted",
                "expert_placement_apply_success",
                "expert_placement_apply_worker_count",
                "expert_placement_apply_worker_success_count",
                "expert_placement_apply_duration_ms",
                "expert_placement_apply_reason",
                "expert_placement_verify_hook_available",
                "expert_placement_verify_attempted",
                "expert_placement_verify_success",
                "expert_placement_verify_worker_count",
                "expert_placement_verify_worker_success_count",
                "expert_placement_verify_reason",
                "expert_placement_contract_seen_by_runtime",
                "expert_placement_contract_seen_by_all_workers",
                "expert_placement_contract_seen_worker_count",
                "expert_placement_contract_seen_worker_total",
                "expert_placement_physical_weight_migration",
                "expert_placement_runtime_moved_expert_shards",
                "expert_placement_runtime_moved_weight_bytes",
                "expert_placement_runtime_remap_duration_ms",
                "expert_placement_runtime_active_request_remap",
                "expert_placement_runtime_step_boundary_barrier",
                "expert_placement_runtime_physical_host_ids_observed",
                "expert_placement_runtime_cross_node_weight_migration",
                "expert_placement_runtime_cross_node_moved_expert_shards",
                "expert_placement_runtime_cross_node_moved_weight_bytes",
                "expert_placement_runtime_verification_level",
                "expert_placement_runtime_verified_placement",
                "expert_placement_runtime_can_verify_physical_placement",
                "expert_placement_runtime_can_remap_live_ep_rank",
                "expert_placement_runtime_can_measure_all_to_all",
                "expert_placement_runtime_all_to_all_counters_available",
                "expert_placement_runtime_all_to_all_collective_calls",
                "expert_placement_runtime_all_to_all_dispatch_calls",
                "expert_placement_runtime_all_to_all_combine_calls",
                "expert_placement_runtime_all_to_all_observed_input_bytes",
                "expert_placement_runtime_all_to_all_observed_output_bytes",
                "expert_placement_runtime_all_to_all_internode_calls",
                "expert_placement_runtime_all_to_all_measurement_kind",
                "expert_placement_runtime_capability_reason",
                "reparallelization_execution_model",
                "reparallelization_execution_model_reason",
                "expert_placement_execution_model",
                "expert_placement_execution_model_reason",
                "expert_placement_runtime_contract_mode",
                "expert_placement_live_migration_enabled",
                "expert_placement_quiescent_remap_enabled",
                "expert_placement_physical_migration_required",
                "moe_route_histogram_available",
                "moe_route_histogram_source",
                "moe_route_histogram_kind",
                "effective_expert_parallel_size",
                "runtime_effective_expert_parallel_size",
                "expert_parallel_size_source",
                "expert_physical_replication_factor",
                "vllm_data_parallel_size",
                "sllm_replica_count",
            }
        }

    async def _deployment_expert_placement_runtime_status(
        self,
        deployment,
    ) -> Dict[str, Any]:
        metadata_rows: List[Dict[str, Any]] = []
        for handle in getattr(deployment, "instances", {}).values():
            metadata = await self._context_migration_target_metadata(handle)
            if metadata:
                metadata_rows.append(metadata)

        def count_truthy(key: str) -> int:
            return sum(1 for row in metadata_rows if row.get(key))

        def compact_values(key: str) -> str:
            values = sorted(
                {
                    str(row.get(key))
                    for row in metadata_rows
                    if row.get(key) not in (None, "")
                }
            )
            return ",".join(values)

        def sum_int(key: str) -> int:
            return sum(int(row.get(key, 0) or 0) for row in metadata_rows)

        def max_float(key: str) -> float:
            return max(
                (float(row.get(key, 0.0) or 0.0) for row in metadata_rows),
                default=0.0,
            )

        return {
            "metadata_count": len(metadata_rows),
            "apply_hook_available_count": count_truthy(
                "expert_placement_apply_hook_available"
            ),
            "apply_attempted_count": count_truthy(
                "expert_placement_apply_attempted"
            ),
            "apply_success_count": count_truthy(
                "expert_placement_apply_success"
            ),
            "apply_reasons": compact_values(
                "expert_placement_apply_reason"
            ),
            "verify_hook_available_count": count_truthy(
                "expert_placement_verify_hook_available"
            ),
            "verify_attempted_count": count_truthy(
                "expert_placement_verify_attempted"
            ),
            "verify_success_count": count_truthy(
                "expert_placement_verify_success"
            ),
            "verify_reasons": compact_values(
                "expert_placement_verify_reason"
            ),
            "contract_seen_count": count_truthy(
                "expert_placement_contract_seen_by_runtime"
            ),
            "contract_seen_all_workers_count": count_truthy(
                "expert_placement_contract_seen_by_all_workers"
            ),
            "contract_seen_worker_count": sum(
                int(row.get("expert_placement_contract_seen_worker_count", 0) or 0)
                for row in metadata_rows
            ),
            "contract_seen_worker_total": sum(
                int(row.get("expert_placement_contract_seen_worker_total", 0) or 0)
                for row in metadata_rows
            ),
            "physical_weight_migration_count": count_truthy(
                "expert_placement_physical_weight_migration"
            ),
            "runtime_moved_expert_shards": sum_int(
                "expert_placement_runtime_moved_expert_shards"
            ),
            "runtime_moved_weight_bytes": sum_int(
                "expert_placement_runtime_moved_weight_bytes"
            ),
            "runtime_remap_duration_ms": max_float(
                "expert_placement_runtime_remap_duration_ms"
            ),
            "active_request_remap_count": count_truthy(
                "expert_placement_runtime_active_request_remap"
            ),
            "step_boundary_barrier_count": count_truthy(
                "expert_placement_runtime_step_boundary_barrier"
            ),
            "physical_host_ids_observed_count": count_truthy(
                "expert_placement_runtime_physical_host_ids_observed"
            ),
            "cross_node_weight_migration_count": count_truthy(
                "expert_placement_runtime_cross_node_weight_migration"
            ),
            "cross_node_moved_expert_shards": sum_int(
                "expert_placement_runtime_cross_node_moved_expert_shards"
            ),
            "cross_node_moved_weight_bytes": sum_int(
                "expert_placement_runtime_cross_node_moved_weight_bytes"
            ),
            "verification_levels": compact_values(
                "expert_placement_runtime_verification_level"
            ),
            "verified_placement_count": count_truthy(
                "expert_placement_runtime_verified_placement"
            ),
            "can_verify_physical_placement_count": count_truthy(
                "expert_placement_runtime_can_verify_physical_placement"
            ),
            "can_remap_live_ep_rank_count": count_truthy(
                "expert_placement_runtime_can_remap_live_ep_rank"
            ),
            "can_measure_all_to_all_count": count_truthy(
                "expert_placement_runtime_can_measure_all_to_all"
            ),
            "all_to_all_counters_available_count": count_truthy(
                "expert_placement_runtime_all_to_all_counters_available"
            ),
            "all_to_all_collective_calls": sum_int(
                "expert_placement_runtime_all_to_all_collective_calls"
            ),
            "all_to_all_dispatch_calls": sum_int(
                "expert_placement_runtime_all_to_all_dispatch_calls"
            ),
            "all_to_all_combine_calls": sum_int(
                "expert_placement_runtime_all_to_all_combine_calls"
            ),
            "all_to_all_observed_input_bytes": sum_int(
                "expert_placement_runtime_all_to_all_observed_input_bytes"
            ),
            "all_to_all_observed_output_bytes": sum_int(
                "expert_placement_runtime_all_to_all_observed_output_bytes"
            ),
            "all_to_all_internode_calls": sum_int(
                "expert_placement_runtime_all_to_all_internode_calls"
            ),
            "all_to_all_measurement_kinds": compact_values(
                "expert_placement_runtime_all_to_all_measurement_kind"
            ),
            "runtime_expert_placement_available_count": count_truthy(
                "runtime_expert_placement_available"
            ),
            "runtime_expert_placement_worker_count": sum_int(
                "runtime_expert_placement_worker_count"
            ),
            "runtime_expert_placement_shard_count": sum_int(
                "runtime_expert_placement_shard_count"
            ),
            "capability_reasons": compact_values(
                "expert_placement_runtime_capability_reason"
            ),
            "plan_applied_count": count_truthy(
                "expert_placement_plan_applied"
            ),
            "plan_verified_count": count_truthy(
                "expert_placement_plan_verified"
            ),
            "contract_reasons": compact_values(
                "expert_placement_contract_reason"
            ),
            "reparallelization_execution_models": compact_values(
                "reparallelization_execution_model"
            ),
            "expert_placement_execution_models": compact_values(
                "expert_placement_execution_model"
            ),
            "expert_placement_execution_reasons": compact_values(
                "expert_placement_execution_model_reason"
            ),
            "expert_placement_contract_modes": compact_values(
                "expert_placement_runtime_contract_mode"
            ),
            "expert_placement_live_migration_count": count_truthy(
                "expert_placement_live_migration_enabled"
            ),
            "expert_placement_quiescent_remap_count": count_truthy(
                "expert_placement_quiescent_remap_enabled"
            ),
            "expert_placement_physical_migration_required_count": (
                count_truthy("expert_placement_physical_migration_required")
            ),
        }

    @staticmethod
    def _placement_marker_value(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> Optional[str]:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value)
            if text:
                return text
        return None

    @classmethod
    def _placement_marker_from_payload(
        cls,
        payload: Optional[Mapping[str, Any]],
    ) -> Dict[str, Optional[str]]:
        payload = payload or {}
        return {
            "placement_epoch": cls._placement_marker_value(
                payload,
                "target_placement_epoch",
                "placement_epoch",
                "target_placement_version",
                "placement_version",
                "target_expert_placement_epoch",
                "expert_placement_epoch",
            ),
            "expert_placement_fingerprint": cls._placement_marker_value(
                payload,
                "target_expert_placement_fingerprint",
                "expert_placement_fingerprint",
            ),
            "placement_source": cls._placement_marker_value(
                payload,
                "target_placement_source",
                "placement_source",
            ),
        }

    @classmethod
    def _compare_placement_markers(
        cls,
        expected_payload: Optional[Mapping[str, Any]],
        current_payload: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        expected = cls._placement_marker_from_payload(expected_payload)
        current = cls._placement_marker_from_payload(current_payload)
        expected_epoch = expected.get("placement_epoch")
        expected_fingerprint = expected.get("expert_placement_fingerprint")
        if expected_epoch is None and expected_fingerprint is None:
            return {
                "attempted": False,
                "verified": False,
                "compatible": True,
                "stale": False,
                "reason": "expected_placement_marker_unavailable",
                "mismatch_keys": [],
                "expected": expected,
                "current": current,
            }

        missing_keys = []
        mismatch_keys = []
        for key in ("placement_epoch", "expert_placement_fingerprint"):
            expected_value = expected.get(key)
            if expected_value is None:
                continue
            current_value = current.get(key)
            if current_value is None:
                missing_keys.append(key)
            elif current_value != expected_value:
                mismatch_keys.append(key)

        if mismatch_keys or missing_keys:
            return {
                "attempted": True,
                "verified": not missing_keys,
                "compatible": False,
                "stale": bool(mismatch_keys),
                "reason": (
                    "target_placement_changed"
                    if mismatch_keys
                    else "current_placement_marker_unavailable"
                ),
                "mismatch_keys": mismatch_keys,
                "missing_keys": missing_keys,
                "expected": expected,
                "current": current,
            }

        return {
            "attempted": True,
            "verified": True,
            "compatible": True,
            "stale": False,
            "reason": "stable_target_placement",
            "mismatch_keys": [],
            "expected": expected,
            "current": current,
        }

    async def _verify_target_placement_handshake(
        self,
        expected_payload: Optional[Mapping[str, Any]],
        target: InstanceHandle,
        operation: str,
    ) -> Dict[str, Any]:
        expected = self._placement_marker_from_payload(expected_payload)
        if (
            expected.get("placement_epoch") is None
            and expected.get("expert_placement_fingerprint") is None
        ):
            check = self._compare_placement_markers(expected_payload, {})
        else:
            current_metadata = await self._context_migration_target_metadata(
                target
            )
            check = self._compare_placement_markers(
                expected_payload,
                current_metadata,
            )
        check.update(
            {
                "operation": operation,
                "target_instance_id": target.instance_id,
                "target_node_id": str(target.node_id or ""),
            }
        )
        return check

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
                    concurrency=max(0, int(instance.concurrency)),
                    max_queue_length=max(0, int(instance.max_queue_length)),
                    queue_depth=max(0, int(instance.concurrency)),
                    metadata=await self._context_migration_target_metadata(
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
        placement_handshake_checks = []
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

            placement_check = await self._verify_target_placement_handshake(
                plan,
                target,
                operation="context_migration_prefix_warmup",
            )
            placement_handshake_checks.append(placement_check)
            if not placement_check.get("compatible", True):
                skipped.append(
                    {
                        "source_instance_id": source_id,
                        "target_instance_id": target_id,
                        "reason": placement_check.get(
                            "reason", "target_placement_not_stable"
                        ),
                        "placement_handshake": placement_check,
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
            "placement_handshake_checks": placement_handshake_checks,
            "placement_handshake_attempts": sum(
                1
                for check in placement_handshake_checks
                if check.get("attempted")
            ),
            "placement_handshake_successes": sum(
                1
                for check in placement_handshake_checks
                if check.get("attempted") and check.get("compatible")
            ),
            "placement_handshake_failures": sum(
                1
                for check in placement_handshake_checks
                if check.get("attempted") and not check.get("compatible")
            ),
            "placement_handshake_stale": sum(
                1
                for check in placement_handshake_checks
                if check.get("stale")
            ),
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
        decision["context_source_count"] = len(sources)
        decision["context_target_count"] = len(targets)
        decision["candidate_component_costs_enabled"] = (
            self.emit_context_migration_candidate_costs
        )
        if self.emit_context_migration_candidate_costs:
            decision["candidate_component_costs"] = (
                build_candidate_component_costs(
                    sources=sources,
                    targets=targets,
                    planner_config=planner_config,
                )
            )
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
        notice_time_s: Optional[float] = None,
        deadline_time_s: Optional[float] = None,
        trace_event_time_s: Optional[float] = None,
        trace_deadline_time_s: Optional[float] = None,
        grace_period_s: Optional[float] = None,
    ):
        spot_event_id = f"preempt-{uuid.uuid4().hex}"
        matches = await self._matching_inference_instances(
            node_id=node_id, instance_id=instance_id
        )
        marked_instances = []
        timing_metadata = {
            "spot_event_id": spot_event_id,
            "preemption_event_id": spot_event_id,
            "notice_time_s": notice_time_s,
            "deadline_time_s": deadline_time_s,
            "trace_event_time_s": trace_event_time_s,
            "trace_deadline_time_s": trace_deadline_time_s,
            "grace_period_s": grace_period_s,
        }
        for instance in matches:
            from_state = instance.state.value
            if deadline_time_s is not None:
                instance.preemption_deadline_time_s = float(deadline_time_s)
            await instance.mark_preempting()
            token_batches = await self._capture_current_tokens(instance)
            prompt_tokens = max(
                0,
                int(
                    self.backend_config.get(
                        "spotserve_expected_prompt_tokens", 0
                    )
                    or 0
                ),
            )
            generated_tokens = [
                max(0, len(tokens) - prompt_tokens)
                for tokens in token_batches
            ]
            progress_min = int(
                self.backend_config.get(
                    "spotserve_preemption_progress_min_tokens", 0
                )
                or 0
            )
            progress_max = int(
                self.backend_config.get(
                    "spotserve_preemption_progress_max_tokens", 0
                )
                or 0
            )
            progress_metadata = {
                **timing_metadata,
                "preemption_progress_observed": bool(token_batches),
                "preemption_expected_prompt_tokens": prompt_tokens,
                "preemption_inflight_request_count": len(token_batches),
                "preemption_generated_tokens_per_request": generated_tokens,
                "preemption_generated_tokens_min": (
                    min(generated_tokens) if generated_tokens else 0
                ),
                "preemption_generated_tokens_max": (
                    max(generated_tokens) if generated_tokens else 0
                ),
                "preemption_generated_tokens_avg": (
                    sum(generated_tokens) / len(generated_tokens)
                    if generated_tokens
                    else 0.0
                ),
                "preemption_progress_window_min_tokens": progress_min,
                "preemption_progress_window_max_tokens": progress_max,
                "preemption_requests_in_progress_window": sum(
                    1
                    for tokens in generated_tokens
                    if progress_min <= tokens <= progress_max
                ),
            }
            self._emit_metric(
                make_instance_state_event(
                    model=self.model_name,
                    instance_id=instance.instance_id,
                    node_id=instance.node_id,
                    from_state=from_state,
                    to_state=instance.state.value,
                    reason="trace_event",
                    **progress_metadata,
                )
            )
            marked_instances.append(instance.instance_id)
            logger.info(
                f"Marked instance {instance.instance_id} as preempting "
                f"for model {self.model_name}"
            )
        event_state_marked_at_s = time.time()
        replanning = await self._replan_after_spot_event(
            event="preempt",
            node_id=node_id,
            instance_id=instance_id,
            matches=matches,
            spot_event_id=spot_event_id,
            event_state_marked_at_s=event_state_marked_at_s,
        )
        context_migration = await self._plan_context_migration_after_spot_event(
            event="preempt",
            matches=matches,
        )
        return {
            "model": self.model_name,
            "event": "preempt",
            "spot_event_id": spot_event_id,
            "preemption_event_id": spot_event_id,
            "instances": marked_instances,
            "notice_time_s": notice_time_s,
            "deadline_time_s": deadline_time_s,
            "trace_event_time_s": trace_event_time_s,
            "trace_deadline_time_s": trace_deadline_time_s,
            "grace_period_s": grace_period_s,
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
        spot_event_id = f"add-{uuid.uuid4().hex}"
        updates = {
            normalized_node_id: {"state": READY, **dict(node_info or {})}
        }
        event_state_marked_at_s = time.time()
        replanning = await self._replan_after_spot_event(
            event="add",
            node_id=normalized_node_id,
            instance_id=None,
            matches=[],
            worker_node_updates=updates,
            spot_event_id=spot_event_id,
            event_state_marked_at_s=event_state_marked_at_s,
        )
        return {
            "model": self.model_name,
            "event": "add",
            "spot_event_id": spot_event_id,
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
        notice_time_s: Optional[float] = None,
        deadline_time_s: Optional[float] = None,
        trace_event_time_s: Optional[float] = None,
        trace_deadline_time_s: Optional[float] = None,
        grace_period_s: Optional[float] = None,
        auto_deadline: bool = False,
    ):
        matches = await self._matching_inference_instances(
            node_id=node_id, instance_id=instance_id
        )
        marked_instances = []
        timing_metadata = {
            "notice_time_s": notice_time_s,
            "deadline_time_s": deadline_time_s,
            "trace_event_time_s": trace_event_time_s,
            "trace_deadline_time_s": trace_deadline_time_s,
            "grace_period_s": grace_period_s,
            "auto_deadline": auto_deadline,
        }
        for instance in matches:
            if not auto_deadline:
                await self._capture_current_tokens(instance)
            await self._set_instance_state(
                instance,
                InstanceState.DEAD,
                reason="trace_dead",
                metadata=timing_metadata,
            )
            if auto_deadline:
                # Revoke source-owned KV; do not leave it readable after the
                # simulated provider deadline or delete the Kubernetes pod.
                actor = instance.backend_instance
                if actor is not None and hasattr(actor, "_ray_actor_id"):
                    ray.kill(actor, no_restart=True)
                elif actor is not None and hasattr(actor, "shutdown"):
                    await self._call_backend_method(actor, "shutdown")
                instance.backend_instance = None
                if self.backend != "dummy":
                    await self.model_loading_scheduler.deallocate_resource.remote(
                        self.model_name, instance.instance_id,
                        {**self.resource_requirements, "num_gpus": instance.num_gpu},
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
            "notice_time_s": notice_time_s,
            "deadline_time_s": deadline_time_s,
            "trace_event_time_s": trace_event_time_s,
            "trace_deadline_time_s": trace_deadline_time_s,
            "grace_period_s": grace_period_s,
            "auto_deadline": auto_deadline,
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
        if self.backend == "vllm" and completed_tokens:
            if completed_tokens > len(state.tokens):
                raise NativeKVRecoveryError("committed_output_exceeds_visible_tokens")
            restore_request["_spotserve_committed_output_tokens"] = list(
                state.tokens[-completed_tokens:]
            )
            restore_request["_spotserve_original_prompt_tokens"] = len(state.tokens) - completed_tokens
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
            planner_config=self.stateful_recovery_config,
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
        Dict[str, Any],
        bool,
    ]:
        counters = {
            "state_restore_attempts": 0,
            "state_restore_successes": 0,
            "state_restored_tokens": 0,
            "supports_state_restore": False,
            "state_kind": "",
            "state_restore_reason": "",
            "state_restored_blocks": 0,
            "state_restore_staged": False,
            "state_restore_started_at_s": 0.0,
            "state_restore_finished_at_s": 0.0,
            "state_restore_duration_ms": 0.0,
        }
        if state is None:
            if self.require_native_kv_restore:
                raise NativeKVRecoveryError("native_kv_snapshot_missing")
            return None, None, counters, False
        if self.require_native_kv_restore and (
            state.state_kind != "vllm_kv_snapshot" or not state.supports_restore
        ):
            raise NativeKVRecoveryError("native_kv_snapshot_not_restorable")
        counters["state_kind"] = state.state_kind
        counters["supports_state_restore"] = bool(state.supports_restore)

        placement_handshake = None
        if isinstance(target_selection, Mapping):
            selected_candidate = target_selection.get("selected_candidate")
            if isinstance(selected_candidate, Mapping):
                placement_handshake = (
                    await self._verify_target_placement_handshake(
                        selected_candidate,
                        target_instance,
                        operation="stateful_recovery_restore",
                    )
                )
                updated_selection = dict(target_selection)
                updated_candidate = dict(selected_candidate)
                updated_candidate.update(
                    {
                        "placement_handshake_attempted": (
                            placement_handshake.get("attempted", False)
                        ),
                        "placement_handshake_verified": (
                            placement_handshake.get("verified", False)
                        ),
                        "placement_handshake_compatible": (
                            placement_handshake.get("compatible", True)
                        ),
                        "placement_handshake_stale": (
                            placement_handshake.get("stale", False)
                        ),
                        "placement_handshake_reason": (
                            placement_handshake.get("reason", "")
                        ),
                        "placement_handshake_mismatch_keys": (
                            placement_handshake.get("mismatch_keys", [])
                        ),
                        "current_placement_epoch": (
                            placement_handshake.get("current", {}).get(
                                "placement_epoch"
                            )
                        ),
                        "current_expert_placement_fingerprint": (
                            placement_handshake.get("current", {}).get(
                                "expert_placement_fingerprint"
                            )
                        ),
                    }
                )
                updated_selection["selected_candidate"] = updated_candidate
                updated_selection["placement_handshake"] = placement_handshake
                if not placement_handshake.get("compatible", True):
                    updated_selection["action"] = "fallback_token_replay"
                    updated_selection["reason"] = placement_handshake.get(
                        "reason", "target_placement_not_stable"
                    )
                target_selection = updated_selection

        planner_allows_restore = (
            target_selection is None
            or target_selection.get("action") == "restore_state"
        )
        restore_supported = planner_allows_restore and await (
            self._backend_supports_state_restore(target_instance)
        )
        counters["supports_state_restore"] = bool(restore_supported)
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
            target_selection=target_selection,
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
            restore_started_at = time.time()
            counters["state_restore_started_at_s"] = restore_started_at
            restore_result = await self._restore_inference_state(
                target_instance, state, request_data
            )
            restore_finished_at = time.time()
            counters["state_restore_finished_at_s"] = restore_finished_at
            counters["state_restore_duration_ms"] = (
                restore_finished_at - restore_started_at
            ) * 1000
            try:
                restored_blocks = int(
                    restore_result.get("restored_blocks", 0) or 0
                )
            except (TypeError, ValueError):
                restored_blocks = 0
            counters["state_restore_reason"] = str(
                restore_result.get("reason")
                or (
                    "state_restore_staged"
                    if restore_result.get("staged")
                    else decision.reason
                )
            )
            counters["state_restored_blocks"] = (
                restored_blocks if restore_result.get("restored") else 0
            )
            counters["state_restore_staged"] = bool(
                restore_result.get("staged", False)
            )
            if restore_result.get("staged"):
                try:
                    expected_blocks = int(
                        restore_result.get("expected_blocks", 0) or 0
                    )
                except (TypeError, ValueError):
                    expected_blocks = 0
                counters["state_restore_expected_blocks"] = expected_blocks
            if restore_result.get("restored"):
                if state.state_kind == "vllm_kv_snapshot" and restored_blocks <= 0:
                    counters["state_restore_reason"] = (
                        "vllm_kv_restore_empty"
                    )
                    if self.require_native_kv_restore:
                        raise NativeKVRecoveryError("native_kv_restore_empty")
                    return None, [list(state.tokens)], counters, True
                counters["state_restore_successes"] = 1
                counters["state_restored_tokens"] = decision.recovered_tokens
                return state, None, counters, False
            if restore_result.get("staged"):
                if self.require_native_kv_restore and int(
                    restore_result.get("expected_blocks", 0) or 0
                ) <= 0:
                    raise NativeKVRecoveryError("native_kv_restore_staging_empty")
                return state, None, counters, False

            counters["state_restore_reason"] = str(
                restore_result.get("reason")
                or decision.reason
            )

        if self.require_native_kv_restore:
            raise NativeKVRecoveryError(
                f"native_kv_restore_required:{counters['state_restore_reason'] or decision.reason}"
            )
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
        self._record_reparallelization_request_start(request_start)

        def finish_request_latency_ms() -> float:
            latency_ms = (time.time() - request_start) * 1000
            self._record_reparallelization_request_latency(latency_ms)
            return latency_ms

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
        supports_state_restore = False
        state_kind = ""
        state_restore_reason = ""
        state_restored_blocks = 0
        state_restore_staged = False
        state_restore_started_at_s = 0.0
        state_restore_finished_at_s = 0.0
        state_restore_duration_ms = 0.0
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
                        supports_state_restore = bool(
                            state_counters.get(
                                "supports_state_restore",
                                supports_state_restore,
                            )
                        )
                        state_kind = str(
                            state_counters.get("state_kind", state_kind) or ""
                        )
                        state_restore_reason = str(
                            state_counters.get(
                                "state_restore_reason",
                                state_restore_reason,
                            )
                            or ""
                        )
                        state_restored_blocks += int(
                            state_counters.get("state_restored_blocks", 0)
                            or 0
                        )
                        state_restore_staged = (
                            state_restore_staged
                            or bool(
                                state_counters.get(
                                    "state_restore_staged", False
                                )
                            )
                        )
                        if state_counters.get("state_restore_started_at_s"):
                            if state_restore_started_at_s == 0.0:
                                state_restore_started_at_s = float(
                                    state_counters[
                                        "state_restore_started_at_s"
                                    ]
                                )
                        if state_counters.get("state_restore_finished_at_s"):
                            state_restore_finished_at_s = float(
                                state_counters[
                                    "state_restore_finished_at_s"
                                ]
                            )
                        state_restore_duration_ms += float(
                            state_counters.get(
                                "state_restore_duration_ms", 0.0
                            )
                            or 0.0
                        )
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
                        try:
                            kv_restored_blocks = int(
                                kv_restore.get("restored_blocks", 0) or 0
                            )
                        except (TypeError, ValueError):
                            kv_restored_blocks = 0
                        if not kv_restore.get("restored"):
                            kv_restored_blocks = 0
                        state_restored_blocks += kv_restored_blocks
                        state_restore_reason = str(
                            kv_restore.get("reason")
                            or state_restore_reason
                            or ""
                        )
                        state_kind = state_snapshot.state_kind or state_kind
                        if kv_restore.get("restored") and kv_restored_blocks > 0:
                            if int(kv_restore.get("cached_tokens", 0) or 0) > 0:
                                receipt = {
                                    "type": "native_kv_receipt", "model": self.model_name,
                                    "request_id": request_id,
                                    "source_instance_id": recovery_state_source_instance_id,
                                    "target_instance_id": instance_id,
                                    "target_node_id": instance.node_id,
                                    "cached_tokens": int(kv_restore["cached_tokens"]),
                                    "restored_blocks": kv_restored_blocks,
                                    "receipt_kind": "completed_native_request",
                                }
                                self._emit_metric(receipt)
                                if self.require_native_kv_restore:
                                    self.native_kv_receipts[request_id] = receipt
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
                            if self.require_native_kv_restore:
                                raise NativeKVRecoveryError(
                                    "native_kv_attach_failed:no_cached_prefix"
                                )
                    if self.require_native_kv_restore and state_snapshot is not None and (
                        not kv_restore.get("restored")
                        or int(kv_restore.get("cached_tokens", 0) or 0) <= 0
                    ):
                        raise NativeKVRecoveryError("native_kv_receipt_required")
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
                                latency_ms=finish_request_latency_ms(),
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
                                supports_state_restore=(
                                    supports_state_restore
                                ),
                                state_kind=state_kind,
                                state_restore_reason=state_restore_reason,
                                state_restored_blocks=state_restored_blocks,
                                state_restore_staged=state_restore_staged,
                                state_restore_started_at_s=(
                                    state_restore_started_at_s
                                ),
                                state_restore_finished_at_s=(
                                    state_restore_finished_at_s
                                ),
                                state_restore_duration_ms=(
                                    state_restore_duration_ms
                                ),
                            )
                        )
                        return result

                    failed_attempts += 1
                    if reparallelized:
                        migration_state = inflight.get("migration_state")
                        recovery_state = migration_state
                        recovery_state_source_instance_id = instance_id
                        force_state_recovery = (
                            migration_state is not None
                            and self.recovery_policy == RecoveryPolicy.STATEFUL_RECOVERY
                        )
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

                    if self.require_native_kv_restore and (
                        recovery_state is None or not recovery_state.supports_restore
                        or recovery_state.state_kind != "vllm_kv_snapshot"
                    ):
                        raise NativeKVRecoveryError("native_kv_snapshot_required_for_retry")
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
                                latency_ms=finish_request_latency_ms(),
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
                                supports_state_restore=(
                                    supports_state_restore
                                ),
                                state_kind=state_kind,
                                state_restore_reason=state_restore_reason,
                                state_restored_blocks=state_restored_blocks,
                                state_restore_staged=state_restore_staged,
                                state_restore_started_at_s=(
                                    state_restore_started_at_s
                                ),
                                state_restore_finished_at_s=(
                                    state_restore_finished_at_s
                                ),
                                state_restore_duration_ms=(
                                    state_restore_duration_ms
                                ),
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
                            latency_ms=finish_request_latency_ms(),
                            retry_count=attempts - 1,
                            failed_attempts=failed_attempts,
                            state_restore_attempts=state_restore_attempts,
                            state_restore_successes=state_restore_successes,
                            state_restore_fallback=state_restore_fallback,
                            state_restored_tokens=state_restored_tokens,
                            supports_state_restore=supports_state_restore,
                            state_kind=state_kind,
                            state_restore_reason=state_restore_reason,
                            state_restored_blocks=state_restored_blocks,
                            state_restore_staged=state_restore_staged,
                            state_restore_started_at_s=(
                                state_restore_started_at_s
                            ),
                            state_restore_finished_at_s=(
                                state_restore_finished_at_s
                            ),
                            state_restore_duration_ms=(
                                state_restore_duration_ms
                            ),
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
                                latency_ms=finish_request_latency_ms(),
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
                                supports_state_restore=(
                                    supports_state_restore
                                ),
                                state_kind=state_kind,
                                state_restore_reason=state_restore_reason,
                                state_restored_blocks=state_restored_blocks,
                                state_restore_staged=state_restore_staged,
                                state_restore_started_at_s=(
                                    state_restore_started_at_s
                                ),
                                state_restore_finished_at_s=(
                                    state_restore_finished_at_s
                                ),
                                state_restore_duration_ms=(
                                    state_restore_duration_ms
                                ),
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
        await self._cancel_start_tasks()
        # stop all inference instances
        # return all unfinished requests
        while not self.request_queue.empty():
            request_data, done_event = await self.request_queue.get()
            done_event.set_result({"error": "Instance cancelled"})

        async with self.instance_management_lock:
            inference_instances = {}
            for pool in (
                self.starting_inference_instances,
                self.ready_inference_instances,
                self.deleting_inference_instances,
                self.failed_inference_instances,
            ):
                inference_instances.update(pool)
                pool.clear()
            ft_instances = {}
            for pool in (self.starting_ft_instances, self.ready_ft_instances):
                ft_instances.update(pool)
                pool.clear()
            deleted_instance_id = list(inference_instances.keys()) + list(
                ft_instances.keys()
            )
        delete_tasks = [
            self._shutdown_instance_handle(instance_id, instance)
            for instance_id, instance in inference_instances.items()
        ]
        delete_tasks.extend(
            self._shutdown_instance_handle(
                instance_id, instance, deallocate_resources=False
            )
            for instance_id, instance in ft_instances.items()
        )
        await asyncio.gather(*delete_tasks)
        await self._clear_scheduler_model_state()

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
            if self._reparallelization_execution_active:
                await asyncio.sleep(self.loop_interval)
                continue
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
        task = self.loop.create_task(self._start_instance(instance_id))
        self._track_start_task(
            instance_id, task, self.instance_start_tasks
        )

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
        task = self.loop.create_task(self._start_ft_instance(instance_id))
        self._track_start_task(instance_id, task, self.ft_start_tasks)
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _start_instance(self, instance_id):
        instance = None
        resources_allocated = False
        # Each concurrent startup must receive its own reservation snapshot.
        startup_backend_config = dict(self.backend_config)
        allocated_nodes = []
        node_allocations = {}
        try:
            async with self.instance_management_lock:
                if instance_id not in self.starting_inference_instances:
                    logger.error(f"Instance {instance_id} not found")
                    return
                instance = self.starting_inference_instances[instance_id]
            if self.backend == "dummy":
                startup_node = "control"
                startup_placement = {}
            else:
                # Now ask model loading scheduler to load the model
                logger.info(
                    f"Allocating resources for model {self.model_name} on instance {instance_id}"
                )
                distributed_targets = self.backend_config.get(
                    "spotserve_target_worker_nodes", []
                )
                if isinstance(distributed_targets, str):
                    distributed_targets = [
                        value.strip()
                        for value in distributed_targets.split(",")
                        if value.strip()
                    ]
                distributed_allocation = bool(
                    self.backend == "vllm"
                    and self.backend_config.get(
                        "spotserve_distributed_gpu_allocation", False
                    )
                    and self.backend_config.get(
                        "spotserve_vllm_ray_dp_owns_gpus", False
                    )
                    and isinstance(distributed_targets, list)
                    and len(distributed_targets) > 1
                )
                if distributed_allocation:
                    allocation = (
                        await self.model_loading_scheduler.allocate_distributed_resource.remote(
                            self.model_name,
                            instance_id,
                            self.resource_requirements,
                            [str(node) for node in distributed_targets],
                            False,
                        )
                    )
                    if not isinstance(allocation, Mapping):
                        raise RuntimeError(
                            "scheduler_distributed_allocation_result_invalid"
                        )
                    startup_node = str(allocation.get("node_id") or "")
                    resources_allocated = True
                    allocated_nodes = [
                        str(node)
                        for node in allocation.get("target_node_ids", [])
                    ]
                    if not startup_node or not allocated_nodes:
                        raise RuntimeError(
                            "scheduler_did_not_reserve_worker_nodes"
                        )
                    startup_backend_config[
                        "spotserve_reserved_worker_nodes"
                    ] = allocated_nodes
                    node_allocations = dict(allocation.get("node_allocations", {}))
                    startup_backend_config[
                        "spotserve_distributed_node_allocations"
                    ] = node_allocations
                else:
                    startup_node = (
                        await self.model_loading_scheduler.allocate_resource.remote(
                            self.model_name, instance_id, self.resource_requirements
                        )
                    )
                resources_allocated = True
                startup_placement = worker_placement_options(startup_node)
            async with instance.lock:
                instance.node_id = startup_node
                instance.member_node_ids = allocated_nodes or [str(startup_node)]
                instance.node_allocations = node_allocations or {
                    str(startup_node): int(self.resource_requirements["num_gpus"])
                }
            startup_config = {
                "num_cpus": self.resource_requirements["num_cpus"],
                "num_gpus": (
                    0
                    if (
                        self.backend == "vllm"
                        and self.backend_config.get(
                            "spotserve_vllm_ray_dp_owns_gpus", False
                        )
                    )
                    else self.resource_requirements["num_gpus"]
                ),
                **startup_placement,
            }
            logger.info(
                f"Startup config: {startup_config}, {self.backend_config}"
            )

            starter_options = dict(startup_placement)

            if self.backend == "dummy":
                from sllm.backends.dummy_backend import DummyBackend

                instance.backend_instance = DummyBackend(
                    self.model_name, startup_backend_config
                )
            else:
                instance.backend_instance = await start_instance.options(
                    **starter_options
                ).remote(
                    instance_id,
                    self.backend,
                    self.model_name,
                    startup_backend_config,
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
                if instance_id not in self.starting_inference_instances:
                    cleanup_started_instance = True
                else:
                    cleanup_started_instance = False
                    self.ready_inference_instances[instance_id] = instance
                    self.starting_inference_instances.pop(instance_id)
            if cleanup_started_instance:
                logger.info(
                    "Instance %s finished startup after it was untracked; "
                    "shutting it down",
                    instance_id,
                )
                await self._shutdown_instance_handle(
                    instance_id,
                    instance,
                    deallocate_resources=resources_allocated,
                )
                return
            return instance_id
        except asyncio.CancelledError:
            logger.info(
                "Cancelled startup for instance %s of model %s",
                instance_id,
                self.model_name,
            )
            if instance is not None:
                async with self.instance_management_lock:
                    self.starting_inference_instances.pop(instance_id, None)
                if instance.backend_instance is not None:
                    try:
                        await self._stop_backend(
                            instance.backend_instance, "shutdown"
                        )
                    except Exception:
                        logger.exception(
                            "Failed to shutdown cancelled backend actor %s "
                            "for model %s",
                            instance_id,
                            self.model_name,
                        )
                if resources_allocated and self.backend != "dummy":
                    try:
                        await self.model_loading_scheduler.deallocate_resource.remote(
                            self.model_name,
                            instance_id,
                            self.resource_requirements,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to deallocate resources for cancelled "
                            "instance %s of model %s",
                            instance_id,
                            self.model_name,
                        )
            raise
        except Exception as exc:
            logger.exception(
                "Failed to start instance %s for model %s",
                instance_id,
                self.model_name,
            )
            if instance is None:
                return
            failure_reason = str(exc)
            if len(failure_reason) > 2000:
                failure_reason = failure_reason[:2000] + "...<truncated>"
            setattr(instance, "failure_reason", failure_reason)
            await self._set_instance_state(
                instance, InstanceState.DEAD, reason="startup_failed"
            )
            async with self.instance_management_lock:
                self.starting_inference_instances.pop(instance_id, None)
                self.failed_inference_instances[instance_id] = instance
            if instance.backend_instance is not None:
                try:
                    await self._stop_backend(
                        instance.backend_instance, "shutdown"
                    )
                except Exception:
                    logger.exception(
                        "Failed to shutdown failed backend actor %s for "
                        "model %s",
                        instance_id,
                        self.model_name,
                    )
            if resources_allocated and self.backend != "dummy":
                try:
                    await self.model_loading_scheduler.deallocate_resource.remote(
                        self.model_name,
                        instance_id,
                        self.resource_requirements,
                    )
                except Exception:
                    logger.exception(
                        "Failed to deallocate resources for failed instance "
                        "%s of model %s",
                        instance_id,
                        self.model_name,
                    )
            return

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

        startup_placement = worker_placement_options(startup_node)
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            **startup_placement,
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                **startup_placement
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

    async def _shutdown_instance_handle(
        self,
        instance_id: str,
        instance: InstanceHandle,
        deallocate_resources: bool = True,
    ):
        await instance.mark_dead()
        if instance.backend_instance is not None:
            try:
                await self._stop_backend(instance.backend_instance, "shutdown")
            except Exception:
                logger.exception(
                    "Failed to shutdown backend actor %s for model %s",
                    instance_id,
                    self.model_name,
                )
        if deallocate_resources and self.backend != "dummy":
            await self.model_loading_scheduler.deallocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pools = (self.ready_ft_instances, self.starting_ft_instances)
            else:
                pools = (
                    self.ready_inference_instances,
                    self.starting_inference_instances,
                    self.deleting_inference_instances,
                    self.failed_inference_instances,
                )
            instance = None
            for pool in pools:
                if instance_id in pool:
                    instance = pool.pop(instance_id)
                    break
            if instance is None:
                logger.error(f"Instance {instance_id} not found")
                return
        await self._shutdown_instance_handle(instance_id, instance)
        return
