"""Concrete vLLM worker deployment for dynamic re-parallelization.

The planner is intentionally runtime-neutral.  This adapter is the deployment
boundary used by the router: it allocates Ray worker resources, starts real
``VllmBackend`` actors with the planned parallel shape, waits for engine
initialisation, snapshots/aborts tracked requests, and releases the old
deployment only after traffic switches.
"""

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

import ray

from sllm.inference_instance import start_instance
from sllm.utils import InstanceHandle

from .reparallelization import ParallelPlan

logger = logging.getLogger("ray")

MaybeAsync = Callable[..., Any]


async def _call(fn: MaybeAsync, *args: Any, **kwargs: Any) -> Any:
    value = fn(*args, **kwargs)
    if isinstance(value, Awaitable) or inspect.isawaitable(value):
        return await value
    return value


@dataclass
class VllmDeployment:
    """A ready or draining set of vLLM actors managed as one plan."""

    plan: ParallelPlan
    instances: Dict[str, InstanceHandle] = field(default_factory=dict)
    backend_config: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, int] = field(default_factory=dict)


class VllmDeploymentAdapter:
    """Create, switch, drain, and stop real vLLM Ray worker deployments."""

    def __init__(
        self,
        model_name: str,
        backend_config: Mapping[str, Any],
        resource_requirements: Mapping[str, int],
        scheduler: Any,
        traffic_switcher: MaybeAsync,
        request_migrator: Optional[MaybeAsync] = None,
        max_queue_length: int = 1,
        drain_timeout_s: float = 30.0,
        migration_completion_waiter: Optional[MaybeAsync] = None,
    ) -> None:
        self.model_name = model_name
        self.backend_config = dict(backend_config)
        self.resource_requirements = dict(resource_requirements)
        self.scheduler = scheduler
        self.traffic_switcher = traffic_switcher
        self.request_migrator = request_migrator
        self.last_request_migration: Optional[Dict[str, Any]] = None
        self.max_queue_length = max(1, int(max_queue_length))
        self.drain_timeout_s = max(0.1, float(drain_timeout_s))
        self.migration_completion_waiter = migration_completion_waiter

    async def wait_for_migration_completion(
        self, deployment: Optional[VllmDeployment]
    ) -> None:
        """Keep source NIXL agents alive until target retries attach."""
        if deployment is None or self.migration_completion_waiter is None:
            return
        await _call(
            self.migration_completion_waiter,
            deployment,
            timeout_s=self.drain_timeout_s,
        )

    def _plan_backend_config(self, plan: ParallelPlan) -> Dict[str, Any]:
        config = dict(self.backend_config)
        config["tensor_parallel_size"] = plan.tensor_parallel_size
        config["pipeline_parallel_size"] = plan.pipeline_parallel_size
        config["data_parallel_size"] = plan.data_parallel_size
        config["vllm_data_parallel_size"] = plan.data_parallel_size
        config["replica_count"] = plan.replica_count
        config["sllm_replica_count"] = plan.replica_count
        planned_ep_size = plan.effective_expert_parallel_size
        config["planned_effective_expert_parallel_size"] = planned_ep_size
        config["planned_expert_parallel_size"] = planned_ep_size
        config["expert_parallel_size_source"] = (
            "derived_from_tp_dp"
            if plan.enable_expert_parallel
            else "disabled"
        )
        config["expert_parallel_size_verified"] = False
        config["enable_expert_parallel"] = plan.enable_expert_parallel
        config["placement_epoch"] = plan.placement_epoch
        config["reparallelization_execution_model"] = "actor_recreate"
        config["reparallelization_execution_model_reason"] = (
            "vllm_actor_recreate"
        )
        expert_placement_plan = (
            dict(plan.expert_placement_plan)
            if isinstance(plan.expert_placement_plan, Mapping)
            else {}
        )
        if expert_placement_plan:
            config["expert_placement_plan"] = expert_placement_plan
            config["expert_placement_execution_model"] = (
                "expert_aware_actor_recreate"
            )
            config["expert_placement_execution_model_reason"] = (
                "logical_expert_placement_plan_carried_into_recreated_actor"
            )
            config["expert_placement_runtime_contract_mode"] = (
                "observe_only_contract"
            )
            config["expert_placement_live_migration_enabled"] = False
            config["expert_placement_physical_migration_required"] = False
            config["placement_source"] = str(
                expert_placement_plan.get(
                    "placement_source",
                    "logical_reparallelization_planner",
                )
            )
            fingerprint = expert_placement_plan.get("placement_fingerprint")
            if fingerprint:
                config["expert_placement_fingerprint"] = str(fingerprint)
                config["expert_placement_plan_fingerprint"] = str(fingerprint)
                config["expert_placement_contract_fingerprint"] = str(
                    fingerprint
                )
            config["expert_placement_contract_available"] = True
            config["expert_placement_contract_source"] = config[
                "placement_source"
            ]
            config["expert_placement_contract_epoch"] = plan.placement_epoch
            config["expert_placement_plan_applied"] = False
            config["expert_placement_plan_verified"] = False
            placement_snapshot = expert_placement_plan.get(
                "expert_placement_snapshot"
            )
            if isinstance(placement_snapshot, Mapping):
                config["expert_placement_snapshot"] = dict(
                    placement_snapshot
                )
        return config

    def _replica_gpu_count(self, plan: ParallelPlan) -> int:
        replica_gpus = (
            plan.tensor_parallel_size
            * plan.pipeline_parallel_size
            * plan.data_parallel_size
        )
        if replica_gpus <= 0:
            raise ValueError("parallel plan has no GPUs per replica")
        expected = replica_gpus * max(plan.replica_count, 1)
        if plan.num_gpus < expected:
            raise ValueError(
                "parallel plan num_gpus is smaller than its replica shape: "
                f"{plan.num_gpus} < {expected}"
            )
        return replica_gpus

    async def _known_scheduler_node_ids(self) -> Optional[set[str]]:
        scheduler_snapshot = getattr(self.scheduler, "_get_worker_nodes", None)
        if scheduler_snapshot is None:
            return None
        try:
            remote = getattr(scheduler_snapshot, "remote", None)
            worker_nodes = (
                await remote()
                if remote is not None
                else await scheduler_snapshot()
            )
        except Exception:
            logger.info(
                "Could not query scheduler worker nodes before vLLM replan",
                exc_info=True,
            )
            return None
        if not isinstance(worker_nodes, Mapping):
            return None
        return {str(node_id) for node_id in worker_nodes}

    async def create_workers(self, plan: ParallelPlan) -> VllmDeployment:
        if plan.backend != "vllm":
            raise ValueError(
                "VllmDeploymentAdapter only supports backend='vllm', got "
                f"{plan.backend!r}"
            )
        if plan.model_name != self.model_name:
            raise ValueError(
                f"plan model {plan.model_name!r} does not match "
                f"{self.model_name!r}"
            )

        replica_gpus = self._replica_gpu_count(plan)
        backend_config = self._plan_backend_config(plan)
        resource_requirements = {
            "num_cpus": max(0, int(self.resource_requirements.get("num_cpus", 1))),
            "num_gpus": replica_gpus,
        }
        deployment = VllmDeployment(
            plan=plan,
            backend_config=backend_config,
            resource_requirements=resource_requirements,
        )
        target_nodes = list(plan.target_nodes)
        known_scheduler_node_ids = await self._known_scheduler_node_ids()

        try:
            for replica in range(max(plan.replica_count, 1)):
                instance_id = (
                    f"{self.model_name}_reparallel_{uuid.uuid4().hex[:12]}"
                )
                target_node_id = (
                    target_nodes[replica % len(target_nodes)]
                    if target_nodes
                    else None
                )
                if (
                    target_node_id is not None
                    and known_scheduler_node_ids is not None
                    and str(target_node_id) not in known_scheduler_node_ids
                ):
                    raise RuntimeError(
                        "target_worker_node_not_in_scheduler_snapshot: "
                        f"{target_node_id}"
                    )
                allocation_kwargs = {
                    "model_name": self.model_name,
                    "instance_id": instance_id,
                    "resources": resource_requirements,
                }
                if target_node_id is not None:
                    allocation_kwargs["target_node_id"] = target_node_id
                startup_node = await _call(
                    self.scheduler.allocate_resource.remote,
                    **allocation_kwargs,
                )
                startup_resources = {
                    "worker_node": 0.1,
                    f"worker_id_{startup_node}": 0.1,
                }
                startup_config = {
                    **resource_requirements,
                    "resources": startup_resources,
                }
                actor = await _call(
                    start_instance.options(resources=startup_resources).remote,
                    instance_id,
                    "vllm",
                    self.model_name,
                    backend_config,
                    startup_config,
                )
                handle = InstanceHandle(
                    instance_id=instance_id,
                    max_queue_length=self.max_queue_length,
                    num_gpu=replica_gpus,
                    node_id=str(startup_node),
                    backend_instance=actor,
                )
                await _call(actor.init_backend.remote)
                await handle.mark_ready(node_id=str(startup_node))
                deployment.instances[instance_id] = handle
                logger.info(
                    "Started vLLM re-parallelized instance %s on node %s "
                    "with TP=%s DP=%s PP=%s EP=%s",
                    instance_id,
                    startup_node,
                    plan.tensor_parallel_size,
                    plan.data_parallel_size,
                    plan.pipeline_parallel_size,
                    plan.enable_expert_parallel,
                )
        except Exception:
            await self.stop_workers(deployment)
            raise
        return deployment

    async def ready_workers(
        self, deployment: VllmDeployment, plan: ParallelPlan
    ) -> bool:
        if deployment.plan != plan or not deployment.instances:
            return False
        for handle in deployment.instances.values():
            if not handle.ready or handle.state.value != "ready":
                return False
            try:
                metadata = await _call(
                    handle.backend_instance.get_runtime_metadata.remote,
                    instance_id=handle.instance_id,
                    node_id=handle.node_id or "",
                )
            except Exception:
                logger.exception("vLLM worker readiness probe failed")
                return False
            if not isinstance(metadata, Mapping):
                return False
        return True

    async def switch_workers(
        self, deployment: VllmDeployment, plan: ParallelPlan
    ) -> Any:
        if deployment.plan != plan:
            raise ValueError("deployment plan changed before traffic switch")
        return await _call(self.traffic_switcher, deployment, plan)

    async def drain_workers(self, deployment: Optional[VllmDeployment]) -> None:
        if deployment is None:
            return
        self.last_request_migration = None
        # Stop new allocations first; tracked requests already running on the
        # old handles remain visible to the migration callback below.
        for handle in deployment.instances.values():
            await handle.mark_draining()
        if self.request_migrator is not None:
            self.last_request_migration = await _call(
                self.request_migrator, deployment
            )
        deadline = asyncio.get_running_loop().time() + self.drain_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if all(handle.concurrency <= 0 for handle in deployment.instances.values()):
                return
            await asyncio.sleep(0.1)
        logger.warning(
            "vLLM deployment drain timed out for model %s; stopping actors",
            self.model_name,
        )

    async def stop_workers(self, deployment: Optional[VllmDeployment]) -> None:
        if deployment is None:
            return
        for instance_id, handle in list(deployment.instances.items()):
            actor = handle.backend_instance
            try:
                if actor is not None:
                    await _call(actor.stop.remote)
            except Exception:
                logger.exception("Failed stopping vLLM actor %s", instance_id)
            finally:
                try:
                    if actor is not None:
                        ray.kill(actor, no_restart=True)
                except Exception:
                    logger.exception("Failed killing vLLM actor %s", instance_id)
                try:
                    await _call(
                        self.scheduler.deallocate_resource.remote,
                        self.model_name,
                        instance_id,
                        deployment.resource_requirements,
                    )
                except Exception:
                    logger.exception(
                        "Failed releasing resources for vLLM actor %s",
                        instance_id,
                    )

    def snapshot(self, instances: Mapping[str, InstanceHandle]) -> VllmDeployment:
        """Capture the router's active actors as the executor's old target."""
        plan = ParallelPlan(
            model_name=self.model_name,
            backend="vllm",
            tensor_parallel_size=max(
                1, int(self.backend_config.get("tensor_parallel_size", 1) or 1)
            ),
            data_parallel_size=max(
                1, int(self.backend_config.get("data_parallel_size", 1) or 1)
            ),
            pipeline_parallel_size=max(
                1, int(self.backend_config.get("pipeline_parallel_size", 1) or 1)
            ),
            replica_count=max(1, len(instances)),
            enable_expert_parallel=bool(
                self.backend_config.get("enable_expert_parallel", False)
            ),
            num_gpus=sum(int(handle.num_gpu or 0) for handle in instances.values()),
            target_nodes=[
                str(handle.node_id)
                for handle in instances.values()
                if handle.node_id is not None
            ],
            placement_epoch=max(
                0, int(self.backend_config.get("placement_epoch", 0) or 0)
            ),
            expert_placement_plan=(
                self.backend_config.get("expert_placement_plan")
                if isinstance(
                    self.backend_config.get("expert_placement_plan"),
                    Mapping,
                )
                else None
            ),
            reason="active_deployment",
        )
        return VllmDeployment(
            plan=plan,
            instances=dict(instances),
            backend_config=dict(self.backend_config),
            resource_requirements=dict(self.resource_requirements),
        )
