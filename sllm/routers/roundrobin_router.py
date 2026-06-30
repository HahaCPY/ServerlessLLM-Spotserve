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
from typing import Any, Dict, List, Optional, Tuple

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.spot.metrics import (
    JsonlMetricsWriter,
    make_instance_state_event,
    make_request_event,
)
from sllm.spot.recovery_policy import RecoveryPolicy, normalize_policy

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
        return {
            "model": self.model_name,
            "event": "preempt",
            "instances": marked_instances,
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
        return {
            "model": self.model_name,
            "event": "recover",
            "instances": recovered_instances,
        }

    async def mark_instance_recovered(self, instance_id: str):
        return await self.handle_recover(instance_id=instance_id)

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
            await self._set_instance_state(
                instance, InstanceState.DEAD, reason="trace_dead"
            )
            marked_instances.append(instance.instance_id)
            logger.info(
                f"Marked instance {instance.instance_id} as dead "
                f"for model {self.model_name}"
            )
        return {
            "model": self.model_name,
            "event": "dead",
            "instances": marked_instances,
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

    async def _counts_toward_capacity(self, instance: InstanceHandle) -> bool:
        async with instance.lock:
            return instance.state in {
                InstanceState.STARTING,
                InstanceState.READY,
                InstanceState.BUSY,
            }

    async def _call_backend(
        self,
        instance: InstanceHandle,
        request_data: dict,
        action: str,
        replay_tokens: Optional[List[List[int]]] = None,
    ):
        await self._ensure_lora_adapter(instance, request_data)
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
        request_start = time.time()
        attempts = 0
        failed_attempts = 0
        assigned_instances = []
        recovered_tokens = 0
        recovery_fallback = False
        replay_tokens = None

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

        try:
            while True:
                instance = None
                attempts += 1
                try:
                    instance_id, instance = (
                        await self._allocate_instance_for_request()
                    )
                    assigned_instances.append(instance_id)
                    logger.info(
                        f"{request_data}, type: {type(request_data)}, "
                        f"attempt: {attempts}"
                    )
                    result = await self._call_backend(
                        instance,
                        request_data,
                        action,
                        replay_tokens=replay_tokens,
                    )
                    logger.info("Finished processing request")

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
                            )
                        )
                        return result

                    failed_attempts += 1
                    if self._result_is_preempted(result):
                        await self._set_instance_state(
                            instance,
                            InstanceState.PREEMPTING,
                            reason="backend_preempted",
                        )
                        replay_tokens = self._tokens_from_preempted_result(
                            result
                        )
                        completed_tokens = result.get("completed_tokens", 0)
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
                            )
                        )
                        return result

                    if (
                        self.recovery_policy
                        != RecoveryPolicy.GENERATED_TOKEN_REPLAY
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
                        )
                    )
                    return {"error": str(e)}
                except Exception as e:
                    failed_attempts += 1
                    logger.error(f"Request attempt failed: {e}")
                    if instance is not None:
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
                            )
                        )
                        return {"error": str(e)}

                    if (
                        self.recovery_policy
                        != RecoveryPolicy.GENERATED_TOKEN_REPLAY
                    ):
                        replay_tokens = None
                finally:
                    if instance is not None:
                        await self._release_instance_request(instance)
        finally:
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
