import asyncio

import pytest

from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.schedulers.fcfs_scheduler import FcfsScheduler
from sllm.spot.reparallelization import ParallelPlan, plan_dynamic_reparallelization
from sllm.spot.worker_pool import reserved_worker_ips
from sllm.utils import InstanceHandle


def nodes(count):
    return {str(i): {"total_gpu": 1, "free_gpu": 1, "state": "ready"}
            for i in range(count)}


@pytest.mark.parametrize("count", [8, 12])
def test_gpu_pool_is_not_an_instance_or_fixed_size(count):
    pool = nodes(count)
    pool["0"]["state"] = "preempting"
    pool["1"]["free_gpu"] = 0
    allocation = FcfsScheduler._distributed_gpu_allocation(
        pool, list(pool), requested_gpus=2, require_all_target_nodes=False)
    assert set(allocation) == set(sorted(set(pool) - {"0", "1"})[:2])
    assert sum(allocation.values()) == 2


@pytest.mark.asyncio
async def test_non_primary_rank_preemption_matches_distributed_instance():
    router = RoundRobinRouter(
        "m", {"num_cpus": 1, "num_gpus": 2}, "dummy", {}, {})
    handle = InstanceHandle("ep2", 2, 2, node_id="0", member_node_ids=["0", "1"])
    router.ready_inference_instances[handle.instance_id] = handle
    matches = await router._matching_inference_instances(node_id="1")
    assert matches == [handle]
    assert router._spot_target_node_ids("1", matches) == ["1"]
    assert router._spot_target_node_ids(None, matches) == ["0", "1"]
    assert (await router.get_instance_states())["ep2"]["member_node_ids"] == ["0", "1"]


def decide(alpha, rows, count=8, **planner):
    return plan_dynamic_reparallelization(
        "m", nodes(count), model_config={"backend": "vllm",
            "backend_capability": {"supported_configs": rows}},
        planner_config={"selection_policy": "spotserve_algorithm1",
                        "arrival_rate_req_s": alpha, **planner})


def row(gpus, throughput, latency, batch=4):
    return {"tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "data_parallel_size": gpus, "replica_count": 1, "num_gpus": gpus,
            "enable_expert_parallel": True, "batch_size": batch,
            "throughput_estimate_req_s": throughput, "latency_estimate_ms": latency}


def test_algorithm1_uses_min_latency_not_largest_gpu_or_weighted_score():
    result = decide(3, [row(2, 4, 100), row(4, 10, 500)])
    assert result["selected_total_gpus"] == 2
    assert result["workload_cost_model"]["selection_branch"] == "min_latency_meeting_arrival_rate"
    plan = ParallelPlan.from_dict(result["parallel_plan"])
    assert plan.batch_size == 4
    assert plan.to_dict()["batch_size"] == 4


def test_algorithm1_overload_maximizes_profiled_throughput():
    result = decide(20, [row(2, 8, 100), row(4, 7, 10)])
    assert result["selected_total_gpus"] == 2
    assert result["workload_cost_model"]["selection_branch"] == "max_throughput"


def test_algorithm1_does_not_invent_missing_profiles():
    result = decide(1, [row(4, 0, 100)])
    assert result["action"] == "no_capacity"
    assert result["workload_cost_model"]["selection_branch"] == "no_valid_profile"


def test_algorithm1_latency_tie_uses_fewer_gpus():
    assert decide(1, [row(4, 8, 100), row(2, 4, 100)])["selected_total_gpus"] == 2


def test_algorithm1_explicit_md1_approximation_changes_queue_cost():
    result = decide(3.9, [row(2, 4, 100), row(4, 10, 500)], queue_model="md1_aggregate")
    assert result["selected_total_gpus"] == 4
    assert result["selected_config"]["queue_penalty_ms"] > 0


def ray_workers():
    return [{"Alive": True, "Resources": {f"worker_id_{i}": 1, "GPU": 1},
             "NodeManagerAddress": f"10.0.0.{i + 1}"} for i in range(8)]


def test_reserved_workers_resolve_pod_ips_not_physical_host_count():
    assert reserved_worker_ips(["2", "3"], {"2": 1, "3": 1}, ray_workers(), 2) == [
        "10.0.0.3", "10.0.0.4"]


def test_reserved_worker_ip_placement_rejects_ambiguous_or_partial_ownership():
    pool = ray_workers()
    pool[1]["NodeManagerAddress"] = pool[0]["NodeManagerAddress"]
    with pytest.raises(ValueError, match="distinct_ray_pod_ips"):
        reserved_worker_ips(["0", "1"], {"0": 1, "1": 1}, pool, 2)
    pool[0]["Resources"]["GPU"] = 2
    with pytest.raises(ValueError, match="whole_worker_gpu_reservation"):
        reserved_worker_ips(["0"], {"0": 1}, pool, 1)


@pytest.mark.asyncio
async def test_trace_deadline_fires_while_notice_handler_is_still_busy(monkeypatch):
    import sllm.spot.preemption_simulator as simulator
    from sllm.spot.trace_reader import SpotEvent

    order = []
    class Method:
        def __init__(self, name):
            self.name = name
        async def remote(self, **kwargs):
            order.append(self.name)
            if self.name == "preempt":
                await asyncio.sleep(0.04)
                order.append("preempt_finished")
            return {}
    class Controller:
        handle_preemption = Method("preempt")
        handle_instance_dead = Method("dead")
        handle_add = Method("add")
    class Ray:
        @staticmethod
        def get_actor(*args):
            return Controller()
    monkeypatch.setattr(simulator, "ray", Ray())
    monkeypatch.setattr(simulator, "load_spot_trace", lambda *args, **kwargs: [
        SpotEvent(0, "preempt", node_id="0", grace_period_s=0.005),
        SpotEvent(0.01, "add", node_id="1"),
    ])
    result = await simulator.replay_trace("unused.jsonl")
    assert order.index("dead") < order.index("preempt_finished")
    assert order.index("add") < order.index("preempt_finished")
    assert result["auto_dead_dispatched"] == 1


@pytest.mark.asyncio
async def test_multiple_ep_instances_have_disjoint_actual_reservations(monkeypatch):
    import sllm.spot.vllm_deployment_adapter as module
    pool = nodes(8)
    class Remote:
        def __init__(self, fn):
            self.remote = fn
    async def snapshot():
        return pool
    async def allocate(**kwargs):
        allocation = FcfsScheduler._distributed_gpu_allocation(
            pool, kwargs["target_node_ids"], kwargs["resources"]["num_gpus"],
            kwargs["require_all_target_nodes"])
        for worker, count in allocation.items():
            pool[worker]["free_gpu"] -= count
        return {"node_id": next(iter(allocation)), "target_node_ids": list(allocation),
                "node_allocations": allocation}
    class Scheduler:
        _get_worker_nodes = Remote(snapshot)
        allocate_distributed_resource = Remote(allocate)
    configs = []
    class Actor:
        async def init():
            return None
        init_backend = Remote(init)
    class Starter:
        def options(self, **kwargs):
            return self
        async def remote(self, instance_id, backend, model, config, startup):
            configs.append(config)
            return Actor()
    monkeypatch.setattr(module, "start_instance", Starter())
    adapter = module.VllmDeploymentAdapter("m", {
        "spotserve_distributed_gpu_allocation": True,
        "spotserve_vllm_ray_dp_owns_gpus": True,
    }, {"num_cpus": 1, "num_gpus": 2}, Scheduler(), lambda *args: None)
    deployment = await adapter.create_workers(ParallelPlan(
        "m", "vllm", 1, data_parallel_size=2, replica_count=2,
        enable_expert_parallel=True, num_gpus=4, target_nodes=list(pool)))
    assert [set(h.worker_node_ids()) for h in deployment.instances.values()] == [
        {"0", "1"}, {"2", "3"}]
    assert configs[0]["spotserve_reserved_worker_nodes"] == ["0", "1"]
    assert configs[1]["spotserve_reserved_worker_nodes"] == ["2", "3"]
    assert configs[0] is not configs[1]
    assert adapter.snapshot(deployment.instances).plan.target_nodes == ["0", "1", "2", "3"]


@pytest.mark.asyncio
async def test_deadline_revokes_source_after_it_has_left_ready_pool():
    from sllm.utils import InstanceState
    router = RoundRobinRouter("m", {"num_cpus": 1, "num_gpus": 2}, "dummy", {}, {})
    stopped = []
    class Actor:
        async def shutdown(self):
            stopped.append("source")
    source = InstanceHandle("source", 2, 2, node_id="0", member_node_ids=["0", "1"],
                            backend_instance=Actor())
    router.deleting_inference_instances[source.instance_id] = source
    result = await router.handle_dead(node_id="1", auto_deadline=True)
    assert stopped == ["source"]
    assert source.state == InstanceState.DEAD
    assert result["instances"] == ["source"]


@pytest.mark.asyncio
async def test_committed_plan_controls_scaler_and_keeps_source_discoverable():
    from types import SimpleNamespace

    from sllm.spot.vllm_deployment_adapter import VllmDeployment

    router = RoundRobinRouter(
        "m", {"num_cpus": 1, "num_gpus": 2}, "dummy", {}, {})
    source = InstanceHandle(
        "source", 2, 2, node_id="0", member_node_ids=["0", "1"])
    target = InstanceHandle(
        "target", 2, 4, node_id="4", member_node_ids=["4", "5", "6", "7"])
    router.ready_inference_instances = {"source": source}
    router.auto_scaling_config = {
        "min_instances": 2, "max_instances": 2, "target": 8}
    router.vllm_deployment_adapter = SimpleNamespace()
    plan = ParallelPlan("m", "vllm", 1, data_parallel_size=4, replica_count=1)
    deployment = VllmDeployment(
        plan, {"target": target}, {"data_parallel_size": 4}, {"num_gpus": 4})

    await router._switch_vllm_deployment(deployment, plan)

    assert router.auto_scaling_config == {
        "min_instances": 1, "max_instances": 1, "target": 8}
    assert router.ready_inference_instances == {"target": target}
    assert router.deleting_inference_instances == {"source": source}
    assert await router._matching_inference_instances(node_id="1") == [source]


@pytest.mark.asyncio
async def test_export_timeout_does_not_call_same_hook_again():
    from sllm.spot.vllm_deployment_adapter import VllmDeployment
    router = RoundRobinRouter("m", {"num_cpus": 1, "num_gpus": 1}, "dummy", {},
                             {"reparallelization_config": {"state_export_timeout_s": 0.001}})
    calls = []
    class Actor:
        async def abort_request(self, **kwargs):
            return {"aborted": True}
    source = InstanceHandle("source", 2, 1, node_id="0", backend_instance=Actor())
    async def export(*args, **kwargs):
        calls.append("export")
        await asyncio.sleep(1)
    router._capture_inference_state = export
    router.inflight_requests["req"] = {
        "instance_id": "source", "instance": source, "request_id": "req",
        "request_data": {"request_id": "req"}}
    deployment = VllmDeployment(ParallelPlan("m", "vllm", 1), {"source": source})
    result = await router._prepare_reparallelization_requests(deployment)
    assert calls == ["export"]
    assert result["state_exported"] == 0
    assert result["abort_requested"] == 1
    assert router.inflight_requests["req"]["migration_state"] is None


@pytest.mark.asyncio
async def test_http_grace_deadline_does_not_wait_for_notice_response(monkeypatch, tmp_path):
    import json
    from benchmarks.spotserve import run_benchmark as module
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"time": 0, "event": "preempt", "node_id": "1",
                                "grace_period_s": 0.005}) + "\n")
    calls = []
    def post(endpoint, payload, timeout):
        calls.append(payload)
        return {"success": True}
    async def mocked_transport(fn, *args):
        result = fn(*args)
        if args[1]["event"] == "preempt":
            await asyncio.sleep(0.03)
            calls.append({"event": "notice_response"})
        return result
    monkeypatch.setattr(module, "post_json", post)
    monkeypatch.setattr(module.asyncio, "to_thread", mocked_transport)
    await module.replay_trace_over_http(str(path), 1, "http://localhost/v1/chat/completions",
                                        tmp_path / "log", 1, "auto", "sllm")
    assert [row["event"] for row in calls] == ["preempt", "dead", "notice_response"]
    assert calls[1]["auto_deadline"] is True
    assert calls[1]["deadline_time_s"] == calls[0]["deadline_time_s"]
