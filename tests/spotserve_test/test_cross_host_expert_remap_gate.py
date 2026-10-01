import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_spotserve_cross_host_expert_remap.py"
)
SIMULATED_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_spotserve_simulated_cross_host_remap.py"
)


def load_gate_module():
    spec = importlib.util.spec_from_file_location("cross_host_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_simulated_gate_module():
    spec = importlib.util.spec_from_file_location(
        "simulated_cross_host_gate", SIMULATED_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ray_node(worker_id, node_id, address, host_marker):
    return {
        "Alive": True,
        "NodeID": node_id,
        "NodeManagerAddress": address,
        "Resources": {
            "GPU": 1,
            f"worker_id_{worker_id}": 1,
            f"spotserve_physical_host_{host_marker}": 1,
        },
    }


def test_cross_host_gate_accepts_distinct_physical_hosts(monkeypatch):
    gate = load_gate_module()
    monkeypatch.setattr(
        gate.ray,
        "nodes",
        lambda: [
            ray_node("0", "ray-a", "10.0.0.11", "host-a"),
            ray_node("1", "ray-b", "10.0.0.12", "host-b"),
        ],
    )

    workers = gate.target_worker_nodes(["0", "1"])

    assert [worker["worker_id"] for worker in workers] == ["0", "1"]
    assert len({worker["physical_host_id"] for worker in workers}) == 2


def test_cross_host_gate_rejects_two_workers_on_same_physical_host(monkeypatch):
    gate = load_gate_module()
    monkeypatch.setattr(
        gate.ray,
        "nodes",
        lambda: [
            ray_node("0", "ray-a", "10.0.0.11", "same-host"),
            ray_node("1", "ray-b", "10.0.0.12", "same-host"),
        ],
    )

    with pytest.raises(RuntimeError, match="distinct physical-host markers"):
        gate.target_worker_nodes(["0", "1"])


def test_cross_host_gate_builds_complete_ep2_owner_swap():
    gate = load_gate_module()
    plan = gate.swapped_plan(
        "model",
        {
            "layer:0/expert:0": 0,
            "layer:0/expert:1": 1,
            "layer:1/expert:0": 0,
            "layer:1/expert:1": 1,
        },
    )

    assert plan["require_cross_node"] is True
    assert plan["expert_placement_physical_migration_required"] is True
    assert plan["covered_expert_count"] == 4
    assert plan["expert_to_target_rank"] == {
        "layer:0/expert:0": "replica:0/ep-rank:1",
        "layer:0/expert:1": "replica:0/ep-rank:0",
        "layer:1/expert:0": "replica:0/ep-rank:1",
        "layer:1/expert:1": "replica:0/ep-rank:0",
    }


def simulated_ray_node(worker_id, node_id, address, failure_domain):
    node = ray_node(
        worker_id,
        node_id,
        address,
        "one-physical-host",
    )
    node["Resources"][f"spotserve_failure_domain_{failure_domain}"] = 1
    node["Resources"]["GPU"] = 2
    return node


def test_simulated_host_gate_accepts_one_physical_host_with_two_domains(
    monkeypatch,
):
    gate = load_simulated_gate_module()
    monkeypatch.setattr(
        gate.ray,
        "nodes",
        lambda: [
            simulated_ray_node("0", "ray-a", "10.89.0.2", "domain-a"),
            simulated_ray_node("1", "ray-b", "10.89.0.3", "domain-b"),
        ],
    )

    workers = gate.simulated_worker_nodes(["0", "1"])

    assert len({worker["physical_host_id"] for worker in workers}) == 1
    assert len({worker["failure_domain_id"] for worker in workers}) == 2


def test_simulated_host_gate_rejects_distinct_physical_hosts(monkeypatch):
    gate = load_simulated_gate_module()
    first = simulated_ray_node("0", "ray-a", "10.89.0.2", "domain-a")
    second = simulated_ray_node("1", "ray-b", "10.89.0.3", "domain-b")
    second["Resources"].pop("spotserve_physical_host_one-physical-host")
    second["Resources"]["spotserve_physical_host_other-host"] = 1
    monkeypatch.setattr(gate.ray, "nodes", lambda: [first, second])

    with pytest.raises(RuntimeError, match="one shared physical-host marker"):
        gate.simulated_worker_nodes(["0", "1"])
