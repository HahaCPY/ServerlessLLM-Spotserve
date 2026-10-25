"""CPU-only tests for managed-Ray control-plane placement."""

from types import SimpleNamespace

from sllm import utils as sllm_utils


def test_managed_ray_head_without_cpu_allows_worker_fallback(monkeypatch):
    monkeypatch.setenv("SLLM_ALLOW_RAY_NODE_ID_WORKERS", "1")
    monkeypatch.setattr(
        sllm_utils,
        "_alive_ray_nodes",
        lambda: [{"NodeID": "head-node", "Resources": {}}],
    )
    monkeypatch.setattr(
        sllm_utils.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "head-node"),
    )

    options = sllm_utils.control_plane_placement_options()
    strategy = options["scheduling_strategy"]

    assert strategy.node_id == "head-node"
    assert strategy.soft is True


def test_cpu_less_head_triggers_fallback_inside_worker(monkeypatch):
    monkeypatch.delenv("SLLM_ALLOW_RAY_NODE_ID_WORKERS", raising=False)
    monkeypatch.setattr(
        sllm_utils,
        "_alive_ray_nodes",
        lambda: [
            {
                "NodeID": "head-node",
                "Resources": {
                    "memory": 4e9,
                    "node:__internal_head__": 1.0,
                },
            },
            {
                "NodeID": "gpu-worker",
                "Resources": {"CPU": 4.0, "GPU": 1.0},
            },
        ],
    )
    monkeypatch.setattr(
        sllm_utils.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "gpu-worker"),
    )

    options = sllm_utils.control_plane_placement_options()
    strategy = options["scheduling_strategy"]

    assert strategy.node_id == "gpu-worker"
    assert strategy.soft is True
