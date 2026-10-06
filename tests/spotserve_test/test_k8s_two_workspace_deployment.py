import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_node(*, node_id, address, resources):
    return {
        "Alive": True,
        "NodeID": node_id,
        "NodeManagerAddress": address,
        "Resources": resources,
    }


def test_manifest_renderer_replaces_all_placeholders():
    renderer = load_script("render_k8s_two_workspace.py")
    template = renderer.DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    values = {
        "WORKSPACE_A_NAMESPACE": "workspace-a",
        "WORKSPACE_B_NAMESPACE": "workspace-b",
        "IMAGE": "registry/sllm@sha256:" + "a" * 64,
        "IMAGE_DIGEST": "sha256:" + "a" * 64,
        "MODEL_PVC_A": "model-a",
        "MODEL_PVC_B": "model-b",
        "RESULTS_PVC_A": "results-a",
        "CLUSTER_NAME": "spotserve-k8",
        "HEAD_SERVICE": "spotserve-ray-head",
        "MEM_POOL_SIZE": "4GB",
    }
    rendered = renderer.render_manifest(template, values)
    assert "__" not in rendered
    assert "namespace: workspace-a" in rendered
    assert "namespace: workspace-b" in rendered
    assert "ordinal + 0" in rendered
    assert "ordinal + 4" in rendered
    assert rendered.count('nvidia.com/gpu: "1"') == 4
    assert "spotserve-ray-head.workspace-a.svc.cluster.local:6379" in rendered
    assert "fieldPath: status.podIP" in rendered
    assert "fieldPath: spec.nodeName" in rendered


def test_renderer_requires_immutable_image_by_default():
    renderer = load_script("render_k8s_two_workspace.py")
    with pytest.raises(ValueError, match="immutable"):
        renderer.immutable_image("registry/sllm:latest", False)
    assert renderer.immutable_image("registry/sllm:latest", True) == (
        "registry/sllm:latest"
    )


def test_pool_verifier_accepts_four_workers_per_workspace():
    verifier = load_script("verify_k8s_two_workspace_pool.py")
    namespace_a = "workspace-a"
    namespace_b = "workspace-b"
    marker_a = verifier.marker(
        "spotserve_failure_domain_", f"workspace:{namespace_a}"
    )
    marker_b = verifier.marker(
        "spotserve_failure_domain_", f"workspace:{namespace_b}"
    )
    nodes = [
        ray_node(
            node_id="head",
            address="10.0.0.1",
            resources={"control_node": 1},
        )
    ]
    for index in range(8):
        workspace_marker = marker_a if index < 4 else marker_b
        physical_marker = f"spotserve_physical_host_{index // 4}"
        nodes.append(
            ray_node(
                node_id=f"worker-{index}",
                address=f"10.0.0.{index + 2}",
                resources={
                    "GPU": 1,
                    "worker_node": 1,
                    f"worker_id_{index}": 1,
                    workspace_marker: 1,
                    physical_marker: 1,
                },
            )
        )

    result = verifier.validate_nodes(nodes, namespace_a, namespace_b)
    assert result["status"] == "passed"
    assert result["gpu_count"] == 8
    assert result["workspace_worker_counts"] == {"A": 4, "B": 4}
    assert result["physical_host_count"] == 2
    assert result["cross_physical_host_available"] is True


def test_pool_verifier_rejects_duplicate_worker_ids():
    verifier = load_script("verify_k8s_two_workspace_pool.py")
    namespace_a = "workspace-a"
    namespace_b = "workspace-b"
    marker_a = verifier.marker(
        "spotserve_failure_domain_", f"workspace:{namespace_a}"
    )
    marker_b = verifier.marker(
        "spotserve_failure_domain_", f"workspace:{namespace_b}"
    )
    nodes = [ray_node(node_id="head", address="10.0.0.1", resources={"control_node": 1})]
    for index in range(8):
        current_id = 0 if index == 1 else index
        nodes.append(ray_node(
            node_id=f"worker-{index}", address=f"10.0.0.{index + 2}",
            resources={
                "GPU": 1,
                f"worker_id_{current_id}": 1,
                marker_a if index < 4 else marker_b: 1,
                f"spotserve_physical_host_{index // 4}": 1,
            },
        ))
    with pytest.raises(ValueError, match="duplicate worker ID"):
        verifier.validate_nodes(nodes, namespace_a, namespace_b)


def test_adapted_trace_never_exceeds_eight_slots():
    import json

    trace_path = (
        REPO_ROOT
        / "examples/spotserve/spot_trace_k8_two_workspace_churn.jsonl"
    )
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    ready = {str(index) for index in range(8)}
    minimum = len(ready)
    for row in rows:
        node_id = row["node_id"]
        if row["event"] == "preempt":
            ready.discard(node_id)
        elif row["event"] == "recover":
            ready.add(node_id)
        minimum = min(minimum, len(ready))
        assert len(ready) <= 8
    assert minimum == 5
    assert ready == {str(index) for index in range(8)}
