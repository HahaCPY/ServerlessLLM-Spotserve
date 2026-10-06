#!/usr/bin/env python3
"""Read-only verification for a two-workspace eight-GPU Ray pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def marker(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def worker_id(resources: Mapping[str, Any]) -> str:
    ids = sorted(
        key.removeprefix("worker_id_")
        for key, value in resources.items()
        if key.startswith("worker_id_") and float(value or 0) > 0
    )
    if len(ids) != 1:
        raise ValueError(f"each GPU worker needs one worker_id_* resource: {ids}")
    return ids[0]


def validate_nodes(
    nodes: Sequence[Mapping[str, Any]],
    workspace_a: str,
    workspace_b: str,
) -> dict[str, Any]:
    alive = [node for node in nodes if node.get("Alive", False)]
    heads = [
        node
        for node in alive
        if float((node.get("Resources") or {}).get("control_node", 0) or 0) > 0
    ]
    if len(heads) != 1:
        raise ValueError(f"expected one live control head, observed {len(heads)}")
    if float((heads[0].get("Resources") or {}).get("GPU", 0) or 0) != 0:
        raise ValueError("control head must advertise zero GPUs")

    gpu_nodes = [
        node
        for node in alive
        if float((node.get("Resources") or {}).get("GPU", 0) or 0) > 0
    ]
    if len(gpu_nodes) != 8:
        raise ValueError(f"expected eight live GPU Ray nodes, observed {len(gpu_nodes)}")

    expected_a = marker("spotserve_failure_domain_", f"workspace:{workspace_a}")
    expected_b = marker("spotserve_failure_domain_", f"workspace:{workspace_b}")
    rows = []
    seen_ids = set()
    seen_addresses = set()
    physical_markers = set()
    workspace_counts = {"A": 0, "B": 0}
    for node in gpu_nodes:
        resources = node.get("Resources") or {}
        if float(resources.get("GPU", 0) or 0) != 1:
            raise ValueError("every GPU worker pod must advertise exactly one GPU")
        current_worker_id = worker_id(resources)
        if current_worker_id in seen_ids:
            raise ValueError(f"duplicate worker ID: {current_worker_id}")
        seen_ids.add(current_worker_id)
        address = str(node.get("NodeManagerAddress") or "")
        if not address or address in seen_addresses:
            raise ValueError("GPU workers must advertise distinct pod IPs")
        seen_addresses.add(address)
        current_physical = sorted(
            key for key in resources if key.startswith("spotserve_physical_host_")
        )
        if len(current_physical) != 1:
            raise ValueError(
                f"worker {current_worker_id} needs one physical-host marker"
            )
        physical_markers.update(current_physical)
        if float(resources.get(expected_a, 0) or 0) > 0:
            workspace = "A"
        elif float(resources.get(expected_b, 0) or 0) > 0:
            workspace = "B"
        else:
            raise ValueError(
                f"worker {current_worker_id} lacks the expected workspace marker"
            )
        workspace_counts[workspace] += 1
        rows.append(
            {
                "worker_id": current_worker_id,
                "workspace": workspace,
                "ray_node_id": str(node.get("NodeID") or ""),
                "pod_ip": address,
                "physical_host_marker": current_physical[0],
            }
        )

    if seen_ids != {str(index) for index in range(8)}:
        raise ValueError(f"expected worker IDs 0..7, observed {sorted(seen_ids)}")
    if workspace_counts != {"A": 4, "B": 4}:
        raise ValueError(f"expected four workers per workspace: {workspace_counts}")
    for row in rows:
        numeric_id = int(row["worker_id"])
        expected_workspace = "A" if numeric_id < 4 else "B"
        if row["workspace"] != expected_workspace:
            raise ValueError(
                f"worker {numeric_id} belongs to workspace {row['workspace']}, "
                f"expected {expected_workspace}"
            )

    return {
        "status": "passed",
        "head_count": 1,
        "gpu_worker_count": 8,
        "gpu_count": 8,
        "workspace_worker_counts": workspace_counts,
        "physical_host_count": len(physical_markers),
        "cross_physical_host_available": len(physical_markers) >= 2,
        "workers": sorted(rows, key=lambda row: int(row["worker_id"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-a-namespace", required=True)
    parser.add_argument("--workspace-b-namespace", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-namespace", default="sllm")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ray is required to verify the live pool") from exc
    ray.init(
        address=args.ray_address,
        namespace=args.ray_namespace,
        ignore_reinit_error=True,
    )
    result = validate_nodes(
        ray.nodes(), args.workspace_a_namespace, args.workspace_b_namespace
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
