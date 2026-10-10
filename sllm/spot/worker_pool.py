"""Translate scheduler reservations to vLLM Ray-DP placement constraints."""
from typing import Any, Mapping, Sequence


def reserved_worker_ips(
    worker_ids: Sequence[str],
    allocations: Mapping[str, int],
    ray_nodes: Sequence[Mapping[str, Any]],
    expected_gpus: int,
) -> list[str]:
    ids = list(dict.fromkeys(str(value) for value in worker_ids))
    if not ids or set(ids) != set(allocations):
        raise ValueError("worker_reservation_membership_mismatch")
    if sum(allocations.values()) != expected_gpus:
        raise ValueError("worker_reservation_gpu_count_mismatch")
    ips = []
    for worker_id in ids:
        matches = [node for node in ray_nodes if node.get("Alive", False)
                   and node.get("Resources", {}).get(f"worker_id_{worker_id}", 0) > 0]
        if not matches:
            matches = [
                node for node in ray_nodes
                if node.get("Alive", False)
                and str(node.get("NodeID") or "") == worker_id
                and int(float(node.get("Resources", {}).get("GPU", 0) or 0)) > 0
            ]
        if len(matches) != 1:
            raise ValueError(f"reserved_worker_not_uniquely_alive:{worker_id}")
        node = matches[0]
        count = allocations[worker_id]
        if count <= 0 or count != int(count):
            raise ValueError(f"invalid_worker_gpu_reservation:{worker_id}")
        # IP-only placement cannot enforce a fractional reservation on a
        # multi-GPU worker. The formal contract is one Ray worker per GPU.
        if int(node.get("Resources", {}).get("GPU", 0)) != count:
            raise ValueError(f"ip_placement_requires_whole_worker_gpu_reservation:{worker_id}")
        ip = str(node.get("NodeManagerAddress") or "")
        if not ip or ip in ips:
            raise ValueError("reserved_workers_require_distinct_ray_pod_ips")
        ips.append(ip)
    return ips
