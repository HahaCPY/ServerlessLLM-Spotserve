"""Verify physical expert remap across two distinct GPU hosts.

This gate intentionally fails on a same-host Ray cluster.  Each target worker
node must advertise one node-local ``spotserve_physical_host_*`` resource.
"""

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import ray


HOST_RESOURCE_PREFIX = "spotserve_physical_host_"


def post_json(endpoint, path, payload, timeout):
    request = Request(
        f"{endpoint.rstrip('/')}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def target_worker_nodes(worker_ids):
    selected = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        matched = [
            worker_id
            for worker_id in worker_ids
            if float(resources.get(f"worker_id_{worker_id}", 0) or 0) > 0
        ]
        if not matched:
            continue
        markers = sorted(
            key[len(HOST_RESOURCE_PREFIX) :]
            for key, value in resources.items()
            if key.startswith(HOST_RESOURCE_PREFIX) and float(value or 0) > 0
        )
        if len(markers) != 1:
            raise RuntimeError(
                "Each cross-host worker must expose exactly one node-local "
                f"physical-host marker; node={node.get('NodeID')} markers={markers}"
            )
        selected.append(
            {
                "worker_id": matched[0],
                "ray_node_id": str(node.get("NodeID") or ""),
                "address": str(node.get("NodeManagerAddress") or ""),
                "physical_host_id": markers[0],
                "gpu_count": float(resources.get("GPU", 0) or 0),
            }
        )
    by_worker = {row["worker_id"]: row for row in selected}
    missing = sorted(set(worker_ids) - set(by_worker))
    if missing:
        raise RuntimeError(f"Missing Ray worker nodes: {missing}; observed={selected}")
    selected = [by_worker[worker_id] for worker_id in worker_ids]
    host_ids = {row["physical_host_id"] for row in selected}
    addresses = {row["address"] for row in selected}
    node_ids = {row["ray_node_id"] for row in selected}
    if len(host_ids) != len(selected):
        raise RuntimeError(
            "Cross-host validation requires distinct physical-host markers; "
            f"workers={selected}"
        )
    if len(addresses) != len(selected) or len(node_ids) != len(selected):
        raise RuntimeError(
            "Cross-host validation requires distinct Ray nodes and addresses; "
            f"workers={selected}"
        )
    if any(row["gpu_count"] < 1 for row in selected):
        raise RuntimeError(f"Every target worker needs one Ray GPU: {selected}")
    return selected


def wait_for_instance(model_name, timeout, *, require_idle=False):
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        try:
            router = ray.get_actor(model_name, namespace="models")
            latest = ray.get(router.get_instance_states.remote(), timeout=15)
            for instance_id, state in latest.items():
                ready = state.get("pool") == "ready" and state.get("state") == "ready"
                idle = int(state.get("concurrency", 0) or 0) == 0
                if ready and (idle or not require_idle):
                    return instance_id, state
        except (ValueError, ray.exceptions.GetTimeoutError):
            pass
        time.sleep(0.5)
    qualifier = "idle ready" if require_idle else "ready"
    raise TimeoutError(f"No {qualifier} instance for {model_name}; states={latest}")


def runtime_profile(instance_id, state):
    actor = ray.get_actor(instance_id, namespace="sllm")
    metadata = ray.get(
        actor.get_runtime_metadata.remote(
            instance_id=instance_id,
            node_id=str(state.get("node_id") or ""),
        ),
        timeout=120,
    )
    return metadata["model_resource_profile"]


def expert_owners(profile):
    shards = profile.get("runtime_expert_placement_shards", {})
    if not profile.get("runtime_expert_placement_available") or not shards:
        raise RuntimeError("vLLM did not return actual expert placement")
    owners = {}
    for expert_key, rows in shards.items():
        ranks = sorted(
            {
                int(row["ep_rank"])
                for row in rows
                if isinstance(row, dict) and row.get("ep_rank") is not None
            }
        )
        if len(ranks) != 1:
            raise RuntimeError(
                f"Cross-host fixed-EP gate expects one owner per expert: "
                f"{expert_key}={ranks}"
            )
        owners[str(expert_key)] = ranks[0]
    return owners


def swapped_plan(model_name, owners):
    rank_count = len(set(owners.values()))
    if rank_count != 2 or set(owners.values()) != {0, 1}:
        raise RuntimeError(f"Cross-host gate requires an EP2 layout: {owners}")
    target = {
        key: f"replica:0/ep-rank:{1 - rank}"
        for key, rank in sorted(owners.items())
    }
    fingerprint = hashlib.sha256(
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "model_name": model_name,
        "placement_epoch": 1,
        "placement_source": "cross_host_validation",
        "placement_fingerprint": fingerprint,
        "expert_to_target_rank": target,
        "expert_to_target_ranks": {key: [value] for key, value in target.items()},
        "required_expert_count": len(target),
        "covered_expert_count": len(target),
        "planned_shard_count": len(target),
        "target_rank_count": 2,
        "expert_physical_replication_factor": 1,
        "sllm_replica_count": 1,
        "live_expert_remap": True,
        "allow_active_requests": False,
        "require_cross_node": True,
        "physical_migration_required": True,
        "expert_placement_physical_migration_required": True,
        "target_parallel_plan": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 2,
            "enable_expert_parallel": True,
            "effective_expert_parallel_size": 2,
        },
    }


def shifted_plan(
    model_name,
    owners,
    *,
    rank_count,
    rank_shift,
    require_cross_node=False,
    require_cross_failure_domain=False,
):
    observed_ranks = set(owners.values())
    if observed_ranks != set(range(rank_count)):
        raise RuntimeError(
            f"Expected EP{rank_count} owners, observed={owners}"
        )
    target = {
        key: f"replica:0/ep-rank:{(rank + rank_shift) % rank_count}"
        for key, rank in sorted(owners.items())
    }
    fingerprint = hashlib.sha256(
        json.dumps(target, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "model_name": model_name,
        "placement_epoch": 1,
        "placement_source": "failure_domain_validation",
        "placement_fingerprint": fingerprint,
        "expert_to_target_rank": target,
        "expert_to_target_ranks": {key: [value] for key, value in target.items()},
        "required_expert_count": len(target),
        "covered_expert_count": len(target),
        "planned_shard_count": len(target),
        "target_rank_count": rank_count,
        "expert_physical_replication_factor": 1,
        "sllm_replica_count": 1,
        "live_expert_remap": True,
        "allow_active_requests": False,
        "require_cross_node": bool(require_cross_node),
        "require_cross_failure_domain": bool(require_cross_failure_domain),
        "physical_migration_required": True,
        "expert_placement_physical_migration_required": True,
        "target_parallel_plan": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": rank_count,
            "enable_expert_parallel": True,
            "effective_expert_parallel_size": rank_count,
        },
    }


def infer(endpoint, model_name, timeout):
    response = post_json(
        endpoint,
        "v1/chat/completions",
        {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain expert parallel inference in two sentences.",
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        },
        timeout,
    )
    if response.get("error") or not response.get("choices"):
        raise RuntimeError(f"Inference failed: {response}")


def cleanup_model(endpoint, model_name, instance_ids):
    try:
        post_json(endpoint, "delete", {"model": model_name}, 120)
    finally:
        for instance_id in instance_ids:
            try:
                ray.kill(ray.get_actor(instance_id, namespace="sllm"), no_restart=True)
            except ValueError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/spotserve/"
            "config-vllm-cross-host-expert-remap-performance.json"
        ),
    )
    parser.add_argument("--model-path", default="/models/Qwen2-MoE-Tiny")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8343")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--event-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--target-worker-id", action="append", default=[])
    parser.add_argument("--require-internode-a2a", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/spotserve_cross_host_expert_remap_performance/report.json"
        ),
    )
    args = parser.parse_args()

    worker_ids = args.target_worker_id or ["0", "1"]
    if len(worker_ids) != 2 or len(set(worker_ids)) != 2:
        raise RuntimeError("Exactly two distinct target worker IDs are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = f"vllm-cross-host-remap-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["backend_config"]["pretrained_model_name_or_path"] = args.model_path
    config["backend_config"]["spotserve_target_worker_nodes"] = worker_ids
    config["router_config"]["metrics_path"] = str(
        args.output.with_suffix(".jsonl").resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto", namespace="sllm")
    workers = target_worker_nodes(worker_ids)
    registered = False
    instance_ids = set()
    try:
        registered = True
        post_json(args.endpoint, "register", config, args.ready_timeout)
        instance_id, state = wait_for_instance(model_name, args.ready_timeout)
        instance_ids.add(instance_id)
        infer(args.endpoint, model_name, args.request_timeout)
        instance_id, state = wait_for_instance(
            model_name, args.ready_timeout, require_idle=True
        )
        before = runtime_profile(instance_id, state)
        before_owners = expert_owners(before)
        plan = swapped_plan(model_name, before_owners)

        actor = ray.get_actor(instance_id, namespace="sllm")
        apply_result = ray.get(
            actor.apply_expert_placement_plan.remote(
                expert_placement_plan=plan
            ),
            timeout=args.event_timeout,
        )
        if not (
            apply_result.get("success")
            and apply_result.get("applied")
            and apply_result.get("verified")
        ):
            raise RuntimeError(f"Cross-host remap failed: {apply_result}")

        after_apply = runtime_profile(instance_id, state)
        after_owners = expert_owners(after_apply)
        changed = {
            key: {"before_ep_rank": before_owners[key], "after_ep_rank": rank}
            for key, rank in after_owners.items()
            if before_owners.get(key) != rank
        }
        required_runtime = {
            "verified_placement": after_apply.get(
                "expert_placement_runtime_verified_placement"
            ),
            "physical_weight_migration": after_apply.get(
                "expert_placement_physical_weight_migration"
            ),
            "cross_node_weight_migration": after_apply.get(
                "expert_placement_runtime_cross_node_weight_migration"
            ),
            "physical_host_count": after_apply.get(
                "expert_placement_runtime_physical_host_count"
            ),
            "ray_node_count": after_apply.get(
                "expert_placement_runtime_ray_node_count"
            ),
            "cross_node_moved_shards": after_apply.get(
                "expert_placement_runtime_cross_node_moved_expert_shards"
            ),
            "cross_node_moved_bytes": after_apply.get(
                "expert_placement_runtime_cross_node_moved_weight_bytes"
            ),
        }
        if not required_runtime["verified_placement"]:
            raise RuntimeError(
                f"Runtime placement was not verified: {required_runtime}"
            )
        if not required_runtime["physical_weight_migration"]:
            raise RuntimeError(f"No physical weight migration: {required_runtime}")
        if not required_runtime["cross_node_weight_migration"]:
            raise RuntimeError(f"No cross-host weight migration: {required_runtime}")
        if int(required_runtime["physical_host_count"] or 0) < 2:
            raise RuntimeError(
                f"Runtime observed fewer than two hosts: {required_runtime}"
            )
        if int(required_runtime["ray_node_count"] or 0) < 2:
            raise RuntimeError(
                f"Runtime observed fewer than two Ray nodes: {required_runtime}"
            )
        if int(required_runtime["cross_node_moved_shards"] or 0) < 1:
            raise RuntimeError(f"No expert shard crossed hosts: {required_runtime}")
        if int(required_runtime["cross_node_moved_bytes"] or 0) < 1:
            raise RuntimeError(f"No expert bytes crossed hosts: {required_runtime}")
        if not changed:
            raise RuntimeError("Expert ownership did not change")

        internode_before = int(
            after_apply.get("all_to_all_internode_calls", 0) or 0
        )
        infer(args.endpoint, model_name, args.request_timeout)
        final_profile = runtime_profile(instance_id, state)
        internode_after = int(final_profile.get("all_to_all_internode_calls", 0) or 0)
        if args.require_internode_a2a and internode_after <= internode_before:
            raise RuntimeError(
                "No post-remap internode A2A call was observed: "
                f"before={internode_before}, after={internode_after}"
            )

        report = {
            "model": model_name,
            "instance_id": instance_id,
            "physical_cross_host_verified": True,
            "target_workers": workers,
            "changed_expert_count": len(changed),
            "changed_experts": changed,
            "runtime": required_runtime,
            "runtime_physical_host_ids": after_apply.get(
                "expert_placement_runtime_physical_host_ids", []
            ),
            "runtime_ray_node_ids": after_apply.get(
                "expert_placement_runtime_ray_node_ids", []
            ),
            "internode_a2a_calls_before": internode_before,
            "internode_a2a_calls_after": internode_after,
            "internode_a2a_observed": internode_after > internode_before,
            "apply_result": apply_result,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "Cross-host expert remap verified: "
            f"hosts={required_runtime['physical_host_count']}, "
            f"changed_experts={len(changed)}, "
            f"cross_host_bytes={required_runtime['cross_node_moved_bytes']}, "
            f"internode_a2a_calls={internode_after - internode_before}; "
            f"report={args.output}"
        )
    finally:
        if registered:
            try:
                cleanup_model(args.endpoint, model_name, instance_ids)
            except Exception as exc:
                print(f"Cleanup warning for {model_name}: {exc}")


if __name__ == "__main__":
    main()
