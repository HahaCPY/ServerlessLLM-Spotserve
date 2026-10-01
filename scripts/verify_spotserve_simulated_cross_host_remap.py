"""Verify a 2+2 GPU expert remap across two simulated host domains."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import ray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_spotserve_cross_host_expert_remap import (
    HOST_RESOURCE_PREFIX,
    cleanup_model,
    expert_owners,
    infer,
    post_json,
    runtime_profile,
    shifted_plan,
    wait_for_instance,
)


FAILURE_DOMAIN_RESOURCE_PREFIX = "spotserve_failure_domain_"


def simulated_worker_nodes(worker_ids):
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
        physical_markers = sorted(
            key[len(HOST_RESOURCE_PREFIX) :]
            for key, value in resources.items()
            if key.startswith(HOST_RESOURCE_PREFIX) and float(value or 0) > 0
        )
        domain_markers = sorted(
            key[len(FAILURE_DOMAIN_RESOURCE_PREFIX) :]
            for key, value in resources.items()
            if key.startswith(FAILURE_DOMAIN_RESOURCE_PREFIX)
            and float(value or 0) > 0
        )
        if len(physical_markers) != 1 or len(domain_markers) != 1:
            raise RuntimeError(
                "Each simulated host needs one physical marker and one "
                f"failure-domain marker: node={node.get('NodeID')}"
            )
        selected.append(
            {
                "worker_id": matched[0],
                "ray_node_id": str(node.get("NodeID") or ""),
                "address": str(node.get("NodeManagerAddress") or ""),
                "physical_host_id": physical_markers[0],
                "failure_domain_id": domain_markers[0],
                "gpu_count": float(resources.get("GPU", 0) or 0),
            }
        )
    by_worker = {row["worker_id"]: row for row in selected}
    missing = sorted(set(worker_ids) - set(by_worker))
    if missing:
        raise RuntimeError(f"Missing simulated worker nodes: {missing}")
    selected = [by_worker[worker_id] for worker_id in worker_ids]
    if len({row["physical_host_id"] for row in selected}) != 1:
        raise RuntimeError(
            "Simulated-host gate requires one shared physical-host marker"
        )
    if len({row["failure_domain_id"] for row in selected}) != len(selected):
        raise RuntimeError(
            "Simulated-host gate requires distinct failure-domain markers"
        )
    if len({row["ray_node_id"] for row in selected}) != len(selected):
        raise RuntimeError("Simulated hosts must be distinct Ray nodes")
    if len({row["address"] for row in selected}) != len(selected):
        raise RuntimeError("Simulated hosts must have distinct container IPs")
    if any(row["gpu_count"] < 2 for row in selected):
        raise RuntimeError(
            f"Each simulated host needs two visible Ray GPUs: {selected}"
        )
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/spotserve/"
            "config-vllm-simulated-cross-host-expert-remap-performance.json"
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
            "results/spotserve_simulated_cross_host_expert_remap_performance/"
            "report.json"
        ),
    )
    args = parser.parse_args()

    worker_ids = args.target_worker_id or ["0", "1"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = f"vllm-simulated-host-remap-{uuid.uuid4().hex[:8]}"
    config["model"] = model_name
    config["backend_config"]["pretrained_model_name_or_path"] = args.model_path
    config["backend_config"]["spotserve_target_worker_nodes"] = worker_ids
    config["router_config"]["metrics_path"] = str(
        args.output.with_suffix(".jsonl").resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ray.init(address="auto", namespace="sllm")
    workers = simulated_worker_nodes(worker_ids)
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
        plan = shifted_plan(
            model_name,
            before_owners,
            rank_count=4,
            rank_shift=2,
            require_cross_failure_domain=True,
        )

        actor = ray.get_actor(instance_id, namespace="sllm")
        apply_result = ray.get(
            actor.apply_expert_placement_plan.remote(
                expert_placement_plan=plan
            ),
            timeout=args.event_timeout,
        )
        if not all(
            apply_result.get(key) for key in ("success", "applied", "verified")
        ):
            raise RuntimeError(f"Simulated-host remap failed: {apply_result}")

        after_apply = runtime_profile(instance_id, state)
        after_owners = expert_owners(after_apply)
        changed = {
            key: {"before_ep_rank": before_owners[key], "after_ep_rank": rank}
            for key, rank in after_owners.items()
            if before_owners.get(key) != rank
        }
        runtime = {
            "host_mode": after_apply.get(
                "expert_placement_runtime_host_mode"
            ),
            "physical_host_count": after_apply.get(
                "expert_placement_runtime_physical_host_count"
            ),
            "failure_domain_count": after_apply.get(
                "expert_placement_runtime_failure_domain_count"
            ),
            "ray_node_count": after_apply.get(
                "expert_placement_runtime_ray_node_count"
            ),
            "verified_placement": after_apply.get(
                "expert_placement_runtime_verified_placement"
            ),
            "physical_weight_migration": after_apply.get(
                "expert_placement_physical_weight_migration"
            ),
            "cross_physical_host_migration": after_apply.get(
                "expert_placement_runtime_cross_node_weight_migration"
            ),
            "cross_failure_domain_migration": after_apply.get(
                "expert_placement_runtime_cross_failure_domain_weight_migration"
            ),
            "cross_failure_domain_moved_shards": after_apply.get(
                "expert_placement_runtime_cross_failure_domain_moved_expert_shards"
            ),
            "cross_failure_domain_moved_bytes": after_apply.get(
                "expert_placement_runtime_cross_failure_domain_moved_weight_bytes"
            ),
        }
        if runtime["host_mode"] != "simulated_failure_domain":
            raise RuntimeError(f"Incorrect runtime host mode: {runtime}")
        if int(runtime["physical_host_count"] or 0) != 1:
            raise RuntimeError(f"Expected one physical host: {runtime}")
        if int(runtime["failure_domain_count"] or 0) != 2:
            raise RuntimeError(f"Expected two simulated hosts: {runtime}")
        if int(runtime["ray_node_count"] or 0) < 2:
            raise RuntimeError(f"Expected two Ray nodes: {runtime}")
        if not runtime["verified_placement"]:
            raise RuntimeError(f"Runtime placement not verified: {runtime}")
        if not runtime["physical_weight_migration"]:
            raise RuntimeError(f"Expert weights did not move: {runtime}")
        if runtime["cross_physical_host_migration"]:
            raise RuntimeError(f"Run was mislabeled as physical cross-host: {runtime}")
        if not runtime["cross_failure_domain_migration"]:
            raise RuntimeError(f"No cross-domain expert movement: {runtime}")
        if int(runtime["cross_failure_domain_moved_shards"] or 0) < 1:
            raise RuntimeError(f"No expert shard crossed domains: {runtime}")
        if int(runtime["cross_failure_domain_moved_bytes"] or 0) < 1:
            raise RuntimeError(f"No expert bytes crossed domains: {runtime}")
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
                "No inter-container A2A call was observed: "
                f"before={internode_before}, after={internode_after}"
            )

        report = {
            "model": model_name,
            "instance_id": instance_id,
            "host_mode": "simulated_failure_domain",
            "physical_cross_host_verified": False,
            "simulated_cross_host_verified": True,
            "target_workers": workers,
            "changed_expert_count": len(changed),
            "changed_experts": changed,
            "runtime": runtime,
            "internode_a2a_calls_before": internode_before,
            "internode_a2a_calls_after": internode_after,
            "inter_container_a2a_observed": internode_after > internode_before,
            "apply_result": apply_result,
        }
        args.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "Simulated cross-host expert remap verified: "
            "physical_hosts=1, simulated_hosts=2, "
            f"changed_experts={len(changed)}, "
            "cross_domain_bytes="
            f"{runtime['cross_failure_domain_moved_bytes']}, "
            f"inter_container_a2a_calls={internode_after - internode_before}; "
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
