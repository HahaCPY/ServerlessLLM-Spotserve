# SpotServe MoE Cross-host Experiment

This guide validates physical expert-weight movement between two distinct GPU
hosts. It is deliberately stricter than the same-host Podman setup: two
containers on one machine do not pass the cross-host gate.

## Topology

Use three Ray nodes:

```text
control host       GPU host A          GPU host B
sllm head          worker_id_0         worker_id_1
0 GPUs             1 visible GPU       1 visible GPU
                   host marker A       host marker B
```

The control host may be a CPU-only machine. The two GPU workers must have
different physical IP addresses and different `--physical-host-id` values.
The worker script exposes one GPU per host so vLLM Ray-DP `fill` placement
creates one EP rank on each physical host.

## Prerequisites

- All hosts can reach the control host on TCP 6379, 6700-6701 and
  10002-10100. NCCL/Gloo must also be allowed between the two GPU hosts; on a
  trusted private experiment network, allowing inter-host TCP traffic is the
  least surprising setup.
- The same container image, including the SpotServe vLLM patch, exists on all
  three hosts. Use an immutable tag or digest, not independently changing
  `latest` tags.
- `/work/spotserve-models/Qwen2-MoE-Tiny/config.json` exists on both GPU hosts.
- Each GPU host has one free GPU with enough memory for Qwen2-MoE-Tiny.
- The repository containing the benchmark runner exists on the control host.

Build and publish the image once, then pull the same tag on every host:

```bash
IMAGE=registry.example.org/serverlessllm/sllm:spotserve-cross-host
podman build -t "$IMAGE" .
podman push "$IMAGE"
```

On every host:

```bash
IMAGE=registry.example.org/serverlessllm/sllm:spotserve-cross-host
podman pull "$IMAGE"
podman image inspect "$IMAGE" --format '{{.Digest}}'
```

The printed digest must match across hosts.

## Start The Cluster

Replace the example addresses with private addresses reachable by every host.

On the control host:

```bash
IMAGE=registry.example.org/serverlessllm/sllm:spotserve-cross-host
scripts/deploy_spotserve_cross_host.sh head \
  --node-ip 10.0.0.10 \
  --image "$IMAGE" \
  --repo-root /work/containers/cpy/ServerlessLLM-Spotserve
```

On GPU host A:

```bash
IMAGE=registry.example.org/serverlessllm/sllm:spotserve-cross-host
scripts/deploy_spotserve_cross_host.sh worker \
  --head-address 10.0.0.10:6379 \
  --node-ip 10.0.0.11 \
  --worker-id 0 \
  --physical-host-id gpu-host-a \
  --model-folder /work/spotserve-models \
  --gpu-device 0 \
  --image "$IMAGE"
```

On GPU host B:

```bash
IMAGE=registry.example.org/serverlessllm/sllm:spotserve-cross-host
scripts/deploy_spotserve_cross_host.sh worker \
  --head-address 10.0.0.10:6379 \
  --node-ip 10.0.0.12 \
  --worker-id 1 \
  --physical-host-id gpu-host-b \
  --model-folder /work/spotserve-models \
  --gpu-device 0 \
  --image "$IMAGE"
```

The physical host IDs are hashed before being advertised as Ray custom
resources. They establish host identity without copying the driver's
environment into remote vLLM actors.

## Preflight

On the control host:

```bash
podman exec sllm_cross_host_head bash -lc '
/opt/venvs/head/bin/python - <<'"'"'PY'"'"'
import json
import ray

ray.init(address="auto", namespace="sllm")
rows = []
for node in ray.nodes():
    if not node.get("Alive"):
        continue
    resources = node.get("Resources") or {}
    rows.append({
        "address": node.get("NodeManagerAddress"),
        "gpu": resources.get("GPU", 0),
        "worker_0": resources.get("worker_id_0", 0),
        "worker_1": resources.get("worker_id_1", 0),
        "host_markers": sorted(
            key for key in resources
            if key.startswith("spotserve_physical_host_")
        ),
    })
print(json.dumps(rows, indent=2))
PY
'
```

Expected: one row with `worker_0=1`, one different-address row with
`worker_1=1`, one GPU on each, and a different host marker on each row.

## Run The Gate

On the control host:

```bash
podman exec sllm_cross_host_head bash -lc '
cd /workspace/ServerlessLLM-Spotserve &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_cross_host_expert_remap_performance.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 240 \
  --trace-event-timeout 600 \
  --ray-address auto \
  --ray-namespace sllm
'
```

The gate performs this sequence:

```text
verify two physical hosts and two Ray nodes
-> deploy one Ray-DP2 / EP2 model across worker 0 and worker 1
-> run inference and read actual runtime expert ownership
-> swap EP-rank ownership for every expert
-> physically transfer and verify expert tensors
-> run post-remap inference
-> require an observed internode A2A call
```

A valid success line looks like:

```text
Cross-host expert remap verified: hosts=2, changed_experts=..., cross_host_bytes=..., internode_a2a_calls=...;
```

The report is written to:

```text
results/spotserve_cross_host_expert_remap_performance/report.json
```

Do not accept the run unless all of these are true or positive:

```text
physical_cross_host_verified
runtime.verified_placement
runtime.physical_weight_migration
runtime.cross_node_weight_migration
runtime.physical_host_count >= 2
runtime.ray_node_count >= 2
runtime.cross_node_moved_shards > 0
runtime.cross_node_moved_bytes > 0
internode_a2a_observed
```

## What This Proves

Passing this gate proves physical expert-weight movement and post-remap MoE
inference across two real hosts, plus runtime-observed internode A2A activity.
It does not by itself prove that internode traffic is lower than a baseline;
that requires a paired baseline/remap workload and NIC or runtime counter
comparison on the same hosts.

The repository includes the implementation and fail-closed gate, but a final
passing report can only be produced on a real multi-host cluster.
