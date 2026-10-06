# Two-workspace K8s pool: 1 Ray head + 8 one-GPU workers

This package joins two Kubernetes workspaces, each capped at four GPUs, into
one Ray cluster. Workspace A hosts the zero-GPU ServerlessLLM/Ray head and GPU
workers `0..3`; workspace B hosts GPU workers `4..7`.

This is usable only when both workspaces are in the same Kubernetes cluster,
or otherwise have routable Pod networks and cross-workspace DNS/service
connectivity. Deploying the manifests proves neither cross-pod NIXL restore
nor SpotServe recovery; those remain capability gates.

## Prerequisites

- Two existing namespaces with a four-GPU quota each.
- Permission to deploy into both namespaces from one `kubectl` context.
- A container image built from this repository. Formal runs must use an
  immutable `@sha256:` image reference.
- A model PVC in each namespace containing the same checkpoint at the same
  absolute path. If workers may land on different physical nodes, the storage
  class must allow the same volume to be mounted read-only from all of those
  nodes (for example RWX/ROX where supported); a node-local RWO volume can
  leave pods unschedulable. The experiment preflight verifies checkpoint
  hashes after the pods start.
- A results PVC in workspace A.
- Unrestricted bidirectional Pod networking between both namespaces. Ray,
  NCCL and NIXL use Pod IPs and dynamic data-plane connections; exposing only
  the head Service is insufficient when a default-deny NetworkPolicy exists.

The manifests do not create namespaces or PVCs and do not modify quotas.

## Git checkout and model data

You may clone the whole repository in both workspaces, and both checkouts must
use the exact same commit. Kubernetes pods, however, run the files baked into
the container image; a `git pull` on a workspace filesystem does not update an
already running pod unless that checkout is explicitly mounted. For a formal
run, build and push one image from the selected commit, use its immutable
`@sha256:` reference in both workspaces, and let the preflight reject any image
or checkpoint mismatch.

Do **not** delete the model checkpoint merely because the repository is
cloned or updated. Keep an identical checkpoint in the two namespace-local
model PVCs, mounted at `/models` as read-only. The formal runner hashes the
model files observed by every GPU worker. Fields such as
`delete_models_before_run` delete a ServerlessLLM deployment/actor through the
HTTP API; they do not delete checkpoint files from the PVC.

## Render

From the repository root:

```bash
python scripts/render_k8s_two_workspace.py \
  --workspace-a-namespace <workspace-a-namespace> \
  --workspace-b-namespace <workspace-b-namespace> \
  --image '<registry>/serverlessllm@sha256:<digest>' \
  --image-digest 'sha256:<digest>' \
  --model-pvc-a <model-pvc-in-a> \
  --model-pvc-b <model-pvc-in-b> \
  --results-pvc-a <results-pvc-in-a> \
  --output-dir /tmp/spotserve-k8-rendered
```

The command writes:

- `spotserve-two-workspace-8gpu.yaml`
- `rendered-values.json`
- `spot_trace_k8_two_workspace_churn.jsonl`

It refuses to overwrite these files unless `--force` is provided. A mutable
image tag is rejected unless `--allow-mutable-image` is explicitly used for a
non-formal smoke.

## Network check before apply

Confirm that workspace B can resolve and reach the future workspace-A head
Service name:

```text
<head-service>.<workspace-a-namespace>.svc.cluster.local:6379
```

If either namespace has default-deny NetworkPolicy, ask the cluster operator
to allow all ingress and egress between pods labelled with the selected
`spotserve.openai/cluster` value. Do not assume that opening TCP 6379 alone is
enough: Ray workers, NCCL collectives and NIXL transfers communicate directly
over Pod IPs.

## Apply and observe

Applying this rendered file changes the two Kubernetes workspaces:

```bash
kubectl apply -f /tmp/spotserve-k8-rendered/spotserve-two-workspace-8gpu.yaml
```

Observe without restarting anything:

```bash
kubectl get pods -n <workspace-a-namespace> -o wide
kubectl get pods -n <workspace-b-namespace> -o wide
kubectl logs -n <workspace-a-namespace> deploy/<cluster-name>-head
kubectl logs -n <workspace-a-namespace> statefulset/<cluster-name>-worker-a
kubectl logs -n <workspace-b-namespace> statefulset/<cluster-name>-worker-b
```

The worker StatefulSets derive stable IDs from their ordinals:

```text
workspace A: 0, 1, 2, 3
workspace B: 4, 5, 6, 7
```

Every worker advertises its Pod IP to Ray, the Kubernetes `spec.nodeName` as
its physical-host marker, and its workspace namespace as a separate failure
domain marker. Workers do not use `hostNetwork`.

## Verify the live Ray pool

Run inside the head pod or another process connected to the same Ray cluster:

```bash
/opt/venvs/head/bin/python /app/scripts/verify_k8s_two_workspace_pool.py \
  --workspace-a-namespace <workspace-a-namespace> \
  --workspace-b-namespace <workspace-b-namespace> \
  --ray-address auto --ray-namespace sllm \
  --output /results/two-workspace-pool.json
```

The verifier is read-only. It requires exactly:

- one live zero-GPU `control_node`;
- eight live one-GPU Ray nodes;
- unique worker IDs `0..7` and unique Pod IPs;
- four workers carrying workspace A's failure-domain marker;
- four workers carrying workspace B's failure-domain marker;
- one physical-host marker on every worker.

`cross_physical_host_available=true` means at least two Kubernetes
`spec.nodeName` values were observed. It does not prove that every worker is a
separate failure domain.

## Smoke, then recovery gate

Run the existing non-preempting smoke first from the head pod:

```bash
/opt/venvs/head/bin/python -u /app/scripts/run_k8s_moe_f1_f2.py \
  --config benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 1 \
  --phase smoke \
  --model-path /models/Qwen1.5-MoE-A2.7B-Chat \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --ray-address auto --ray-namespace sllm \
  --output-dir /results/k8s-smoke
```

Only after smoke passes should the cross-worker core probe be attempted. The
current repository documents a fail-closed cross-worker KV limitation in
`docs/spotserve-k8s-core-verification.md`; a successful eight-worker startup
must not be reported as cross-node KV recovery.

## One-command F1/F2 pipeline

The container image includes one fail-closed entry point. Run it inside the
workspace-A head pod after all eight workers are READY:

```bash
/opt/venvs/head/bin/python -u /app/scripts/run_k8s_two_workspace_f1_f2.py \
  --workspace-a-namespace <workspace-a-namespace> \
  --workspace-b-namespace <workspace-b-namespace> \
  --config /app/benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json \
  --repeats 3 \
  --output-dir /results/k8s-f1-f2-$(date +%Y%m%d-%H%M%S) \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --ray-address auto --ray-namespace sllm
```

The wrapper runs these gates in order and stops at the first failure:

1. Verify exactly one zero-GPU head and workers `0..7`, split 4+4.
2. Run a short two-EP2 inference smoke.
3. Measure and freeze the DP2/EP2, DP3/EP3 and DP4/EP4 profiles.
4. Verify live cross-worker KV recovery and dynamic target selection.
5. Run the pre-registered F1 and F2 matrix and render the final report.

The formal matrix uses Qwen1.5-MoE-A2.7B, an 8192-token input, a 1024-token
output, and schedules preemption near output token 512 using the measured
decode profile. F1 contains Rerouting, Reparallelization, Original SpotServe
and MoE-SpotServe. F2 contains Original SpotServe, reparallelization-only,
migration-only and full MoE-SpotServe. Each treatment has three paired formal
repeats. Completed valid runs are reused on resume; a failed smoke or KV probe
is never retried automatically.

Main outputs under `--output-dir` are:

- `pipeline-summary.md` and `pipeline-summary.json`: Chinese stage summary.
- `results.md`: F1/F2 aggregate report.
- `run-ledger.json`: immutable per-run index and summary metrics.
- `runs/`: per-treatment raw requests, metrics, metadata and analyzer output.
- `preflight/`, `profiles/`, `pilot/`, `smoke-attempts/` and
  `core-probe-attempts/`: gate evidence.

## Adapted eight-slot churn trace

`examples/spotserve/spot_trace_k8_two_workspace_churn.jsonl` adapts the supplied
trace to a fixed pool of eight allocated GPU slots:

```text
t=0       workers 0..7 already exist (capacity 8)
t=90      worker 4 PREEMPTING; auto-DEAD at t=120 (capacity 7)
t=150     workers 1 and 5 PREEMPTING; auto-DEAD at t=180 (capacity 5)
t=210     worker 4 RECOVER (capacity 6)
t=270     worker 1 RECOVER (capacity 7)
t=330     worker 5 RECOVER (capacity 8)
```

The trace omits the original fourth add because it would raise capacity to 9,
above the combined quota. It also omits `DONE`, which is not a supported trace
event. Each multi-node event is expanded into one JSONL row per worker.

`recover` is a logical capacity event. For a real replacement-pod experiment,
do not send it until the replacement pod has joined Ray and advertises the
expected worker ID. A trace event cannot create Kubernetes capacity.

The trace is a multi-churn capability input, not yet the preregistered F1/F2
trace. The formal runner currently generates a single workload-aligned
preemption dynamically from the discovered initial EP2 membership.
