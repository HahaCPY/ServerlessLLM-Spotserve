# SpotServe Version 2: Recover Dispatch + Node Health

## Goal

Version 2 closes the main control-plane gap from Version 1:

```text
preempt / recover / dead
```

are now all dispatched through the controller and reflected in both router
instance state and scheduler node health.

This version does not implement spot-risk ranking, dynamic reparallelization,
KV cache migration, expert-aware scheduling, or cloud spot-provider integration.

## Implemented

### Recover Dispatch

Files:

```text
sllm/spot/preemption_simulator.py
sllm/controller.py
sllm/routers/roundrobin_router.py
sllm/utils.py
```

Trace replay now dispatches `recover` events:

```text
trace recover
  -> preemption_simulator
  -> controller.handle_recover()
  -> router.handle_recover()
  -> PREEMPTING instances become READY
```

Important behavior:

- `PREEMPTING` instances can recover to `READY`.
- `DEAD` instances are not revived by recover.
- recovered instances can accept new requests again.
- recover emits an instance-state metric with reason `trace_recover`.

### Scheduler Node Health

Files:

```text
sllm/utils.py
sllm/schedulers/fcfs_scheduler.py
sllm/schedulers/storage_aware_scheduler.py
sllm/controller.py
```

Version 2 adds scheduler-level node health:

```text
READY
PREEMPTING
DEAD
```

Node-level trace events update scheduler state:

```text
preempt(node_id) -> scheduler.mark_node_preempting(node_id)
recover(node_id) -> scheduler.mark_node_recovered(node_id)
dead(node_id)    -> scheduler.mark_node_dead(node_id)
```

Both FCFS and storage-aware scheduling now skip nodes that are not `READY`.

This is a hard health filter only. It does not rank nodes by spot risk.

### CLI Trace Replay

Files:

```text
sllm/cli/clic.py
sllm/cli/_cli_utils.py
```

Version 2 adds:

```bash
sllm replay-trace \
  --trace examples/spotserve/spot_trace_sample.jsonl \
  --speedup 10 \
  --ray-address auto \
  --ray-namespace sllm
```

The CLI wraps:

```bash
python -m sllm.spot.preemption_simulator
```

### Repeatable Setup Script

File:

```text
scripts/prepare_spotserve.sh
```

The script automates:

- build `sllm_head`
- recreate `sllm_head`
- wait for Ray
- wait for the SLLM HTTP API
- copy benchmark artifacts into `/tmp/spotserve-work`
- deploy the three dummy policy models
- print `sllm status`

Useful fast path when only benchmark files changed:

```bash
scripts/prepare_spotserve.sh --skip-build --skip-recreate
```

## How To Run

Prepare the environment:

```bash
scripts/prepare_spotserve.sh
```

Run the long benchmark from the head container:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_long.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --ray-address auto \
  --ray-namespace sllm
'
```

Replay the sample trace directly:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/sllm replay-trace \
  --trace examples/spotserve/spot_trace_sample.jsonl \
  --speedup 10 \
  --ray-address auto \
  --ray-namespace sllm
'
```

Expected trace log behavior:

```text
Replaying spot event: ... event='preempt' ...
Replaying spot event: ... event='recover' ...
Replaying spot event: ... event='dead' ...
Trace replay finished: ...
```

## Definition Of Done

Version 2 is complete when:

- `recover` is dispatched to the controller.
- the scheduler tracks `READY`, `PREEMPTING`, and `DEAD` node health.
- routers can recover matching `PREEMPTING` instances to `READY`.
- `DEAD` instances are not recovered.
- `sllm replay-trace` can replay synthetic traces from the CLI.
