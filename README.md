# ServerlessLLM-Spotserve

This repository is a research branch of
[ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM). It extends the
ServerlessLLM control plane with SpotServe-style recovery experiments for spot
GPU serving.

The current goal is not to reproduce the full SpotServe system in one step.
Instead, this branch builds and validates the control-plane pieces needed for
preemption-aware serving:

- synthetic spot trace replay
- worker and node health state
- preemption, recovery, and dead-node dispatch
- retry and generated-token replay policies
- request-level recovery metrics
- dummy, transformers, vLLM dense, and vLLM MoE validation paths
- benchmark and report generation

Generated-token replay in this branch is best-effort prompt/token replay. It is
not true KV cache migration.

## Repository Status

| Area | Status | Notes |
| --- | --- | --- |
| Control-plane prototype | Done | Preemption-aware routing, policy configs, metrics, benchmark reports. |
| Recover dispatch and node health | Done | `preempt`, `recover`, and `dead` events update router and scheduler state. |
| Recovery correctness validation | Done | Forced dummy failures prove retry/replay metrics are actually triggered. |
| vLLM dense black-box integration | Done | Dense vLLM models deploy and run under synthetic traces. |
| Dynamic reparallelization planner | Prototype | Planner code and synthetic config exist; it does not rebuild workers online. |
| vLLM MoE black-box integration | Done | MoE configs, benchmark matrices, and reports validate vLLM as a black-box backend. |

Detailed notes live in:

- `docs/spotserve-first-version.md`
- `docs/spotserve-version2.md`
- `docs/spotserve-version3.md`
- `docs/spotserve-version4.md`
- `docs/spotserve-version5.md`
- `docs/spotserve-version5-vllm-moe.md`
- `CPY-plan.md`

## Layout

```text
sllm/spot/                       Spot trace, metrics, recovery, planner helpers
sllm/routers/                    Recovery-aware router behavior
sllm/schedulers/                 Node-health-aware scheduling
examples/spotserve/              Model deployment configs and spot traces
benchmarks/spotserve/            Benchmark matrices, workloads, runner
scripts/prepare_spotserve.sh     Repeatable local/container setup helper
scripts/analyze_spotserve_*      Summary and report generation
docs/spotserve-*.md              Implementation notes by milestone
results/                         Local benchmark output
```

## Ports

The compose setup maps the container API port to the host:

| Location | API URL |
| --- | --- |
| Host shell | `http://127.0.0.1:8344` |
| Inside `sllm_head` | `http://127.0.0.1:8343` |

Most benchmark commands in this README run inside `sllm_head`, so they use
port `8343`.

## Quick Start: Dummy SpotServe Benchmark

Build, recreate the head container, copy benchmark files, and deploy the three
standard dummy policy models:

```bash
scripts/prepare_spotserve.sh --deploy-set standard
```

Run the long dummy benchmark from inside the head container:

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

The standard matrix is useful for plumbing validation. If all policies finish
with `46/46` successes, that proves the trace, controller, router, and report
path can run to completion. It does not prove generated-token replay is better
than retry.

## Recovery Correctness Benchmark

Use this benchmark when you need to prove retry/replay actually triggered.

Prepare only the correctness dummy models:

```bash
scripts/prepare_spotserve.sh --deploy-set correctness
```

Run:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_recovery_correctness.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 30 \
  --skip-trace
'
```

Expected result shape:

```text
dummy-correctness-none: successes=0/2, failed_attempts=2, retries=0, recovered_tokens=0, fallbacks=0
dummy-correctness-naive-retry: successes=2/2, failed_attempts=2, retries=2, recovered_tokens=0, fallbacks=0
dummy-correctness-token-replay: successes=2/2, failed_attempts=2, retries=2, recovered_tokens>0, fallbacks>=0
```

This is the main correctness check for Version 3.

## vLLM Dense Black-box Benchmark

The vLLM dense path requires a GPU worker. The tested model is:

```text
Qwen/Qwen3-0.6B
```

Prepare:

```bash
export MODEL_FOLDER=$PWD/model
scripts/prepare_spotserve.sh --deploy-set vllm-dense
```

Run:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 120 \
  --ray-address auto \
  --ray-namespace sllm
'
```

The vLLM dense benchmark validates black-box control-plane compatibility. It is
normal for synthetic trace runs to show:

```text
failed_attempts=0, retries=0, recovered_tokens=0, fallbacks=0
```

That means the trace changed routing and instance state without killing an
in-flight vLLM generation. Use the recovery correctness benchmark above when
you need forced mid-generation failure evidence.

## vLLM MoE Black-box Benchmark

The vLLM MoE path validates the same SpotServe-style control plane against a
single MoE model served by vLLM as a black box.

Prepare:

```bash
export MODEL_FOLDER=$PWD/model
export SPOTSERVE_VLLM_MOE_MODEL=Qwen/Qwen1.5-MoE-A2.7B
export SPOTSERVE_VLLM_MOE_LOAD_FORMAT=auto
export SPOTSERVE_VLLM_MOE_TP=1
scripts/prepare_spotserve.sh --deploy-set vllm-moe
```

Run the MoE-only matrix:

```bash
podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
'
```

Run the dense-vs-MoE comparison:

```bash
scripts/prepare_spotserve.sh --deploy-set vllm-blackbox

podman exec sllm_head bash -lc '
cd /tmp/spotserve-work &&
/opt/venvs/head/bin/python benchmarks/spotserve/run_benchmark.py \
  --config benchmarks/spotserve/benchmark_matrix_vllm_dense_vs_moe.yaml \
  --endpoint http://127.0.0.1:8343/v1/chat/completions \
  --request-timeout 180 \
  --ray-address auto \
  --ray-namespace sllm
'
```

The MoE path intentionally does not implement expert routing, expert migration,
or MoE-specific recovery optimization.

## Dynamic Reparallelization Planner Prototype

The planner prototype lives in:

```text
sllm/spot/reparallelization.py
```

It can summarize available GPUs and generate a candidate parallel config after
`preempt`, `recover`, or `dead` events. The current prototype is planning-only:
it records what a new tensor/pipeline/data parallel layout should be, but it
does not restart vLLM workers or apply the new layout online.

Related files:

```text
examples/spotserve/config-dummy-reparallelization.json
examples/spotserve/spot_trace_reparallelization.jsonl
benchmarks/spotserve/benchmark_matrix_reparallelization.yaml
tests/spotserve_test/test_reparallelization_planner.py
```

## Reports and Results

Every benchmark run writes a timestamped directory under `results/` inside the
container workdir, for example:

```text
results/spotserve/2026-06-30_07-20-13_dummy-no-preemption/
```

Each run contains:

- `run_metadata.json`
- `raw_requests.jsonl`
- `summary.json`
- `summary.csv`
- `report.html`
- `report.md`
- `router_request_metrics.jsonl`, when matching router metrics are available

You can regenerate reports manually:

```bash
python scripts/analyze_spotserve_benchmark.py results/spotserve/<run_id>
python scripts/plot_spotserve_benchmark.py results/spotserve/<run_id>
```

## Common Commands

Check container status:

```bash
podman ps -a --filter name=sllm
```

Check Ray from the head container:

```bash
podman exec sllm_head /opt/venvs/head/bin/ray status
```

Check deployed models:

```bash
podman exec sllm_head /opt/venvs/head/bin/sllm status
```

Call the OpenAI-compatible endpoint from the host:

```bash
curl http://127.0.0.1:8344/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dummy-token-replay",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16
  }'
```

Fast setup path after editing only benchmark/config files:

```bash
scripts/prepare_spotserve.sh --skip-build --skip-recreate --deploy-set standard
```

## Troubleshooting

`ModuleNotFoundError: No module named 'ray'`

Run trace benchmarks inside `sllm_head`, or use `--skip-trace` for workloads
that do not need Ray trace replay from the host process.

`Endpoint health check failed ... Connection refused`

Use the right port for where the command is running. Host commands use `8344`;
commands inside `sllm_head` use `8343`. Also verify that `sllm_head` is still
running with `podman ps -a --filter name=sllm`.

`Cannot register model`

Check Ray status and deployed actors:

```bash
podman exec sllm_head /opt/venvs/head/bin/ray status
podman exec sllm_head /opt/venvs/head/bin/sllm status
```

For vLLM dense, also make sure the worker container has GPU resources and the
model path is visible through `MODEL_FOLDER`.

`sllm_head` exits with code `139`

This usually means the Ray/vLLM process crashed under resource pressure. For
vLLM dense, reduce the number of deployed policy aliases or restart with the
dedicated `vllm-dense` deploy set instead of `all`.

`/work` is full

Podman overlay storage can fill `/work` even when the repository is small. Check:

```bash
du -h --max-depth=1 /work/containers/cpy/storage 2>/dev/null | sort -h
```

Only prune storage that belongs to your own `/work/containers/cpy` area.

## Development Checks

Run lightweight syntax checks:

```bash
python -m py_compile \
  sllm/spot/reparallelization.py \
  sllm/spot/metrics.py \
  benchmarks/spotserve/run_benchmark.py \
  scripts/analyze_spotserve_benchmark.py \
  scripts/plot_spotserve_benchmark.py
```

Run the planner unit test:

```bash
python -m pytest tests/spotserve_test/test_reparallelization_planner.py
```

## Roadmap

The active roadmap is in `CPY-plan.md`. Near-term work focuses on:

- dynamic reparallelization beyond planning-only output
- low-cost context migration planning
- stateful inference recovery
- spot-risk-aware scheduling
- expert-aware MoE recovery

## Upstream ServerlessLLM

This branch is based on ServerlessLLM, an Apache-2.0 project for low-latency
serverless LLM inference with fast checkpoint loading and GPU multiplexing.

Useful upstream links:

- Documentation: <https://serverlessllm.github.io>
- Paper: <https://www.usenix.org/system/files/osdi24-fu.pdf>
- Original project: <https://github.com/ServerlessLLM/ServerlessLLM>

If you use the upstream ServerlessLLM work in research, cite:

```bibtex
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={OSDI'24},
  year={2024}
}
```

## License

Apache 2.0. See `LICENSE`.
