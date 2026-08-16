# Trace baseline vs. SpotServe KV restore

## Run

- Model: `/work/spotserve-models/Qwen2-MoE-Tiny`
- Trace: `examples/spotserve/spot_trace_local_three_gpu_add_remove_churn.jsonl`
- Host GPUs: 0, 1, 3 (GPU 2 was occupied by an unrelated container)
- Container fleet: maximum 3 live containers in this run
- Trace states: `add`, `remove`, `DONE`; 14 trace events
- Physical cross-node transfer: no; this is a same-host multi-container simulation
- One source preemption/migration was triggered by the first source `remove` event

The same model, trace, request, token delay, and worker settings were run once
in each mode. The first earlier trial was discarded because it used generated
output-token count as the context length; this report uses `computed_tokens`,
which is the full source context length.

## Results

| Metric | Original baseline (full recompute) | SpotServe (NIXL KV restore) |
|---|---:|---:|
| Run status | passed | passed |
| Harness elapsed | 1231.763 s | 1230.516 s |
| Trace execution elapsed | 1217.79 s | 1216.61 s |
| Source computed context | 66 tokens | 66 tokens |
| Source completed output | 2 tokens | 2 tokens |
| Source KV blocks | 5 | 5 |
| Restored KV blocks | 0 | 5 |
| Context tokens recomputed after migration | 66 | 0 |
| Context tokens saved | — | 66 |
| Migration interval | 22.334 s | 22.180 s |
| Restore successes | 0 (not used by baseline) | 1 |
| Restore fallbacks | 0 | 0 |
| Target continued after source stop | yes | yes |

The migration interval differed by only 0.154 s. The full run time was
dominated by starting and stopping vLLM containers, so this one-run result
should be interpreted as a functional KV-restore comparison, not a production
throughput or latency benchmark. The main measured benefit is that SpotServe
avoided recomputing all 66 already-computed context tokens while preserving
the request after source preemption.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0,1,3 \
PYTHONPATH=/work/containers/s112060021/Qwen3/vllm:/work/containers/s112060021/Qwen3/ServerlessLLM-Spotserve \
/work/containers/s112060021/Qwen3/vllm/.venv/bin/python -u \
tests/spotserve_test/run_trace_baseline_vs_spotserve.py \
  --model /work/spotserve-models/Qwen2-MoE-Tiny \
  --trace examples/spotserve/spot_trace_local_three_gpu_add_remove_churn.jsonl \
  --gpus 0 1 3 --trace-speedup 1000 --token-delay-s 0.1 \
  --timeout-s 360 \
  --output /tmp/spotserve-trace-baseline-vs-nixl-tiny-v2.json
```

The raw JSON result is `/tmp/spotserve-trace-baseline-vs-nixl-tiny-v2.json`.
This run validates the trace-driven source/target handoff and NIXL restore;
it does not yet measure a physical cross-node link or a live planner plan
reselection/traffic-switch benchmark.
