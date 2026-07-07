# SpotServe Benchmark Report

## Run Metadata

- Run: `2026-07-02_16-09-04_vllm-moe-tp2-preemption-none`
- Policy: `none`
- Backend: `vllm`
- Model: `vllm-moe-none`
- Trace: `examples/spotserve/spot_trace_vllm_moe_none.jsonl`
- Workload: `benchmarks/spotserve/workloads/vllm_moe_trace.jsonl`

## Summary

| Requests | Success Rate | P50 | P95 | P99 | Throughput |
|---:|---:|---:|---:|---:|---:|
| 6 | 100.00% | 8722.58 ms | 19838.50 ms | 19838.50 ms | 0.19 req/s |

## Recovery Correctness

| Router Metrics Rows | Triggered Requests | Failed Attempts | Retry Count | Recovered Tokens | Fallbacks | Replay Succeeded |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Instance State Metrics

| Instance Events | Preempting | Ready | Dead | Draining |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |

Open `report.html` for the visual report.
