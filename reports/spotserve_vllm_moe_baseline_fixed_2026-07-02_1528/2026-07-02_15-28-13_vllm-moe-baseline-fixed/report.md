# SpotServe Benchmark Report

## Run Metadata

- Run: `2026-07-02_15-28-13_vllm-moe-baseline-fixed`
- Policy: `none`
- Backend: `vllm`
- Model: `vllm-moe-baseline`
- Trace: `None`
- Workload: `benchmarks/spotserve/workloads/vllm_moe_trace.jsonl`

## Summary

| Requests | Success Rate | P50 | P95 | P99 | Throughput |
|---:|---:|---:|---:|---:|---:|
| 6 | 100.00% | 1668.80 ms | 2187.66 ms | 2187.66 ms | 0.19 req/s |

## Recovery Correctness

| Router Metrics Rows | Triggered Requests | Failed Attempts | Retry Count | Recovered Tokens | Fallbacks | Replay Succeeded |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Instance State Metrics

| Instance Events | Preempting | Ready | Dead | Draining |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |

Open `report.html` for the visual report.
