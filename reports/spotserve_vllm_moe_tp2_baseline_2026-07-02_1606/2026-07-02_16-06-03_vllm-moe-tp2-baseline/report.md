# SpotServe Benchmark Report

## Run Metadata

- Run: `2026-07-02_16-06-03_vllm-moe-tp2-baseline`
- Policy: `none`
- Backend: `vllm`
- Model: `vllm-moe-baseline`
- Trace: `None`
- Workload: `benchmarks/spotserve/workloads/vllm_moe_trace.jsonl`

## Summary

| Requests | Success Rate | P50 | P95 | P99 | Throughput |
|---:|---:|---:|---:|---:|---:|
| 6 | 100.00% | 1674.61 ms | 2179.37 ms | 2179.37 ms | 0.19 req/s |

## Recovery Correctness

| Router Metrics Rows | Triggered Requests | Failed Attempts | Retry Count | Recovered Tokens | Fallbacks | Replay Succeeded |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Instance State Metrics

| Instance Events | Preempting | Ready | Dead | Draining |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |

Open `report.html` for the visual report.
