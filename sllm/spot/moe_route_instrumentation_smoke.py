"""Real vLLM smoke for SpotServe MoE route instrumentation."""

import asyncio
import json
import os
import time
from typing import Any

from sllm.backends.vllm_backend import VllmBackend


def _first_moe_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        metadata = row.get("metadata", {})
        if _has_runtime_topk_histogram(metadata):
            return metadata
    return {}


def _has_runtime_topk_histogram(metadata: dict[str, Any]) -> bool:
    histogram = metadata.get("per_request_expert_route_histogram")
    return (
        metadata.get("moe_route_histogram_available")
        and metadata.get("moe_route_histogram_source") == "vllm_runtime_topk"
        and metadata.get("moe_route_histogram_kind") == "runtime_observed_topk"
        and isinstance(histogram, dict)
        and bool(histogram)
    )


async def main() -> None:
    model_path = os.getenv(
        "SPOTSERVE_MOE_ROUTE_INSTRUMENTATION_MODEL",
        "/work/spotserve-models/Qwen2-MoE-Tiny",
    )
    backend = VllmBackend(
        "moe-route-instrumentation-smoke",
        {
            "pretrained_model_name_or_path": model_path,
            "load_format": "auto",
            "torch_dtype": "float16",
            "gpu_memory_utilization": float(
                os.getenv("SPOTSERVE_MOE_ROUTE_GPU_MEMORY_UTILIZATION", "0.10")
            ),
            "max_model_len": int(
                os.getenv("SPOTSERVE_MOE_ROUTE_MAX_MODEL_LEN", "256")
            ),
            "max_num_seqs": 2,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "enable_prefix_caching": False,
            "enable_moe_route_instrumentation": True,
            "enforce_eager": True,
            "trust_remote_code": True,
            "trace_debug": True,
            "stop_timeout_s": 1.0,
        },
    )
    started = time.monotonic()
    await backend.init_backend()

    request_id = "moe-route-instrumentation-smoke-1"
    request_task = asyncio.create_task(
        backend.generate(
            {
                "request_id": request_id,
                "model": "moe-route-instrumentation-smoke",
                "prompt": (
                    "Route this request through a tiny MoE model and continue "
                    "with deterministic short text."
                ),
                "max_tokens": 64,
                "ignore_eos": True,
                "temperature": 0.0,
                "_spotserve_token_delay_s": 0.05,
            }
        )
    )

    observed: dict[str, Any] = {}
    last_metadata: dict[str, Any] = {}
    deadline = time.monotonic() + float(
        os.getenv("SPOTSERVE_MOE_ROUTE_OBSERVE_TIMEOUT_S", "60")
    )
    try:
        while time.monotonic() < deadline:
            request_metadata = await backend._request_runtime_moe_metadata(
                request_id
            )
            if request_metadata:
                last_metadata = request_metadata
            if _has_runtime_topk_histogram(request_metadata):
                observed = request_metadata
                break

            request_context = await backend.get_request_kv_metadata(
                request_id=request_id,
                instance_id="moe-route-instrumentation-smoke",
                node_id="local",
            )
            context_metadata = request_context.get("metadata", {})
            if isinstance(context_metadata, dict) and context_metadata:
                last_metadata = context_metadata
            if (
                isinstance(context_metadata, dict)
                and _has_runtime_topk_histogram(context_metadata)
            ):
                observed = context_metadata
                break

            rows = await backend.get_context_metadata(
                instance_id="moe-route-instrumentation-smoke",
                node_id="local",
            )
            if rows:
                row_metadata = rows[0].get("metadata", {})
                if isinstance(row_metadata, dict) and row_metadata:
                    last_metadata = row_metadata
            observed = _first_moe_metadata(rows)
            if observed:
                break
            if request_task.done():
                break
            await asyncio.sleep(0.1)

        result = await asyncio.wait_for(request_task, timeout=120)
        report = {
            "status": "passed" if observed else "failed",
            "elapsed_s": round(time.monotonic() - started, 3),
            "model_path": model_path,
            "request_id": request_id,
            "moe_route_histogram_source": observed.get(
                "moe_route_histogram_source",
                last_metadata.get("moe_route_histogram_source", "unavailable"),
            ),
            "moe_route_histogram_kind": observed.get(
                "moe_route_histogram_kind",
                last_metadata.get("moe_route_histogram_kind", "unavailable"),
            ),
            "per_request_expert_route_histogram": observed.get(
                "per_request_expert_route_histogram",
                last_metadata.get("per_request_expert_route_histogram", {}),
            ),
            "reason": observed.get("reason", last_metadata.get("reason")),
            "response_error": result.get("error"),
        }
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if not observed:
            raise AssertionError(report)
    finally:
        await backend.request_trace.delete_request(request_id)
        await backend.stop()


if __name__ == "__main__":
    asyncio.run(main())
