"""Native sparse-MoE EP2 -> EP1 graceful rank-removal feasibility only."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import traceback


async def rank_snapshots(engine):
    core = engine.engine_core
    groups = await asyncio.gather(
        *[
            core._call_utility_async(
                "collective_rpc",
                "get_elastic_ep_phase1_snapshot",
                90,
                (),
                {},
                engine=identity,
            )
            for identity in core.core_engines
        ]
    )
    return [snapshot for group in groups for snapshot in group]


def coverage(snapshots):
    by_layer = {}
    for snapshot in snapshots:
        for layer in snapshot["layers"]:
            entry = by_layer.setdefault(
                layer["layer"],
                {"expected": set(range(layer["total_experts"])), "present": set()},
            )
            entry["present"].update(layer["expert_ids"])
    return {
        str(layer): {
            "expected_count": len(entry["expected"]),
            "present_count": len(entry["present"]),
            "missing": sorted(entry["expected"] - entry["present"]),
        }
        for layer, entry in sorted(by_layer.items())
    }


async def generate(engine, prompt, request_id):
    from vllm import SamplingParams

    last = None
    async for result in engine.generate(
        prompt,
        SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, seed=0),
        request_id,
    ):
        last = result
    if last is None or not last.outputs:
        raise RuntimeError("inference_returned_no_output")
    return list(last.outputs[0].token_ids)


async def run(args):
    import ray

    from vllm.config.parallel import EPLBConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    report = {
        "experiment": "elastic_ep_phase1_graceful_rank_removal",
        "model": args.model,
        "formal_migration_ablation": False,
        "vllm_source_modified": False,
        "status": "started",
    }
    engine = None
    try:
        ray.init(
            num_cpus=4,
            num_gpus=2,
            object_store_memory=268435456,
            include_dashboard=False,
            _temp_dir="/taskresults/ray",
        )
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive_nodes) != 1:
            raise RuntimeError("expected_one_isolated_ray_node")
        ray_node_ip = str(alive_nodes[0]["NodeManagerAddress"])
        os.environ["VLLM_HOST_IP"] = ray_node_ip
        report["ray_node_ip"] = ray_node_ip
        started = time.monotonic()
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=args.model,
                tensor_parallel_size=1,
                data_parallel_size=2,
                data_parallel_size_local=2,
                data_parallel_backend="ray",
                data_parallel_address=ray_node_ip,
                enable_expert_parallel=True,
                enable_elastic_ep=True,
                enable_eplb=True,
                eplb_config=EPLBConfig(
                    use_async=False,
                    num_redundant_experts=0,
                    step_interval=1000000,
                    window_size=5,
                ),
                all2all_backend="allgather_reducescatter",
                worker_cls=(
                    "scripts.moe_elastic_ep_phase1_worker.ElasticEPPhase1Worker"
                ),
                enforce_eager=True,
                dtype="bfloat16",
                seed=0,
                trust_remote_code=False,
                max_model_len=1024,
                max_num_seqs=4,
                max_num_batched_tokens=1024,
                gpu_memory_utilization=0.7,
                enable_prefix_caching=False,
                disable_log_stats=True,
                moe_backend="triton",
            )
        )
        report["engine_startup_s"] = time.monotonic() - started
        report["before"] = {
            "core_engine_count": len(engine.engine_core.core_engines),
            "rank_snapshots": await rank_snapshots(engine),
            "output_tokens": await generate(engine, args.prompt, "phase1-before"),
        }
        report["before"]["coverage"] = coverage(
            report["before"]["rank_snapshots"]
        )
        if len(report["before"]["rank_snapshots"]) != 2:
            raise RuntimeError("initial_ep_group_did_not_have_two_ranks")
        if any(
            not row["mixture_of_experts_interface"]
            or row["num_moe_layers"] < 1
            or row["expert_weights_layer_count"] != row["num_moe_layers"]
            or len(row["layers"]) != row["num_moe_layers"]
            for row in report["before"]["rank_snapshots"]
        ):
            raise RuntimeError("runtime_moe_eplb_interface_or_layer_mapping_incomplete")
        if any(row["missing"] for row in report["before"]["coverage"].values()):
            raise RuntimeError("initial_ep_group_expert_coverage_incomplete")
        print("MOE_ELASTIC_EP_PHASE1_PROGRESS=initial_group_serving", flush=True)

        scale_started = time.monotonic()
        await asyncio.wait_for(engine.scale_elastic_ep(1), timeout=360)
        report["scale_down_s"] = time.monotonic() - scale_started
        report["after"] = {
            "core_engine_count": len(engine.engine_core.core_engines),
            "rank_snapshots": await rank_snapshots(engine),
            "output_tokens": await generate(engine, args.prompt, "phase1-after"),
        }
        report["after"]["coverage"] = coverage(report["after"]["rank_snapshots"])
        before_rank0 = next(
            row
            for row in report["before"]["rank_snapshots"]
            if row["data_parallel_rank"] == 0
        )
        after_rank0 = next(
            row
            for row in report["after"]["rank_snapshots"]
            if row["data_parallel_rank"] == 0
        )
        report["survivor"] = {
            "same_pid": before_rank0["pid"] == after_rank0["pid"],
            "same_gpu_uuid": before_rank0["gpu_uuid"] == after_rank0["gpu_uuid"],
            "same_model_object_id": (
                before_rank0["model_object_id"]
                == after_rank0["model_object_id"]
            ),
            "ep_group_changed": (
                before_rank0["ep_group"]["device_group_id"]
                != after_rank0["ep_group"]["device_group_id"]
            ),
            "dp_group_changed": (
                before_rank0["dp_group"]["device_group_id"]
                != after_rank0["dp_group"]["device_group_id"]
            ),
        }
        if len(report["after"]["rank_snapshots"]) != 1:
            raise RuntimeError("scaled_group_did_not_have_one_rank")
        if len(report["after"]["coverage"]) != before_rank0["num_moe_layers"]:
            raise RuntimeError("scaled_group_moe_layer_coverage_not_observed")
        if any(row["missing"] for row in report["after"]["coverage"].values()):
            raise RuntimeError("scaled_group_expert_coverage_incomplete")
        if not all(
            report["survivor"][key]
            for key in ("same_pid", "same_gpu_uuid", "same_model_object_id")
        ):
            raise RuntimeError("surviving_rank_was_restarted_or_replaced")
        report["status"] = "passed"
        print("MOE_ELASTIC_EP_PHASE1_PROGRESS=scale_down_serving", flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(exc).__name__,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if args.output:
            Path(args.output).write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print("MOE_ELASTIC_EP_PHASE1_JSON=" + json.dumps(report), flush=True)
        if engine is not None:
            try:
                engine.shutdown()
            except BaseException:
                traceback.print_exc()
        ray.shutdown()
    if report["status"] != "passed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        default="Write a Python function for sorting and explain its tests.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
