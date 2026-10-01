"""EP numerical checks and load sweep; never changes routing or placement."""

import argparse
import asyncio
import json
import time
import traceback

from scripts import moe_diagnostic_trace as trace
from scripts.probe_granite_ep import all_rank_rpc
from scripts.profile_granite_moe_tp import digest


def serialize_logprobs(rows):
    return [{str(token): {"logprob": float(value.logprob), "rank": value.rank}
             for token, value in row.items()} if row is not None else None for row in rows]


async def run(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM
    from pathlib import Path
    calibration = json.loads(Path(args.calibration).read_text())
    cases = [{"id": "previous-repeated17", "tokens": [17]*4096},
             {"id": "previous-repeated18", "tokens": [18]*4096}]
    cases += [{"id": key, "tokens": calibration["source_cases"][key]["prompt_token_ids"]}
              for key in ("train-code", "train-story", "validation-science", "formal-serving")]
    report = {"status":"running", "tp":args.tp,"dp":args.dp,"ep_flag":args.ep,"dtype":args.dtype,
              "correctness_generations":[],"teacher_forced":[],"performance":[],"profiles":[],
              "formal_experiment":False,"no_skew_or_router_changes":True,
              "tolerances":{"teacher_forced_max_abs_logprob":0.15,
                  "router_weight_max_abs":1e-5,"router_boundary_tie":1e-5,
                  "expert_oracle_relative_l2":0.03,"expert_oracle_max_abs":"0.02 + 0.1 * reference_max_abs"}}
    engine = None
    try:
        with trace.span("ep_engine_constructor"):
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                model=args.model,tensor_parallel_size=args.tp,data_parallel_size=args.dp,
                data_parallel_size_local=args.dp,enable_expert_parallel=args.ep,
                all2all_backend="allgather_reducescatter", enforce_eager=True,
                dtype=args.dtype,seed=0,trust_remote_code=False,cpu_offload_gb=0,
                max_model_len=8704,max_num_seqs=8,max_num_batched_tokens=2048,
                gpu_memory_utilization=0.7,enable_prefix_caching=False,
                enable_return_routed_experts=False,async_scheduling=False,
                moe_backend="triton",disable_log_stats=True,
                worker_cls="scripts.moe_ep_correctness_worker.EPCorrectnessDiagnosticWorker"))
        report["placement"] = await all_rank_rpc(engine,"get_expert_placement_evidence")
        report["startup"] = await all_rank_rpc(engine,"get_diagnostic_snapshot")

        async def generate(case, output, label, index, with_probs=False, teacher=None):
            tokens = case["tokens"] + (teacher or [])
            options = dict(temperature=0,max_tokens=output,ignore_eos=True,seed=0)
            if with_probs:
                options["logprobs"] = 20
            if teacher is not None:
                options["prompt_logprobs"] = 10
            started = time.monotonic()
            prefix = []
            first = None
            kwargs={"data_parallel_rank":index%args.dp} if args.dp>1 else {}
            async for item in engine.generate(TokensPrompt(prompt_token_ids=tokens),
                    SamplingParams(**options),label+"-"+case["id"]+f"-{index}",**kwargs):
                current = list(item.outputs[0].token_ids)
                if current[:len(prefix)]!=prefix or len(current)<len(prefix):
                    raise ValueError("stream prefix changed")
                prefix = current
                if prefix and first is None:
                    first=time.monotonic()
            if len(prefix)!=output or item.outputs[0].finish_reason!="length":
                raise ValueError("incorrect generation length/finish")
            row={"case":case["id"],"prompt_sha256":digest(case["tokens"]),"prompt_tokens":len(case["tokens"]),
                 "output_token_ids":prefix,"output_sha256":digest(prefix),"latency_s":time.monotonic()-started,
                 "ttft_s":first-started,"stream_and_length_passed":True}
            if with_probs:
                row["logprobs"]=serialize_logprobs(item.outputs[0].logprobs)
            if teacher is not None:
                rows=item.prompt_logprobs[len(case["tokens"]):len(tokens)]
                if len(rows)!=len(teacher) or any(row is None for row in rows):
                    raise ValueError("missing teacher-forced token probabilities")
                row["teacher_token_ids"]=teacher
                row["teacher_prompt_sha256"]=digest(tokens)
                row["teacher_logprobs"]=serialize_logprobs(rows)
            return row

        if args.collective_gate:
            small=[{"id":"collective-train-code","tokens":calibration["source_cases"]["train-code"]["prompt_token_ids"][:32]},
                   {"id":"collective-train-story","tokens":calibration["source_cases"]["train-story"]["prompt_token_ids"][:32]}]
            await all_rank_rpc(engine,"set_collective_diagnostics",(True,))
            await all_rank_rpc(engine,"set_correctness_diagnostics",(True,))
            rows=await asyncio.gather(*[generate(case,1,"collective-gate",i) for i,case in enumerate(small)])
            report["collective_gate"]={"requests":rows,
                "collectives":await all_rank_rpc(engine,"collective_snapshot"),
                "runtime_correctness":await all_rank_rpc(engine,"correctness_snapshot")}
            await all_rank_rpc(engine,"set_correctness_diagnostics",(False,))
            await all_rank_rpc(engine,"set_collective_diagnostics",(False,))
        await all_rank_rpc(engine,"set_diagnostic_measurement",(None,))
        for start in range(0,len(cases),2):
            await all_rank_rpc(engine,"set_correctness_diagnostics",(True,))
            rows=await asyncio.gather(*[generate(case,args.correctness_output,f"correctness-{start}",i,True)
                                        for i,case in enumerate(cases[start:start+2])])
            report["correctness_generations"].extend(rows)
            report.setdefault("runtime_correctness",[]).append({"cases":[r["case"] for r in rows],
                "ranks":await all_rank_rpc(engine,"correctness_snapshot")})
            await all_rank_rpc(engine,"set_correctness_diagnostics",(False,))
            trace.emit("phase2_correctness_batch",tp=args.tp,dp=args.dp,ep=args.ep,cases=[r["case"] for r in rows])
        reference=(json.loads(Path(args.reference).read_text())["report"]["correctness_generations"]
                   if args.reference else report["correctness_generations"])
        refs={r["case"]:r for r in reference}
        # Common TP continuation is teacher-forced for every mode, including TP.
        for start in range(0,len(cases),2):
            rows=await asyncio.gather(*[generate(case,1,f"teacher-{start}",i,False,
                                                refs[case["id"]]["output_token_ids"])
                                        for i,case in enumerate(cases[start:start+2])])
            report["teacher_forced"].extend(rows)
        # Large concurrent prefill/dispatch correctness check, identical cases
        # and fixed order in all modes; no expert hotness is manufactured.
        high_cases=[cases[i%len(cases)] for i in range(8)]
        await all_rank_rpc(engine,"set_correctness_diagnostics",(True,))
        high=await asyncio.gather(*[generate(case,min(128,args.correctness_output),"routing-load-c8",i)
                                   for i,case in enumerate(high_cases)])
        report["high_routing_load"]={"concurrency":8,"generations":high,
            "ranks":await all_rank_rpc(engine,"correctness_snapshot")}
        await all_rank_rpc(engine,"set_correctness_diagnostics",(False,))
        if args.batch_shape_gate:
            # A normally occurring capacity-probe anomaly: two requests with
            # the *same* ordinary train-code prefix may differ within one TP
            # engine. Record the first 128 streamed tokens and top-k logprobs
            # in all three modes without changing sampling, router or experts.
            ordinary=[{"id":key,"tokens":calibration["source_cases"][key]["prompt_token_ids"][:512]}
                for key in ("train-code","train-story","validation-science","formal-serving")]
            batch=[ordinary[i%len(ordinary)] for i in range(8)]
            outputs=await asyncio.gather(*[generate(case,128,"normal-c8-shape",i,True)
                                            for i,case in enumerate(batch)])
            report["batch_shape_gate"]={"concurrency":8,"output_tokens":128,
                "normal_train_code_duplicated_at_indices":[0,4],"requests":outputs}
        print("MOE_PHASE2_CORRECTNESS_JSON="+json.dumps({key:value for key,value in report.items()
            if key not in ("profiles","performance","startup","placement")}),flush=True)
        trace.emit("phase2_correctness_complete",tp=args.tp,dp=args.dp,ep=args.ep)

        if args.correctness_only:
            report["status"]="passed"
            return report

        for prompt in (512,4096):
            for output in (1,128):
                for concurrency in (1,2,4,8):
                    batch_cases=[{"id":f"shape-{i}","tokens":[17+i%2]*prompt} for i in range(concurrency)]
                    async def batch(label):
                        started=time.monotonic()
                        rows=await asyncio.gather(*[generate(case,output,label,i)
                                                   for i,case in enumerate(batch_cases)])
                        return {"batch_wall_s":time.monotonic()-started,"requests":rows}
                    label=f"p{prompt}-o{output}-c{concurrency}"
                    await batch("warm-"+label)
                    samples=[await batch(f"sample-{label}-{i}") for i in range(3)]
                    report["performance"].append({"prompt_tokens":prompt,"output_tokens":output,
                        "concurrency":concurrency,"samples":samples})
                    if (prompt==512 and output==128) or (prompt==4096 and output==1 and concurrency in (1,8)):
                        await all_rank_rpc(engine,"start_kernel_profiler")
                        await all_rank_rpc(engine,"set_diagnostic_measurement",(label,))
                        profiled=await batch("profile-"+label)
                        snapshots=await all_rank_rpc(engine,"get_diagnostic_snapshot")
                        kernels=await all_rank_rpc(engine,"stop_kernel_profiler")
                        await all_rank_rpc(engine,"set_diagnostic_measurement",(None,))
                        report["profiles"].append({"prompt_tokens":prompt,"output_tokens":output,
                            "concurrency":concurrency,"sample":profiled,"ranks":snapshots,"kernel_profiles":kernels})
                    trace.emit("phase2_cell_complete",tp=args.tp,dp=args.dp,ep=args.ep,
                               prompt=prompt,output=output,concurrency=concurrency)
        report["status"]="passed"
    except Exception:
        report["status"]="failed"
        report["traceback"]=traceback.format_exc()
    finally:
        if engine is not None:
            engine.shutdown()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--tp",type=int,required=True)
    parser.add_argument("--dp",type=int,default=1)
    parser.add_argument("--ep",action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--dtype",choices=("bfloat16","float16","float32"),default="bfloat16")
    parser.add_argument("--correctness-output",type=int,default=512)
    parser.add_argument("--correctness-only",action="store_true")
    parser.add_argument("--collective-gate",action="store_true")
    parser.add_argument("--batch-shape-gate",action="store_true")
    args=parser.parse_args()
    if args.tp*args.dp!=2:
        raise ValueError("two GPU diagnostic only")
    if args.correctness_output <= 0:
        raise ValueError("correctness output must be positive")
    report=asyncio.run(run(args))
    print("MOE_PHASE2_JSON="+json.dumps(report),flush=True)
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
