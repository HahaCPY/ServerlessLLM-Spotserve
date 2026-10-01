"""Native Granite serving capacity probe; never modifies router, weights or plans."""

import argparse
import asyncio
import json
from pathlib import Path
import time
import traceback

from scripts.probe_granite_ep import all_rank_rpc
from scripts.profile_granite_moe_tp import digest


async def run(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    calibration = json.loads(Path(args.calibration).read_text())
    normal = [calibration["source_cases"][name]["prompt_token_ids"]
              for name in ("train-code","train-story","validation-science","formal-serving")]
    report = {"status":"running", "mode":args.mode, "physical_gpus":args.tp*args.dp,
              "config":{"tp":args.tp,"dp":args.dp,"ep":args.ep,"dtype":"bfloat16",
                        "memory_utilization":0.7,"max_num_seqs":64,"prefix_cache":False},
              "profile":[{"prompt_tokens":p,"output_tokens":n,"concurrency":c}
                         for p,n,values in ((512,128,args.short_concurrency),
                                            (4096,32,args.long_concurrency)) for c in values],
              "normal_inputs_only":True,"formal_comparison":False,"cells":[]}
    engine = None
    try:
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model,tensor_parallel_size=args.tp,data_parallel_size=args.dp,
            data_parallel_size_local=args.dp,enable_expert_parallel=args.ep,
            all2all_backend="allgather_reducescatter",enforce_eager=True,dtype="bfloat16",seed=0,
            trust_remote_code=False,cpu_offload_gb=0,max_model_len=8704,
            max_num_seqs=64,max_num_batched_tokens=8192,gpu_memory_utilization=0.7,
            enable_prefix_caching=False,enable_return_routed_experts=False,
            async_scheduling=False,moe_backend="triton",disable_log_stats=True,
            worker_cls="scripts.moe_ep_correctness_worker.EPCorrectnessDiagnosticWorker"))
        report["startup_memory"] = await all_rank_rpc(engine,"capacity_snapshot")
        report["placement"] = await all_rank_rpc(engine,"get_expert_placement_evidence")

        async def barrier(cell, repetition):
            if not args.barrier_dir:
                return
            directory = Path(args.barrier_dir)
            directory.mkdir(parents=True,exist_ok=True)
            label = f"{cell}-{repetition}"
            (directory/f"{label}-{args.worker_id}.ready").touch(exist_ok=False)
            deadline=time.monotonic()+180
            while len(list(directory.glob(f"{label}-*.ready"))) != 2:
                if time.monotonic()>deadline: raise TimeoutError("two replicas did not reach barrier")
                await asyncio.sleep(0.1)
            go=directory/f"{label}.go"
            if args.worker_id=="0": go.touch(exist_ok=False)
            while not go.exists():
                if time.monotonic()>deadline: raise TimeoutError("barrier coordinator did not release")
                await asyncio.sleep(0.1)

        async def request(tokens,index,label,output):
            start=time.monotonic(); first=None; previous=[]; last=None
            kwargs={"data_parallel_rank":index%args.dp} if args.dp>1 else {}
            async for item in engine.generate(TokensPrompt(prompt_token_ids=tokens),
                SamplingParams(temperature=0,max_tokens=output,ignore_eos=True,seed=0),
                f"capacity-{label}-{args.worker_id}-{index}",**kwargs):
                current=list(item.outputs[0].token_ids)
                if current[:len(previous)]!=previous or len(current)<len(previous):
                    raise ValueError("stream token prefix changed")
                previous=current
                if first is None and current: first=time.monotonic()
                last=item
            finish=time.monotonic()
            if last is None or len(previous)!=output or last.outputs[0].finish_reason!="length":
                raise ValueError("missing token / wrong finish reason")
            return {"prompt_sha256":digest(tokens),"output_sha256":digest(previous),
                "output_token_ids":previous,
                "output_count":len(previous),"latency_s":finish-start,"ttft_s":first-start,
                "tpot_s":(finish-first)/max(output-1,1),"passed_stream":True}

        async def batch(p,n,c,label,repetition):
            tokens=[normal[index%len(normal)][:p] for index in range(c)]
            if any(len(token)!=p for token in tokens):
                raise ValueError("calibration has insufficient normal prompt tokens")
            await barrier(label,repetition)
            start=time.monotonic()
            rows=await asyncio.wait_for(asyncio.gather(*[
                request(token,i,f"{label}-{repetition}",n)
                for i,token in enumerate(tokens)]),timeout=600)
            wall=time.monotonic()-start
            return {"wall_s":wall,"tokens_per_s":c*n/wall,"requests":rows}

        for p,n,values in ((512,128,args.short_concurrency),(4096,32,args.long_concurrency)):
            for c in values:
                label=f"p{p}-n{n}-c{c}"
                cell={"prompt_tokens":p,"output_tokens":n,"concurrency":c,"status":"running"}
                report["cells"].append(cell)
                try:
                    await batch(p,n,c,label,"warm")
                    samples=[]
                    for rep in range(args.samples):
                        await all_rank_rpc(engine,"reset_capacity_peak")
                        sample=await batch(p,n,c,label,str(rep))
                        sample["rank_memory"] = await all_rank_rpc(engine,"capacity_snapshot")
                        samples.append(sample)
                    cell.update({"samples":samples,"status":"complete"})
                    print("MOE_CAPACITY_PROGRESS="+json.dumps({"mode":args.mode,"worker":args.worker_id,
                        "prompt":p,"output":n,"concurrency":c,"wall_s":samples[-1]["wall_s"]}),flush=True)
                except Exception as error:
                    cell.update({"status":"failed","error":repr(error)})
                    break
        report["status"]="passed" if all(cell["status"]=="complete" for cell in report["cells"]) else "partial"
    except Exception:
        report["status"]="failed"
        report["traceback"]=traceback.format_exc()
    finally:
        if engine is not None: engine.shutdown()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--mode",required=True)
    parser.add_argument("--tp",type=int,required=True)
    parser.add_argument("--dp",type=int,default=1)
    parser.add_argument("--ep",action="store_true")
    parser.add_argument("--samples",type=int,default=2)
    parser.add_argument("--short-concurrency",type=int,nargs="*",default=(8,16,32,48))
    parser.add_argument("--long-concurrency",type=int,nargs="*",default=(8,16,24,32,48))
    parser.add_argument("--barrier-dir")
    parser.add_argument("--worker-id",default="0")
    args=parser.parse_args()
    if args.tp*args.dp not in (1,2) or args.samples<1: raise ValueError("bounded 1/2-GPU probe")
    report=asyncio.run(run(args))
    print("MOE_CAPACITY_JSON="+json.dumps(report),flush=True)
    return 0 if report["status"] in ("passed","partial") else 1


if __name__=="__main__": raise SystemExit(main())
