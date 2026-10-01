"""Minimum same-prefix layer diagnosis on idle GPU0/1, TP then TP+EP."""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid
from scripts.probe_granite_ep import all_rank_rpc
from scripts.run_moe_tp_microbenchmark import snapshot


async def inside(args):
    from scripts.analyze_moe_phase2_diagnostics import first_difference
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.inputs import TokensPrompt
    data=json.loads(Path(args.calibration).read_text()); out=Path(args.ep_dir)
    reference=json.loads((out/"tp2-reference.json").read_text())["report"]["correctness_generations"]
    marker="MOE_PHASE2_CORRECTNESS_JSON="
    other=json.loads(next(l[len(marker):] for l in (out/"tp2-ep2.log").read_text().splitlines()
                          if l.startswith(marker)))["correctness_generations"]
    other={r["case"]:r for r in other}
    engine=AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model,tensor_parallel_size=2,
        enable_expert_parallel=args.ep,enforce_eager=True,dtype="bfloat16",seed=0,
        trust_remote_code=False,cpu_offload_gb=0,max_model_len=8704,max_num_seqs=1,
        max_num_batched_tokens=2048,gpu_memory_utilization=0.7,enable_prefix_caching=False,
        enable_return_routed_experts=False,async_scheduling=False,moe_backend="triton",
        disable_log_stats=True,worker_cls="scripts.moe_layer_trace_worker.LayerTraceDiagnosticWorker"))
    try:
        await all_rank_rpc(engine,"enable_layer_trace")
        result={"status":"passed","ep_flag":args.ep,"cases":[]}
        for ref in reference:
            position=first_difference(ref["output_token_ids"],other[ref["case"]]["output_token_ids"])
            if position is None: continue
            case=ref["case"]
            prompt=([17 if case.endswith("17") else 18]*4096 if case.startswith("previous-repeated")
                    else data["source_cases"][case]["prompt_token_ids"])
            tokens=prompt+ref["output_token_ids"][:position]
            await all_rank_rpc(engine,"reset_layer_trace")
            async for output in engine.generate(TokensPrompt(prompt_token_ids=tokens),
                    SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=20),case): pass
            probabilities={str(k):float(v.logprob) for k,v in output.outputs[0].logprobs[0].items()}
            result["cases"].append({"case":case,"prefix_tokens":len(tokens),
                "next_token":int(output.outputs[0].token_ids[0]),"logprobs":probabilities,
                "ranks":await all_rank_rpc(engine,"get_layer_trace")})
        return result
    finally:
        engine.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True); parser.add_argument("--calibration",required=True)
    parser.add_argument("--ep-dir",required=True); parser.add_argument("--compiler-cache",required=True)
    parser.add_argument("--inside",action="store_true"); parser.add_argument("--ep",action="store_true")
    args=parser.parse_args()
    if args.inside:
        print("MOE_LAYER_TRACE_JSON="+json.dumps(asyncio.run(inside(args))),flush=True); return 0
    before=snapshot(); uuids={int(r.split(",")[0]):canonical_gpu_uuid(r.split(",")[1].strip()) for r in before["gpus"]}
    if any(canonical_gpu_uuid(r.split(",")[0].strip()) in {uuids[0],uuids[1]} for r in before["processes"]):
        raise ValueError("GPU0/1 occupied; other processes will not be stopped")
    repo=Path(__file__).resolve().parents[1]; vllm=repo.parent/"vllm"; out=Path(args.ep_dir).resolve()
    for ep,label in ((False,"tp2"),(True,"tp2-ep2")):
        name="spotserve-layer-diag-"+uuid.uuid4().hex[:16]
        command=["podman","run","--rm","--name",name,"--network","none","--shm-size","1g"]
        for gpu in (0,1): command += ["--device",f"nvidia.com/gpu={gpu}"]
        for src,dst,mode in ((repo,repo,"ro"),(vllm,vllm,"ro"),(args.model,args.model,"ro"),
            ("/usr/local/cuda-13.0","/usr/local/cuda","ro"),(Path(args.compiler_cache).resolve(),"/taskcache","rw")):
            command += ["--volume",f"{src}:{dst}:{mode}"]
        env={"PYTHONPATH":f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
            "OMP_NUM_THREADS":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
            "NCCL_SOCKET_IFNAME":"lo","VLLM_HOST_IP":"127.0.0.1","VLLM_USE_V2_MODEL_RUNNER":"0",
            "TRITON_CACHE_DIR":"/taskcache/triton","CUDA_CACHE_PATH":"/taskcache/cuda",
            "FLASHINFER_WORKSPACE_BASE":"/taskcache/flashinfer-ep-phase2-20260915"}
        for key,value in env.items(): command += ["--env",f"{key}={value}"]
        command += ["localhost/spotserve-python312-nixl:latest",str(vllm/".venv/bin/python"),"-u","-m",
            "scripts.run_moe_ep_layer_trace","--inside","--model",args.model,"--calibration",str(Path(args.calibration).resolve()),
            "--ep-dir",str(out),"--compiler-cache","/taskcache"]
        if ep: command.append("--ep")
        row={"command":command,"status":"running"}; process=None
        try:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            with (out/(label+"-layer-trace.log")).open("x") as log:
                for line in process.stdout:
                    log.write(line); log.flush()
                    if line.startswith("MOE_LAYER_TRACE_JSON="): row["report"]=json.loads(line.split("=",1)[1])
            row["exit_code"]=process.wait(timeout=30)
            if row["exit_code"] or "report" not in row: raise RuntimeError("layer diagnostic failed")
            row["status"]="passed"
            print("MOE_LAYER_TRACE_PROGRESS="+json.dumps({"mode":label,"status":"passed"}),flush=True)
        finally:
            if process is not None and process.poll() is None:
                subprocess.run(["podman","stop","--time","5",name],capture_output=True,timeout=30)
                process.wait(timeout=30)
            write_artifact(out/(label+"-layer-trace.json"),row)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
