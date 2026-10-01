"""Owned GPUs: same normal traffic across TP2, EP2 and two TP1 full replicas."""

import argparse
import json
from pathlib import Path
import queue
import re
import statistics
import subprocess
import threading
import time
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid,digest
from scripts.run_moe_tp_microbenchmark import snapshot


def summary_cell(samples, total_concurrency, output_tokens):
    req=[r for sample in samples for r in sample["requests"]]
    latency=sorted(r["latency_s"] for r in req)
    wall=[s["wall_s"] for s in samples]
    return {"total_concurrency":total_concurrency,"output_tokens":output_tokens,
        "samples":len(samples),"wall_mean_s":statistics.mean(wall),
        "wall_sd_s":statistics.stdev(wall) if len(wall)>1 else 0,
        "generated_tokens_per_s":statistics.mean(total_concurrency*output_tokens/x for x in wall),
        "latency_mean_s":statistics.mean(latency),"latency_p95_s":latency[min(len(latency)-1,int(0.95*len(latency)))],
        "ttft_mean_s":statistics.mean(r["ttft_s"] for r in req),
        "tpot_mean_s":statistics.mean(r["tpot_s"] for r in req),
        "completed_requests":len(req),"stream_checks_pass":all(r["passed_stream"] for r in req)}


def config_summary(reports, replicas):
    answer=[]
    for index,cell in enumerate(reports[0]["cells"]):
        peers=[r["cells"][index] for r in reports]
        if any(p["status"]!="complete" for p in peers):
            answer.append({"prompt_tokens":cell["prompt_tokens"],"output_tokens":cell["output_tokens"],
                           "total_concurrency":sum(p["concurrency"] for p in peers),
                           "status":"incomplete","errors":[p.get("error") for p in peers]})
            continue
        combined=[]
        for paired in zip(*(p["samples"] for p in peers)):
            combined.append({"wall_s":max(s["wall_s"] for s in paired),
                "requests":[request for sample in paired for request in sample["requests"]]})
        answer.append({"prompt_tokens":cell["prompt_tokens"],"output_tokens":cell["output_tokens"],
            "status":"complete",**summary_cell(combined,sum(p["concurrency"] for p in peers),
                                                  cell["output_tokens"])})
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--compiler-cache",required=True)
    parser.add_argument("--gpu-indices",type=int,nargs=2,default=[2,3])
    parser.add_argument("--samples",type=int,default=2)
    parser.add_argument("--short-concurrency",type=int,nargs="*",default=[8,16,32,48])
    parser.add_argument("--long-concurrency",type=int,nargs="*",default=[8,16,24,32,48])
    args=parser.parse_args()
    if len(set(args.gpu_indices))!=2 or any(c<2 or c%2 for c in args.short_concurrency+args.long_concurrency):
        raise ValueError("all full-replica totals must be even and >=2")
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]; vllm=repo.parent/"vllm"
    calibration=Path(args.calibration).resolve(); cache=Path(args.compiler_cache).resolve()
    initial=snapshot(); uuids={int(r.split(",")[0]):canonical_gpu_uuid(r.split(",")[1].strip()) for r in initial["gpus"]}
    expected={uuids[index] for index in args.gpu_indices}
    if len(expected)!=2 or any(canonical_gpu_uuid(r.split(",")[0].strip()) in expected for r in initial["processes"]):
        raise ValueError("GPUs occupied/missing; no other GPU jobs will be stopped")
    protocol={"physical_gpu_indices":args.gpu_indices,"gpu_uuids":sorted(expected),
        "model":args.model,"calibration_sha256":digest(json.loads(calibration.read_text())),
        "fixed_prompt_profiles":[[512,128],[4096,32]],
        "total_concurrencies":{"short":args.short_concurrency,"long":args.long_concurrency},
        "independent_engines_per_configuration":{"tp2":1,"tp2-ep2":1,"dp2-ep2":1,"tp1-full-replicas":2},
        "sample_repeats_same_engine":args.samples,"compiler_cache":str(cache),
        "no_skew_or_planner_changes":True,"not_formal_ab":True,"no_gpu_mps":True}
    write_artifact(out/"protocol.json",protocol)
    reports={}

    def launch(mode,tp,dp,ep,gpu,worker_id,barrier):
        name="spotserve-capacity-diag-"+uuid.uuid4().hex[:14]
        command=["podman","run","--rm","--name",name,"--network","none","--shm-size","2g"]
        for index in gpu: command += ["--device",f"nvidia.com/gpu={index}"]
        for src,dst,permission in ((repo,repo,"ro"),(vllm,vllm,"ro"),(args.model,args.model,"ro"),
            ("/usr/local/cuda-13.0","/usr/local/cuda","ro"),(cache,"/taskcache","rw"),(out,out,"rw")):
            command += ["--volume",f"{src}:{dst}:{permission}"]
        env={"PYTHONPATH":f"{repo}/scripts/diagnostic_site:{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
            "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1",
            "VLLM_USE_V2_MODEL_RUNNER":"0","VLLM_HOST_IP":"127.0.0.1","NCCL_SOCKET_IFNAME":"lo",
            "TRITON_CACHE_DIR":"/taskcache/triton","CUDA_CACHE_PATH":"/taskcache/cuda",
            "FLASHINFER_WORKSPACE_BASE":"/taskcache/flashinfer-ep-phase2-20260915",
            "MOE_DIAG_EARLY_IMPORTS":"1"}
        for key,value in env.items(): command += ["--env",f"{key}={value}"]
        command += ["localhost/spotserve-python312-nixl:latest",str(vllm/".venv/bin/python"),"-u","-m",
            "scripts.probe_moe_ep_capacity","--model",args.model,"--calibration",str(calibration),
            "--mode",mode,"--tp",str(tp),"--dp",str(dp),"--samples",str(args.samples),
            "--worker-id",worker_id,"--short-concurrency"]
        shorts=[c//2 for c in args.short_concurrency] if mode=="tp1-full-replicas" else args.short_concurrency
        longs=[c//2 for c in args.long_concurrency] if mode=="tp1-full-replicas" else args.long_concurrency
        command += [str(c) for c in shorts]+["--long-concurrency"]+[str(c) for c in longs]
        if ep: command += ["--ep"]
        if barrier: command += ["--barrier-dir",str(out/"full-replica-barriers")]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        return {"name":name,"mode":mode,"gpu":gpu,"worker_id":worker_id,"command":command,
                "process":process,"environment_before":snapshot()}

    def collect(owned):
        label=owned["mode"]+("-"+owned["worker_id"] if owned["mode"]=="tp1-full-replicas" else "")
        process=owned["process"]; q=queue.Queue()
        def drain():
            for line in process.stdout: q.put(line)
            q.put(None)
        threading.Thread(target=drain,daemon=True).start()
        data={key:value for key,value in owned.items() if key!="process"}
        deadline=time.monotonic()+3600
        try:
            with (out/(label+".log")).open("x") as log:
                while True:
                    if time.monotonic()>deadline: raise TimeoutError("capacity probe exceeds owned container deadline")
                    try: line=q.get(timeout=1)
                    except queue.Empty: continue
                    if line is None: break
                    log.write(line);log.flush()
                    if line.startswith("MOE_CAPACITY_JSON="):
                        data["report"]=json.loads(line.split("=",1)[1])
                    elif line.startswith("MOE_CAPACITY_PROGRESS="):
                        print(line.strip(),flush=True)
            data["exit_code"]=process.wait(timeout=30)
            if data["exit_code"] or data.get("report",{}).get("status") not in ("passed","partial"):
                raise RuntimeError("capacity probe failure; log retained")
            events=(out/(label+".log")).read_text()
            match=re.search(r"GPU KV cache size:\s*([0-9,]+) tokens",events)
            if match: data["runtime_kv_capacity_tokens_per_group"]=int(match.group(1).replace(",",""))
            data["status"]=data["report"]["status"]
        except Exception as exc:
            data["status"]="failed"; data["error"]=repr(exc)
        finally:
            if process.poll() is None:
                subprocess.run(["podman","stop","--time","5",owned["name"]],capture_output=True,timeout=30)
                process.wait(timeout=30)
            data["environment_after"]=snapshot()
            write_artifact(out/(label+".json"),data)
        return data

    for mode,tp,dp,ep in (("tp2",2,1,False),("tp2-ep2",2,1,True),("dp2-ep2",1,2,True)):
        owned=launch(mode,tp,dp,ep,args.gpu_indices,"0",False)
        row=collect(owned); reports[mode]=[row]
        if row["status"]=="failed": break
    if all(mode in reports and reports[mode][0].get("status")!="failed"
           for mode in ("tp2","tp2-ep2","dp2-ep2")):
        # Each independently complete TP1 model uses one GPU; paired start
        # barriers align the same offered load, never pretending to be EP ranks.
        owned=[launch("tp1-full-replicas",1,1,False,[gpu],str(i),True)
               for i,gpu in enumerate(args.gpu_indices)]
        results=[None,None]
        threads=[threading.Thread(target=lambda k,task:results.__setitem__(k,collect(task)),args=(i,task))
                 for i,task in enumerate(owned)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        reports["tp1-full-replicas"]=results
    summaries={label:config_summary([r["report"] for r in group],len(group))
               for label,group in reports.items() if all("report" in r for r in group)}
    write_artifact(out/"summary.json",{"protocol":protocol,"summaries":summaries,
        "raw_status":{label:[r["status"] for r in group] for label,group in reports.items()},
        "environment_final":snapshot(),"limitations":[
            "two repeats per fresh engine are not deployment replicates",
            "fixed closed-loop batch; saturation is tested concurrency, not open-loop SLO maximum",
            "EP throughput exploratory until correctness gate released"]})
    return 0 if len(summaries)==4 else 1


if __name__=="__main__": raise SystemExit(main())
