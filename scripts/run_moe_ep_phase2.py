"""Owned two-GPU containers, native TP reference then two EP modes."""

import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid,digest
from scripts.run_moe_architecture_diagnostics import diagnostic_log_events,validate_placement
from scripts.run_moe_tp_microbenchmark import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--compiler-cache",required=True)
    parser.add_argument("--gpu-indices",type=int,nargs=2,default=[2,3])
    parser.add_argument("--dtype",choices=("bfloat16","float16","float32"),default="bfloat16")
    parser.add_argument("--correctness-output",type=int,default=512)
    parser.add_argument("--correctness-only",action="store_true")
    parser.add_argument("--collective-gate",action="store_true")
    parser.add_argument("--batch-shape-gate",action="store_true")
    args=parser.parse_args()
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]; vllm=repo.parent/"vllm"
    calibration=Path(args.calibration).resolve(); cache=Path(args.compiler_cache).resolve()
    initial=snapshot(); uuids={int(r.split(",")[0]):canonical_gpu_uuid(r.split(",")[1].strip()) for r in initial["gpus"]}
    if len(set(args.gpu_indices)) != 2 or any(index not in uuids for index in args.gpu_indices):
        raise ValueError("two distinct physical GPU indices are required")
    expected={uuids[index] for index in args.gpu_indices}
    if any(canonical_gpu_uuid(r.split(",")[0].strip()) in expected for r in initial["processes"]):
        raise ValueError("test GPUs occupied; other jobs will not be stopped")
    protocol={"groups":args.gpu_indices,"model":args.model,"calibration_sha256":digest(json.loads(calibration.read_text())),
              "no_formal_comparison":True,"no_vllm_source_edits":True,
              "same_inputs_in_all_modes":True,"no_skew_placement_or_planner_changes":True,
              "dtype":args.dtype,"correctness_output":args.correctness_output,
              "correctness_only":args.correctness_only,"collective_gate":args.collective_gate,
              "batch_shape_gate":args.batch_shape_gate}
    write_artifact(out/"protocol.json",protocol)
    for label,tp,dp,ep in (("tp2-reference",2,1,False),("tp2-ep2",2,1,True),("dp2-ep2",1,2,True)):
        name="spotserve-ep-diag-"+uuid.uuid4().hex[:16]
        command=["podman","run","--rm","--name",name,"--network","none","--shm-size","2g"]
        for gpu in args.gpu_indices: command += ["--device",f"nvidia.com/gpu={gpu}"]
        for src,dst,mode in ((repo,repo,"ro"),(vllm,vllm,"ro"),(args.model,args.model,"ro"),
                             ("/usr/local/cuda-13.0","/usr/local/cuda","ro"),(cache,"/taskcache","rw")):
            command += ["--volume",f"{src}:{dst}:{mode}"]
        env={"PYTHONPATH":f"{repo}/scripts/diagnostic_site:{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
            "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1",
            "VLLM_USE_V2_MODEL_RUNNER":"0","VLLM_HOST_IP":"127.0.0.1","NCCL_SOCKET_IFNAME":"lo",
            "TRITON_CACHE_DIR":"/taskcache/triton","CUDA_CACHE_PATH":"/taskcache/cuda",
            "FLASHINFER_WORKSPACE_BASE":"/taskcache/flashinfer-ep-phase2-20260915",
            "MOE_DIAG_EARLY_IMPORTS":"1"}
        for key,value in env.items(): command += ["--env",f"{key}={value}"]
        command += ["localhost/spotserve-python312-nixl:latest",str(vllm/".venv/bin/python"),"-u","-m",
            "scripts.probe_moe_ep_phase2","--model",args.model,"--calibration",str(calibration),
            "--tp",str(tp),"--dp",str(dp),"--dtype",args.dtype,
            "--correctness-output",str(args.correctness_output)]
        if args.correctness_only: command.append("--correctness-only")
        if args.collective_gate: command.append("--collective-gate")
        if args.batch_shape_gate: command.append("--batch-shape-gate")
        if ep: command += ["--ep","--reference",str(out/"tp2-reference.json")]
        row={"label":label,"command":command,"status":"running","environment_before":snapshot()}
        messages=queue.Queue(); process=None
        try:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            def drain():
                for line in process.stdout: messages.put(line)
                messages.put(None)
            thread=threading.Thread(target=drain,daemon=True); thread.start()
            deadline=time.monotonic()+1800
            with (out/(label+".log")).open("x") as log:
                while True:
                    if time.monotonic()>deadline: raise TimeoutError("diagnostic timeout")
                    try: line=messages.get(timeout=1)
                    except queue.Empty: continue
                    if line is None: break
                    log.write(line); log.flush()
                    if line.startswith("MOE_PHASE2_JSON="): row["report"]=json.loads(line.split("=",1)[1])
                    else:
                        for event in diagnostic_log_events(line):
                            if event["event"].startswith("phase2_"):
                                print("MOE_EP_PHASE2_PROGRESS="+json.dumps({"label":label,**event}),flush=True)
            row["exit_code"]=process.wait(timeout=30)
            if row["exit_code"] or row.get("report",{}).get("status")!="passed":
                raise RuntimeError("native engine or diagnostic failed; inspect preserved report/log")
            if args.dtype=="bfloat16":
                validate_placement(row["report"],expected)
                row["checkpoint_samples_verified"] = True
            else:
                # The on-disk BF16 scalar digest cannot match a deliberately
                # converted FP16/FP32 tensor. Keep structural and native
                # collective evidence distinct from checkpoint byte hashes.
                placement=row["report"]["placement"]
                if {canonical_gpu_uuid(r["actual_gpu_uuid"]) for r in placement} != expected:
                    raise ValueError("actual GPU identities differ")
                if any(r["model_class"]!="GraniteMoeForCausalLM" or len(r["expert_modules"])!=32
                       or any(not m["weight_device"].startswith("cuda:") for m in r["expert_modules"])
                       for r in placement):
                    raise ValueError("converted native model or GPU weights missing")
                if ep:
                    for layer in range(32):
                        groups=[set(next(m for m in r["expert_modules"] if m["layer"]==layer)["global_expert_ids"])
                                for r in placement]
                        if groups[0]&groups[1] or groups[0]|groups[1]!=set(range(40)):
                            raise ValueError("converted EP experts fail disjoint coverage")
                row["checkpoint_samples_verified"] = False
                row["warning"] = "diagnostic dtype conversion; on-disk BF16 byte digest not applicable"
            row["placement_verified"]=True; row["status"]="passed"
        except Exception as error:
            row["status"]="failed"; row["error"]=repr(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                subprocess.run(["podman","stop","--time","5",name],capture_output=True,timeout=30)
                process.wait(timeout=30)
            row["environment_after"]=snapshot(); write_artifact(out/(label+".json"),row)
    write_artifact(out/"completed.json",{"status":"passed","protocol":protocol,"environment_final":snapshot()})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
