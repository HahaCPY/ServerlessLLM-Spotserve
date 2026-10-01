"""Own GPU0 diagnostic container; preserve FP32 stdout and parsed results."""

import argparse
import json
from pathlib import Path
import subprocess
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid
from scripts.run_moe_tp_microbenchmark import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--ep-dir",required=True)
    parser.add_argument("--mode",choices=("tp2-ep2","dp2-ep2"),required=True)
    parser.add_argument("--layer-trace",action="store_true")
    args=parser.parse_args(); out=Path(args.ep_dir).resolve()
    suffix="-fp32-layer-trace" if args.layer_trace else "-fp32"
    before=snapshot(); physical=canonical_gpu_uuid(next(r.split(",")[1].strip() for r in before["gpus"] if r.startswith("0,")))
    if any(canonical_gpu_uuid(r.split(",")[0].strip())==physical for r in before["processes"]):
        raise ValueError("GPU0 occupied; no other process will be stopped")
    repo=Path(__file__).resolve().parents[1]; vllm=repo.parent/"vllm"
    name="spotserve-fp32-oracle-"+uuid.uuid4().hex[:16]
    command=["podman","run","--rm","--name",name,"--network","none","--shm-size","1g",
             "--device","nvidia.com/gpu=0"]
    for src,dst in ((repo,repo),(vllm,vllm),(args.model,args.model),("/usr/local/cuda-13.0","/usr/local/cuda")):
        command += ["--volume",f"{src}:{dst}:ro"]
    for key,value in {"PYTHONPATH":f"{vllm}/.venv/lib/python3.12/site-packages:{repo}",
            "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"1"}.items():
        command += ["--env",f"{key}={value}"]
    command += ["localhost/spotserve-python312-nixl:latest",str(vllm/".venv/bin/python"),"-u","-m",
        "scripts.probe_granite_fp32_prefix","--model",args.model,"--calibration",str(Path(args.calibration).resolve()),
        "--ep-dir",str(out),"--modes",args.mode]
    if args.layer_trace: command.append("--layer-trace")
    row={"mode":args.mode,"status":"running","command":command,"environment_before":before}
    process=None
    try:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        with (out/(args.mode+suffix+".log")).open("x") as log:
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith("MOE_FP32_PROGRESS="): print(line.rstrip(),flush=True)
                if line.startswith("MOE_FP32_JSON="): row["report"]=json.loads(line.split("=",1)[1])
        row["exit_code"]=process.wait(timeout=30)
        if row["exit_code"] or row.get("report",{}).get("status") not in ("passed","not_needed"):
            raise RuntimeError("FP32 reference failed; inspect preserved log")
        if row["report"]["status"]=="passed" and canonical_gpu_uuid(row["report"]["gpu_uuid"])!=physical:
            raise ValueError("oracle GPU identity differs")
        row["status"]="passed"
    except Exception as error:
        row["status"]="failed"; row["error"]=repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(["podman","stop","--time","5",name],capture_output=True,timeout=30)
            process.wait(timeout=30)
        row["environment_after"]=snapshot(); write_artifact(out/(args.mode+suffix+".json"),row)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
