"""Same native recovery configuration, isolated FlashInfer miss then cache hits."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from scripts.moe_gpu_runtime import GPUPool, write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid, digest
from scripts.run_moe_architecture_diagnostics import cold_breakdown, diagnostic_log_events
from scripts.run_moe_tp_microbenchmark import snapshot


def cache_inventory(root):
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*.so"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compiler-cache", required=True)
    parser.add_argument("--warm-repeats", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shared_cache = Path(args.compiler_cache).resolve()
    compiler_cache_initially_empty = not shared_cache.exists() or not any(shared_cache.rglob("*"))
    flashinfer = shared_cache / "flashinfer-diagnostic-20260915"
    flashinfer.mkdir(parents=True,exist_ok=False)
    data = json.loads(Path(args.calibration).read_text())
    initial = snapshot()
    uuids = {int(r.split(",")[0]): canonical_gpu_uuid(r.split(",")[1].strip()) for r in initial["gpus"]}
    if any(canonical_gpu_uuid(r.split(",")[0].strip())==uuids[1] for r in initial["processes"]):
        raise ValueError("GPU1 occupied; existing GPU processes will not be stopped")
    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    env = {"FLASHINFER_WORKSPACE_BASE": "/taskcache/" + flashinfer.name,
           "MOE_DIAG_EARLY_IMPORTS": "1",
           "PYTHONPATH": f"{repo}/scripts/diagnostic_site:{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}"}
    protocol = {"model":args.model, "revision":args.revision, "calibration_sha256":digest(data),
        "physical_gpu":1,"unchanged_native_recovery":True, "flashinfer_cache":str(flashinfer),
        "other_compiler_cache":str(shared_cache), "compiler_cache_initially_empty":compiler_cache_initially_empty,
        "cache_cold_definition":("new empty FlashInfer/Triton/CUDA cache root" if compiler_cache_initially_empty else
            "new empty FlashInfer workspace only; existing Triton/CUDA caches unchanged"),
        "storage_state":"no OS cache purge; physical checkpoint reads observed separately",
        "cold_cache_repeats":1,"warm_cache_repeats":args.warm_repeats,
        "diagnostic_env":env,"no_planner_router_placement_workload_changes":True}
    write_artifact(out / "protocol.json",protocol)
    result = {"status":"running","protocol":protocol,"runs":[]}
    pool = GPUPool(out / "workers",args.model,args.revision,uuids,timeout=900,
                   compiler_cache=shared_cache,diagnostic_env=env)
    try:
        for repeat in range(args.warm_repeats+1):
            label = "cache-cold-r0" if repeat==0 else f"cache-hit-r{repeat-1}"
            before = cache_inventory(flashinfer)
            worker = pool.launch(label,[1],capture=True,startup_profiling=True)
            began = time.monotonic()
            worker.warm(data["source_cases"]["train-code"]["frozen"]["tokens"])
            ended = time.monotonic()
            diagnostic = worker.inspection["startup_diagnostics"][0]
            worker.close()
            frontend = [e for line in worker.log_lines for e in diagnostic_log_events(line)]
            row = cold_breakdown(worker.evidence(),diagnostic,frontend,worker.started,began,ended)
            row.update({"label":label,"cache_state":"cold" if repeat==0 else "hit_candidate",
                        "diagnostics":diagnostic,"cache_before":before,"cache_after":cache_inventory(flashinfer),
                        "warmup_sample":worker.records[-1],"runtime":worker.runtime})
            io = [e for e in diagnostic["startup"]["events"] if e["event"]=="loading_io_observation"][0]
            row["loading_io_deltas"] = {k:io["after"][k]-v for k,v in io["before"].items() if k in io["after"]}
            row["cache_libraries_unchanged"] = row["cache_before"]==row["cache_after"]
            result["runs"].append(row)
            write_artifact(out/(label+".json"),row)
            print("MOE_CACHE_PROGRESS="+json.dumps({"label":label,"total_s":row["total_s"],
                "sampler_s":row["nested_worker_spans_s"]["runner__dummy_sampler_run"],
                "libraries_unchanged":row["cache_libraries_unchanged"]}),flush=True)
        hits=result["runs"][1:]
        result["warm_cache_summary"]={"mean_s":statistics.mean(r["total_s"] for r in hits),
                                      "sample_sd_s":statistics.stdev(r["total_s"] for r in hits) if len(hits)>1 else None}
        result["cold_cache_total_s"]=result["runs"][0]["total_s"]
        result["status"]="passed"
    finally:
        pool.close()
        result["environment_final"]=snapshot()
        write_artifact(out/"cache_diagnostics.json",result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
