"""Auditable numeric comparisons; cache states are never pooled."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.moe_gpu_runtime import write_artifact
from scripts.run_moe_architecture_diagnostics import diagnostic_log_events


def first_difference(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),
                min(len(a),len(b)) if len(a)!=len(b) else None)


def verify_collective_gate(report):
    """Check first native two-rank collective at identical tiny inputs.

    This is a conservation test, not an assertion that TP and EP BF16 logits
    must match exactly. Capture is disabled for timed serving measurements.
    """
    gate = report.get("collective_gate")
    if not gate:
        return {"available": False, "passes": False, "reason": "not captured"}
    ranks = sorted(gate["collectives"], key=lambda r:r["rank"])
    if len(ranks) != 2:
        return {"available": True, "passes": False, "reason": "expected two physical ranks"}
    checks = []

    def rows(record):
        return record["values"]

    def compare(label, actual, expected, *, exact=False):
        def flatten(value):
            if isinstance(value, list):
                for child in value: yield from flatten(child)
            else:
                yield value
        a, e = list(flatten(actual)), list(flatten(expected))
        if len(a) != len(e):
            result = {"label": label, "passed": False, "actual_values": len(a),
                      "expected_values": len(e), "reason": "shape mismatch"}
        else:
            scale = sum(v*v for v in e)**0.5
            rel = (sum((x-y)**2 for x,y in zip(a,e))**0.5) / max(scale,1e-8)
            maximum = max((abs(x-y) for x,y in zip(a,e)), default=0)
            refmax = max((abs(v) for v in e), default=0)
            result = {"label": label, "relative_l2": rel, "max_abs": maximum,
                "reference_max_abs": refmax, "actual_values": len(a),
                "passed": (maximum == 0 if exact else
                    rel <= 0.03 and maximum <= 0.02 + 0.1 * refmax)}
        checks.append(result)

    records = [rank["records"] for rank in ranks]
    if all("agrs_dispatch" in row for row in records):
        for key in ("hidden", "weights", "ids"):
            full = sum((rows(row["agrs_dispatch"]["input"][key]) for row in records), [])
            for i, row in enumerate(records):
                compare(f"agrs_dispatch_{key}_rank{i}",
                        rows(row["agrs_dispatch"]["output"][key]), full, exact=True)
        if not all("agrs_combine" in row for row in records):
            checks.append({"label":"agrs_combine_pair", "passed":False,
                           "reason":"only dispatch captured"})
        else:
            inputs = [rows(row["agrs_combine"]["input"]) for row in records]
            local_lengths = [len(rows(row["agrs_dispatch"]["input"]["hidden"])) for row in records]
            if len({len(item) for item in inputs}) != 1 or sum(local_lengths) != len(inputs[0]):
                checks.append({"label":"agrs_combine_shape", "passed":False,
                               "reason":"nonmatching rank buffer / local chunk lengths"})
            else:
                summed = [[x+y for x,y in zip(a,b)] for a,b in zip(*inputs)]
                start = 0
                for i, (record, length) in enumerate(zip(records,local_lengths)):
                    compare(f"agrs_combine_rank{i}",rows(record["agrs_combine"]["output"]),
                            summed[start:start+length])
                    start += length
    elif any("agrs_dispatch" in row or "agrs_combine" in row for row in records):
        checks.append({"label":"agrs_pair", "passed":False,"reason":"only one rank captured AGRS"})
    if all("moe_final_reduce" in row for row in records):
        reductions = [row["moe_final_reduce"] for row in records]
        inputs = [rows(r["input"]) for r in reductions]
        if len(inputs[0]) == len(inputs[1]):
            full = [[x+y for x,y in zip(a,b)] for a,b in zip(*inputs)]
            for i, reduction in enumerate(reductions):
                actual = rows(reduction["output"])
                # Native helper calls the outer TP group: TP2+EP2 has
                # moe_config.tp_size=1 but actual TP world_size=2, whereas
                # DP2+EP2 has moe_config.ep_size=2 and outer TP world_size=1.
                if (reduction.get("actual_tp_group_size", report.get("tp",1)) == 1 or
                    reduction.get("is_sequence_parallel",False) or
                    reduction.get("skip_final_all_reduce",False) or
                    reduction.get("fused_output_is_reduced",False)):
                    compare(f"moe_final_noop_rank{i}",actual,inputs[i],exact=True)
                else:
                    compare(f"moe_final_reduce_rank{i}",actual,full)
        else:
            checks.append({"label":"moe_final_reduce_shape", "passed":False})
    elif any("moe_final_reduce" in row for row in records):
        checks.append({"label":"moe_final_reduce_pair", "passed":False})
    snapshots = gate["runtime_correctness"]
    for i, rank in enumerate(snapshots):
        totals = rank["router_totals"]
        checks.append({"label":f"tiny_routing_rank{i}", "passed":
            totals.get("assignments",0)>0 and all(totals.get(k,0)==0 for k in
                ("invalid_ids","duplicate_rows","nonfinite_weights","topk_non_tie_different_rows"))})
        checks.append({"label":f"tiny_expert_rank{i}", "passed":
            bool(rank["expert_oracle"]) and all(row["passes_predeclared_tolerance"] and
                row["nonfinite_output_values"]==0 for row in rank["expert_oracle"])})
    return {"available":True,"passes":bool(checks) and all(r["passed"] for r in checks),
            "checks": checks, "rank_expert_coverage":[r["global_expert_ids"] for r in ranks],
            "scope":"first tiny native collective per mode and first expert-kernel call per layer; not exhaustive all-token proof"}


def duplicate_normal_request_gate(report):
    gate=report.get("batch_shape_gate")
    if not gate:
        return {"available":False}
    a,b=(gate["requests"][i] for i in gate["normal_train_code_duplicated_at_indices"])
    if a["prompt_sha256"]!=b["prompt_sha256"]:
        raise ValueError("duplicate gate used different prompt tokens")
    first=first_difference(a["output_token_ids"],b["output_token_ids"])
    result={"available":True,"same_prompt":True,"first_different_token_1based":
        first+1 if first is not None else None,"exact_output":first is None,
        "scope":"normal duplicated input within same C8 engine batch; not a full correct-answer oracle"}
    if first is None:
        return result
    if a["output_token_ids"][:first]!=b["output_token_ids"][:first]:
        raise AssertionError("first difference has unequal previous prefixes")
    x,y=a["logprobs"][first],b["logprobs"][first]
    shared=set(x)&set(y)
    errors=[abs(x[token]["logprob"]-y[token]["logprob"]) for token in shared]
    def top2(values):
        scores=sorted([(v["logprob"],k) for k,v in values.items()],reverse=True)
        return {"top_id":scores[0][1],"top_logprob":scores[0][0],
                "top2_margin":scores[0][0]-scores[1][0] if len(scores)>1 else None}
    result.update({"common_previous_tokens":first,"reference_first_token":a["output_token_ids"][first],
        "second_first_token":b["output_token_ids"][first],"shared_logprob_ids":len(shared),
        "max_shared_logprob_abs_error":max(errors) if errors else None,
        "topk_first_request":top2(x),"topk_second_request":top2(y)})
    return result


def logprob_comparison(reference,observed,tolerance=0.15):
    assert len(reference)==len(observed)
    rows=[]
    for index,(a,b) in enumerate(zip(reference,observed)):
        shared=set(a)&set(b)
        error=max((abs(a[k]["logprob"]-b[k]["logprob"]) for k in shared),default=float("inf"))
        atop=max(v["logprob"] for v in a.values()); btop=max(v["logprob"] for v in b.values())
        abest={k for k,v in a.items() if atop-v["logprob"]<=1e-7}
        bbest={k for k,v in b.items() if btop-v["logprob"]<=1e-7}
        values=sorted({v["logprob"] for v in a.values()},reverse=True)
        gap=values[0]-values[1] if len(values)>1 else 0
        different=not bool(abest&bbest)
        rows.append({"token_index_1based":index+1,"max_shared_logprob_error":error,
                     "top1_sets_disjoint":different,"reference_top1_ids":sorted(abest),
                     "observed_top1_ids":sorted(bbest),"reference_gap_to_next_distinct_score":gap,
                     "passes_abs_tolerance":error<=tolerance,
                     "top1_change_within_error_bound":not different or gap<=2*error+1e-7})
    return {"tokens":len(rows),"max_shared_logprob_error":max(r["max_shared_logprob_error"] for r in rows),
            "top1_disjoint_positions":sum(r["top1_sets_disjoint"] for r in rows),
            "abs_tolerance_failures":sum(not r["passes_abs_tolerance"] for r in rows),
            "unexplained_top1_changes":sum(not r["top1_change_within_error_bound"] for r in rows),
            "per_token":rows,"scope":"shared returned top-k/true-token logprobs, not full-vocabulary KL"}


def runtime_correctness(report):
    snapshots=[r for batch in report["runtime_correctness"] for r in batch["ranks"]]
    snapshots += report["high_routing_load"]["ranks"]
    result=[]
    for s in snapshots:
        t=s["router_totals"]; oracle=s["expert_oracle"]
        passed=(s["router_objects"]==32 and len(oracle)==32 and t.get("assignments",0)>0
                and all(t.get(k,0)==0 for k in ("invalid_ids","duplicate_rows","nonfinite_weights","topk_non_tie_different_rows"))
                and t.get("max_weight_reference_error",1)<=1e-5
                and t.get("max_weight_sum_error",1)<=1e-5
                and all(o["passes_predeclared_tolerance"] and o["nonfinite_output_values"]==0 for o in oracle))
        result.append({"local_rank":s["local_rank"],"router_totals":t,"expert_kernel_objects":len(oracle),
                       "max_expert_relative_l2":max((o["relative_l2_error"] for o in oracle),default=None),
                       "passes_checks":passed})
    return {"all_pass":all(s["passes_checks"] for s in result),"snapshots":result,
            "scope":"all routing rows plus 16 expert-output rows/object including tail; not exhaustive all-token correctness proof"}


def kernel_category(name):
    # CUDA device annotations can duplicate the real kernel's duration.
    if name.startswith(("MoEDiag:","nccl:","aten:")):
        return None
    if "ncclDevKernel" in name:
        for kind in ("AllGather","ReduceScatter","AllReduce"):
            if kind in name: return "collective_"+kind
        return "collective_other"
    if "fused_moe_kernel" in name: return "expert_gemm"
    if "moe_align" in name: return "dispatch_alignment"
    if "moeTopK" in name or "moeSoftmax" in name: return "router"
    if "flash_fwd" in name or "flash_attn" in name: return "attention"
    if "gemv" in name or "gemm" in name or "Gemm" in name: return "dense_gemm"
    if name.startswith(("Memcpy","Memset")): return "memory_copy"
    return "other_kernels"


def profile_breakdown(report):
    ordinary={(c["prompt_tokens"],c["output_tokens"],c["concurrency"]):
        statistics.mean(s["batch_wall_s"] for s in c["samples"]) for c in report["performance"]}
    cells=[]
    for cell in report["profiles"]:
        key=(cell["prompt_tokens"],cell["output_tokens"],cell["concurrency"])
        ranks=[]
        for kernel,spans in zip(cell["kernel_profiles"],cell["ranks"]):
            totals={}; calls={}; excluded=[]
            for event in kernel["cuda_kernels"]:
                category=kernel_category(event["name"])
                if category is None:
                    excluded.append(event["name"]); continue
                totals[category]=totals.get(category,0)+event["self_device_us"]/1000
                calls[category]=calls.get(category,0)+event["calls"]
            ranks.append({"local_rank":kernel["local_rank"],"actual_kernel_ms":totals,"kernel_calls":calls,
                "excluded_device_annotations":excluded,"instrumented_spans":spans["current"]["cpu_totals"],
                "instrumented_gpu_intervals":spans["current"]["gpu_totals"]})
        cells.append({"prompt_tokens":key[0],"output_tokens":key[1],"concurrency":key[2],
            "ordinary_batch_mean_s":ordinary[key],"profile_batch_s":cell["sample"]["batch_wall_s"],
            "profiler_wall_multiplier":cell["sample"]["batch_wall_s"]/ordinary[key],"ranks":ranks,
            "scope":"top-80 recorded CUDA event types; annotations excluded; per-rank kernel work, NOT an exclusive end-to-end critical-path partition"})
    return cells


def vector_relative_l2(a,b):
    assert len(a)==len(b)
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))/max(math.sqrt(sum(y*y for y in b)),1e-12)


def layer_comparison(reference,observed):
    rows=[]
    for key in sorted(reference,key=int):
        a,b=reference[key],observed[key]
        scores=sorted(a["router_logits"],reverse=True)
        rows.append({"layer":int(key),"input_equal":a["input"]==b["input"],
            "router_logits_equal":a["router_logits"]==b["router_logits"],
            "selected_sets_equal":set(a["selected_ids"])==set(b["selected_ids"]),
            "reference_ids":a["selected_ids"],"observed_ids":b["selected_ids"],
            "reference_router_logit_top8_top9_gap":scores[7]-scores[8],
            "router_logit_max_abs_error":max(abs(x-y) for x,y in zip(a["router_logits"],b["router_logits"])),
            **{name+"_relative_l2":vector_relative_l2(b[name],a[name]) for name in ("input","moe_output","decoder_output")}})
    return {"first_input_difference":next((r["layer"] for r in rows if not r["input_equal"]),None),
        "first_router_logit_difference":next((r["layer"] for r in rows if not r["router_logits_equal"]),None),
        "first_selected_set_difference":next((r["layer"] for r in rows if not r["selected_sets_equal"]),None),"layers":rows}


def refined_cache_stages(row,all_process_events=None):
    p=row["clock_provenance"]; events=row["diagnostics"]["startup"]["events"]
    entry=next(e["monotonic_s"] for e in events if e["event"]=="diagnostic_python_entry")
    ctor=next(e["monotonic_s"] for e in events if e["event"]=="span_start" and e["name"]=="worker_constructor")
    starts={e["span_id"]:e for e in events if e["event"]=="span_start"}
    cuda=[]
    for e in events:
        if e["event"]=="span_end" and e["name"]=="early_cuda_lazy_initialization":
            start=starts[e["span_id"]]
            parent=starts.get(start.get("parent_id"),{})
            if parent.get("name")!="early_cuda_lazy_initialization":
                assert entry<=start["monotonic_s"]<=e["monotonic_s"]<=ctor
                cuda.append(e["duration_s"])
    major=dict(row["major_nonoverlapping_stages_s"])
    residual=major.pop("engine_bootstrap_and_other")
    major["engine_core_process_spawn_and_preworker_bootstrap"]=entry-p["engine_start"]
    major["worker_python_imports_except_cuda_lazy_init"]=ctor-entry-sum(cuda)
    major["cuda_lazy_initialization_outermost"]=sum(cuda)
    major["engine_bootstrap_after_worker_other"]=residual-(ctor-p["engine_start"])
    # The frontend also initializes a CUDA context before spawning the worker.
    # Its spans live in the raw frontend log, not the worker's RPC snapshot.
    if all_process_events is None:
        log_path=row.get("existing_worker_evidence",{}).get("log_path")
        all_process_events=([e for line in Path(log_path).read_text().splitlines()
                             for e in diagnostic_log_events(line)] if log_path else [])
    all_starts={(e["pid"],e["span_id"]):e for e in all_process_events if e["event"]=="span_start"}
    parent_cuda=0.0
    worker_pid=next(e["pid"] for e in events if e["event"]=="diagnostic_python_entry") if "pid" in events[0] else None
    for e in all_process_events:
        if e["event"]!="span_end" or e["name"]!="early_cuda_lazy_initialization" or e["pid"]==worker_pid:
            continue
        start=all_starts[(e["pid"],e["span_id"])]
        parent=all_starts.get((e["pid"],start.get("parent_id")),{})
        if parent.get("name")=="early_cuda_lazy_initialization":
            continue
        assert p["engine_start"]<=start["monotonic_s"]<=e["monotonic_s"]<=entry
        parent_cuda+=e["duration_s"]
    if parent_cuda:
        major["engine_core_process_spawn_and_preworker_bootstrap"]-=parent_cuda
    major["cuda_lazy_initialization_all_processes_outermost"]=major.pop("cuda_lazy_initialization_outermost")+parent_cuda
    assert min(major.values())>=0 and abs(sum(major.values())-row["total_s"])<1e-6
    return major


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-dir",required=True)
    parser.add_argument("--cache-dir",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(); ep=Path(args.ep_dir); cache=Path(args.cache_dir)
    raw={label:json.loads((ep/(label+".json")).read_text()) for label in ("tp2-reference","tp2-ep2","dp2-ep2")}
    reports={label:row["report"] for label,row in raw.items()}
    control=reports["tp2-reference"]; cgen={r["case"]:r for r in control["correctness_generations"]}
    cteacher={r["case"]:r for r in control["teacher_forced"]}
    summary={"scope":"diagnostic only; no formal EP latency conclusions before correctness resolution",
             "ep":{},"cache":{},"source_sha256":{},"layer_traces":[],"fp32_prefix_oracles":{}}
    for label,r in reports.items():
        out={"runtime_execution_status":raw[label]["status"],"runtime_correctness":runtime_correctness(r),
             "free_generation_comparison":[],"teacher_forced_comparison":[],"performance":[],"profiles":r["profiles"],
             "kernel_breakdown":profile_breakdown(r),"high_routing_generation_comparison":[],"formal_correctness_gate":"not_released"}
        for observed in r["correctness_generations"]:
            reference=cgen[observed["case"]]
            assert reference["prompt_sha256"]==observed["prompt_sha256"]
            position=first_difference(reference["output_token_ids"],observed["output_token_ids"])
            common=reference["output_token_ids"][:position] if position is not None else reference["output_token_ids"]
            diff={"case":observed["case"],"first_difference_1based":None if position is None else position+1,
                  "output_tokens":len(observed["output_token_ids"]),"common_generated_prefix_tokens":len(common)}
            if position is not None:
                diff.update({"tp_token":reference["output_token_ids"][position],"ep_token":observed["output_token_ids"][position],
                             "common_prefix_position_probs":logprob_comparison([reference["logprobs"][position]],[observed["logprobs"][position]])})
            out["free_generation_comparison"].append(diff)
        for observed,reference in zip(r["high_routing_load"]["generations"],control["high_routing_load"]["generations"]):
            assert reference["prompt_sha256"]==observed["prompt_sha256"]
            index=first_difference(reference["output_token_ids"],observed["output_token_ids"])
            out["high_routing_generation_comparison"].append({"case":observed["case"],
                "first_difference_1based":None if index is None else index+1,
                "stream_and_length_passed":observed["stream_and_length_passed"]})
        for observed in r["teacher_forced"]:
            ref=cteacher[observed["case"]]
            assert observed["teacher_prompt_sha256"]==ref["teacher_prompt_sha256"]
            comparison=logprob_comparison(ref["teacher_logprobs"],observed["teacher_logprobs"])
            out["teacher_forced_comparison"].append({"case":observed["case"],**comparison})
        for i,c in enumerate(r["performance"]):
            vals=[s["batch_wall_s"] for s in c["samples"]]
            base=statistics.mean(s["batch_wall_s"] for s in control["performance"][i]["samples"])
            mean=statistics.mean(vals)
            out["performance"].append({"prompt_tokens":c["prompt_tokens"],"output_tokens":c["output_tokens"],
                "concurrency":c["concurrency"],"batch_mean_s":mean,"batch_sample_sd_s":statistics.stdev(vals),
                "generated_tokens_per_s":c["concurrency"]*c["output_tokens"]/mean,"relative_to_tp2_percent":100*(mean/base-1),
                "formal_conclusion":False})
        summary["ep"][label]=out
        path=ep/(label+".json"); summary["source_sha256"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary["tp_self_decode_vs_teacher_forced"]=[{"case":r["case"],**logprob_comparison(cgen[r["case"]]["logprobs"],r["teacher_logprobs"])} for r in control["teacher_forced"]]
    cd=json.loads((cache/"cache_diagnostics.json").read_text())
    hits=cd["runs"][1:]
    summary["cache"]={"cold_cache_total_s":cd["runs"][0]["total_s"],"warm_cache_total":cd["warm_cache_summary"],
        "hits_libraries_unchanged":all(r["cache_libraries_unchanged"] for r in hits),
        "cold_refined_stages":refined_cache_stages(cd["runs"][0]),
        "hit_refined_stages":[refined_cache_stages(r) for r in hits],
        "hit_stage_summary":{},"hit_nested_summary":{},
        "hit_weight_loading_share_percent":[r["weight_loading_share_percent"] for r in hits],
        "hit_model_loading_upper_bound_percent":[r["entire_model_loading_upper_bound_percent"] for r in hits]}
    for key in summary["cache"]["hit_refined_stages"][0]:
        vals=[r[key] for r in summary["cache"]["hit_refined_stages"]]
        summary["cache"]["hit_stage_summary"][key]={"mean_s":statistics.mean(vals),"sample_sd_s":statistics.stdev(vals)}
    for key in ("checkpoint_and_weight_loading","distributed_initialization","runner__dummy_run","runner__dummy_sampler_run","flashinfer_build_and_load","flashinfer_ninja"):
        vals=[r["nested_worker_spans_s"][key] for r in hits]
        summary["cache"]["hit_nested_summary"][key]={"mean_s":statistics.mean(vals),"sample_sd_s":statistics.stdev(vals)}
    cp=cache/"cache_diagnostics.json"; summary["source_sha256"][str(cp)]=hashlib.sha256(cp.read_bytes()).hexdigest()
    tp_trace=ep/"tp2-layer-trace.json"; ep_trace=ep/"tp2-ep2-layer-trace.json"
    if tp_trace.exists() and ep_trace.exists():
        a=json.loads(tp_trace.read_text())["report"]["cases"]; b=json.loads(ep_trace.read_text())["report"]["cases"]
        for reference,observed in zip(a,b):
            assert reference["case"]==observed["case"] and reference["prefix_tokens"]==observed["prefix_tokens"]
            summary["layer_traces"].append({"case":reference["case"],"prefix_tokens":reference["prefix_tokens"],
                "tp_recomputed_next_token":reference["next_token"],"ep_recomputed_next_token":observed["next_token"],
                "ranks":[layer_comparison(x["layers"],y["layers"]) for x,y in zip(reference["ranks"],observed["ranks"])],
                "scope":"native BF16 last query at identical common prefix; recomputed prefill is not identical to cached autoregressive decode"})
        for path in (tp_trace,ep_trace): summary["source_sha256"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for label in ("tp2-ep2","dp2-ep2"):
        path=ep/(label+"-fp32.json")
        if path.exists():
            summary["fp32_prefix_oracles"][label]=json.loads(path.read_text())["report"]
            summary["source_sha256"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_artifact(Path(args.output),summary)
    print(json.dumps({"output":args.output,"cache":summary["cache"]["warm_cache_total"],
                      "ep_runtime_checks":{k:v["runtime_correctness"]["all_pass"] for k,v in summary["ep"].items()}}))


if __name__=="__main__":
    main()
