"""Independent same-checkpoint FP32 oracle at first common-prefix divergences."""

import argparse
import json
from pathlib import Path
import time

from scripts.analyze_moe_phase2_diagnostics import first_difference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--calibration",required=True)
    parser.add_argument("--ep-dir",required=True)
    parser.add_argument("--modes",nargs="+",choices=("tp2-ep2","dp2-ep2"),default=["tp2-ep2","dp2-ep2"])
    parser.add_argument("--layer-trace",action="store_true",help="Copy last-query native HF layer vectors, without changing computation")
    args=parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1)
    calibration=json.loads(Path(args.calibration).read_text())
    ep=Path(args.ep_dir)
    control=json.loads((ep/"tp2-reference.json").read_text())["report"]["correctness_generations"]
    refs={r["case"]:r for r in control}
    tasks={}
    for label in args.modes:
        path=ep/(label+".json")
        if path.exists():
            observed=json.loads(path.read_text())["report"]["correctness_generations"]
        else:
            marker="MOE_PHASE2_CORRECTNESS_JSON="
            partial=next((line[len(marker):] for line in (ep/(label+".log")).read_text().splitlines()
                          if line.startswith(marker)),None)
            if partial is None:
                raise ValueError("requested mode has no completed correctness snapshot")
            observed=json.loads(partial)["correctness_generations"]
        for row in observed:
            ref=refs[row["case"]]; index=first_difference(ref["output_token_ids"],row["output_token_ids"])
            if index is None:
                continue
            case=row["case"]
            prompt=([17 if case.endswith("17") else 18]*4096 if case.startswith("previous-repeated")
                    else calibration["source_cases"][case]["prompt_token_ids"])
            task=tasks.setdefault((case,index),{"case":case,"index":index,
                "prefix":prompt+ref["output_token_ids"][:index],"comparisons":[]})
            task["comparisons"].append({"mode":label,"tp_token":ref["output_token_ids"][index],
                "ep_token":row["output_token_ids"][index],"tp_logprobs":ref["logprobs"][index],
                "ep_logprobs":row["logprobs"][index]})
    if not tasks:
        print("MOE_FP32_JSON="+json.dumps({"status":"not_needed","divergences":[]})); return 0
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
        local_files_only=True,trust_remote_code=False,attn_implementation="eager").eval().to("cuda")
    if type(model).__name__!="GraniteMoeForCausalLM" or model.config.num_local_experts!=40:
        raise ValueError("not the same Granite MoE architecture")
    layer_trace={}
    if args.layer_trace:
        for index,layer in enumerate(model.model.layers):
            def before(module,inputs,key=index):
                layer_trace[str(key)]={"input":inputs[0].reshape(-1,1536)[-1].float().cpu().tolist()}
            def after(module,inputs,output,key=index):
                layer_trace[str(key)]["moe_output"]=output.reshape(-1,1536)[-1].float().cpu().tolist()
            def router(module,inputs,output,key=index):
                ids,weights,logits=output
                layer_trace[str(key)].update({"router_logits":logits[-1].float().cpu().tolist(),
                    "selected_ids":ids[-1].cpu().tolist(),"selected_weights":weights[-1].float().cpu().tolist()})
            def decoder(module,inputs,output,key=index):
                layer_trace[str(key)]["decoder_output"]=output.reshape(-1,1536)[-1].float().cpu().tolist()
            layer.block_sparse_moe.register_forward_pre_hook(before)
            layer.block_sparse_moe.register_forward_hook(after)
            layer.block_sparse_moe.router.register_forward_hook(router)
            layer.register_forward_hook(decoder)
    report={"status":"running","model_class":type(model).__name__,"dtype":"float32",
        "tf32_enabled":torch.backends.cuda.matmul.allow_tf32,"load_s":time.monotonic()-started,
        "gpu_uuid":str(torch.cuda.get_device_properties(0).uuid),"chunk_tokens":256,
        "scope":"HF FP32 cross-implementation oracle at first common-prefix divergence, not exhaustive vLLM FP32 reference",
        "layer_trace_enabled":args.layer_trace,"divergences":[]}
    with torch.inference_mode():
        for task in tasks.values():
            layer_trace.clear()
            past=None; result=None; began=time.monotonic()
            for offset in range(0,len(task["prefix"]),256):
                ids=torch.tensor([task["prefix"][offset:offset+256]],device="cuda")
                result=model(input_ids=ids,past_key_values=past,use_cache=True,logits_to_keep=1)
                past=result.past_key_values
            logits=result.logits[0,-1].float(); lp=torch.log_softmax(logits,-1)
            topvals,topids=lp.topk(64)
            observed={str(i):float(v) for i,v in zip(topids.cpu().tolist(),topvals.cpu().tolist())}
            comparisons=[]
            for c in task["comparisons"]:
                candidates=set(c["tp_logprobs"])|set(c["ep_logprobs"])
                probabilities={i:float(lp[int(i)].item()) for i in candidates}
                comparisons.append({**c,"fp32_candidate_logprobs":probabilities,
                    "tp_max_shared_fp32_logprob_error":max(abs(v["logprob"]-probabilities[i]) for i,v in c["tp_logprobs"].items()),
                    "ep_max_shared_fp32_logprob_error":max(abs(v["logprob"]-probabilities[i]) for i,v in c["ep_logprobs"].items()),
                    "fp32_tp_minus_ep_token_logprob":probabilities[str(c["tp_token"])]-probabilities[str(c["ep_token"])]})
            report["divergences"].append({"case":task["case"],"first_difference_1based":task["index"]+1,
                "prefix_tokens":len(task["prefix"]),"fp32_top64_logprobs":observed,
                "fp32_top1_token":int(logits.argmax().item()),"comparisons":comparisons,
                "oracle_s":time.monotonic()-began})
            if args.layer_trace:
                report["divergences"][-1]["layers"]=dict(layer_trace)
            print("MOE_FP32_PROGRESS="+json.dumps({"case":task["case"],"token_1based":task["index"]+1,
                "oracle_s":report["divergences"][-1]["oracle_s"]}),flush=True)
            del past,result
            torch.cuda.empty_cache()
    report["status"]="passed"
    print("MOE_FP32_JSON="+json.dumps(report),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
