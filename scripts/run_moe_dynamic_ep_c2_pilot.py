"""C2 4096/512/around-256 native EP source and sequential EP recovery pilot."""

import argparse
import json
from pathlib import Path
import time
import traceback

from scripts.check_granite_native_freeze import validate_route_histogram
from scripts.moe_gpu_runtime import GPUPool, write_artifact
from scripts.run_granite_moe_replay_pilot import digest
from scripts.run_moe_calibrated_experiments import GPU_UUIDS
from scripts.run_moe_dynamic_ep_recovery_pilot import free_group


def log(**fields):
    print("EP_C2_PROGRESS=" + json.dumps(fields), flush=True)


def generate_pair(worker, ids, prompts, lengths, dp=1, installed=False, freeze=0):
    began = time.monotonic()
    for index, rid in enumerate(ids):
        worker.send({"op": "generate", "request_id": rid,
                     "token_ids": None if installed else prompts[index],
                     "use_installed_replay": installed, "max_new_tokens": lengths[index],
                     "pause_after_new_tokens": freeze,
                     "data_parallel_rank": index if dp > 1 else 0})
    outputs, ended = {rid: [] for rid in ids}, {}
    while len(ended) < 2:
        event = worker.wait(lambda e: e.get("event") == "output"
                            and e.get("request_id") in ids)
        rid = event["request_id"]
        tokens = list(event["cumulative_token_ids"])
        if tokens[:len(outputs[rid])] != outputs[rid]:
            raise ValueError("stream revised delivered C2 tokens")
        outputs[rid] = tokens
        if event.get("finished"):
            if len(tokens) != lengths[ids.index(rid)] or event.get("finish_reason") != "length":
                raise ValueError("C2 output length/finish reason mismatch")
            ended[rid] = event["host_received_monotonic_s"]
    return {"batch_wall_s": max(ended.values()) - began, "outputs": outputs,
            "per_request_latency_s": {rid: ended[rid] - began for rid in ids}}


def native_freeze_pair(source, prompts, refs, config):
    ids = ("freeze0", "freeze1")
    for rid, prompt in zip(ids, prompts):
        source.send({"op": "generate", "request_id": rid, "token_ids": prompt,
                     "max_new_tokens": 512, "pause_after_new_tokens": 256})
    visible = {rid: [] for rid in ids}
    while True:
        event = source.wait(lambda e: e.get("event") in ("output", "paused")
                            and e.get("request_id") in ids)
        rid = event["request_id"]
        if event["event"] == "paused":
            break
        tokens = list(event["cumulative_token_ids"])
        if tokens[:len(visible[rid])] != visible[rid] or event.get("finished"):
            raise ValueError("C2 source streamed revision/completed before freeze")
        visible[rid] = tokens
    notice = time.monotonic()
    source.command_event({"op": "pause_generation"}, "generation_paused")
    pause_ack_s = time.monotonic() - notice
    with source.event_condition:
        for event in source.events:
            if event.get("event") == "output" and event.get("request_id") in ids:
                rid = event["request_id"]
                tokens = list(event["cumulative_token_ids"])
                if len(tokens) > len(visible[rid]):
                    visible[rid] = tokens
    cases = []
    for index, rid in enumerate(ids):
        metadata = source.command_event({"op": "metadata", "request_id": rid},
                                        "metadata")["result"]
        output = list(metadata.get("output_tokens", []))
        count = metadata.get("completed_tokens")
        if (not metadata.get("found") or metadata.get("prompt_tokens") != prompts[index]
                or metadata.get("tokens") != prompts[index] + output
                or count != len(output) or not 240 <= count <= 256
                or output != refs["outputs"][f"reference{index}"][:count]
                or output[:len(visible[rid])] != visible[rid]
                or not metadata.get("allocated_kv_block_count")):
            raise ValueError("C2 source exact frozen prefix/KV/reference gate failed")
        routes = validate_route_histogram(metadata, config["num_hidden_layers"],
                                          config["num_local_experts"],
                                          config["num_experts_per_tok"])
        cases.append({"prompt_token_ids": prompts[index], "frozen_output_token_ids": output,
                      "frozen_prefix_token_ids": prompts[index] + output,
                      "frozen_prefix_sha256": digest(prompts[index] + output),
                      "completed_output_tokens": count, "remaining_tokens": 512 - count,
                      "client_visible_tokens": len(visible[rid]),
                      "allocated_kv_block_count": metadata["allocated_kv_block_count"],
                      "routes": routes})
    return {"notice_to_pause_ack_s": pause_ack_s, "cases": cases}


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--compiler-cache", required=True)
    ap.add_argument("--pairs", nargs="*", default=["23"])
    ap.add_argument("--shapes", nargs="*", default=["tp2dp1", "tp1dp2"])
    ap.add_argument("--source-group", default="01")
    ap.add_argument("--prior-source-json")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    source_group = [int(ch) for ch in args.source_group]
    if len(source_group) != 2 or len(set(source_group)) != 2:
        raise ValueError("source group must name two distinct GPUs")
    report = {"scope": "C2_P4096_N512_freeze_around256_EP_recovery_pilot",
              "status": "running", "source_group": source_group, "source_shape": "TP2/DP1/EP2",
              "candidate_results": [], "original_moe_aware_A_B_completed": False,
              "recovery": "host_tokens_GPU_replay_no_direct_NIXL_KV"}
    pool = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                                   trust_remote_code=False)
        seeds = [tokenizer.encode(text) for text in (
            "Explain expert routing, KV cache and GPU scheduling. ",
            "Describe sparse language model recovery after GPU loss. ")]
        prompts = [(seed * (4096 // len(seed) + 1))[:4096] for seed in seeds]
        report["prompt_sha256"] = [digest(prompt) for prompt in prompts]
        free_group(sorted(set(([int(ch) for pair in args.pairs for ch in pair]
                               if args.prior_source_json else
                               source_group + [int(ch) for pair in args.pairs for ch in pair]))))
        pool = GPUPool(out / "worker_logs", args.model, args.revision, GPU_UUIDS,
                       timeout=900, compiler_cache=args.compiler_cache, max_num_seqs=2)
        if args.prior_source_json:
            prior = json.loads(Path(args.prior_source_json).read_text())
            refs, freeze = prior["reference"], prior["freeze"]
            if (prior["runtime"]["runtime"]["enable_expert_parallel"] is not True
                    or [digest(c["prompt_token_ids"]) for c in freeze["cases"]]
                    != report["prompt_sha256"]):
                raise ValueError("prior C2 EP source evidence does not match")
            report["source"] = prior
            report["prior_source_json"] = args.prior_source_json
            write_artifact(out / "source.json", prior)
            log(phase="source_reused")
        else:
            log(phase="source_launch")
            source = pool.launch("source-" + args.source_group, source_group,
                                 capture=True, tp=2, dp=1, ep=True)
            source.warm(prompts[0])
            warm = generate_pair(source, ["warm0", "warm1"], prompts, [32, 32])
            refs = generate_pair(source, ["reference0", "reference1"], prompts, [512, 512])
            freeze = native_freeze_pair(source, prompts, refs, pool.config)
            report["source"] = {"warm": warm, "reference": refs, "freeze": freeze,
                                "runtime": source.evidence()}
            write_artifact(out / "source.json", report["source"])
            log(phase="source_verified", freeze_counts=[c["completed_output_tokens"]
                 for c in freeze["cases"]], reference_batch_wall_s=refs["batch_wall_s"])
            source.close()
        cases = freeze["cases"]
        prefixes = [case["frozen_prefix_token_ids"] for case in cases]
        for pair in args.pairs:
            group = [int(ch) for ch in pair]
            for shape in args.shapes:
                tp, dp = (2, 1) if shape == "tp2dp1" else (1, 2)
                name = f"target-{pair}-{shape}"
                row = {"candidate": name, "group": group, "tp": tp, "dp": dp,
                       "EP_enabled": True, "status": "running"}
                target = None
                try:
                    free_group(group)
                    log(phase="target_launch", candidate=name)
                    began = time.monotonic()
                    target = pool.launch(name, group, capture=False, tp=tp, dp=dp, ep=True)
                    row["engine_ready_s"] = target.cold_to_engine_ready_s
                    row["warm"] = generate_pair(target, [name+"warm0", name+"warm1"],
                                                prefixes, [32, 32], dp=dp)
                    row["warmed_ready_s"] = time.monotonic() - began
                    began_install = time.monotonic()
                    for i in range(2):
                        ack = target.command_event({"op": "install_replay_tokens",
                            "request_id": name+str(i), "token_ids": prefixes[i]},
                            "replay_tokens_installed")
                        if (ack["frozen_prefix_sha256"] != cases[i]["frozen_prefix_sha256"]
                                or ack["gpu_kv_restored"]):
                            raise ValueError("C2 handoff prefix mismatch")
                    row["host_handoff_s"] = time.monotonic() - began_install
                    row["remaining"] = generate_pair(target, [name+"0", name+"1"],
                        prefixes, [case["remaining_tokens"] for case in cases],
                        dp=dp, installed=True)
                    row["launch_to_all_final_s"] = (row["warmed_ready_s"]
                        + row["host_handoff_s"] + row["remaining"]["batch_wall_s"])
                    row["exact_reference_match"] = [
                        cases[i]["frozen_output_token_ids"]
                        + row["remaining"]["outputs"][name+str(i)]
                        == refs["outputs"]["reference"+str(i)] for i in range(2)]
                    row["status"] = "passed" if all(row["exact_reference_match"]) else "divergent"
                except Exception:
                    row["status"] = "failed"
                    row["traceback"] = traceback.format_exc()
                finally:
                    if target is not None:
                        target.close()
                        row["runtime"] = target.evidence()
                    write_artifact(out / (name + ".json"), row)
                    report["candidate_results"].append(row)
                    log(phase="target_done", candidate=name, status=row["status"],
                        all_final_s=row.get("launch_to_all_final_s"),
                        error=row.get("traceback", "")[-250:])
        report["status"] = ("passed" if all(r["status"] == "passed"
                                           for r in report["candidate_results"]) else "partial")
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        if pool is not None:
            pool.close()
        write_artifact(out / "report.json", report)
        log(phase="finished", status=report["status"], error=report.get("traceback", "")[-400:])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
