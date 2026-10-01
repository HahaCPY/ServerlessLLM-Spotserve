"""One real EP source freeze and sequential EP placement/shape recovery pilots.

This records candidate execution costs; it does not label them formal O/M A/B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from scripts.moe_gpu_runtime import GPUPool, write_artifact
from scripts.run_granite_moe_replay_pilot import digest
from scripts.run_moe_calibrated_experiments import GPU_UUIDS


def progress(**fields):
    print("DYNAMIC_EP_PILOT=" + json.dumps(fields), flush=True)


def free_group(group):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = {int(bits[0]): (int(bits[1]), int(bits[2])) for line in result.stdout.splitlines()
            if (bits := [part.strip() for part in line.split(",")])}
    if any(rows[g][0] > 1500 or rows[g][1] > 10 for g in group):
        raise RuntimeError("GPU group occupied; preserving other users' workloads: " + str(rows))
    return rows


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compiler-cache", required=True)
    parser.add_argument("--source-json", help="Reuse an independently verified live EP source prefix")
    parser.add_argument("--pairs", nargs="*", default=["12", "13", "23"])
    parser.add_argument("--shapes", nargs="*", default=["tp2dp1", "tp1dp2"])
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--freeze-tokens", type=int, default=64)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "single_event_EP_candidate_recovery_pilot_not_formal_original_moe_A_B",
              "status": "running", "model": args.model, "revision": args.revision,
              "source_group": [0, 1], "source_shape": "TP2/DP1/EP2",
              "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
              "freeze_tokens": args.freeze_tokens, "recovery_method": "host_tokens_GPU_replay",
              "NIXL_KV_restore": False, "candidate_results": [],
              "source": None, "environment": None}
    pool = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                                   trust_remote_code=False)
        seed = tokenizer.encode("Explain serving a sparse language model on changing GPU resources. ")
        prompt = (seed * (args.prompt_tokens // len(seed) + 1))[:args.prompt_tokens]
        report["prompt_sha256"] = digest(prompt)
        report["environment"] = free_group([1, 2, 3] if args.source_json else [0, 1, 2, 3])
        pool = GPUPool(out / "worker_logs", args.model, args.revision, GPU_UUIDS,
                       timeout=600, compiler_cache=args.compiler_cache)
        if args.source_json:
            prior = json.loads(Path(args.source_json).read_text())
            reference, freeze = prior["reference"], prior["freeze"]
            if (freeze["frozen"]["prompt_sha256"] != digest(prompt)
                    or freeze["frozen"]["completed_tokens"] != args.freeze_tokens
                    or len(reference["output_token_ids"]) != args.output_tokens
                    or not prior["worker"]["runtime"]["enable_expert_parallel"]
                    or prior["worker"]["physical_gpu_indices"] != [0, 1]):
                raise ValueError("prior live EP source evidence does not match this pilot")
            report["source"] = prior
            report["prior_source_artifact"] = args.source_json
            report["prior_source_sha256"] = digest(prior)
            write_artifact(out / "source.json", prior)
            progress(phase="source_reused", artifact=args.source_json)
        else:
            progress(phase="source_launch")
            source = pool.launch("source-01", [0, 1], capture=True, tp=2, dp=1, ep=True)
            source.warm(prompt)
            reference = source.measure("source-reference", prompt, args.output_tokens)
            freeze = source.freeze("source-freeze", prompt, output_tokens=args.output_tokens,
                                   threshold=args.freeze_tokens)
            if freeze["client_prefix"] != reference["output_token_ids"][:args.freeze_tokens]:
                raise ValueError("EP source frozen prefix differs from uninterrupted reference")
            source.abort_resume("source-freeze")
            report["source"] = {"reference": reference, "freeze": freeze,
                                "worker": source.evidence()}
            write_artifact(out / "source.json", report["source"])
            progress(phase="source_verified", frozen=freeze["frozen"]["completed_tokens"],
                     route_sha=freeze["routes"]["histogram_sha256"])
            source.close()
        tokens = freeze["frozen"]["tokens"]
        remaining = freeze["frozen"]["remaining_tokens"]
        for pair in args.pairs:
            group = [int(char) for char in pair]
            for shape in args.shapes:
                tp, dp = (2, 1) if shape == "tp2dp1" else (1, 2)
                label = f"target-{pair}-{shape}"
                row = {"candidate": label, "pair": group, "tp": tp, "dp": dp,
                       "ep": True, "status": "running"}
                worker = None
                try:
                    free_group(group)
                    progress(phase="target_launch", candidate=label)
                    started = time.monotonic()
                    worker = pool.launch(label, group, capture=False, tp=tp, dp=dp, ep=True)
                    row["engine_ready_s"] = worker.cold_to_engine_ready_s
                    worker.warm(tokens)
                    row["warmed_ready_s"] = time.monotonic() - started
                    install_start = time.monotonic()
                    ack = worker.command_event({"op": "install_replay_tokens",
                        "request_id": label, "token_ids": tokens}, "replay_tokens_installed")
                    row["host_install_s"] = time.monotonic() - install_start
                    if (ack["frozen_prefix_sha256"] != freeze["frozen"]["full_prefix_sha256"]
                            or ack["token_count"] != len(tokens) or ack["gpu_kv_restored"]):
                        raise ValueError("host token handoff verification failed")
                    resumed = worker.measure(label, output_tokens=remaining, installed=True)
                    row["remaining_service_s"] = resumed["latency_s"]
                    row["candidate_launch_to_final_s"] = (row["warmed_ready_s"]
                        + row["host_install_s"] + row["remaining_service_s"])
                    row["matches_uninterrupted_source"] = (
                        freeze["client_prefix"] + resumed["output_token_ids"]
                        == reference["output_token_ids"])
                    row["resume"] = resumed
                    row["handoff_ack"] = ack
                    row["status"] = "passed" if row["matches_uninterrupted_source"] else "divergent"
                except Exception:
                    row["status"] = "failed"
                    row["traceback"] = traceback.format_exc()
                finally:
                    if worker is not None:
                        worker.close()
                        row["worker"] = worker.evidence()
                    write_artifact(out / (label + ".json"), row)
                    report["candidate_results"].append(row)
                    progress(phase="target_done", candidate=label, status=row["status"],
                             candidate_launch_to_final_s=row.get("candidate_launch_to_final_s"),
                             error=row.get("traceback", "")[-300:])
        report["status"] = "passed" if all(r["status"] == "passed"
                                            for r in report["candidate_results"]) else "partial"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    finally:
        if pool is not None:
            pool.close()
        write_artifact(out / "report.json", report)
        progress(phase="finished", status=report["status"],
                 error=report.get("traceback", "")[-500:])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
