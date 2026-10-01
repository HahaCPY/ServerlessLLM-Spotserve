"""Audited TP service sweep; no preemption, fleet planner, or network mutation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time
import traceback
import uuid

from scripts.moe_gpu_runtime import write_artifact
from scripts.profile_granite_moe_tp import canonical_gpu_uuid, digest


ORDERS = ((1, 2), (2, 1), (1, 2))
GROUPS = {1: [2], 2: [2, 3]}
PROMPTS = [512, 2048, 4096, 8192]
OUTPUTS = [1, 64, 256]


def summarize_stage(runs):
    cells = defaultdict(dict)
    for run in runs:
        profile = run["profile"]
        if run["status"] != "passed" or profile["status"] != "passed":
            raise ValueError("failed workers cannot become service candidates")
        for cell in profile["cells"]:
            key = (cell["prompt_tokens"], cell["concurrency"], cell["max_new_tokens"])
            cells[key].setdefault(profile["tp"], []).append((run["block"], cell))
    rows = []
    for (prompt, concurrency, output), by_tp in sorted(cells.items()):
        result = {"prompt_tokens": prompt, "concurrency": concurrency,
                  "output_tokens": output, "profiles": {}}
        for tp in (1, 2):
            blocks = sorted(by_tp[tp])
            if len(blocks) != 3 or [b for b, _ in blocks] != [0, 1, 2]:
                raise ValueError("each cell needs three independent engine blocks per TP")
            block_values, ttfts, decodes, hashes, prompts = [], [], [], [], []
            for _, cell in blocks:
                if cell["status"] != "passed" or len(cell["samples"]) != 3:
                    raise ValueError("each block needs three complete warmed batches")
                block_values.append(statistics.mean(s["batch_wall_s"] for s in cell["samples"]))
                requests = [r for s in cell["samples"] for r in s["requests"]]
                ttfts.append(statistics.mean(r["ttft_s"] for r in requests))
                decodes.append(statistics.mean(r["latency_s"] - r["ttft_s"] for r in requests))
                hashes.append([r["output_sha256"] for r in requests])
                prompts.append([r["prompt_sha256"] for r in requests])
            result["profiles"][str(tp)] = {
                "independent_engine_blocks": 3, "warmed_batches_per_block": 3,
                "batch_wall_mean_s": statistics.mean(block_values),
                "batch_wall_sample_sd_s": statistics.stdev(block_values),
                "block_mean_samples_s": block_values,
                "ttft_mean_s": statistics.mean(ttfts),
                "decode_after_first_mean_s": statistics.mean(decodes),
                "per_remaining_token_mean_s": statistics.mean(decodes) / (output - 1) if output > 1 else None,
                "output_hashes": hashes, "prompt_hashes": prompts,
            }
        one, two = result["profiles"]["1"], result["profiles"]["2"]
        if one["prompt_hashes"] != two["prompt_hashes"]:
            raise ValueError("TP candidates did not receive identical inputs")
        deltas = [b - a for a, b in zip(one["block_mean_samples_s"], two["block_mean_samples_s"])]
        mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
        guard = max(0.001, 2 * sd)
        winner = ("tp2" if max(deltas) < 0 and mean < -guard else
                  "tp1" if min(deltas) > 0 and mean > guard else "unresolved")
        result.update({"paired_tp2_minus_tp1_s": deltas, "paired_mean_s": mean,
                       "paired_sample_sd_s": sd, "repeatability_guard_s": guard,
                       "repeatable_winner": winner,
                       "relative_tp2_minus_tp1_percent": 100 * mean / one["batch_wall_mean_s"],
                       "outputs_identical_across_tp": one["output_hashes"] == two["output_hashes"]})
        rows.append(result)
    return rows


def crossover_gate(rows):
    longer = [r for r in rows if r["output_tokens"] > 1]
    wins = [r for r in longer if r["repeatable_winner"] == "tp2"
            and r["outputs_identical_across_tp"]]
    losses = [r for r in longer if r["repeatable_winner"] == "tp1"
              and r["outputs_identical_across_tp"]]
    return {"multi_token_crossover_observed": bool(wins and losses),
            "tp2_multi_token_winning_cells": len(wins),
            "tp1_multi_token_winning_cells": len(losses),
            "single_token_only_tp2_advantage": not wins and any(
                r["output_tokens"] == 1 and r["repeatable_winner"] == "tp2" for r in rows),
            "gate_definition": "both signs across multi-token cells; all three paired block differences agree; mean exceeds max(1ms,2*sample_SD); matching outputs",
            "statistical_significance_claimed": False,
            "routing_benefit_established": False, "fleet_experiment_eligible": False}


def verify_profile(profile, group, expected_uuids, config_hash, concurrency):
    observed = profile["observed_runtime"]
    expected_runtime = {"tensor_parallel_size": len(group), "pipeline_parallel_size": 1,
                        "data_parallel_size": 1, "enable_expert_parallel": False,
                        "dtype": "torch.bfloat16", "routing_capture": True,
                        "max_model_len": 8704, "max_num_seqs": max(concurrency),
                        "max_num_batched_tokens": 2048, "enable_prefix_caching": False,
                        "cpu_offload_gb": 0, "enforce_eager": True, "async_scheduling": False}
    if observed != expected_runtime:
        raise ValueError("independent engine runtime observation does not match protocol")
    if (profile["status"] != "passed" or profile["config_sha256"] != config_hash
            or profile["tp"] != len(group) or not profile["runtime_moe_module_verified"]
            or not profile["routing_capture"] or profile["async_scheduling"] is not False
            or profile["cpu_offload_gb"] != 0 or profile["dtype"] != "bfloat16"
            or not profile["enforce_eager"] or profile["enable_prefix_caching"]
            or profile["max_num_seqs"] != max(concurrency)
            or profile["max_model_len"] != 8704
            or profile["execution"]["physical_gpu_indices"] != group
            or sorted(profile["execution"]["actual_visible_gpu_uuids"]) != sorted(expected_uuids)):
        raise ValueError("actual runtime/GPU/MoE differs from registered microbenchmark")
    expected = {(p, c, n) for p in PROMPTS for c in concurrency for n in OUTPUTS}
    actual = [(c["prompt_tokens"], c["concurrency"], c["max_new_tokens"]) for c in profile["cells"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("partial or duplicated sweep cannot pass")


def snapshot():
    observations = {}
    for name, query in {
            "gpus": "--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version",
            "processes": "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"}.items():
        row = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=15)
        observations[name] = row.stdout.splitlines()
    observations["observed_epoch_s"] = time.time()
    return observations


def run_worker(args, out, stage, concurrency, block, tp, gpu_uuids, config_hash):
    group = GROUPS[tp]
    before = snapshot()
    if any(canonical_gpu_uuid(line.split(",")[0].strip()) in {gpu_uuids[g] for g in group}
           for line in before["processes"] if line.strip()):
        raise RuntimeError("target GPU has another compute process; it will not be stopped")
    name = "spotserve-micro-" + uuid.uuid4().hex[:16]
    repo = Path(__file__).resolve().parents[1]
    vllm = repo.parent / "vllm"
    python = vllm / ".venv/bin/python"
    label = f"{stage}-block{block}-tp{tp}"
    command = ["podman", "run", "--rm", "--name", name, "--network", "none", "--shm-size", "1g"]
    for gpu in group:
        command += ["--device", f"nvidia.com/gpu={gpu}"]
    for source, target, mode in [(repo, repo, "ro"), (vllm, vllm, "ro"),
                                 (Path(args.model), Path(args.model), "ro"),
                                 (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"), "ro"),
                                 (Path(args.compiler_cache), Path("/taskcache"), "rw")]:
        command += ["--volume", f"{source}:{target}:{mode}"]
    env = {"PYTHONPATH": f"{vllm}/.venv/lib/python3.12/site-packages:{vllm}:{repo}",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "0", "NCCL_SOCKET_IFNAME": "lo",
           "TRITON_CACHE_DIR": "/taskcache/triton", "CUDA_CACHE_PATH": "/taskcache/cuda"}
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [args.image, str(python), "-u", "-m", "scripts.profile_granite_moe_tp",
                "--model", args.model, "--model-revision", args.model_revision,
                "--tp", str(tp), "--prompt-tokens", *map(str, PROMPTS),
                "--concurrency", *map(str, concurrency), "--output-tokens", *map(str, OUTPUTS),
                "--max-model-len", "8704", "--repeats", "3", "--capture-routes",
                "--synchronous-scheduling", "--retain-output-tokens", "--physical-gpu-indices",
                *map(str, group), "--physical-gpu-uuids", *(gpu_uuids[g] for g in group)]
    result = {"status": "running", "label": label, "stage": stage, "block": block,
              "command": command, "environment_before": before, "timed_jit_warnings": []}
    process, reader = None, None
    events = queue.Queue()
    measuring = None
    try:
        with (out / (label + ".log")).open("x") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            def read():
                for line in process.stdout:
                    events.put(line)
                events.put(None)
            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            deadline = time.monotonic() + args.worker_timeout_s
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("microbenchmark worker timed out")
                try:
                    line = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if line.startswith("MOE_TP_PROFILE_EVENT="):
                    row = json.loads(line.split("=", 1)[1])
                    if row["phase"] == "measurement_start":
                        measuring = row
                    elif row["phase"] == "measurement_end":
                        measuring = None
                    if row["phase"] in {"engine_start", "cell_complete", "shutdown_end"}:
                        print("MOE_MICRO_PROGRESS=" + json.dumps({"label": label, **row}), flush=True)
                if "Triton kernel JIT compilation during inference:" in line or "Autotuning process starts" in line:
                    if measuring is not None:
                        result["timed_jit_warnings"].append({"measurement": measuring, "message": line.strip()})
                if line.startswith("MOE_TP_PROFILE_JSON="):
                    result["profile"] = json.loads(line.split("=", 1)[1])
            result["exit_code"] = process.wait(timeout=15)
        if result["exit_code"] or "profile" not in result or result["timed_jit_warnings"]:
            raise ValueError("worker failed, has no complete artifact, or JIT occurred during timing")
        verify_profile(result["profile"], group, [gpu_uuids[g] for g in group], config_hash, concurrency)
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt):
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(["podman", "stop", "--time", "5", name], capture_output=True, timeout=30)
            process.wait(timeout=15)
        if reader is not None:
            reader.join(timeout=2)
        result["environment_after"] = snapshot()
        write_artifact(out / (label + ".json"), result)
    return result


def curves_svg(rows, concurrency):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="340" viewBox="0 0 1080 340">',
             '<rect width="1080" height="340" fill="white"/>',
             f'<text x="20" y="20" font-family="sans-serif">Warmed batch service, C={concurrency}; blue TP1, orange TP2; error bars: engine-block sample SD</text>']
    for panel, output in enumerate(OUTPUTS):
        cells = sorted((r for r in rows if r["concurrency"] == concurrency and r["output_tokens"] == output),
                       key=lambda r: r["prompt_tokens"])
        if not cells:
            continue
        left, top, width, height = 55 + panel * 355, 60, 280, 220
        maximum = max(p["batch_wall_mean_s"] + p["batch_wall_sample_sd_s"]
                      for r in cells for p in r["profiles"].values()) * 1.15
        parts += [f'<text x="{left}" y="42" font-family="sans-serif">Output={output} tokens</text>',
                  f'<path d="M {left} {top} V {top+height} H {left+width}" stroke="#555" fill="none"/>']
        for tick in range(5):
            y = top + height - height * tick / 4
            parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{maximum*tick/4:.3f}s</text>')
        for tp, color in (("1", "#1f77b4"), ("2", "#ff7f0e")):
            points = []
            for i, row in enumerate(cells):
                x = left + width * i / max(1, len(cells) - 1)
                p = row["profiles"][tp]
                y = top + height - p["batch_wall_mean_s"] / maximum * height
                dy = p["batch_wall_sample_sd_s"] / maximum * height
                points.append(f"{x},{y}")
                parts.append(f'<path d="M{x},{y-dy} V{y+dy}" stroke="{color}"/>')
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
                if tp == "1":
                    parts.append(f'<text x="{x}" y="{top+height+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{row["prompt_tokens"]}</text>')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left+width/2}" y="325" text-anchor="middle" font-family="sans-serif" font-size="12">Input tokens (discrete cells)</text>')
    return "\n".join(parts + ["</svg>", ""])


def report_markdown(result):
    lines = ["# Granite MoE TP=1／2 獨立 microbenchmark", "", f"狀態：{result['status']}。", "",
             "只量 warmed service；沒有中斷、KV／expert 搬移或 fleet 決策。兩種 TP 都使用相同 MoE checkpoint。",
             "TP1 固定 GPU2；TP2 固定 GPU2/3。每 TP 三個 fresh engine blocks，每 cell 每 block 三個 warmed batches。",
             "Synthetic repeated-token prompts；同 cell 的輸入 hashes 跨 TP 相同。沒有把 warmed batches 當正式 spot runs。",
             "服務時間排除 startup／暖機；TTFT 是 frontend 收到首 token，不是純 GPU prefill。",
             "同步標準 scheduler、routing capture=true、BF16、V1/eager/TRITON、prefix cache off。不是 NativeFreezeScheduler，不能直接沿用成旧 formal profile。",
             "repeatability gate 為描述性放行，不宣稱統計顯著；三對 engine blocks 不能證明一般化效益。", ""]
    for stage in result["stages"]:
        lines += [f"## {stage['name']}", "", f"狀態：{stage['status']}。", ""]
        for c in stage["concurrency"]:
            lines += [f"![C={c} service curves]({stage['name']}-c{c}-curves.svg)", ""]
        if "rows" not in stage:
            lines += ["未完成，不補造平均值。", ""]
            continue
        lines += ["| Input | C | Output | TP1 秒：mean ± SD | TP2 秒：mean ± SD | TP2−TP1 % | Repeatable winner | Output 相同 |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
        for row in stage["rows"]:
            a, b = row["profiles"]["1"], row["profiles"]["2"]
            lines.append(f"| {row['prompt_tokens']} | {row['concurrency']} | {row['output_tokens']} | "
                         f"{a['batch_wall_mean_s']:.4f} ± {a['batch_wall_sample_sd_s']:.4f} | "
                         f"{b['batch_wall_mean_s']:.4f} ± {b['batch_wall_sample_sd_s']:.4f} | "
                         f"{row['relative_tp2_minus_tp1_percent']:+.2f} | {row['repeatable_winner']} | "
                         f"{row['outputs_identical_across_tp']} |")
        lines += ["", "Gate：`" + json.dumps(stage["gate"], ensure_ascii=False) + "`", ""]
    lines += ["## 下一步", "", "找到交叉也不等於 MoE 收益：下一步須獨立驗證 generic／routing／shuffled 的排序與選擇。",
              "若測試範圍內無多 token 交叉，停止 fleet 擴充；不强迫 TP2，不推論所有模型／硬體都不適合 TP2。",
              "C>1 service 通過亦不代表既有 C=1 native freeze/recovery 已支援併發。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compiler-cache", required=True)
    parser.add_argument("--image", default="localhost/spotserve-python312-nixl:latest")
    parser.add_argument("--worker-timeout-s", type=float, default=1800)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    Path(args.compiler_cache).mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.model, "config.json").read_text())
    if config.get("architectures") != ["GraniteMoeForCausalLM"] or not (
            0 < config.get("num_experts_per_tok", 0) < config.get("num_local_experts", 0)):
        raise ValueError("non-sparse/dense checkpoint is not eligible")
    environment = snapshot()
    gpu_uuids = {int(r.split(",")[0]): canonical_gpu_uuid(r.split(",")[1].strip())
                 for r in environment["gpus"]}
    if set(gpu_uuids) != {0, 1, 2, 3}:
        raise ValueError("four visible GPU environment required")
    protocol = {"model": args.model, "revision": args.model_revision, "config_sha256": digest(config),
                "prompt_tokens": PROMPTS, "output_tokens": OUTPUTS, "orders": ORDERS,
                "physical_groups": GROUPS, "fresh_engine_blocks_per_tp": 3,
                "warmed_batches_per_cell_per_block": 3, "warm_status_controlled": True,
                "conditional_expand_to_C2_C4_if_no_multi_token_crossover": True,
                "repeatability_guard": "all 3 paired deltas same sign; mean exceeds max(1ms,2*SD)",
                "formal_spot_experiment": False, "network_mutation": False}
    write_artifact(out / "protocol.json", protocol)
    write_artifact(out / "environment_initial.json", environment)
    result = {"status": "running", "protocol": protocol, "stages": []}
    try:
        for name, concurrency in (("c1", [1]), ("c2-c4", [2, 4])):
            stage = {"name": name, "concurrency": concurrency, "status": "running", "runs": []}
            result["stages"].append(stage)
            directory = out / name
            directory.mkdir()
            for block, order in enumerate(ORDERS):
                for tp in order:
                    run = run_worker(args, directory, name, concurrency, block, tp, gpu_uuids, digest(config))
                    stage["runs"].append(run)
                    if run["status"] != "passed":
                        raise RuntimeError("failed microbenchmark worker; stop before later stages")
            stage["rows"] = summarize_stage(stage["runs"])
            stage["gate"] = crossover_gate(stage["rows"])
            stage["status"] = "passed"
            write_artifact(out / (name + "-summary.json"), stage)
            for c in concurrency:
                with (out / f"{name}-c{c}-curves.svg").open("x") as plot:
                    plot.write(curves_svg(stage["rows"], c))
            print("MOE_MICRO_STAGE_COMPLETE=" + json.dumps({"stage": name, "gate": stage["gate"]}), flush=True)
            if name == "c1" and stage["gate"]["multi_token_crossover_observed"]:
                break
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt):
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        if result["stages"] and result["stages"][-1]["status"] == "running":
            result["stages"][-1]["status"] = "failed"
    finally:
        result["environment_final"] = snapshot()
        write_artifact(out / "microbenchmark.json", result)
        with (out / "report.md").open("x") as report:
            report.write(report_markdown(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
