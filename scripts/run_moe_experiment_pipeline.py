"""Sequential GPU gates and three-repeat experiments; stop on any failed gate."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.moe_gpu_runtime import write_artifact
from scripts.run_granite_moe_replay_pilot import digest


def summarize_runs(rows, metric):
    result = {}
    for version in ("original", "moe_aware"):
        samples = [r for r in rows if r.get("version") == version]
        if (len(samples) != 3 or {r.get("repeat") for r in samples} != {0, 1, 2}
                or not all(r.get("status") == "passed" and r.get("formal_run") for r in samples)):
            raise ValueError("averages require three independent complete formal runs per version")
        values = [r[metric] for r in samples]
        result[version] = {"n": 3, "mean": statistics.mean(values),
                           "sample_sd": statistics.stdev(values), "samples": values}
    result["paired_moe_minus_original"] = [
        next(r[metric] for r in rows if r["version"] == "moe_aware" and r["repeat"] == i)
        - next(r[metric] for r in rows if r["version"] == "original" and r["repeat"] == i)
        for i in range(3)]
    return result


def formal_statistics(out):
    ablation = json.loads((out / "formal-ablation/formal_ablation.json").read_text())
    overall = json.loads((out / "formal-overall/formal_overall.json").read_text())
    if ablation["status"] != "passed" or overall["status"] != "passed":
        raise ValueError("both formal matrices must pass")
    result = {"formal_complete_runs": len(ablation["runs"]) + len(overall["runs"]), "groups": {}}
    for kind in ("migration", "reparallelization"):
        rows = [r for r in ablation["runs"] if r["kind"] == kind]
        result["groups"][kind] = {metric: summarize_runs(rows, metric) for metric in (
            "notice_to_first_token_s", "notice_to_complete_s", "maximum_delivery_gap_including_handoff_s")}
        result["groups"][kind]["selected_candidates"] = [
            {"repeat": r["repeat"], "version": r["version"],
             "candidate": r["decision"]["selected_candidate"],
             "actual_TP": r["runtime_evidence"]["tensor_parallel_size"],
             "matches_reference": r["matches_uninterrupted_reference"]} for r in rows]
    result["groups"]["overall"] = {metric: summarize_runs(overall["runs"], metric) for metric in (
        "stream_complete_s", "mean_request_latency_s", "output_tokens_per_s", "max_delivery_gap_s")}
    result["groups"]["overall"]["preemptions_per_run"] = [len(r["preemptions"]) for r in overall["runs"]]
    return result


def report_markdown(result):
    lines = ["# MoE × SpotServe v2 GPU 執行報告", "", f"狀態：{result['status']}。", "",
             "兩版都使用相同 Granite MoE checkpoint；generic 與 route-conditioned cost 的共同目標不變。",
             "恢復方式為 host token handoff + GPU replay，不是 direct KV restore；沒有實體 expert placement。",
             "四卡既有 store 未停止，非排他環境。Synthetic 4K+512、C=1，不推論一般線上流量容量。", "",
             "## 階段", "", "| 階段 | 狀態 | 結果 |", "| --- | --- | --- |"]
    for row in result["stages"]:
        lines.append(f"| {row['stage']} | {row['status']} | [{row['artifact']}]({row['artifact']}) |")
    if "statistics" not in result:
        lines += ["", "尚未全部通過，不補造正式三次平均值。", ""]
        return "\n".join(lines)
    stats = result["statistics"]
    lines += ["", f"正式完整 runs：{stats['formal_complete_runs']}；每組每版 n=3。以下為 mean ± sample SD。", "",
              "| 組別 | 指標 | Original | MoE-aware |", "| --- | --- | ---: | ---: |"]
    for kind, group in stats["groups"].items():
        for metric, value in group.items():
            if not isinstance(value, dict) or "original" not in value:
                continue
            a, b = value["original"], value["moe_aware"]
            lines.append(f"| {kind} | {metric} | {a['mean']:.4f} ± {a['sample_sd']:.4f} | "
                         f"{b['mean']:.4f} ± {b['sample_sd']:.4f} |")
    lines += ["", "僅三次 repeats、單一 formal synthetic workload，不能宣稱統計顯著或普遍 MoE 收益。",
              "若兩版同選 TP/target，保留原始選擇，不強迫拉開差距；校準與保留 workload 診斷另見各階段 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compiler-cache", required=True)
    parser.add_argument("--wait-for-calibration-s", type=float, default=0)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_for_calibration_s
    calibration_path = Path(args.calibration)
    while not calibration_path.exists() and time.monotonic() < deadline:
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "passed":
        raise ValueError("completed independent GPU calibration must pass before pipeline")
    if calibration["protocol"].get("deployment_ready_regime") != (
            "all_candidate_shapes_and_routing_capture_warmed_before_every_ready_trial"):
        raise ValueError("all candidates require matched prepopulated compiler-cache ready trials")
    result = {"status": "running", "calibration_sha256": digest(calibration), "stages": []}
    artifacts = {"validate": "cost_validation.json", "validate-shape": "shape_validation.json",
                 "pilot": "pilot.json", "overall-pilot": "overall_pilot.json",
                 "formal-ablation": "formal_ablation.json", "formal-overall": "formal_overall.json"}
    try:
        for stage in artifacts:
            command = [sys.executable, "-u", "-m", "scripts.run_moe_calibrated_experiments",
                       "--model", args.model, "--model-revision", args.model_revision,
                       "--calibration", args.calibration, "--compiler-cache", args.compiler_cache,
                       "--stage", stage, "--output-dir", str(out / stage)]
            if stage in {"overall-pilot", "formal-ablation", "formal-overall"}:
                command += ["--pilot-report", str(out / "pilot/pilot.json")]
            if stage in {"overall-pilot", "formal-overall"}:
                command += ["--shape-report", str(out / "validate-shape/shape_validation.json")]
            if stage == "formal-overall":
                command += ["--overall-pilot-report", str(out / "overall-pilot/overall_pilot.json")]
            row = {"stage": stage, "command": command, "status": "running",
                   "artifact": str(Path(stage) / artifacts[stage]), "started_epoch_s": time.time()}
            result["stages"].append(row)
            print("MOE_PIPELINE_STAGE_STARTED=" + json.dumps(row), flush=True)
            with (out / (stage + ".log")).open("x") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                row["exit_code"] = process.wait()
            artifact = out / row["artifact"]
            row["status"] = json.loads(artifact.read_text())["status"] if artifact.exists() else "failed"
            if row["exit_code"]:
                row["returned_artifact_status"] = row["status"]
                row["status"] = "failed"
            row["completed_epoch_s"] = time.time()
            print("MOE_PIPELINE_STAGE_FINISHED=" + json.dumps(row), flush=True)
            if row["exit_code"] or row["status"] != "passed":
                result["status"] = "failed"
                break
        else:
            result["statistics"] = formal_statistics(out)
            result["status"] = "passed"
    finally:
        if result["status"] == "running":
            result["status"] = "failed"
        write_artifact(out / "pipeline.json", result)
        with (out / "report.md").open("x") as report:
            report.write(report_markdown(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
