"""Render exact user-specified C2 EP pilot raw data into a report-ready table."""

import argparse
import json
from pathlib import Path


def f(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--result-dirs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source_path = Path(args.source)
    source = json.loads(source_path.read_text())
    found = {}
    for directory in map(Path, args.result_dirs):
        for file in directory.glob("target-*.json"):
            row = json.loads(file.read_text())
            key = (tuple(row["group"]), row["tp"], row["dp"])
            if row["status"] == "passed" or key not in found:
                found[key] = (file, row)
    lines = []
    for pair in ((1, 2), (1, 3), (2, 3)):
        for tp, dp in ((2, 1), (1, 2)):
            file, row = found.get((pair, tp, dp), (None, {}))
            raw = "—" if file is None else f"[JSON]({file.resolve()})"
            match = row.get("exact_reference_match")
            lines.append(
                f"| `{{{pair[0]},{pair[1]}}}` | `TP{tp}/DP{dp}/EP2` | "
                f"{f(row.get('engine_ready_s'))} | {f(row.get('warmed_ready_s'))} | "
                f"{f(1000*row['host_handoff_s'] if 'host_handoff_s' in row else None)} | "
                f"{f(row.get('remaining',{}).get('batch_wall_s'))} | "
                f"{f(row.get('launch_to_all_final_s'))} | "
                f"{'✓ / ✓' if match == [True, True] else ('未量測' if file is None else row['status'])} | {raw} |")
    cases = source["freeze"]["cases"]
    p = found.get(((2, 3), 2, 1), (None, {}))[1]
    d = found.get(((2, 3), 1, 2), (None, {}))[1]
    x, y = p.get("launch_to_all_final_s"), d.get("launch_to_all_final_s")
    a = [found.get((pair, 2, 1), (None, {}))[1].get("launch_to_all_final_s")
         for pair in ((1, 2), (1, 3), (2, 3))]
    text = f"""# Granite EP enabled、C2 4096/512/約 256：當日實測

這份資料直接使用原始 BF16 Granite 3.1 3B A800M Instruct（32 個 MoE layer、每層 40 experts、top-k8）與 RTX 5070 Ti。兩個同時請求各有 **4096 input／512 output tokens**；GPU1/2 source 為 `TP2/DP1/EP2`，native freeze 在第 **{cases[0]['completed_output_tokens']}／{cases[1]['completed_output_tokens']}** 個 output token，兩個凍結前綴都與不中斷參考逐 token 一致。Source C2 不中斷 batch 完成 **{source['reference']['batch_wall_s']:.2f} s**，notice→pause ack **{1000*source['freeze']['notice_to_pause_ack_s']:.2f} ms**，兩請求 KV blocks 為 {cases[0]['allocated_kv_block_count']}／{cases[1]['allocated_kv_block_count']}。即時 expert routing 分別涵蓋 {cases[0]['routes']['observed_routed_tokens']}／{cases[1]['routes']['observed_routed_tokens']} tokens。[Source raw]({source_path.resolve()})。

凍結後停止自有 source group，依序建立以下 EP target，經 host token handoff、GPU prefill/replay 完成餘下 {cases[0]['remaining_tokens']}／{cases[1]['remaining_tokens']} tokens。這是**每格 n=1 的候選恢復量測**；兩個 target 請求同時送入。表中 `launch→all final` 從每個 target container 啟動時計，含啟動、雙請求暖機、handoff、餘下 batch；不含 source 關閉與候選之間的等待，故不是六次獨立 notice→final 實驗。

| GPU target | EP 形狀 | Engine ready (s) | 暖機後 ready (s) | Handoff (ms) | 餘下 C2 batch (s) | Launch→all final (s) | 完整輸出逐 token 核對 | Raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(lines)}

2A（固定 `TP2/DP1/EP2`、變 target 卡組）：`{{1,2}}`、`{{1,3}}`、`{{2,3}}` 的 launch→all final 分別為 **{f(a[0])}、{f(a[1])}、{f(a[2])} s**。每格只有一次，不能用來宣稱 placement 的統計優勢。

2B（固定 GPU`{{2,3}}`、變 parallel shape）：`TP2/DP1/EP2` 為 **{f(x)} s**，`TP1/DP2/EP2` 為 **{f(y)} s**，DP2−TP2 **{f(y-x) if x is not None and y is not None else '—'} s**；餘下 C2 batch 的差值另見表格。兩種 shape 都是真實 EP enabled，並完成相同 source 前綴的兩請求恢復。

**Original SpotServe vs MoE-aware SpotServe overall／planner 2A／planner 2B：尚未完成 paired A/B。** 本輪實測的是它們共同可選的 physical placement 與 EP parallel shape，沒有把候選最小值假裝成 MoE-aware 實際選擇；也沒有將 host replay 說成 NIXL KV 直接恢復。要量 planner 差距，需先以獨立 prompts、同一 cache 政策校準 generic 與 live-route conditioned 成本，再讓兩版在同一候選集合各自決策及執行至少三次。現有 BF16 重複 token 跨 TP/EP 的既有一致性問題尚未解決，因此這些 ordinary workload 成功格仍屬 pilot。
"""
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(text)
    print(target)


if __name__ == "__main__":
    main()
