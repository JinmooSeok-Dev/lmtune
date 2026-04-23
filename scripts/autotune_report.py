#!/usr/bin/env python3
"""Summarize data/autotune/experiments.jsonl → top-K markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=ROOT / "data/autotune/experiments.jsonl")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}"); return 1

    rows = []
    for line in args.log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        print("no experiments logged"); return 1

    ok_rows = [r for r in rows if r.get("vllm_started") and r.get("slo_pass")]
    failed_rows = [r for r in rows if not r.get("vllm_started")]
    slo_fail_rows = [r for r in rows if r.get("vllm_started") and not r.get("slo_pass")]

    ok_rows.sort(key=lambda r: r.get("total_score", 0), reverse=True)

    print(f"# Autotune Summary — {args.log.name}\n")
    print(f"- Total experiments: {len(rows)}")
    print(f"- SLO-passing: {len(ok_rows)}")
    print(f"- SLO-failing: {len(slo_fail_rows)}")
    print(f"- vLLM start failures: {len(failed_rows)}\n")

    print("## Top results (by total_score across 3 workloads)\n")
    print("| rank | label | model | total | short | medium | long | mnseq | pfx | chunk | gpumem |")
    print("|-----:|:------|:------|-----:|------:|-------:|-----:|:------|:----|:------|:-------|")
    for i, r in enumerate(ok_rows[: args.top], 1):
        eng = r.get("engine_args") or {}
        w = r.get("workloads") or {}
        sh = (w.get("short") or {}).get("score") or 0
        md = (w.get("medium") or {}).get("score") or 0
        lg = (w.get("long") or {}).get("score") or 0
        model = (r.get("model") or "").split("/")[-1]
        print(f"| {i} | `{r['label']}` | `{model}` | {r['total_score']:.1f} | "
              f"{sh:.1f} | {md:.1f} | {lg:.1f} | "
              f"{eng.get('max_num_seqs')} | {eng.get('enable_prefix_caching')} | "
              f"{eng.get('enable_chunked_prefill')} | {eng.get('gpu_memory_utilization')} |")

    print("\n## Per-workload winners\n")
    for wname in ("short", "medium", "long"):
        if not ok_rows:
            continue
        sorted_w = sorted(
            [r for r in ok_rows if (r.get("workloads") or {}).get(wname, {}).get("score")],
            key=lambda r: (r["workloads"][wname].get("score") or 0),
            reverse=True,
        )
        if sorted_w:
            w_top = sorted_w[0]
            wmetric = w_top["workloads"][wname]
            model = (w_top.get("model") or "").split("/")[-1]
            print(f"- **{wname}**: `{w_top['label']}` ({model}) "
                  f"score={wmetric.get('score'):.1f} "
                  f"throughput={wmetric.get('throughput_tok_avg'):.1f}tok/s "
                  f"ttft_p99={wmetric.get('ttft_p99'):.0f}ms")

    if slo_fail_rows:
        print("\n## SLO-failing experiments\n")
        print("| label | total_score | TTFT/E2E fail detail |")
        print("|:------|-----------:|:---------------------|")
        for r in slo_fail_rows:
            per = []
            for wname, wr in (r.get("workloads") or {}).items():
                if wr.get("slo_pass") is False:
                    per.append(f"{wname}: ttft_p99={wr.get('ttft_p99'):.0f}ms" if wr.get("ttft_p99") else f"{wname}: ?")
            print(f"| `{r['label']}` | {r.get('total_score', 0):.1f} | {'; '.join(per)} |")

    if failed_rows:
        print("\n## vLLM start failures\n")
        for r in failed_rows:
            print(f"- `{r['label']}` — duration={r.get('duration_sec')}s")

    if ok_rows:
        winner = ok_rows[0]
        print(f"\n## Winner\n")
        print(f"`{winner['label']}`  score={winner['total_score']:.1f}")
        print(f"engine_args: `{json.dumps(winner['engine_args'], sort_keys=True)}`")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
