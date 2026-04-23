#!/usr/bin/env python3
"""Per-workload goodput ranking across all experiments (Round 1+2).

Loads experiments_*.jsonl and per (workload, config) pair ranks by score.
Distinguishes per-workload SLO pass (not aggregate).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def rows_from(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", type=Path, required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    all_rows = []
    for p in args.logs:
        for r in rows_from(p):
            r["_src"] = p.name
            all_rows.append(r)

    # Per-workload view
    per_wl: dict[str, list[tuple]] = {"short": [], "medium": [], "long": []}
    for r in all_rows:
        if not r.get("vllm_started"):
            continue
        model = (r.get("model") or "Qwen/Qwen2.5-1.5B-Instruct").split("/")[-1]
        label = r.get("label") or r.get("seq")
        eng = r.get("engine_args") or {}
        for wname, wr in (r.get("workloads") or {}).items():
            if wname not in per_wl:
                continue
            score = wr.get("score") or 0
            slo = wr.get("slo_pass")
            ttft = wr.get("ttft_p99")
            e2e = wr.get("e2e_p99")
            thr = wr.get("throughput_tok_avg")
            per_wl[wname].append((score, slo, thr, ttft, e2e, label, model, eng, r["_src"]))

    print(f"# Per-Workload Goodput Ranking (Round 1 + Round 2)\n")
    print(f"- SLO: ttft_p99 ≤ 500ms, e2e_p99 ≤ 30000ms")
    print(f"- score = throughput_tok × max(0, 1 − ttft_p99/(2·500ms))")
    print(f"- per-workload `slo` field is the workload-level pass (NOT experiment aggregate)\n")

    for wname in ("short", "medium", "long"):
        rows = sorted(per_wl[wname], key=lambda x: x[0], reverse=True)
        print(f"## Workload: `{wname}`\n")
        print("| rank | label | model | score | throughput | ttft_p99 | e2e_p99 | slo | src |")
        print("|-----:|:------|:------|-----:|-----------:|---------:|--------:|:---:|:----|")
        for i, (sc, slo, thr, ttft, e2e, label, model, eng, src) in enumerate(rows[: args.top], 1):
            print(f"| {i} | `{label}` | `{model}` | {sc:.1f} | "
                  f"{thr:.1f} | {ttft:.0f}ms | {e2e*1000:.0f}ms | "
                  f"{'✅' if slo else '❌'} | {src} |")
        slo_pass = [r for r in rows if r[1]]
        if slo_pass:
            w = slo_pass[0]
            print(f"\n**🏆 SLO-passing winner**: `{w[5]}` on `{w[6]}` — score={w[0]:.1f}, "
                  f"throughput={w[2]:.1f}tok/s, TTFT p99={w[3]:.0f}ms\n")
        else:
            print(f"\n(no SLO-passing configuration for this workload)\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
