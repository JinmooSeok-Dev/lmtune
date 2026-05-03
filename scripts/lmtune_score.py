#!/usr/bin/env python3
"""Repeat a bench run N times, compute composite score, emit JSON.

Runs `bench run --json-summary` N times against a (profile, endpoint) pair,
measures CV(throughput_tok.avg), auto-extends to count=5 if CV >= threshold,
and prints a single-line JSON for autoresearch's objective reader.

Score:
    penalty = max(0, 1 - ttft_p99 / (2 * ttft_slo_ms))
    score   = throughput_tok.avg * penalty
    (SLO failed in any individual run -> score = 0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", "--profile", type=Path, required=True)
    ap.add_argument("-e", "--endpoint", type=Path, required=True)
    ap.add_argument("-n", "--count", type=int, default=3)
    ap.add_argument("--cv-threshold", type=float, default=0.10)
    ap.add_argument("--ttft-slo-ms", type=float, default=500.0,
                    help="penalty denominator uses 2 * this value")
    ap.add_argument(
        "--bench-bin",
        default=os.environ.get("LMTUNE_BIN") or os.environ.get("BENCH_BIN") or "lmtune",
    )
    ap.add_argument("--db", type=Path, default=None)
    return ap.parse_args()


def config_hash(profile: Path, endpoint: Path) -> str:
    h = hashlib.sha1()
    h.update(profile.read_bytes())
    h.update(b"\n---\n")
    h.update(endpoint.read_bytes())
    return h.hexdigest()[:8]


def run_once(bench_bin: str, profile: Path, endpoint: Path, db: Path | None) -> dict:
    """Run bench once; parse the last JSON line from stdout."""
    cmd = [bench_bin, "run", "-p", str(profile), "-e", str(endpoint), "--json-summary"]
    if db is not None:
        cmd += ["--db", str(db)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"bench run failed: {proc.stderr.strip()[-400:] or proc.returncode}")
    # Last non-empty line that parses as JSON.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    # Diagnostic: show the tail of stdout + stderr so the caller can see the real cause.
    tail_out = "\n".join(proc.stdout.splitlines()[-5:])
    tail_err = "\n".join(proc.stderr.splitlines()[-5:])
    raise RuntimeError(
        f"no JSON summary line (rc={proc.returncode}); stdout_tail=<{tail_out!r}>; "
        f"stderr_tail=<{tail_err!r}>"
    )


def collect(bench_bin: str, profile: Path, endpoint: Path, n: int, db: Path | None) -> list[dict]:
    out = []
    for i in range(n):
        summary = run_once(bench_bin, profile, endpoint, db)
        summary["_idx"] = i
        out.append(summary)
    return out


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return math.inf
    return statistics.stdev(values) / mean


def aggregate(summaries: list[dict], ttft_slo_ms: float) -> dict:
    """Compute mean metrics + composite score."""
    n = len(summaries)
    all_pass = all(s.get("slo_pass") for s in summaries)

    def avg(key: str) -> float | None:
        xs = [s.get("metrics", {}).get(key) for s in summaries]
        xs = [float(x) for x in xs if x is not None]
        return statistics.fmean(xs) if xs else None

    throughput_tok_avg = avg("throughput_tok.avg")
    ttft_p99 = avg("ttft.p99")
    e2e_p99 = avg("e2e.p99")

    throughputs = [s.get("metrics", {}).get("throughput_tok.avg") for s in summaries]
    throughputs = [float(x) for x in throughputs if x is not None]
    cv_throughput = cv(throughputs)

    if all_pass and throughput_tok_avg is not None and ttft_p99 is not None:
        penalty = max(0.0, 1.0 - ttft_p99 / (2 * ttft_slo_ms))
        score = throughput_tok_avg * penalty
    else:
        score = 0.0

    return {
        "n_runs": n,
        "slo_pass": all_pass,
        "cv_throughput": cv_throughput,
        "throughput_tok_avg": throughput_tok_avg,
        "ttft_p99": ttft_p99,
        "e2e_p99": e2e_p99,
        "score": score,
        "run_ids": [s.get("run_id") for s in summaries],
    }


def main() -> int:
    args = parse_args()
    cfg_hash = config_hash(args.profile, args.endpoint)

    summaries = collect(args.bench_bin, args.profile, args.endpoint, args.count, args.db)
    agg = aggregate(summaries, args.ttft_slo_ms)

    # CV gate: if throughput CV too high, extend to 5 once.
    extended = False
    if agg["cv_throughput"] >= args.cv_threshold and args.count < 5:
        extra = collect(args.bench_bin, args.profile, args.endpoint, 5 - args.count, args.db)
        summaries.extend(extra)
        agg = aggregate(summaries, args.ttft_slo_ms)
        extended = True

    accepted = bool(agg["slo_pass"]) and agg["cv_throughput"] < args.cv_threshold

    out = {
        "config_hash": cfg_hash,
        "profile": args.profile.name,
        "endpoint": args.endpoint.name,
        "ttft_slo_ms": args.ttft_slo_ms,
        "cv_threshold": args.cv_threshold,
        "extended_to_5": extended,
        "accepted": accepted,
        **agg,
    }
    print(json.dumps(out, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
