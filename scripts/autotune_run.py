#!/usr/bin/env python3
"""Autotune orchestrator — apply engine_args, restart vllm, score 3 workloads, log.

Minimal replacement for autoresearch-claude-code, hand-rolled so it can run
without the Claude Code plugin. Reads a list of hypotheses (each a dict of
engine_args) from stdin-json or from --hypotheses, iterates them, and writes
one JSONL line per experiment to --log.

Each experiment:
  1) Write engine_args to `<endpoint>.deployment.engine_args`
  2) `vllm_restart.sh <endpoint>`
  3) For each workload in (short, medium, long):
        bench_score.py -p <profile> -e <endpoint> -n <repeats>
  4) Log: label, engine_args, per-workload metrics, aggregate score
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESTART = ROOT / "scripts" / "vllm_restart.sh"
SCORE = ROOT / "scripts" / "bench_score.py"
PY = ROOT / ".venv" / "bin" / "python"


def load_endpoint(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_endpoint(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def set_endpoint(endpoint_path: Path, engine_args: dict, model: str | None = None) -> None:
    cfg = load_endpoint(endpoint_path)
    cfg.setdefault("deployment", {}).setdefault("engine_args", {})
    cfg["deployment"]["engine_args"] = dict(engine_args)
    if model:
        cfg["model"] = model
        cfg["tokenizer"] = model
    write_endpoint(endpoint_path, cfg)


def restart_vllm(endpoint_path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        ["bash", str(RESTART), str(endpoint_path)],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    return ok, (proc.stdout + proc.stderr).strip().splitlines()[-20:] if not ok else ""


def score_one(profile_path: Path, endpoint_path: Path, repeats: int) -> dict:
    cmd = [str(PY), str(SCORE),
           "-p", str(profile_path),
           "-e", str(endpoint_path),
           "-n", str(repeats)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"error": proc.stderr.strip()[-300:] or "no output"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"error": "non-json output: " + lines[-1][-200:]}


def run_experiment(
    label: str,
    engine_args: dict,
    endpoint_path: Path,
    workload_profiles: list[Path],
    repeats: int,
    model: str | None = None,
) -> dict:
    t0 = time.time()
    set_endpoint(endpoint_path, engine_args, model=model)
    ok, err = restart_vllm(endpoint_path)
    if not ok:
        return {
            "label": label, "engine_args": engine_args,
            "vllm_started": False, "error": err,
            "duration_sec": round(time.time() - t0, 1),
        }

    workloads: dict[str, dict] = {}
    total_score = 0.0
    any_slo_fail = False
    for wp in workload_profiles:
        res = score_one(wp, endpoint_path, repeats)
        workloads[wp.stem] = res
        total_score += float(res.get("score") or 0.0)
        if res.get("slo_pass") is False:
            any_slo_fail = True

    return {
        "label": label,
        "model": model,
        "engine_args": engine_args,
        "vllm_started": True,
        "total_score": total_score,
        "slo_pass": (not any_slo_fail),
        "workloads": workloads,
        "duration_sec": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", type=Path, required=True,
                    help="endpoint YAML; will be rewritten in place")
    ap.add_argument("--profiles-dir", type=Path,
                    default=ROOT / "configs/profiles/autotune")
    ap.add_argument("--workloads", nargs="+", default=["short", "medium", "long"])
    ap.add_argument("--log", type=Path,
                    default=ROOT / "data/autotune/experiments.jsonl")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--hypotheses", type=Path, required=True,
                    help="JSON file; list of {label, engine_args} dicts")
    args = ap.parse_args()

    profiles = [args.profiles_dir / f"{w}.yaml" for w in args.workloads]
    for p in profiles:
        if not p.exists():
            print(f"profile not found: {p}", file=sys.stderr); return 2

    hyps = json.loads(args.hypotheses.read_text(encoding="utf-8"))
    args.log.parent.mkdir(parents=True, exist_ok=True)

    for i, h in enumerate(hyps, 1):
        label = h.get("label") or f"exp-{i}"
        eng = h.get("engine_args") or {}
        model = h.get("model")
        print(f"\n=== [{i}/{len(hyps)}] {label} ===")
        if model:
            print(f"model: {model}")
        print(f"engine_args: {json.dumps(eng, sort_keys=True)}")
        result = run_experiment(label, eng, args.endpoint, profiles, args.repeats, model=model)
        result["seq"] = i
        result["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with args.log.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(result, separators=(",", ":")) + "\n")
        if result.get("vllm_started"):
            print(f"  → total_score={result['total_score']:.1f}  "
                  f"slo_pass={result['slo_pass']}  "
                  f"duration={result['duration_sec']}s")
        else:
            print(f"  → VLLM START FAILED  duration={result['duration_sec']}s")

    print(f"\nall experiments logged to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
