"""trial_runner — worker entrypoint. One-shot evaluation of a TrialPayload.

Two invocation modes:
1. In-process: `run_trial_locally(payload)` called from ProcessPoolBackend's
   child process. Returns a TrialResult directly (pickled back to parent).
2. K8s Job: `python -m bench.orchestrate.trial_runner` reads PARAMS_JSON,
   ENDPOINT_PATH, PROFILE_PATHS, REPEATS from env and prints the resulting
   TrialResult as JSON on stdout for the backend to harvest. (Phase S4)

Both paths build a `BenchScoreObjective` and call it once. If the trial
includes parallelism axes (tp/pp/dp/ep), the caller must have materialized
an endpoint YAML before submission — this runner does not mutate YAMLs.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from bench.orchestrate.backend import TrialPayload, TrialResult
from bench.search.objective import BenchScoreObjective


def run_trial_locally(payload: TrialPayload) -> TrialResult:
    """ProcessPool entrypoint. Pure function: no DuckDB writes, no side effects
    besides running bench runs via BenchScoreObjective (which spawns subprocesses).
    """
    try:
        obj = BenchScoreObjective(
            endpoint_path=payload.endpoint_path,
            profile_paths=payload.profile_paths,
            repeats=payload.repeats,
            ttft_slo_ms=payload.ttft_slo_ms,
        )
        result = obj(payload.params)
        status = (
            "completed" if (result.accepted and not result.error)
            else ("pruned" if result.error and "slo" in (result.error or "").lower() else "crash")
        )
        return TrialResult(
            trial_id=payload.trial_id,
            status=status,
            score=float(result.score) if result.score is not None else None,
            metrics=dict(result.metrics),
            error=result.error,
            backend="process-pool",
            worker_id=str(os.getpid()),
        )
    except Exception as e:  # noqa: BLE001
        return TrialResult(
            trial_id=payload.trial_id,
            status="crash",
            score=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            backend="process-pool",
            worker_id=str(os.getpid()),
        )


def _main_k8s() -> int:
    """K8s Job entrypoint (Phase S4 hookup). Reads env → runs one trial → prints JSON."""
    params = json.loads(os.environ["PARAMS_JSON"])
    payload = TrialPayload(
        trial_id=os.environ["TRIAL_ID"],
        study_id=os.environ["STUDY_ID"],
        seq=int(os.environ.get("TRIAL_SEQ", "0")),
        params=params,
        endpoint_path=os.environ["ENDPOINT_PATH"],
        profile_paths=os.environ["PROFILE_PATHS"].split(":"),
        repeats=int(os.environ.get("REPEATS", "3")),
        ttft_slo_ms=float(os.environ.get("TTFT_SLO_MS", "500.0")),
    )
    result = run_trial_locally(payload)
    # Marshal as JSON; tuple keys in metrics → "metric|workload" strings.
    print(json.dumps({
        "trial_id": result.trial_id,
        "status": result.status,
        "score": result.score,
        "error": result.error,
        "backend": "k8s-job",
        "worker_id": result.worker_id,
        "metrics": {f"{m}|{w or ''}": v for (m, w), v in result.metrics.items()},
    }))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(_main_k8s())
