"""Driver — distributed ask/dispatch/poll/tell loop.

Keeps `workers` trials in-flight by pre-fetching from Study.ask() whenever a
slot is free. When a backend handle resolves, the Driver updates the Study
via tell() (Optuna bookkeeping) and persists via the DuckDBWriterQueue.

Unlike Study.run() (S1 inline), this path never holds the DuckDB connection
during objective evaluation — writes funnel through the writer queue so the
Driver process is the sole DB holder.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from lmtune.orchestrate.backend import TrialBackend, TrialHandle, TrialPayload, TrialResult
from lmtune.search.objective import ObjectiveResult
from lmtune.search.study import Study
from lmtune.search.trial import Trial

log = logging.getLogger(__name__)


def _to_objective_result(result: TrialResult) -> ObjectiveResult:
    return ObjectiveResult(
        score=float(result.score or 0.0),
        metrics=dict(result.metrics),
        error=result.error,
        accepted=(result.status == "completed"),
    )


def run_distributed(
    study: Study,
    backend: TrialBackend,
    *,
    endpoint_path: str | Path,
    profile_paths: list[str | Path],
    max_trials: int,
    repeats: int = 3,
    ttft_slo_ms: float = 500.0,
    poll_interval_s: float = 0.5,
    budget_seconds: float | None = None,
) -> list[Trial]:
    """Drive a Study through `backend` with up to `workers` concurrent trials.
    `workers` is encoded inside `backend` (e.g., ProcessPoolBackend(workers=N)).
    """
    # Driver is the *sole* DuckDB holder. While any worker is in flight the
    # connection is suspended so children can open the DB (bench run writes runs).
    # Writes from the driver (record_trial, tell) briefly resume→write→suspend.
    study.persist_header()
    study.storage.suspend()

    inflight: dict[str, tuple[TrialHandle, Trial, float]] = {}
    completed: list[Trial] = []
    asked = 0
    started_at = time.time()

    workers = int(getattr(backend, "_workers", 1))

    def _deadline_hit() -> bool:
        return budget_seconds is not None and (time.time() - started_at) >= budget_seconds

    def _write(fn_name: str, *args, **kwargs):
        study.storage.resume()
        try:
            getattr(study.storage, fn_name)(*args, **kwargs)
        finally:
            study.storage.suspend()

    def _fill_slots():
        nonlocal asked
        while (
            len(inflight) < workers
            and asked < max_trials
            and not study._exhausted
            and not _deadline_hit()
        ):
            # ask() mutates Optuna state only (no DB); writes happen below.
            try:
                study.storage.resume()
                try:
                    trial = study.ask()   # ask writes the pending trial row
                finally:
                    study.storage.suspend()
            except Exception as e:
                log.info("study %s: ask exhausted: %s", study.study_id, e)
                break
            payload = TrialPayload(
                trial_id=trial.trial_id,
                study_id=trial.study_id,
                seq=trial.seq,
                params=trial.params,
                endpoint_path=str(endpoint_path),
                profile_paths=[str(p) for p in profile_paths],
                repeats=repeats,
                ttft_slo_ms=ttft_slo_ms,
            )
            handle = backend.submit(payload)
            inflight[trial.trial_id] = (handle, trial, time.time())
            asked += 1

    try:
        _fill_slots()
        while inflight:
            finished: list[str] = []
            for tid, (handle, trial, t0) in list(inflight.items()):
                res = backend.poll(handle)
                if res is None:
                    continue
                dt = time.time() - t0
                # DB writes (tell + metrics) run on the driver thread only.
                study.storage.resume()
                try:
                    study.tell(trial, _to_objective_result(res))
                    if trial.metrics:
                        study.storage.record_trial_metrics(trial.trial_id, trial.metrics)
                finally:
                    study.storage.suspend()
                log.info(
                    "study %s trial %d: status=%s score=%s dt=%.1fs",
                    study.study_id, trial.seq, trial.status.value, trial.score, dt,
                )
                completed.append(trial)
                finished.append(tid)
            for tid in finished:
                inflight.pop(tid, None)
            _fill_slots()
            if not inflight:
                break
            time.sleep(poll_interval_s)

    finally:
        if _deadline_hit():
            for _tid, (handle, _trial, _t0) in list(inflight.items()):
                backend.cancel(handle)
        backend.shutdown()
        study.storage.resume()

    study.storage.set_study_status(study.study_id, "completed", finished=True)
    return completed
