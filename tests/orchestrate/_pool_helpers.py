"""Pickle-safe helpers used by test_process_pool.py's multiprocessing tests."""

from __future__ import annotations

from lmtune.orchestrate.backend import TrialPayload, TrialResult


def run_mock_trial(payload: TrialPayload) -> TrialResult:
    s = float(payload.params.get("x", 0)) * 10.0
    return TrialResult(
        trial_id=payload.trial_id,
        status="completed",
        score=s,
        metrics={("x", None): s},
        backend="process-pool",
        worker_id="test",
    )
