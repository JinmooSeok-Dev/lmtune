"""ProcessPoolBackend end-to-end with a mock objective.

The mock replaces ScoreObjective transparently: we spawn the worker with a
TrialPayload whose `endpoint_path` points at a fixture file the runner reads
AND a monkeypatch hook that swaps ScoreObjective for a pure-Python stub.
For simplicity we call `run_trial_locally` directly with a monkeypatched
objective — same function the pool executes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lmtune.orchestrate.backend import TrialPayload
from lmtune.orchestrate.trial_runner import run_trial_locally
from lmtune.search.objective import ObjectiveResult


class _FakeObj:
    def __init__(self, *a, **kw):
        self.params_seen = None

    def __call__(self, params):
        self.params_seen = params
        s = float(params.get("x", 0)) * 10.0
        return ObjectiveResult(score=s, metrics={("x", None): s}, accepted=True)


def _payload(params: dict) -> TrialPayload:
    return TrialPayload(
        trial_id="tr-t1",
        study_id="st-t1",
        seq=1,
        params=params,
        endpoint_path="/does/not/matter",
        profile_paths=["/does/not/matter"],
        repeats=1,
    )


def test_run_trial_locally_happy_path():
    with patch("lmtune.orchestrate.trial_runner.ScoreObjective", _FakeObj):
        r = run_trial_locally(_payload({"x": 3}))
    assert r.status == "completed"
    assert r.score == 30.0
    assert r.backend == "process-pool"


def test_run_trial_locally_returns_crash_on_exception():
    class _Boom:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, p):
            raise RuntimeError("boom")

    with patch("lmtune.orchestrate.trial_runner.ScoreObjective", _Boom):
        r = run_trial_locally(_payload({"x": 1}))
    assert r.status == "crash"
    assert "boom" in (r.error or "")


def test_pool_runs_two_trials_concurrently(tmp_path):
    """Use ProcessPoolBackend with an actual pool; mock objective via a
    top-level callable module so it's picklable."""
    pytest.importorskip("optuna")  # import safety
    from lmtune.orchestrate.backend_process_pool import ProcessPoolBackend
    from tests.orchestrate._pool_helpers import run_mock_trial

    pool = ProcessPoolBackend(workers=2)
    # submit two payloads; each returns score = x*10 via run_mock_trial.
    handles = []
    for x in [2, 5]:
        p = TrialPayload(
            trial_id=f"tr-{x}",
            study_id="st",
            seq=x,
            params={"x": x},
            endpoint_path="x",
            profile_paths=["x"],
            repeats=1,
        )
        handles.append(pool._pool.submit(run_mock_trial, p))

    results = [h.result(timeout=10) for h in handles]
    pool.shutdown()

    scores = sorted(r.score for r in results)
    assert scores == [20.0, 50.0]
    assert all(r.status == "completed" for r in results)
