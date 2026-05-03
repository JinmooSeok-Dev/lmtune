"""Tests for dashboard explainability annotations.

Pure-function tests on `_annotate_trial`, `_compute_axis_diff`, and the
`_StudyView.to_dict()` augmentation. No DB / no Jinja required.
"""
from __future__ import annotations

from lmtune.visualization.dashboard.build import (
    _annotate_trial,
    _compute_axis_diff,
)
from lmtune.visualization.dashboard.schemas import TrialPoint


def _pt(seq: int, score: float | None, params: dict | None = None, tid: str | None = None):
    return TrialPoint(
        trial_id=tid or f"t{seq}",
        seq=seq,
        score=score,
        params=params or {},
        metrics={},
    )


def test_annotate_first_trial_is_first():
    p = _pt(1, score=10.0)
    out = _annotate_trial(p, running_best=None, direction="maximize", strategy="random", n_startup=0)
    assert out["outcome"] == "first"
    assert out["delta_pct_vs_best"] == 0.0
    assert out["phase"] == "random"


def test_annotate_new_best_maximize():
    p = _pt(5, score=20.0)
    out = _annotate_trial(p, running_best=10.0, direction="maximize", strategy="tpe", n_startup=10)
    assert "new best" in out["outcome"]
    assert out["delta_pct_vs_best"] == 100.0
    # seq 5 is within startup → label as warmup
    assert "warmup" in out["phase"].lower()


def test_annotate_worse_minimize():
    p = _pt(15, score=200.0)
    out = _annotate_trial(p, running_best=100.0, direction="minimize", strategy="tpe", n_startup=10)
    assert out["outcome"] == "↳ exploring"
    assert out["delta_pct_vs_best"] == 100.0
    # past warmup
    assert out["phase"] == "TPE"


def test_annotate_no_score():
    p = _pt(3, score=None)
    out = _annotate_trial(p, running_best=10.0, direction="maximize", strategy="random", n_startup=0)
    assert out["outcome"] == "no score"
    assert out["delta_pct_vs_best"] is None


def test_annotate_strategies_phase_labels():
    p = _pt(1, score=1.0)
    for strat, expected in [
        ("random", "random"),
        ("grid", "grid"),
        ("lhc", "LHC"),
    ]:
        out = _annotate_trial(p, None, "maximize", strat, 0)
        assert out["phase"] == expected


def test_axis_diff_identifies_winner_and_diff():
    pts = [
        _pt(1, score=10.0, params={"a": 1, "b": "x"}, tid="w1"),
        _pt(2, score=20.0, params={"a": 2, "b": "x"}, tid="w2"),  # winner (max)
        _pt(3, score=15.0, params={"a": 1, "b": "y"}, tid="w3"),
    ]
    rows = _compute_axis_diff(pts, direction="maximize", top_k=3)
    # winner is first row
    assert rows[0]["is_winner"] is True
    assert rows[0]["trial_id"] == "w2"
    assert rows[0]["diff"] == {}
    # second row diffs on `a`
    second = next(r for r in rows[1:] if r["trial_id"] == "w3")
    assert "a" in second["diff"]
    assert second["diff"]["a"] == (2, 1)
    assert "b" in second["diff"]


def test_axis_diff_empty_when_no_completed():
    rows = _compute_axis_diff([_pt(1, score=None)], direction="maximize")
    assert rows == []


def test_axis_diff_minimize_picks_smallest():
    pts = [
        _pt(1, score=10.0, params={"x": 1}, tid="a"),
        _pt(2, score=5.0, params={"x": 2}, tid="b"),  # winner (min)
        _pt(3, score=15.0, params={"x": 3}, tid="c"),
    ]
    rows = _compute_axis_diff(pts, direction="minimize")
    assert rows[0]["trial_id"] == "b"
    assert rows[0]["is_winner"] is True
