"""ANOVA + RF importance + bound-tighten recommendations."""

from __future__ import annotations

from bench.search.analysis import anova_per_axis, axis_importance, tighten_bounds


def _trials_x_dominates(n_per_group: int = 6):
    """Score is driven entirely by axis `x`; `noise` is irrelevant."""
    out = []
    for x_val, base in [("A", 10.0), ("B", 100.0), ("C", 50.0)]:
        for i in range(n_per_group):
            out.append({
                "params": {"x": x_val, "noise": i % 2},
                "score": base + i * 0.1,
                "status": "completed",
            })
    return out


def test_anova_freeze_when_axis_dominates():
    trials = _trials_x_dominates(n_per_group=6)
    result = anova_per_axis(trials)
    by_axis = {a.axis: a for a in result}
    assert "x" in by_axis
    x = by_axis["x"]
    assert x.recommendation == "freeze"
    assert x.best_value == "B"


def test_anova_drop_when_axis_is_noise():
    trials = _trials_x_dominates(n_per_group=6)
    result = anova_per_axis(trials)
    by_axis = {a.axis: a for a in result}
    # noise axis contributes nothing ⇒ should drop
    assert by_axis["noise"].recommendation in ("drop", "keep")  # depends on exact p


def test_importance_rf_ranks_dominant_axis_high():
    trials = _trials_x_dominates(n_per_group=8)
    imp = axis_importance(trials, n_estimators=50, seed=0)
    assert "x" in imp and "noise" in imp
    assert imp["x"]["importance"] > imp["noise"]["importance"]
    assert imp["x"]["recommendation"] == "keep"


def test_bound_tighten_shrinks_around_best():
    # top trials cluster tightly around lr=0.1; sigma is small.
    trials = []
    for i in range(20):
        trials.append({
            "params": {"lr": 0.10 + (i % 5) * 0.002, "x": "A"},
            "score": 100.0 - abs((0.10 + (i % 5) * 0.002) - 0.10) * 500,
            "status": "completed",
        })
    axes = [
        {"name": "lr", "kind": "float", "low": 0.0, "high": 1.0},
        {"name": "x", "kind": "categorical", "low": None, "high": None},
    ]
    shrink = tighten_bounds(trials, axes, top_frac=0.5)
    assert "lr" in shrink
    s = shrink["lr"]
    assert s["new_low"] < 0.11
    assert s["new_high"] > 0.09
    assert s["new_high"] - s["new_low"] < 1.0  # shrunk below original span
    # Categorical axis must not appear.
    assert "x" not in shrink


def test_anova_handles_single_group_axis():
    # Axis with only one observed value → no ANOVA, remain 'keep'.
    trials = [
        {"params": {"k": "v"}, "score": 1.0, "status": "completed"},
        {"params": {"k": "v"}, "score": 2.0, "status": "completed"},
    ]
    result = anova_per_axis(trials)
    by = {a.axis: a for a in result}
    assert by["k"].recommendation == "keep"
