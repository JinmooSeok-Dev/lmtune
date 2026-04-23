"""Phase S5 tests: NSGA-II / native samplers / Sobol / pareto plot."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from bench.search.space import Axis, SearchSpace
from bench.search.samplers.native import NativeLHCSampler, NativeRandomSampler, NativeTPESampler
from bench.search.samplers.nsga2 import make_nsga2
from bench.visualization.plots.pareto import non_dominated, plot_pareto
from bench.visualization.plots.sobol_bar import plot_sobol
from bench.visualization.plots.search_trace import plot_search_trace


# ------------------ Native samplers ------------------

def _mixed() -> SearchSpace:
    return SearchSpace(name="m", axes=[
        Axis("x", "categorical", values=[1, 2, 4, 8]),
        Axis("p", "bool"),
        Axis("lr", "float", low=0.01, high=0.5),
    ])


def _obj(p):
    return p["x"] * (1.0 if p["p"] else 0.5) - abs(p["lr"] - 0.1) * 20


def test_native_random_covers_all_categories():
    s = NativeRandomSampler(_mixed(), seed=0)
    xs = set()
    for _ in range(30):
        xs.add(s.ask()["x"])
    assert xs == {1, 2, 4, 8}


def test_native_lhc_stratifies():
    sp = _mixed()
    s = NativeLHCSampler(sp, n_samples=16, seed=1)
    seen_x = []
    for _ in range(16):
        seen_x.append(s.ask()["x"])
    assert set(seen_x) == {1, 2, 4, 8}


def test_native_tpe_beats_random_on_synthetic():
    def run(sampler, n=30):
        scores = []
        for _ in range(n):
            p = sampler.ask()
            s = _obj(p)
            if hasattr(sampler, "tell"):
                sampler.tell(p, s)
            scores.append(s)
        return scores

    r = run(NativeRandomSampler(_mixed(), seed=42))
    t = run(NativeTPESampler(_mixed(), seed=42, n_startup_trials=8))
    # Top-5 mean should favor TPE by a clear margin.
    r_top = sum(sorted(r, reverse=True)[:5]) / 5
    t_top = sum(sorted(t, reverse=True)[:5]) / 5
    assert t_top > r_top * 1.2   # ≥20% better


# ------------------ Pareto (non-dominated) ------------------

def test_pareto_front_identifies_dominated_points():
    # directions: maximize obj1, minimize obj2
    pts = [[10, 5], [5, 3], [8, 2], [3, 8]]
    nd = non_dominated(pts, directions=["maximize", "minimize"])
    # (8,2) dominates (5,3) and (10,5) is non-dominated; (3,8) is clearly dominated.
    assert 2 in nd       # (8, 2)
    assert 0 in nd       # (10, 5)
    assert 3 not in nd   # (3, 8)


def test_pareto_plot_writes_file(tmp_path: Path):
    pts = [[1.0, 2.0], [2.0, 1.0], [0.5, 0.5]]
    out = tmp_path / "pareto.png"
    p = plot_pareto(pts, directions=["maximize", "minimize"], out_path=out)
    assert p.exists() and p.stat().st_size > 0


# ------------------ NSGA-II ------------------

def test_nsga2_sampler_constructs():
    s = make_nsga2(seed=0, population_size=8)
    assert s is not None


# ------------------ Sobol ------------------

def test_sobol_recovers_dominant_axis():
    """Synthetic: y = 10*a + 0.01*b + noise. Sobol should put a >> b."""
    from bench.search.analysis.sobol import sobol_from_history

    trials = []
    import random
    rng = random.Random(0)
    for _ in range(200):
        a = rng.uniform(0, 1)
        b = rng.uniform(0, 1)
        y = 10 * a + 0.01 * b + rng.gauss(0, 0.05)
        trials.append({"params": {"a": a, "b": b}, "score": y, "status": "completed"})
    axes = [
        {"name": "a", "kind": "float", "low": 0.0, "high": 1.0},
        {"name": "b", "kind": "float", "low": 0.0, "high": 1.0},
    ]
    results = sobol_from_history(trials, axes, n_saltelli=256, surrogate_n_estimators=100)
    by = {r.axis: r for r in results}
    assert "a" in by and "b" in by
    assert by["a"].ST > by["b"].ST * 5


def test_sobol_returns_empty_on_categorical_only():
    from bench.search.analysis.sobol import sobol_from_history
    trials = [
        {"params": {"x": "A"}, "score": 1.0, "status": "completed"},
        {"params": {"x": "B"}, "score": 2.0, "status": "completed"},
    ] * 5
    axes = [{"name": "x", "kind": "categorical", "low": None, "high": None}]
    assert sobol_from_history(trials, axes) == []


# ------------------ search trace plot ------------------

def test_search_trace_plot(tmp_path: Path):
    p = plot_search_trace(
        seqs=[1, 2, 3, 4, 5],
        scores=[0.3, 0.5, None, 0.7, 0.6],
        direction="maximize",
        out_path=tmp_path / "trace.png",
    )
    assert p.exists()
