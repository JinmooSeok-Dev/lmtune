from __future__ import annotations

import pytest

from bench.search.samplers import grid_search_space
from bench.search.samplers.lhc import lhc_samples
from bench.search.space import Axis, SearchSpace


def _space() -> SearchSpace:
    return SearchSpace(
        name="t",
        axes=[
            Axis("x", "categorical", values=[1, 2, 4, 8]),
            Axis("p", "bool"),
            Axis("lr", "float", low=0.0, high=1.0),
        ],
    )


def test_grid_rejects_float_axis():
    sp = _space()
    with pytest.raises(ValueError, match="grid does not support kind 'float'"):
        grid_search_space(sp)


def test_grid_discrete_only():
    sp = SearchSpace(
        name="g",
        axes=[
            Axis("x", "categorical", values=[1, 2]),
            Axis("p", "bool"),
        ],
    )
    g = grid_search_space(sp)
    assert g == {"x": [1, 2], "p": [False, True]}


def test_lhc_covers_domain():
    sp = _space()
    samples = lhc_samples(sp, n_samples=16, seed=42)
    assert len(samples) == 16
    # Each x option should appear at least once (Latin Hypercube stratifies)
    xs = [s["x"] for s in samples]
    assert set(xs) == {1, 2, 4, 8}
    # lr should be within bounds
    assert all(0.0 <= s["lr"] < 1.0 for s in samples)


def test_lhc_deterministic_seed():
    sp = _space()
    a = lhc_samples(sp, n_samples=8, seed=7)
    b = lhc_samples(sp, n_samples=8, seed=7)
    assert a == b


def test_lhc_log_uniform_range():
    sp = SearchSpace(
        name="log",
        axes=[Axis("lr", "log_uniform", low=1e-5, high=1.0)],
    )
    samples = lhc_samples(sp, n_samples=20, seed=0)
    assert all(1e-5 <= s["lr"] <= 1.0 for s in samples)
    # With LHC over a 5-decade log space, expect at least some small and some large
    vals = sorted(s["lr"] for s in samples)
    assert vals[0] < 1e-3
    assert vals[-1] > 1e-1
