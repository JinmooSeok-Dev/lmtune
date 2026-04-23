"""BoTorch sampler smoke — small synthetic 2D continuous problem."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bench.search import CallableObjective, SearchSpace, Study, StudyConfig
from bench.search.space import Axis
from bench.storage.duckdb_store import DuckDBStore


botorch = pytest.importorskip("optuna_integration")


def test_botorch_sampler_runs_on_continuous_space():
    sp = SearchSpace(
        name="gp",
        axes=[
            Axis("x", "float", low=-2.0, high=2.0),
            Axis("y", "float", low=-2.0, high=2.0),
        ],
    )
    store = DuckDBStore(Path(tempfile.mkdtemp()) / "bench.duckdb")
    cfg = StudyConfig(name="gp-smoke", strategy="botorch", space=sp, seed=0)
    study = Study(cfg, store)
    ts = study.run(
        CallableObjective(lambda p: -((p["x"] - 1) ** 2 + (p["y"] - 1) ** 2)),
        max_trials=15,
    )
    # Should converge near (1, 1) → best score close to 0.
    top = max(t.score for t in ts if t.score is not None)
    assert top > -1.5, f"BoTorch failed to improve; top={top}"
    store.close()


def test_botorch_sampler_accepts_bo_alias():
    """'bo' and 'gp' should route to the same BoTorch sampler."""
    from bench.search.samplers import make_sampler

    sp = SearchSpace(name="s", axes=[Axis("x", "float", low=0.0, high=1.0)])
    for alias in ("botorch", "gp", "bo"):
        sampler, prefetch = make_sampler(alias, sp, seed=0)
        assert sampler is not None
        assert prefetch is None
