"""TPE / CMA-ES / UCB1 sanity — synthetic objectives."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench.search import CallableObjective, SearchSpace, Study, StudyConfig
from bench.search.space import Axis
from bench.storage.duckdb_store import DuckDBStore


def _sp_mixed() -> SearchSpace:
    return SearchSpace(
        name="mix",
        axes=[
            Axis("x", "categorical", values=[1, 2, 4, 8]),
            Axis("p", "bool"),
            Axis("lr", "float", low=0.01, high=0.2),
        ],
    )


def _sp_cat() -> SearchSpace:
    return SearchSpace(
        name="cat",
        axes=[
            Axis("arm", "categorical", values=["A", "B", "C", "D"]),
            Axis("flag", "bool"),
        ],
    )


def _fake_store(tmp: Path) -> DuckDBStore:
    return DuckDBStore(tmp / "bench.duckdb")


def _obj_mixed(params):
    return params["x"] * (1.0 if params["p"] else 0.5) - abs(params["lr"] - 0.1) * 10


def test_tpe_converges_better_than_random_on_average():
    """Over N=30 trials on a mixed space, TPE's top-1 should match or beat Random."""
    def top_under(strategy, seed):
        store = _fake_store(Path(tempfile.mkdtemp()))
        cfg = StudyConfig(name=f"{strategy}-{seed}", strategy=strategy, space=_sp_mixed(), seed=seed)
        s = Study(cfg, store)
        ts = s.run(CallableObjective(_obj_mixed), max_trials=30)
        store.close()
        return max(t.score for t in ts if t.score is not None)

    wins = 0
    for seed in range(1, 6):
        if top_under("tpe", seed) >= top_under("random", seed):
            wins += 1
    # At least 3/5 seeds — TPE should not be systematically worse.
    assert wins >= 3


def test_cma_es_runs_on_continuous_axes():
    # CMA-ES accepts categorical axes via the independent-sampler fallback; we just
    # verify the end-to-end loop completes without raising.
    store = _fake_store(Path(tempfile.mkdtemp()))
    cfg = StudyConfig(name="cma", strategy="cma_es", space=_sp_mixed(), seed=1)
    s = Study(cfg, store)
    ts = s.run(CallableObjective(_obj_mixed), max_trials=12)
    assert len(ts) >= 8  # startup phase may early-stop a trial or two
    # At least one completed trial with a finite score
    assert any(t.score is not None and t.score > 0 for t in ts)


def test_ucb1_favors_high_reward_arm():
    """On a pure-categorical space where arm 'D' is best, UCB1 should concentrate."""
    store = _fake_store(Path(tempfile.mkdtemp()))
    cfg = StudyConfig(name="ucb", strategy="ucb", space=_sp_cat(), seed=0)
    s = Study(cfg, store)
    rewards = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 10.0}
    ts = s.run(
        CallableObjective(lambda p: rewards[p["arm"]] * (1.2 if p["flag"] else 1.0)),
        max_trials=24,
    )
    assert len(ts) == 24
    picked = [t.params["arm"] for t in ts]
    # After many trials, D should be picked substantially more often than A/B.
    assert picked.count("D") >= picked.count("A")
    assert picked.count("D") >= picked.count("B")
