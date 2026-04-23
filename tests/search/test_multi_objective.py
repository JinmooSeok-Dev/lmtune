"""Multi-objective Study + ParetoObjective on a synthetic 2-obj problem."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench.search import CallableObjective, SearchSpace, Study, StudyConfig
from bench.search.objective import ObjectiveResult
from bench.search.objective_pareto import ObjectiveKey, ParetoObjective
from bench.search.space import Axis
from bench.storage.duckdb_store import DuckDBStore


def _space() -> SearchSpace:
    return SearchSpace(
        name="p",
        axes=[Axis("x", "float", low=-2.0, high=2.0),
              Axis("y", "float", low=-2.0, high=2.0)],
    )


def _base_obj(p):
    return ObjectiveResult(
        score=0.0,
        metrics={
            ("obj1", None): -((p["x"] - 1) ** 2 + (p["y"] - 1) ** 2),  # maximize
            ("obj2", None): abs(p["x"]) + abs(p["y"]),                 # minimize
        },
        accepted=True,
    )


def test_nsga2_multi_objective_finds_pareto_front(tmp_path: Path):
    store = DuckDBStore(tmp_path / "p.duckdb")
    pareto = ParetoObjective(
        CallableObjective(_base_obj),
        [ObjectiveKey("obj1", None, "maximize"),
         ObjectiveKey("obj2", None, "minimize")],
    )
    cfg = StudyConfig(
        name="nsga2", strategy="nsga2", space=_space(),
        directions=["maximize", "minimize"], seed=0,
    )
    study = Study(cfg, store)
    trials = study.run(pareto, max_trials=25)

    # Optuna's best_trials holds the current Pareto front.
    front = study._optuna_study.best_trials
    assert len(trials) == 25
    assert len(front) >= 3  # non-trivial front
    # Every front member should have the two values stored.
    for t in front:
        assert t.values is not None and len(t.values) == 2


def test_trial_metrics_skip_non_scalar_values_sentinel(tmp_path: Path):
    """The ParetoObjective stashes the tuple under ('_values', None).
    DB persistence must silently skip it (list is not a scalar)."""
    store = DuckDBStore(tmp_path / "p.duckdb")
    store.record_study(
        study_id="st-mv",
        name="mv",
        strategy="nsga2",
        metric_name="tuple",
        direction="maximize",
    )
    store.record_trial(
        "tr-mv", "st-mv", 1, {"x": 1}, status="completed",
        score=0.5, backend="inline", completed=True,
    )
    # Mix scalar + list; the list must be dropped, scalar must persist.
    store.record_trial_metrics(
        "tr-mv",
        {("obj1", None): 1.0, ("_values", None): [1.0, 2.0], ("obj2", "short"): 3.0},
    )
    got = store.get_trial_metrics("tr-mv")
    assert "obj1" in got and got["obj1"]["aggregate"] == 1.0
    assert "obj2" in got and got["obj2"]["short"] == 3.0
    assert "_values" not in got
    store.close()
