from __future__ import annotations

from pathlib import Path

from lmtune.search import CallableObjective, SearchSpace, Study, StudyConfig
from lmtune.search.space import Axis
from lmtune.storage.duckdb_store import DuckDBStore


def _store(tmp_path: Path) -> DuckDBStore:
    return DuckDBStore(tmp_path / "lmtune.duckdb")


def test_study_grid_exhausts_and_completes(tmp_path: Path):
    sp = SearchSpace(
        name="g",
        axes=[
            Axis("x", "categorical", values=[1, 2, 4]),
            Axis("p", "bool"),
        ],
    )
    # grid = 3 * 2 = 6
    store = _store(tmp_path)
    cfg = StudyConfig(name="grid-demo", strategy="grid", space=sp)
    study = Study(cfg, store)
    trials = study.run(
        CallableObjective(lambda p: p["x"] * (2.0 if p["p"] else 1.0)),
        max_trials=100,
    )
    assert len(trials) == 6
    hdr = store.get_study(study.study_id)
    assert hdr[8] == "completed"  # status column
    top = store.top_trials(study.study_id, direction="maximize", k=1)
    assert top[0][3] == 8.0  # x=4, p=True


def test_study_random_deterministic(tmp_path: Path):
    sp = SearchSpace(
        name="r",
        axes=[Axis("lr", "float", low=0.0, high=1.0)],
    )
    store_a = _store(tmp_path / "a")
    store_b = _store(tmp_path / "b")

    def run(store):
        cfg = StudyConfig(name="rnd", strategy="random", space=sp, seed=7)
        s = Study(cfg, store)
        ts = s.run(CallableObjective(lambda p: -abs(p["lr"] - 0.3)), max_trials=8)
        return [t.params["lr"] for t in ts]

    assert run(store_a) == run(store_b)


def test_study_lhc_enqueues_prefetch_first(tmp_path: Path):
    sp = SearchSpace(
        name="l",
        axes=[Axis("x", "categorical", values=[1, 2, 4, 8])],
    )
    store = _store(tmp_path)
    cfg = StudyConfig(name="lhc", strategy="lhc", space=sp, seed=1, n_samples=4)
    study = Study(cfg, store)
    trials = study.run(CallableObjective(lambda p: float(p["x"])), max_trials=4)
    xs = [t.params["x"] for t in trials]
    # LHC stratifies: all 4 categorical options must appear in exactly 4 trials
    assert sorted(xs) == [1, 2, 4, 8]


def test_study_warmstart_seeds_top_trial(tmp_path: Path):
    sp = SearchSpace(
        name="w",
        axes=[Axis("x", "categorical", values=[1, 2, 4, 8, 16])],
    )
    store = _store(tmp_path)
    cfg = StudyConfig(name="warm", strategy="random", space=sp, seed=3)
    study = Study(cfg, store)
    # Tell optuna x=16 already scored very high.
    study.enqueue_warmstart([({"x": 16}, 999.0)])
    ts = study.run(CallableObjective(lambda p: float(p["x"])), max_trials=5)
    # The first trial's optuna-known best should include 16 because we enqueued it.
    assert any(t.params["x"] == 16 for t in ts)


def test_study_feasibility_skips_infeasible_candidates(tmp_path: Path):
    """Feasibility wiring — infeasible 후보 (TP=16 single-node) 는 helmfile
    redeploy 0회로 즉시 prune. 결과 trial 들은 모두 feasible (TP ≤ 8)."""
    from lmtune.search.feasibility import Environment

    sp = SearchSpace(
        name="b3-mini",
        axes=[
            Axis("tensor_parallel_size", "categorical", values=[1, 2, 4, 8, 16]),
        ],
        feasibility_constraints=[
            {
                "id": "c2_tp_single_node",
                "rule": "tensor_parallel_size <= environment.npus_per_server",
                "message": "TP must fit single node",
            },
        ],
    )
    store = _store(tmp_path)
    cfg = StudyConfig(
        name="feas",
        strategy="grid",
        space=sp,
        seed=0,
        context={"environment": Environment.b200_single_node()},
    )
    study = Study(cfg, store)
    ts = study.run(
        CallableObjective(lambda p: float(p["tensor_parallel_size"])),
        max_trials=8,
    )
    # All completed trials are feasible (TP ≤ 8).
    assert ts, "expected at least one feasible trial"
    for t in ts:
        assert t.params["tensor_parallel_size"] <= 8, t.params
    # The infeasible TP=16 was pruned by ask() before objective ran.
    assert study._infeasible_count >= 1


def test_study_feasibility_disabled_when_no_environment(tmp_path: Path):
    """No environment in context → feasibility checker NOT installed → 모든
    grid 조합이 그대로 실행 (안전한 default 동작)."""
    sp = SearchSpace(
        name="b3-mini",
        axes=[Axis("tensor_parallel_size", "categorical", values=[1, 8, 16])],
        feasibility_constraints=[
            {
                "id": "c2_tp_single_node",
                "rule": "tensor_parallel_size <= environment.npus_per_server",
            },
        ],
    )
    store = _store(tmp_path)
    cfg = StudyConfig(name="no-env", strategy="grid", space=sp)
    study = Study(cfg, store)
    ts = study.run(
        CallableObjective(lambda p: float(p["tensor_parallel_size"])),
        max_trials=10,
    )
    assert {t.params["tensor_parallel_size"] for t in ts} == {1, 8, 16}
    assert study._infeasible_count == 0
