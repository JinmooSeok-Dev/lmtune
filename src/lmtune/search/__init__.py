"""Search framework (Phase S).

Public API:
    SearchSpace, Axis            — search space declaration (YAML-backed)
    Trial                         — one (params → score) candidate
    Study                         — owns sampler + history + Optuna engine
    Objective                     — wraps bench_score.py (subprocess) or any callable
"""

from bench.search.space import Axis, SearchSpace, load_space
from bench.search.trial import Trial, TrialStatus
from bench.search.objective import Objective, CallableObjective, BenchScoreObjective, ObjectiveResult
from bench.search.study import Study, StudyConfig

__all__ = [
    "Axis",
    "SearchSpace",
    "load_space",
    "Trial",
    "TrialStatus",
    "Objective",
    "CallableObjective",
    "BenchScoreObjective",
    "ObjectiveResult",
    "Study",
    "StudyConfig",
]
