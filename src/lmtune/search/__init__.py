"""Search framework (Phase S).

Public API:
    SearchSpace, Axis            — search space declaration (YAML-backed)
    Trial                         — one (params → score) candidate
    Study                         — owns sampler + history + Optuna engine
    Objective                     — wraps lmtune_score.py (subprocess) or any callable
"""

from lmtune.search.objective import (
    CallableObjective,
    Objective,
    ObjectiveResult,
    ScoreObjective,
)
from lmtune.search.space import Axis, SearchSpace, load_space
from lmtune.search.study import Study, StudyConfig
from lmtune.search.trial import Trial, TrialStatus

__all__ = [
    "Axis",
    "SearchSpace",
    "load_space",
    "Trial",
    "TrialStatus",
    "Objective",
    "CallableObjective",
    "ScoreObjective",
    "ObjectiveResult",
    "Study",
    "StudyConfig",
]
