"""UCB1 bandit — educational categorical-axis sampler.

Treats each categorical/bool axis as an independent multi-armed bandit and
pulls the arm with the highest UCB1 score:

    UCB1(arm) = avg_reward(arm) + sqrt( 2 ln(N_total) / n_pulls(arm) )

Reference: Auer et al., 2002 "Finite-time analysis of the multiarmed bandit problem".

Limitations (deliberate for S2 pedagogical scope):
- Each axis is treated independently (no joint-arm combination). Good enough
  when axes are near-independent; poor when they interact strongly.
- Continuous / integer axes fall back to Optuna's RandomSampler.
- A fresh UCB1 restarts from zero pulls on a new study (no cross-study transfer).

S5 will replace this with a more principled Thompson sampling or LinUCB variant.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import optuna
from optuna.samplers import BaseSampler, RandomSampler

from lmtune.search.space import SearchSpace


class UCB1Sampler(BaseSampler):
    def __init__(
        self,
        space: SearchSpace,
        seed: int | None = None,
        context: dict | None = None,
        exploration_c: float = math.sqrt(2.0),
    ):
        self._space = space
        self._context = context or {}
        self._rng = random.Random(seed)
        self._fallback = RandomSampler(seed=seed)
        self._c = float(exploration_c)
        # arm stats keyed by (axis_name, value_repr)
        self._pulls: dict[tuple[str, Any], int] = defaultdict(int)
        self._reward_sum: dict[tuple[str, Any], float] = defaultdict(float)
        self._total_pulls: int = 0
        # track what each (in-flight) trial tried so we can credit rewards in after_trial
        self._pending: dict[int, dict[str, Any]] = {}

    # Optuna BaseSampler protocol -----------------------------------------

    def infer_relative_search_space(self, study, trial):
        return {}  # everything goes through sample_independent

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name: str, param_distribution):
        # Only do UCB for categorical-like axes (CategoricalDistribution).
        if not isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            return self._fallback.sample_independent(study, trial, param_name, param_distribution)

        choices: list[Any] = list(param_distribution.choices)
        # Pull an unseen arm first (initialization phase).
        for c in choices:
            key = (param_name, _canon(c))
            if self._pulls[key] == 0:
                value = c
                self._pending.setdefault(trial._trial_id, {})[param_name] = _canon(value)
                return value

        # Otherwise pick argmax UCB1
        N = self._total_pulls
        best_arm, best_score = None, -math.inf
        for c in choices:
            key = (param_name, _canon(c))
            n = self._pulls[key]
            mean = self._reward_sum[key] / n
            bonus = self._c * math.sqrt(math.log(max(N, 1)) / n)
            score = mean + bonus
            if score > best_score:
                best_score, best_arm = score, c
        self._pending.setdefault(trial._trial_id, {})[param_name] = _canon(best_arm)
        return best_arm

    def after_trial(self, study, trial, state, values):
        # Credit the reward (scalar value) to each arm that this trial pulled.
        pulled = self._pending.pop(trial._trial_id, None)
        if pulled is None:
            return
        # If the trial failed or had no value, treat the reward as 0.
        reward = 0.0
        if values is not None and len(values) > 0:
            reward = float(values[0])
        for axis_name, value_repr in pulled.items():
            key = (axis_name, value_repr)
            self._pulls[key] += 1
            self._reward_sum[key] += reward
            self._total_pulls += 1


def _canon(v: Any) -> Any:
    """Canonical representation for dict key (booleans are int-ish in Python)."""
    if isinstance(v, bool):
        return (bool, v)
    return v


def make_ucb1(space: SearchSpace, seed: int | None = None, context: dict | None = None):
    return UCB1Sampler(space=space, seed=seed, context=context)
