"""TPE — Tree-structured Parzen Estimator sampler (Optuna default).

Models two densities g(x) (good trials) and l(x) (bad trials) via Parzen windows
and samples from argmax g(x)/l(x). Good for mixed discrete/continuous spaces.

Reference: Bergstra et al., 2011 (NIPS). Optuna's TPESampler is the canonical
implementation. A pedagogical re-implementation is scheduled for Phase S5.
"""

from __future__ import annotations

import optuna


def make_tpe(seed: int | None = None, n_startup_trials: int = 10) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,  # joint density → usually better for correlated params
        group=True,  # handle conditional (active_if) axes gracefully
    )
