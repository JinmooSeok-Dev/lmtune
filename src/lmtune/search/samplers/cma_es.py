"""CMA-ES — Covariance Matrix Adaptation Evolution Strategy (continuous axes).

Optuna's CmaEsSampler expects continuous axes. Mixed spaces: Optuna delegates
categorical axes to an independent (RandomSampler) fallback, so CMA-ES only
drives the continuous dimensions.

Reference: Hansen & Ostermeier, 2001. CMA-ES adapts an (n x n) covariance
matrix of the mutation distribution — good for rugged continuous landscapes.
"""

from __future__ import annotations

import optuna


def make_cma_es(
    seed: int | None = None,
    n_startup_trials: int = 8,
) -> optuna.samplers.CmaEsSampler:
    return optuna.samplers.CmaEsSampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        warn_independent_sampling=False,
    )
