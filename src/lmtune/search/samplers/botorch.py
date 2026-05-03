"""BoTorch backend — Gaussian Process + acquisition-function sampler.

Wraps `optuna_integration.BoTorchSampler`. Preferred regime (2026-04):
- continuous, mixed continuous + int/categorical (via one-hot inside the GP)
- trial budgets ≥ ~200, or per-trial cost so high that each candidate pick
  deserves expensive GP machinery (qEI/qNEI).

For our typical 20-40 trial autotune runs Optuna TPE is cheaper and wins.
BoTorch becomes attractive when the study scales to thousands of trials or
runs on dense computing where GP-per-pick overhead is negligible. See
`docs/search_tooling_2026-04.md` §3.1 for the trade-off analysis.
"""

from __future__ import annotations


def make_botorch(seed: int | None = None, n_startup_trials: int = 10):
    try:
        from optuna_integration import BoTorchSampler
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "BoTorch sampler requires `optuna-integration[botorch]`. "
            "pip install 'optuna-integration[botorch]'"
        ) from e

    return BoTorchSampler(
        seed=seed,
        n_startup_trials=int(n_startup_trials),
    )
