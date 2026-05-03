"""Pruner adapters — Successive Halving / Hyperband.

Pruners rely on intermediate objective reports (trial.report(v, step) + should_prune()).
In Phase S2 we expose them on the `Study` API so a future per-repeat reporting
hook (S3 + writer_queue) can trigger early stopping. Without intermediate values
they are no-ops.

Reference: Li et al. 2017 "Hyperband: A Novel Bandit-Based Approach to Hyperparameter
Optimization" — async variants are in optuna.pruners.HyperbandPruner.
"""

from __future__ import annotations

import optuna


def make_pruner(kind: str | None = None, **kwargs) -> optuna.pruners.BasePruner | None:
    if kind is None or kind == "none":
        return None
    k = kind.lower()
    if k in ("sh", "successive_halving"):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=kwargs.get("min_resource", 1),
            reduction_factor=kwargs.get("reduction_factor", 3),
        )
    if k == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=kwargs.get("min_resource", 1),
            max_resource=kwargs.get("max_resource", "auto"),
            reduction_factor=kwargs.get("reduction_factor", 3),
        )
    raise ValueError(f"unknown pruner: {kind}")
