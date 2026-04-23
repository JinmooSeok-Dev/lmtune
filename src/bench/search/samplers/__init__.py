"""Sampler adapters — create Optuna samplers from a strategy name.

S1 ships Grid/Random/LHC. S2 adds TPE/CMA-ES + Pruners. S5 ships native
(Optuna-free) implementations for the three listed above for pedagogical
transparency; they live in `*_native.py` and are selected by `_native` suffix.
"""

from __future__ import annotations

from typing import Any

import optuna

from bench.search.space import Axis, SearchSpace


def suggest_from_axis(trial: optuna.Trial, axis: Axis) -> Any:
    """Call the matching `suggest_*` on an Optuna trial for this axis."""
    if axis.kind == "categorical":
        return trial.suggest_categorical(axis.name, axis.values)
    if axis.kind == "bool":
        return trial.suggest_categorical(axis.name, [False, True])
    if axis.kind == "int":
        step = int(axis.step) if axis.step else 1
        return trial.suggest_int(axis.name, int(axis.low), int(axis.high), step=step)
    if axis.kind == "float":
        return trial.suggest_float(axis.name, float(axis.low), float(axis.high))
    if axis.kind == "log_uniform":
        return trial.suggest_float(axis.name, float(axis.low), float(axis.high), log=True)
    raise ValueError(f"axis {axis.name}: unsupported kind {axis.kind}")


def grid_search_space(space: SearchSpace, context: dict | None = None) -> dict[str, list[Any]]:
    """Materialize a discrete grid for Optuna's GridSampler.

    Float axes are not supported by grid; raise early.
    Int axes require `step` to enumerate.
    """
    grid: dict[str, list[Any]] = {}
    for a in space.active_axes(context):
        if a.kind == "categorical":
            grid[a.name] = list(a.values or [])
        elif a.kind == "bool":
            grid[a.name] = [False, True]
        elif a.kind == "int":
            if a.step is None:
                raise ValueError(
                    f"axis {a.name}: int grid requires 'step' (got low={a.low} high={a.high})"
                )
            step = int(a.step)
            grid[a.name] = list(range(int(a.low), int(a.high) + 1, step))
        else:
            raise ValueError(
                f"axis {a.name}: grid does not support kind '{a.kind}'. "
                "Use 'random' or 'lhc' for continuous axes."
            )
    return grid


def make_sampler(
    strategy: str,
    space: SearchSpace,
    *,
    seed: int | None = None,
    context: dict | None = None,
    n_samples: int | None = None,
) -> tuple[optuna.samplers.BaseSampler, list[dict] | None]:
    """Build the Optuna sampler for the requested strategy.

    Returns (sampler, prefetch_params). `prefetch_params` is non-None for LHC:
    the caller should enqueue them via `study.enqueue_trial(...)` in order.
    """
    s = strategy.lower()
    if s == "grid":
        grid = grid_search_space(space, context=context)
        return optuna.samplers.GridSampler(grid, seed=seed), None
    if s == "random":
        return optuna.samplers.RandomSampler(seed=seed), None
    if s == "lhc":
        from bench.search.samplers.lhc import lhc_samples
        if n_samples is None:
            raise ValueError("lhc sampler requires n_samples")
        samples = lhc_samples(space, n_samples=n_samples, seed=seed, context=context)
        # After pre-seeding, downstream uses RandomSampler for any overflow trials.
        return optuna.samplers.RandomSampler(seed=seed), samples
    raise ValueError(f"unknown strategy: {strategy}")
