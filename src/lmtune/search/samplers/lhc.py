"""Latin Hypercube sampling — pre-generates N space-filling samples.

scipy.stats.qmc.LatinHypercube produces points in [0,1)^d with good stratification;
we map each dim back to the axis domain.

Because Optuna has no built-in LHC, we materialize samples up front and let the
caller `study.enqueue_trial(params)` them in order. Trials beyond the N samples
fall back to the underlying sampler (usually Random) set by make_sampler().
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats.qmc import LatinHypercube

from bench.search.space import Axis, SearchSpace


def _project(axis: Axis, u: float) -> Any:
    """Map u ∈ [0,1) to an axis value."""
    if axis.kind == "categorical":
        vals = axis.values or []
        idx = min(int(u * len(vals)), len(vals) - 1)
        return vals[idx]
    if axis.kind == "bool":
        return bool(u >= 0.5)
    if axis.kind == "int":
        low, high = int(axis.low), int(axis.high)
        step = int(axis.step) if axis.step else 1
        # index into [low, low+step, ..., high]
        options = list(range(low, high + 1, step))
        idx = min(int(u * len(options)), len(options) - 1)
        return options[idx]
    if axis.kind == "float":
        return float(axis.low) + u * (float(axis.high) - float(axis.low))
    if axis.kind == "log_uniform":
        lo = math.log(float(axis.low))
        hi = math.log(float(axis.high))
        return math.exp(lo + u * (hi - lo))
    raise ValueError(f"axis {axis.name}: unsupported kind {axis.kind}")


def lhc_samples(
    space: SearchSpace,
    *,
    n_samples: int,
    seed: int | None = None,
    context: dict | None = None,
) -> list[dict[str, Any]]:
    axes = space.active_axes(context)
    d = len(axes)
    if d == 0 or n_samples <= 0:
        return []
    # LatinHypercube(d, seed=...) returns floats in [0,1).
    engine = LatinHypercube(d=d, seed=seed) if seed is not None else LatinHypercube(d=d)
    U = engine.random(n=int(n_samples))
    out: list[dict[str, Any]] = []
    for row in np.asarray(U):
        params: dict[str, Any] = {}
        for i, axis in enumerate(axes):
            params[axis.name] = _project(axis, float(row[i]))
        out.append(params)
    return out
