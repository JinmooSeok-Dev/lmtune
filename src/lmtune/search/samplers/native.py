"""Native (Optuna-free) samplers for pedagogical transparency.

These replicate the Phase S1/S2 samplers using only numpy + scipy, so the
math is right there in the repo — no Optuna abstraction to trace through.

- NativeRandomSampler  — draws each axis independently from its declared
                         distribution.
- NativeLHCSampler     — Latin Hypercube over active axes (scipy.stats.qmc).
- NativeTPESampler     — Tree-structured Parzen Estimator. For each axis,
                         fit two KDEs (scipy.stats.gaussian_kde): ``l(x)``
                         from the top fraction (good trials) and ``g(x)`` from
                         the rest. Sample from l, pick the candidate that
                         maximizes ``l(x)/g(x)``.  Mixed categorical handling
                         uses categorical-proportional selection (Bergstra
                         2011 §4.2).

These are NOT drop-in replacements for Optuna's samplers in CLI — they live
under `bench search start --strategy tpe_native` etc., and the default
remains the Optuna variants (battle-tested).
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats.qmc import LatinHypercube

from lmtune.search.space import Axis, SearchSpace

# --- Random ----------------------------------------------------------------

class NativeRandomSampler:
    def __init__(self, space: SearchSpace, seed: int | None = None):
        self._space = space
        self._rng = random.Random(seed)
        self._np = np.random.default_rng(seed)

    def ask(self, context: dict | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for a in self._space.active_axes(context):
            params[a.name] = _sample_axis(a, self._rng, self._np)
        return params


# --- Latin Hypercube -------------------------------------------------------

class NativeLHCSampler:
    def __init__(self, space: SearchSpace, n_samples: int, seed: int | None = None, context: dict | None = None):
        self._space = space
        axes = space.active_axes(context)
        if not axes:
            self._queue: list[dict] = []
            return
        engine = LatinHypercube(d=len(axes), seed=seed) if seed is not None else LatinHypercube(d=len(axes))
        U = engine.random(n=int(n_samples))
        self._queue = [
            {axes[i].name: _project_u(axes[i], float(U[r, i])) for i in range(len(axes))}
            for r in range(U.shape[0])
        ]

    def ask(self, context: dict | None = None) -> dict[str, Any]:
        if not self._queue:
            raise StopIteration("LHC queue exhausted")
        return self._queue.pop(0)


# --- TPE --------------------------------------------------------------------

class NativeTPESampler:
    """Minimal TPE that handles continuous and categorical axes independently.

    For a categorical axis we compute P_good(c) / P_bad(c) as empirical ratios
    (Laplace-smoothed) and sample from P_good; the candidate selection step
    picks the arg-max ratio among `n_ei_candidates` draws.

    For a continuous axis we fit a KDE over the good / bad pool and sample
    from l(x); we draw `n_ei_candidates` and pick arg-max l(x)/g(x).
    """

    def __init__(
        self,
        space: SearchSpace,
        seed: int | None = None,
        *,
        gamma: float = 0.25,         # top fraction → good pool
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        direction: str = "maximize",
    ):
        self._space = space
        self._rng = random.Random(seed)
        self._np = np.random.default_rng(seed)
        self._gamma = float(gamma)
        self._startup = int(n_startup_trials)
        self._n_cand = int(n_ei_candidates)
        self._history: list[tuple[dict, float]] = []
        self._direction = direction

    def tell(self, params: dict, score: float | None):
        if score is not None and math.isfinite(score):
            self._history.append((dict(params), float(score)))

    def ask(self, context: dict | None = None) -> dict[str, Any]:
        if len(self._history) < self._startup:
            return NativeRandomSampler(self._space, seed=self._rng.randint(0, 2**31)).ask(context)

        # Partition: top-γ are "good" (l), rest are "bad" (g).
        sorted_hist = sorted(self._history, key=lambda t: t[1],
                             reverse=(self._direction == "maximize"))
        n_good = max(1, int(math.ceil(self._gamma * len(sorted_hist))))
        good = sorted_hist[:n_good]
        bad = sorted_hist[n_good:]

        params: dict[str, Any] = {}
        for axis in self._space.active_axes(context):
            params[axis.name] = self._suggest_axis(axis, good, bad)
        return params

    def _suggest_axis(
        self, axis: Axis, good: list[tuple[dict, float]], bad: list[tuple[dict, float]]
    ) -> Any:
        g_vals = [t[0].get(axis.name) for t in good if axis.name in t[0]]
        b_vals = [t[0].get(axis.name) for t in bad if axis.name in t[0]]
        if not g_vals:
            return _sample_axis(axis, self._rng, self._np)

        if axis.kind in ("categorical", "bool"):
            choices = axis.values or [False, True]
            good_counts = {c: 0.5 for c in choices}  # Laplace smoothing
            bad_counts = {c: 0.5 for c in choices}
            for v in g_vals:
                good_counts[v] = good_counts.get(v, 0.5) + 1
            for v in b_vals:
                bad_counts[v] = bad_counts.get(v, 0.5) + 1
            total_g = sum(good_counts.values())
            total_b = sum(bad_counts.values())
            ratios = {c: (good_counts[c] / total_g) / (bad_counts[c] / total_b) for c in choices}
            # Sample from P_good, pick arg-max ratio among candidates.
            cands = self._rng.choices(choices,
                                      weights=[good_counts[c] / total_g for c in choices],
                                      k=self._n_cand)
            return max(cands, key=lambda c: ratios[c])

        # Continuous (float / log_uniform / int)
        g_arr = np.asarray([float(v) for v in g_vals])
        b_arr = np.asarray([float(v) for v in b_vals]) if b_vals else None
        if axis.kind == "log_uniform":
            g_arr = np.log(g_arr)
            if b_arr is not None:
                b_arr = np.log(b_arr)

        try:
            l_kde = gaussian_kde(g_arr)
        except Exception:  # single-value → fall back
            return _sample_axis(axis, self._rng, self._np)
        try:
            g_kde = gaussian_kde(b_arr) if b_arr is not None and len(b_arr) > 1 else None
        except Exception:
            g_kde = None

        samples = l_kde.resample(self._n_cand, seed=self._rng.randint(0, 2**31))[0]
        # clip to axis bounds
        lo = float(axis.low)
        hi = float(axis.high)
        if axis.kind == "log_uniform":
            lo, hi = math.log(lo), math.log(hi)
        samples = np.clip(samples, lo, hi)

        if g_kde is None:
            chosen = float(samples[self._rng.randrange(len(samples))])
        else:
            l_density = l_kde(samples)
            g_density = g_kde(samples) + 1e-12
            ratios = l_density / g_density
            chosen = float(samples[int(np.argmax(ratios))])

        if axis.kind == "log_uniform":
            chosen = math.exp(chosen)
        if axis.kind == "int":
            chosen = int(round(chosen))
            chosen = max(int(axis.low), min(int(axis.high), chosen))
        return chosen


# --- helpers --------------------------------------------------------------

def _sample_axis(axis: Axis, rng: random.Random, np_rng: np.random.Generator) -> Any:
    if axis.kind == "categorical":
        return rng.choice(list(axis.values or []))
    if axis.kind == "bool":
        return rng.random() >= 0.5
    if axis.kind == "int":
        step = int(axis.step) if axis.step else 1
        options = list(range(int(axis.low), int(axis.high) + 1, step))
        return rng.choice(options)
    if axis.kind == "float":
        return rng.uniform(float(axis.low), float(axis.high))
    if axis.kind == "log_uniform":
        lo, hi = math.log(float(axis.low)), math.log(float(axis.high))
        return math.exp(rng.uniform(lo, hi))
    raise ValueError(f"axis {axis.name}: unsupported kind {axis.kind}")


def _project_u(axis: Axis, u: float) -> Any:
    if axis.kind == "categorical":
        vals = list(axis.values or [])
        return vals[min(int(u * len(vals)), len(vals) - 1)]
    if axis.kind == "bool":
        return bool(u >= 0.5)
    if axis.kind == "int":
        low, high = int(axis.low), int(axis.high)
        step = int(axis.step) if axis.step else 1
        options = list(range(low, high + 1, step))
        return options[min(int(u * len(options)), len(options) - 1)]
    if axis.kind == "float":
        return float(axis.low) + u * (float(axis.high) - float(axis.low))
    if axis.kind == "log_uniform":
        lo, hi = math.log(float(axis.low)), math.log(float(axis.high))
        return math.exp(lo + u * (hi - lo))
    raise ValueError(f"axis {axis.name}: unsupported kind {axis.kind}")
