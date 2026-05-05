"""Tighten continuous axis bounds around the best-scoring region.

For each float / log_uniform axis, take the top-k trials (by score) and
compute the value's mean ± k·σ, intersected with the current axis bounds.
If the resulting window shrinks by > `min_shrink_frac`, recommend.

This is a greedy "exploit" step: use it once the signal is clear, not on
tiny studies. Default: top 25% trials, k=1 sigma.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def tighten_bounds(
    trials: list[dict],
    axes: list[dict],  # SearchSpace.axes as dicts (name, kind, low, high)
    *,
    top_frac: float = 0.25,
    k_sigma: float = 1.0,
    min_trials: int = 8,
    min_shrink_frac: float = 0.2,
) -> dict[str, dict[str, Any]]:
    completed = sorted(
        [t for t in trials if t.get("status") == "completed" and t.get("score") is not None],
        key=lambda t: float(t["score"]),
        reverse=True,
    )
    if len(completed) < min_trials:
        return {}
    top_n = max(4, int(len(completed) * top_frac))
    top = completed[:top_n]

    out: dict[str, dict[str, Any]] = {}
    for axis in axes:
        if axis["kind"] not in ("float", "log_uniform"):
            continue
        vals = [t["params"].get(axis["name"]) for t in top if axis["name"] in t.get("params", {})]
        vals = [float(v) for v in vals if v is not None]
        if len(vals) < 3:
            continue
        low, high = float(axis["low"]), float(axis["high"])
        span = high - low
        if span <= 0:
            continue

        if axis["kind"] == "log_uniform":
            lv = [math.log(v) for v in vals]
            mu, sigma = float(np.mean(lv)), float(np.std(lv))
            new_low = max(math.log(low), mu - k_sigma * sigma)
            new_high = min(math.log(high), mu + k_sigma * sigma)
            new_low, new_high = math.exp(new_low), math.exp(new_high)
            rel_shrink = 1.0 - (math.log(new_high) - math.log(new_low)) / (
                math.log(high) - math.log(low)
            )
        else:
            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            new_low = max(low, mu - k_sigma * sigma)
            new_high = min(high, mu + k_sigma * sigma)
            rel_shrink = 1.0 - (new_high - new_low) / span

        if rel_shrink >= min_shrink_frac and new_high > new_low:
            out[axis["name"]] = {
                "old_low": low,
                "old_high": high,
                "new_low": new_low,
                "new_high": new_high,
                "shrink_frac": rel_shrink,
                "mean_top": mu if axis["kind"] != "log_uniform" else math.exp(mu),
                "sigma_top": sigma,
                "recommendation": "shrink",
            }
    return out
