"""Per-axis one-way ANOVA on completed trials.

For each axis we test H0: the mean score is the same across all axis values.
A small p-value means at least one value's mean score differs significantly.

Recommendation rules (simple, conservative):
- p < 0.01 AND axis_has >= 2 populated groups
    → "freeze" at the group with the highest mean (high confidence it dominates)
- p >= 0.05
    → "drop"  candidate (axis doesn't explain score variance)
- otherwise → "keep"

Caveats: one-way ANOVA assumes within-group normality and equal variance. With
small N these assumptions are violated; treat p-values as a rough signal only.
Continuous axes are discretized into 4 quantile bins first (so float/log_uniform
can share the same recommendation surface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import f_oneway


@dataclass(slots=True)
class AxisAnova:
    axis: str
    groups: dict[str, list[float]] = field(default_factory=dict)
    f_stat: float | None = None
    p_value: float | None = None
    recommendation: str = "keep"   # keep | freeze | drop
    best_value: Any = None         # populated when recommendation == freeze


def _group_key(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _bin_continuous(values: list[float], n_bins: int = 4) -> list[str]:
    if not values:
        return []
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 2:
        return [f"{v:.4g}" for v in values]
    out: list[str] = []
    for v in values:
        idx = int(np.searchsorted(qs[1:-1], v, side="right"))
        out.append(f"q{idx}:{qs[idx]:.4g}-{qs[idx+1]:.4g}")
    return out


def anova_per_axis(
    trials: list[dict],
    *,
    p_freeze: float = 0.01,
    p_drop: float = 0.05,
    min_group_size: int = 2,
) -> list[AxisAnova]:
    """trials: list of {"params": {...}, "score": float, "status": str}"""
    completed = [t for t in trials if t.get("status") == "completed" and t.get("score") is not None]
    if not completed:
        return []
    axes = sorted({k for t in completed for k in (t.get("params") or {})})
    results: list[AxisAnova] = []
    for axis in axes:
        raw_values = [t["params"].get(axis) for t in completed if axis in t["params"]]
        scores = [float(t["score"]) for t in completed if axis in t["params"]]
        if not raw_values:
            continue
        # Decide categorization (discretize continuous-looking floats).
        if all(isinstance(v, float) for v in raw_values) and len(set(raw_values)) > 4:
            labels = _bin_continuous([float(v) for v in raw_values])
        else:
            labels = [_group_key(v) for v in raw_values]

        groups: dict[str, list[float]] = {}
        raw_per_group: dict[str, list[Any]] = {}
        for lbl, raw, sc in zip(labels, raw_values, scores, strict=False):
            groups.setdefault(lbl, []).append(sc)
            raw_per_group.setdefault(lbl, []).append(raw)

        usable = {k: v for k, v in groups.items() if len(v) >= min_group_size}
        if len(usable) < 2:
            results.append(
                AxisAnova(axis=axis, groups=groups, recommendation="keep")
            )
            continue

        f, p = f_oneway(*usable.values())
        rec = "keep"
        best_value: Any = None
        if p < p_freeze:
            rec = "freeze"
            # Pick the group with the highest mean; use the first raw value in it.
            best_label = max(usable.keys(), key=lambda k: float(np.mean(usable[k])))
            best_value = raw_per_group[best_label][0]
        elif p >= p_drop:
            rec = "drop"
        results.append(
            AxisAnova(
                axis=axis,
                groups=groups,
                f_stat=float(f) if f == f else None,
                p_value=float(p) if p == p else None,
                recommendation=rec,
                best_value=best_value,
            )
        )
    return results
