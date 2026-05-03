"""Space-narrowing analyses (Phase S2).

Offline, history-based techniques that recommend `freeze`, `drop`, or `shrink`
for each axis based on the study's completed trials.

- anova.py       — per-axis one-way ANOVA: if F is large and p is small, that
                   axis dominates the score variance → keep, else potentially
                   drop. Conversely a tight cluster of best trials at one axis
                   value → recommend `freeze` to that value.
- importance.py  — RandomForestRegressor feature_importances_ on (params → score).
                   Low importance → candidate for `drop`.
- bound_tighten.py — continuous axes: recommend new [low, high] around the best
                     ± k·σ window so subsequent studies sample densely where
                     the signal is.

These tools are composable: the recommendations they emit are merged in
`cli_search.py` for the `bench search prune` subcommand.
"""

from lmtune.search.analysis.anova import AxisAnova, anova_per_axis
from lmtune.search.analysis.bound_tighten import tighten_bounds
from lmtune.search.analysis.importance import axis_importance

__all__ = ["AxisAnova", "anova_per_axis", "axis_importance", "tighten_bounds"]
