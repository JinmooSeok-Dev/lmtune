"""Global sensitivity analysis via Sobol indices (SALib).

Sobol indices decompose output variance into contributions from each input
(first-order, S1) and from each input + all its interactions (total-order, ST).
A large gap between S1 and ST means the input interacts strongly with others.

Unlike our Phase S2 `anova.py` (pairwise groupwise p-value) or
`importance.py` (tree-based), Sobol is a proper variance-based decomposition
that (a) handles continuous axes natively, (b) quantifies interactions, and
(c) gives confidence intervals via bootstrap.

Caveats:
- Sobol requires a specific Saltelli sample design. We can either (i) generate
  the exact `saltelli.sample(problem, N)` set and evaluate objectively — best
  accuracy, costly; or (ii) **post-hoc** fit a RandomForest surrogate on the
  existing trial history and apply Sobol to the surrogate — cheap, approximate.
  We default to (ii) below because studies already spend their budget on TPE/
  NSGA-II, not on a separate Saltelli grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from SALib.analyze.sobol import analyze as sobol_analyze
    from SALib.sample.sobol import sample as sobol_sample
except Exception:  # pragma: no cover
    sobol_analyze = None
    sobol_sample = None

from sklearn.ensemble import RandomForestRegressor


@dataclass(slots=True)
class SobolResult:
    axis: str
    S1: float                    # first-order
    ST: float                    # total-order (includes interactions)
    S1_conf: float               # 95% CI half-width
    ST_conf: float
    interaction_gap: float       # ST - S1: large ⇒ interactions dominate


def _pick_continuous(trials: list[dict], axes_spec: list[dict]) -> list[dict]:
    """Retain only continuous (float / log_uniform / int with step) axes — Sobol
    operates on hyperbox domains."""
    out = []
    for a in axes_spec:
        if a.get("kind") in ("float", "log_uniform") or a.get("kind") == "int" and a.get("low") is not None and a.get("high") is not None:
            out.append(a)
    return out


def sobol_from_history(
    trials: list[dict],
    axes_spec: list[dict],
    *,
    n_saltelli: int = 1024,
    surrogate_n_estimators: int = 300,
    seed: int = 0,
) -> list[SobolResult]:
    """Post-hoc Sobol:
       1. fit a RandomForest (params → score) on completed trials,
       2. generate a Saltelli sample over the continuous axes,
       3. evaluate surrogate + analyze.

    Returns one SobolResult per continuous axis.
    """
    if sobol_analyze is None or sobol_sample is None:
        raise RuntimeError("SALib is not installed (pip install -e '.[search]').")

    completed = [t for t in trials if t.get("status") == "completed" and t.get("score") is not None]
    if len(completed) < 5:
        return []

    cont = _pick_continuous(trials, axes_spec)
    if not cont:
        return []

    # 1) Fit surrogate over the full param space; Sobol only queries continuous dims.
    df = pd.DataFrame([t["params"] for t in completed])
    y = np.asarray([float(t["score"]) for t in completed])
    obj_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_bool_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]
    df_enc = pd.get_dummies(df, columns=obj_cols, drop_first=False) if obj_cols else df.copy()
    for c in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[c]):
            df_enc = df_enc.drop(columns=[c])
    rf = RandomForestRegressor(n_estimators=surrogate_n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(df_enc.values, y)

    # 2) Saltelli sample on continuous axes; for other axes hold fixed at their mean.
    problem = {
        "num_vars": len(cont),
        "names": [a["name"] for a in cont],
        "bounds": [[float(a["low"]), float(a["high"])] for a in cont],
    }
    X = sobol_sample(problem, N=int(n_saltelli), calc_second_order=False, seed=seed)

    # Build a full-feature matrix (mean-fill the non-continuous columns).
    fixed = {}
    for c in df_enc.columns:
        fixed[c] = float(df_enc[c].mean()) if len(df_enc) else 0.0

    cont_names = [a["name"] for a in cont]
    full = np.tile(np.asarray([fixed[c] for c in df_enc.columns]), (X.shape[0], 1))
    for i, n in enumerate(cont_names):
        if n in df_enc.columns:
            full[:, list(df_enc.columns).index(n)] = X[:, i]
    Y = rf.predict(full)

    # 3) Analyze
    Si = sobol_analyze(problem, Y, calc_second_order=False, print_to_console=False)

    results: list[SobolResult] = []
    for i, name in enumerate(cont_names):
        results.append(SobolResult(
            axis=name,
            S1=float(Si["S1"][i]),
            ST=float(Si["ST"][i]),
            S1_conf=float(Si["S1_conf"][i]),
            ST_conf=float(Si["ST_conf"][i]),
            interaction_gap=float(Si["ST"][i] - Si["S1"][i]),
        ))
    return results
