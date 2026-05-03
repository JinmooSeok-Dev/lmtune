"""Feature importance via RandomForestRegressor.

Fit a Random Forest on (one-hot / numeric encoded params → score) and read
`feature_importances_`. Sum per-axis importance across its one-hot levels.

Low importance (< threshold) → `drop` candidate.

Caveat: importance is conditional on the sample distribution. If the study
only explored a corner of the space, axes that matter elsewhere can look
unimportant here. Treat this as a cheap heuristic, not a proof.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def axis_importance(
    trials: list[dict],
    *,
    drop_threshold: float = 0.05,
    n_estimators: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Return {axis: {"importance": float, "recommendation": keep|drop}}."""
    completed = [
        t for t in trials
        if t.get("status") == "completed" and t.get("score") is not None
    ]
    if len(completed) < 5:
        return {}

    df = pd.DataFrame([t["params"] for t in completed])
    y = np.asarray([float(t["score"]) for t in completed])

    # Encode: bool/string categorical → get_dummies; numeric axes stay as-is.
    obj_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_bool_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]
    df_enc = pd.get_dummies(df, columns=obj_cols, drop_first=False) if obj_cols else df.copy()
    # Any remaining non-numeric columns would break the RF fit; coerce or drop.
    for c in df_enc.columns:
        if not pd.api.types.is_numeric_dtype(df_enc[c]):
            df_enc = df_enc.drop(columns=[c])

    if df_enc.empty or df_enc.shape[1] == 0:
        return {}

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(df_enc.values, y)
    feat_imp = dict(zip(df_enc.columns, rf.feature_importances_, strict=False))

    # Re-aggregate by original axis (sum across one-hot levels).
    per_axis: dict[str, float] = {c: 0.0 for c in df.columns}
    for feat, imp in feat_imp.items():
        for c in df.columns:
            if feat == c or feat.startswith(f"{c}_"):
                per_axis[c] += float(imp)
                break

    out: dict[str, dict[str, Any]] = {}
    for axis, imp in per_axis.items():
        out[axis] = {
            "importance": imp,
            "recommendation": "drop" if imp < drop_threshold else "keep",
        }
    return out
