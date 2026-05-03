"""Request row DataFrame 기반 group-by 집계.

Jupyter/외부 스크립트에서 그대로 재사용할 수 있도록 Pandas DataFrame 을 반환.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from bench.runners.base import RequestRow


_DEFAULT_AGGS = ("p50", "p95", "p99", "avg", "min", "max", "count")


def requests_to_dataframe(rows: Iterable[RequestRow]) -> pd.DataFrame:
    records = [
        {
            "req_id": r.req_id, "turn_idx": r.turn_idx, "conversation_id": r.conversation_id,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cached_tokens": r.cached_tokens, "thinking_tokens": r.thinking_tokens,
            "tool_call_count": r.tool_call_count, "tool_result_tokens": r.tool_result_tokens,
            "phase": r.phase, "role": r.role,
            "energy_wh": r.energy_wh, "cost_usd": r.cost_usd,
            "ttft_ms": r.ttft_ms, "itl_mean_ms": r.itl_mean_ms, "e2e_ms": r.e2e_ms,
            "started_at": r.started_at, "completed_at": r.completed_at,
            "status": r.status, "error": r.error,
        }
        for r in rows
    ]
    return pd.DataFrame.from_records(records)


def aggregate(
    df: pd.DataFrame,
    group_by: Sequence[str] | None = None,
    metrics: Sequence[str] = ("ttft_ms", "e2e_ms"),
    aggs: Sequence[str] = _DEFAULT_AGGS,
    buckets: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """지정된 column 들로 group-by + metrics 각각에 대해 aggregation.

    `buckets`={"input_tokens": [0, 1000, 4000, 16000]} 로 넘기면 해당 컬럼을
    구간화해서 group key 로 사용 (예: input_token_bucket).
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    if buckets:
        for col, edges in buckets.items():
            if col not in work.columns:
                continue
            cat_col = f"{col}_bucket"
            work[cat_col] = pd.cut(work[col], bins=edges, include_lowest=True)
            if group_by is None:
                group_by = [cat_col]
            elif cat_col not in group_by:
                group_by = [*group_by, cat_col]

    agg_funcs: dict[str, list] = {}
    for m in metrics:
        if m not in work.columns:
            continue
        agg_funcs[m] = [_agg_fn(a) for a in aggs if _agg_fn(a) is not None]

    if not agg_funcs:
        return pd.DataFrame()

    if group_by:
        grouped = work.groupby(list(group_by), dropna=False).agg(agg_funcs)
    else:
        # 전체 집계 → 1-row DataFrame
        grouped = work.agg(agg_funcs)
    grouped.columns = ["__".join([c for c in col if c]).rstrip("_") for col in grouped.columns.values]
    return grouped.reset_index()


def _agg_fn(name: str):
    name = name.lower()
    if name in ("p50", "p95", "p99"):
        q = float(name[1:]) / 100.0
        def qfn(x, _q=q):
            xs = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if not xs:
                return math.nan
            xs = sorted(xs)
            idx = int(_q * (len(xs) - 1))
            return xs[idx]
        qfn.__name__ = name
        return qfn
    if name == "avg":
        return "mean"
    if name == "count":
        return "count"
    if name in ("min", "max", "std", "median", "sum"):
        return name
    return None


def session_totals_from_requests(df: pd.DataFrame) -> pd.DataFrame:
    """requests → session_id 별 total_*/turn_count 집계 (E1 sessions 테이블 대체용)."""
    if df.empty or "conversation_id" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("conversation_id", dropna=True).agg(
        total_input_tokens=("input_tokens", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        total_cached_tokens=("cached_tokens", "sum"),
        turn_count=("turn_idx", "nunique"),
        tool_call_count=("tool_call_count", "sum"),
        total_cost_usd=("cost_usd", "sum"),
        total_energy_wh=("energy_wh", "sum"),
        errors=("error", lambda s: s.notna().sum()),
    )
    g["success"] = g["errors"] == 0
    return g.reset_index().rename(columns={"conversation_id": "session_id"})
