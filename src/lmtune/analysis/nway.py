"""N-way run 비교 + repeat-run variance (OpenHands 10× 편차 검증).

DB 접근 없이 dict 모음을 받도록 설계 — CLI 가 DuckDB 에서 조회해서 전달한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from lmtune.analysis.distributions import VarianceStats, variance_stats


@dataclass
class NWayTable:
    df: pd.DataFrame            # index: (metric, p), columns: run_id 들
    units: dict[str, str]


def build_nway_table(
    run_metrics: dict[str, dict[str, dict[str, float]]],
    metrics: Sequence[str] | None = None,
    percentiles: Sequence[str] = ("p50", "p95", "p99", "avg"),
) -> NWayTable:
    """`run_id → metric → p → value` → 피벗 DataFrame."""
    rows: list[tuple[str, str, str, float]] = []
    for run_id, mm in run_metrics.items():
        for metric, bucket in mm.items():
            if metrics and metric not in metrics:
                continue
            for p, v in bucket.items():
                if p not in percentiles:
                    continue
                rows.append((metric, p, run_id, float(v)))
    if not rows:
        return NWayTable(df=pd.DataFrame(), units={})
    df = pd.DataFrame(rows, columns=["metric", "p", "run_id", "value"])
    pivot = df.pivot_table(index=["metric", "p"], columns="run_id", values="value")
    return NWayTable(df=pivot, units={})


def variance_across_runs(
    run_metrics: dict[str, dict[str, dict[str, float]]],
    metric: str,
    p: str = "p99",
) -> VarianceStats:
    """같은 metric/percentile 의 N-run 분산 통계 (10× 편차 검증용)."""
    values = []
    for _run_id, mm in run_metrics.items():
        v = (mm.get(metric) or {}).get(p)
        if v is not None:
            values.append(float(v))
    return variance_stats(values)


def nway_to_markdown(table: NWayTable, title: str = "N-way Comparison") -> str:
    if table.df.empty:
        return f"# {title}\n\n(no metrics)\n"
    lines = [f"# {title}", "", table.df.to_markdown(), ""]
    return "\n".join(lines)
