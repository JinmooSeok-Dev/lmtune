"""분포 분석 — histogram / CDF / variance / Zipf·lognormal fit (단순 추정).

scipy/sklearn 없이도 돌도록 pure python/numpy 로 구현. 정밀 fit 이 필요하면
각 함수 내부에서 `scipy.stats` 로 교체.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class VarianceStats:
    n: int
    mean: float
    std: float
    cv: float  # Coefficient of Variation
    min_: float
    max_: float
    p50: float
    iqr: float
    iqr_ratio: float  # IQR / median


def variance_stats(values: Iterable[float]) -> VarianceStats:
    xs = sorted(float(v) for v in values if v is not None and not _nan(v))
    n = len(xs)
    if n == 0:
        return VarianceStats(
            0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
        )
    mean = sum(xs) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / n) if n > 1 else 0.0
    p50 = xs[n // 2]
    q1 = xs[n // 4]
    q3 = xs[(3 * n) // 4]
    iqr = q3 - q1
    return VarianceStats(
        n=n,
        mean=mean,
        std=std,
        cv=(std / mean) if mean else math.nan,
        min_=xs[0],
        max_=xs[-1],
        p50=p50,
        iqr=iqr,
        iqr_ratio=(iqr / p50) if p50 else math.nan,
    )


def histogram(values: Iterable[float], bins: int = 30) -> tuple[list[float], list[int]]:
    xs = [float(v) for v in values if v is not None and not _nan(v)]
    if not xs:
        return [], []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [lo, lo + 1e-9], [len(xs)]
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for x in xs:
        i = min(int((x - lo) / width), bins - 1)
        counts[i] += 1
    return edges, counts


def ecdf(values: Iterable[float]) -> tuple[list[float], list[float]]:
    """Empirical CDF: (x, F(x)) 페어."""
    xs = sorted(float(v) for v in values if v is not None and not _nan(v))
    n = len(xs)
    if n == 0:
        return [], []
    return xs, [(i + 1) / n for i in range(n)]


def fit_zipf_s(values: Iterable[int]) -> float:
    """매우 단순한 Zipf shape 추정: `rank * freq = C` 에서 log-log 회귀."""
    from collections import Counter

    xs = [int(v) for v in values if v and v >= 1]
    if len(xs) < 10:
        return math.nan
    freq = Counter(xs)
    pairs = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    # log(rank) vs log(freq) 최소제곱 기울기의 음수
    logs = [(math.log(i + 1), math.log(f)) for i, (_, f) in enumerate(pairs) if f > 0]
    if len(logs) < 3:
        return math.nan
    n = len(logs)
    mx = sum(a for a, _ in logs) / n
    my = sum(b for _, b in logs) / n
    num = sum((a - mx) * (b - my) for a, b in logs)
    den = sum((a - mx) ** 2 for a, _ in logs)
    slope = num / den if den else math.nan
    return -slope


def detect_bimodality(values: Iterable[float]) -> dict[str, float]:
    """단순 bimodality 검출: Hartigan 없이 skewness·kurtosis·peak gap 으로 근사.

    반환: {is_bimodal_hint, peak_gap_ratio, modes_estimate}
    """
    xs = sorted(float(v) for v in values if v is not None and not _nan(v))
    n = len(xs)
    if n < 30:
        return {"is_bimodal_hint": 0.0, "peak_gap_ratio": 0.0}
    edges, counts = histogram(xs, bins=40)
    if not counts:
        return {"is_bimodal_hint": 0.0, "peak_gap_ratio": 0.0}
    # 두 최대 bin 사이 gap 탐지
    sorted_bins = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    i1 = sorted_bins[0]
    i2 = next((i for i in sorted_bins[1:] if abs(i - i1) > 3), None)
    if i2 is None:
        return {"is_bimodal_hint": 0.0, "peak_gap_ratio": 0.0}
    gap = abs(i2 - i1) / len(counts)
    return {
        "is_bimodal_hint": float(gap > 0.2),
        "peak_gap_ratio": gap,
        "mode1_center": (edges[i1] + edges[i1 + 1]) / 2,
        "mode2_center": (edges[i2] + edges[i2 + 1]) / 2,
    }


def _nan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False
