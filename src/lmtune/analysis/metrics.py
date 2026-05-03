from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean, quantiles

from lmtune.runners.base import RequestRow

_P = [0.50, 0.95, 0.99]


def percentiles(values: Sequence[float]) -> dict[str, float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {}
    if len(xs) == 1:
        v = xs[0]
        return {"p50": v, "p95": v, "p99": v, "avg": v}
    qs = quantiles(xs, n=100, method="inclusive")
    out = {
        "p50": qs[49],
        "p95": qs[94],
        "p99": qs[98],
        "avg": mean(xs),
        "min": min(xs),
        "max": max(xs),
    }
    return out


@dataclass
class RequestSummary:
    ttft: dict[str, float]
    itl: dict[str, float]
    e2e: dict[str, float]
    goodput_ratio: float | None = None
    slo_ttft_violations: int = 0
    slo_e2e_violations: int = 0
    total_requests: int = 0


def summarize_requests(
    rows: Iterable[RequestRow],
    ttft_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> RequestSummary:
    rows = list(rows)
    ttfts = [r.ttft_ms for r in rows if r.ttft_ms is not None]
    itls = [r.itl_mean_ms for r in rows if r.itl_mean_ms is not None]
    e2es = [r.e2e_ms for r in rows if r.e2e_ms is not None]

    slo_ttft_vio = 0
    slo_e2e_vio = 0
    goodput = None
    if rows and (ttft_slo_ms is not None or e2e_slo_ms is not None):
        success = 0
        for r in rows:
            ok = True
            if ttft_slo_ms is not None and (r.ttft_ms is None or r.ttft_ms > ttft_slo_ms):
                ok = False
                slo_ttft_vio += 1
            if e2e_slo_ms is not None and (r.e2e_ms is None or r.e2e_ms > e2e_slo_ms):
                ok = False
                slo_e2e_vio += 1
            if ok:
                success += 1
        goodput = success / len(rows)

    return RequestSummary(
        ttft=percentiles(ttfts),
        itl=percentiles(itls),
        e2e=percentiles(e2es),
        goodput_ratio=goodput,
        slo_ttft_violations=slo_ttft_vio,
        slo_e2e_violations=slo_e2e_vio,
        total_requests=len(rows),
    )
