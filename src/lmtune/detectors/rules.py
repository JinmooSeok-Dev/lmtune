from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

from lmtune.analysis.compare import compare_runs
from lmtune.profiles import SLOSpec
from lmtune.runners.base import RequestRow

Severity = str  # "info" | "warning" | "critical"


@dataclass
class Detection:
    detector: str
    severity: Severity
    metric: str | None
    threshold: float | None
    observed: float | None
    message: str

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "severity": self.severity,
            "metric": self.metric,
            "threshold": self.threshold,
            "observed": self.observed,
            "message": self.message,
        }


_OPS = {
    "<=": lambda o, v: o <= v,
    "<":  lambda o, v: o < v,
    ">=": lambda o, v: o >= v,
    ">":  lambda o, v: o > v,
    "==": lambda o, v: o == v,
    "!=": lambda o, v: o != v,
}


def _severity_for(metric: str, observed: float, threshold: float, op: str, default: str) -> str:
    if default == "critical":
        return default
    # 기본 warning 인 경우, "위반 정도"가 1.5배 넘으면 critical 로 승격.
    if op in ("<=", "<") and threshold and observed > 1.5 * threshold:
        return "critical"
    if op in (">=", ">") and threshold and observed < 0.5 * threshold:
        return "critical"
    return default


def detect_slo_violations(
    metrics: dict[str, dict[str, float]], slo: SLOSpec
) -> list[Detection]:
    results: list[Detection] = []
    for chk in slo.resolved_checks():
        observed = (metrics.get(chk.metric) or {}).get(chk.p)
        metric_label = chk.label or f"{chk.metric}.{chk.p}"
        if observed is None:
            results.append(
                Detection(
                    detector="slo",
                    severity="info",
                    metric=f"{chk.metric}.{chk.p}",
                    threshold=chk.value,
                    observed=None,
                    message=f"{metric_label} 측정값 없음 (SLO {chk.op} {chk.value})",
                )
            )
            continue
        op_fn = _OPS[chk.op]
        if not op_fn(observed, chk.value):
            severity = _severity_for(chk.metric, observed, chk.value, chk.op, chk.severity)
            results.append(
                Detection(
                    detector="slo",
                    severity=severity,
                    metric=f"{chk.metric}.{chk.p}",
                    threshold=chk.value,
                    observed=observed,
                    message=f"{metric_label} SLO 위반: {observed:.3f} {chk.op} {chk.value} 불만족",
                )
            )
    return results


def detect_regression(
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_metrics: dict[str, dict[str, float]],
    candidate_metrics: dict[str, dict[str, float]],
    threshold_pct: float = 10.0,
) -> list[Detection]:
    cmp_ = compare_runs(
        baseline_run_id, candidate_run_id, baseline_metrics, candidate_metrics, threshold_pct
    )
    results: list[Detection] = []
    for d in cmp_.regressions:
        sev = "critical" if abs(d.delta_pct) > 2 * threshold_pct else "warning"
        results.append(
            Detection(
                detector="regression",
                severity=sev,
                metric=f"{d.metric}.{d.p}",
                threshold=threshold_pct,
                observed=d.delta_pct,
                message=(
                    f"{d.metric}[{d.p}] {d.baseline:.2f} → {d.candidate:.2f} "
                    f"({d.delta_pct:+.2f}% vs {baseline_run_id})"
                ),
            )
        )
    return results


def detect_iqr_outliers(
    rows: Iterable[RequestRow], attr: str = "ttft_ms", factor: float = 3.0
) -> list[Detection]:
    values = [getattr(r, attr) for r in rows if getattr(r, attr, None) is not None]
    values = [float(v) for v in values]
    if len(values) < 8:
        return []
    values_sorted = sorted(values)
    n = len(values_sorted)
    q1 = values_sorted[n // 4]
    q3 = values_sorted[(3 * n) // 4]
    iqr = q3 - q1
    upper = q3 + factor * iqr
    outliers = [v for v in values if v > upper]
    if not outliers:
        return []
    return [
        Detection(
            detector="iqr",
            severity="warning" if len(outliers) / n < 0.05 else "critical",
            metric=attr,
            threshold=upper,
            observed=max(outliers),
            message=(
                f"{attr}: {len(outliers)}/{n} 요청이 IQR×{factor} 임계 {upper:.1f} 초과 "
                f"(median={median(values):.1f}, max={max(outliers):.1f})"
            ),
        )
    ]


def run_all_rules(
    metrics: dict[str, dict[str, float]],
    rows: Iterable[RequestRow],
    slo: SLOSpec,
    baseline: tuple[str, dict[str, dict[str, float]]] | None = None,
    candidate_run_id: str | None = None,
    regression_threshold_pct: float = 10.0,
) -> list[Detection]:
    dets: list[Detection] = []
    dets.extend(detect_slo_violations(metrics, slo))
    rows = list(rows)
    dets.extend(detect_iqr_outliers(rows, attr="ttft_ms"))
    dets.extend(detect_iqr_outliers(rows, attr="e2e_ms"))
    if baseline and candidate_run_id:
        baseline_run_id, baseline_metrics = baseline
        dets.extend(
            detect_regression(
                baseline_run_id, candidate_run_id, baseline_metrics, metrics,
                threshold_pct=regression_threshold_pct,
            )
        )
    return dets
