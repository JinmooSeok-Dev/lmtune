from __future__ import annotations

from bench.detectors import (
    detect_iqr_outliers,
    detect_regression,
    detect_slo_violations,
    run_all_rules,
)
from bench.profiles import SLOSpec
from bench.runners.base import RequestRow


def test_slo_violation_warning_and_critical():
    # 500 < observed <= 750 => warning (> threshold 이지만 1.5x 미만)
    dets = detect_slo_violations({"ttft": {"p99": 600.0}}, SLOSpec(ttft_p99_ms=500))
    assert any(d.severity == "warning" for d in dets)

    # observed > 1.5x threshold => critical
    dets = detect_slo_violations({"ttft": {"p99": 2000.0}}, SLOSpec(ttft_p99_ms=500))
    assert any(d.severity == "critical" for d in dets)


def test_slo_no_violation():
    metrics = {"ttft": {"p99": 400.0}, "e2e": {"p99": 20000.0}}
    assert not detect_slo_violations(metrics, SLOSpec(ttft_p99_ms=500, e2e_p99_ms=30000))


def test_slo_missing_metric_emits_info():
    dets = detect_slo_violations({}, SLOSpec(ttft_p99_ms=500))
    assert any(d.severity == "info" and "측정값 없음" in d.message for d in dets)


def test_regression_detection():
    baseline = {"ttft": {"p99": 300.0}}
    candidate = {"ttft": {"p99": 500.0}}  # +66%
    dets = detect_regression("b", "c", baseline, candidate, threshold_pct=10.0)
    assert dets and dets[0].severity == "critical"


def test_iqr_outliers_flagged():
    rows = [RequestRow(req_id=f"r{i}", ttft_ms=100 + i * 5) for i in range(30)]
    rows.append(RequestRow(req_id="outlier", ttft_ms=10_000.0))
    dets = detect_iqr_outliers(rows, attr="ttft_ms", factor=3.0)
    assert dets and "IQR" in dets[0].message


def test_run_all_rules_integrates():
    rows = [RequestRow(req_id=f"r{i}", ttft_ms=100 + i * 5, e2e_ms=1000) for i in range(30)]
    metrics = {"ttft": {"p50": 115.0, "p99": 190.0}, "e2e": {"p99": 1000.0}}
    slo = SLOSpec(ttft_p99_ms=150, e2e_p99_ms=2000)
    dets = run_all_rules(metrics, rows, slo)
    # ttft p99 > SLO 150 => violation expected
    assert any(d.detector == "slo" for d in dets)
