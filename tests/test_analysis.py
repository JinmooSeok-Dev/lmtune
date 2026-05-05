from __future__ import annotations

from lmtune.analysis import compare_runs
from lmtune.analysis.metrics import percentiles, summarize_requests
from lmtune.runners.base import RequestRow


def test_percentiles_small_and_large():
    assert percentiles([]) == {}
    one = percentiles([100.0])
    assert one["p50"] == one["p99"] == one["avg"] == 100.0
    many = percentiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
    assert many["p50"] > 0
    assert many["p99"] >= many["p95"] >= many["p50"]
    assert many["avg"] > 0


def test_summarize_requests_goodput():
    rows = [
        RequestRow(req_id=f"r{i}", ttft_ms=100 + i * 10, e2e_ms=500 + i * 50) for i in range(10)
    ]
    s = summarize_requests(rows, ttft_slo_ms=150, e2e_slo_ms=800)
    assert s.total_requests == 10
    assert 0 < s.goodput_ratio <= 1
    assert s.slo_ttft_violations >= 4


def test_compare_runs_detects_regression():
    baseline = {"ttft": {"p50": 100.0, "p99": 300.0}, "throughput_tok": {"avg": 2000.0}}
    candidate = {"ttft": {"p50": 120.0, "p99": 400.0}, "throughput_tok": {"avg": 1500.0}}
    cmp_ = compare_runs("b", "c", baseline, candidate, regression_threshold_pct=10.0)
    names = {(d.metric, d.p) for d in cmp_.regressions}
    assert ("ttft", "p50") in names  # 20% 증가 → 회귀
    assert ("ttft", "p99") in names  # 33% 증가 → 회귀
    assert ("throughput_tok", "avg") in names  # 25% 감소 → 회귀
    assert "Regressions" in cmp_.to_markdown()
