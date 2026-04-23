from __future__ import annotations

from bench.analysis.aggregate import (
    aggregate,
    requests_to_dataframe,
    session_totals_from_requests,
)
from bench.analysis.distributions import (
    detect_bimodality,
    ecdf,
    fit_zipf_s,
    histogram,
    variance_stats,
)
from bench.analysis.nway import build_nway_table, variance_across_runs
from bench.runners.base import RequestRow


def _rows_for_variance():
    return [RequestRow(req_id=f"r{i}", ttft_ms=100 + (i % 5) * 10, e2e_ms=2000 + i * 20) for i in range(50)]


# ---------- Aggregate ----------


def test_requests_to_dataframe_has_agent_cols():
    rows = [
        RequestRow(req_id="a", turn_idx=0, input_tokens=100, cached_tokens=40, phase="exploration"),
        RequestRow(req_id="b", turn_idx=1, input_tokens=200, cached_tokens=160, phase="editing"),
    ]
    df = requests_to_dataframe(rows)
    assert {"cached_tokens", "phase", "thinking_tokens", "tool_call_count"} <= set(df.columns)
    assert df.loc[df.req_id == "a", "phase"].iloc[0] == "exploration"


def test_aggregate_group_by_turn():
    rows = []
    for conv in range(3):
        for turn in range(5):
            rows.append(RequestRow(
                req_id=f"c{conv}t{turn}", turn_idx=turn, conversation_id=str(conv),
                input_tokens=1000 + turn * 500, ttft_ms=200 + turn * 50,
            ))
    df = requests_to_dataframe(rows)
    out = aggregate(df, group_by=["turn_idx"], metrics=["ttft_ms"], aggs=("p50", "avg"))
    assert len(out) == 5
    assert "ttft_ms__p50" in out.columns
    # 턴이 커질수록 TTFT 증가
    assert out.loc[out.turn_idx == 4, "ttft_ms__p50"].iloc[0] > out.loc[out.turn_idx == 0, "ttft_ms__p50"].iloc[0]


def test_aggregate_with_buckets():
    rows = [RequestRow(req_id=f"r{i}", input_tokens=i * 100, ttft_ms=50 + i) for i in range(1, 30)]
    df = requests_to_dataframe(rows)
    out = aggregate(
        df, metrics=["ttft_ms"], aggs=("avg",),
        buckets={"input_tokens": [0, 500, 1500, 3500]},
    )
    assert "input_tokens_bucket" in out.columns
    assert len(out) == 3


def test_session_totals_from_requests():
    rows = [
        RequestRow(req_id="a", conversation_id="s1", turn_idx=0, input_tokens=100, output_tokens=50, cost_usd=0.01),
        RequestRow(req_id="b", conversation_id="s1", turn_idx=1, input_tokens=300, output_tokens=80, cost_usd=0.02),
        RequestRow(req_id="c", conversation_id="s2", turn_idx=0, input_tokens=200, output_tokens=40, cost_usd=0.015),
    ]
    df = requests_to_dataframe(rows)
    out = session_totals_from_requests(df)
    assert set(out["session_id"]) == {"s1", "s2"}
    s1 = out[out.session_id == "s1"].iloc[0]
    assert s1.total_input_tokens == 400
    assert s1.turn_count == 2


# ---------- Distributions ----------


def test_variance_stats_cv_computed():
    vs = variance_stats([100, 100, 100, 100])
    assert vs.cv == 0.0
    vs2 = variance_stats([50, 100, 200, 800])        # 큰 분산
    assert vs2.cv > 0.5


def test_histogram_and_ecdf():
    values = list(range(100))
    edges, counts = histogram(values, bins=10)
    assert sum(counts) == 100
    xs, fs = ecdf(values)
    assert fs[-1] == 1.0
    assert xs[0] <= xs[-1]


def test_fit_zipf_s_positive():
    # Zipf 유사 샘플 (작은 값이 훨씬 많음)
    values = []
    for rank in range(1, 50):
        values.extend([rank] * int(1000 / rank))
    s = fit_zipf_s(values)
    assert 0.5 < s < 3.0


def test_detect_bimodality_hint():
    # 50 근처 50개 + 500 근처 50개
    vals = [50 + (i % 5) for i in range(50)] + [500 + (i % 5) for i in range(50)]
    out = detect_bimodality(vals)
    assert out["is_bimodal_hint"] == 1.0


# ---------- NWay + variance ----------


def test_build_nway_table_pivots_correctly():
    run_metrics = {
        "r1": {"ttft": {"p99": 300}},
        "r2": {"ttft": {"p99": 500}},
        "r3": {"ttft": {"p99": 2000}},         # outlier
    }
    tbl = build_nway_table(run_metrics, metrics=["ttft"])
    assert list(tbl.df.columns) == ["r1", "r2", "r3"]
    assert ("ttft", "p99") in tbl.df.index


def test_variance_across_runs_captures_10x_spread():
    run_metrics = {
        f"r{i}": {"ttft": {"p99": v}}
        for i, v in enumerate([100, 120, 110, 105, 115, 108, 112, 95, 1000])
    }
    vs = variance_across_runs(run_metrics, "ttft", "p99")
    assert vs.max_ / vs.min_ > 9      # 10× 수준 편차 감지
    assert vs.cv > 1.0
