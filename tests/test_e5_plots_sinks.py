from __future__ import annotations

import pandas as pd

from lmtune.profiles import AnalysisSpec, PlotRequest, ProfileSpec
from lmtune.runners.base import RequestRow
from lmtune.visualization import list_plots, list_sinks, render_run_report, sink_write


def _rows_phase_mix():
    rows = []
    for i in range(20):
        phase = ("exploration", "editing", "execution", "verification")[i % 4]
        rows.append(RequestRow(
            req_id=f"r{i}", conversation_id=str(i // 5), turn_idx=i % 5,
            input_tokens=500 + i * 100, output_tokens=50 + i * 5,
            phase=phase, ttft_ms=200 + i * 10, e2e_ms=2000 + i * 50,
        ))
    return rows


# ---------- Plot registry ----------


def test_plot_registry_contains_new_plots():
    kinds = set(list_plots())
    for kind in ("ttft_vs_turn", "ttft_vs_input_len", "cdf", "histogram",
                 "phase_breakdown", "token_snowball", "variance_box"):
        assert kind in kinds, f"missing plot: {kind}"


def test_phase_breakdown_plot_generates(tmp_path):
    from lmtune.visualization.plots import get_plot
    fn = get_plot("phase_breakdown")
    out = fn(_rows_phase_mix(), tmp_path / "phase.png", metric="input_tokens")
    assert out.exists() and out.stat().st_size > 0


def test_cdf_plot_generates(tmp_path):
    from lmtune.visualization.plots import get_plot
    fn = get_plot("cdf")
    out = fn(_rows_phase_mix(), tmp_path / "cdf.png", metric="ttft_ms")
    assert out.exists() and out.stat().st_size > 0


def test_token_snowball_plot_generates(tmp_path):
    from lmtune.visualization.plots import get_plot
    fn = get_plot("token_snowball")
    out = fn(_rows_phase_mix(), tmp_path / "snowball.png")
    assert out.exists()


def test_variance_box_plot_generates(tmp_path):
    from lmtune.visualization.plots import get_plot
    fn = get_plot("variance_box")
    out = fn({f"run{i}": [100 + i * 10 + j for j in range(20)] for i in range(5)},
             tmp_path / "box.png")
    assert out.exists()


# ---------- Sinks ----------


def test_list_sinks_has_formats():
    assert {"csv", "parquet", "json", "markdown", "html", "jupyter"} <= set(list_sinks())


def test_sink_csv_writes(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = sink_write("csv", df, tmp_path / "x.csv")
    assert out.exists()
    assert "a,b" in out.read_text()


def test_sink_html_writes(tmp_path):
    df = pd.DataFrame({"metric": ["ttft"], "p99": [450.0]})
    out = sink_write("html", df, tmp_path / "x.html", title="demo")
    text = out.read_text()
    assert "<table" in text and "demo" in text


def test_sink_parquet_writes(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = sink_write("parquet", df, tmp_path / "x.parquet")
    assert out.exists()


def test_sink_jupyter_writes(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    out = sink_write("jupyter", df, tmp_path / "nb.ipynb", title="demo")
    text = out.read_text()
    assert "cells" in text


# ---------- AnalysisSpec + report 통합 ----------


def test_profile_accepts_analysis_spec():
    p = ProfileSpec.model_validate({
        "slug": "with-analysis", "name": "x", "stage": 3,
        "runner": "aiperf", "mode": "concurrency",
        "workload": {
            "synthetic_input_tokens_mean": 1000, "output_tokens_mean": 200,
            "concurrency": 1, "request_count": 5,
        },
        "analysis": {
            "group_by": ["turn_idx"],
            "metrics": ["ttft", "e2e"],
            "plots": [
                {"kind": "ttft_vs_turn"},
                {"kind": "cdf", "metric": "ttft_ms"},
                {"kind": "phase_breakdown", "metric": "input_tokens"},
            ],
            "sinks": ["markdown", "csv", "html"],
            "derived": [{"name": "prefix_hit_rate"}],
        },
    })
    assert isinstance(p.analysis, AnalysisSpec)
    assert len(p.analysis.plots) == 3
    assert p.analysis.sinks == ["markdown", "csv", "html"]


def test_report_honors_analysis_plots_and_sinks(tmp_path):
    spec = AnalysisSpec(
        group_by=[],
        metrics=["ttft"],
        plots=[
            PlotRequest(kind="cdf", metric="ttft_ms"),
            PlotRequest(kind="phase_breakdown", metric="input_tokens"),
        ],
        sinks=["markdown", "csv", "html"],
    )
    out = render_run_report(
        run_id="r-e5",
        profile_slug="x",
        endpoint_slug="y",
        metrics={"ttft": {"p99": 500.0}},
        rows=_rows_phase_mix(),
        out_dir=tmp_path / "rpt",
        analysis=spec,
    )
    assert out.exists()
    text = out.read_text()
    assert "cdf" in text
    assert "phase_breakdown" in text
    assert (out.parent / "cdf.png").exists()
    assert (out.parent / "phase_breakdown.png").exists()
    assert (out.parent / "requests.csv").exists()
    assert (out.parent / "requests.html").exists()
