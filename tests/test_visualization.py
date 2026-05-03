from __future__ import annotations

from lmtune.runners.base import RequestRow
from lmtune.visualization import plot_ttft_vs_input_len, plot_ttft_vs_turn, render_run_report


def _make_turn_rows(n_conv=3, n_turn=5):
    rows = []
    for c in range(n_conv):
        for t in range(n_turn):
            rows.append(
                RequestRow(
                    req_id=f"c{c}-t{t}",
                    turn_idx=t,
                    conversation_id=str(c),
                    input_tokens=1000 + t * 1500,
                    output_tokens=300,
                    ttft_ms=200.0 + t * 150 + c * 20,
                    e2e_ms=2000.0 + t * 400,
                )
            )
    return rows


def test_plot_ttft_vs_turn_writes_png(tmp_path):
    rows = _make_turn_rows()
    out = plot_ttft_vs_turn(rows, tmp_path / "ttft.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_ttft_vs_input_len_writes_png(tmp_path):
    rows = _make_turn_rows()
    out = plot_ttft_vs_input_len(rows, tmp_path / "len.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_run_report_end_to_end(tmp_path):
    rows = _make_turn_rows()
    report = render_run_report(
        run_id="r1",
        profile_slug="profile-c-agent",
        endpoint_slug="local-vllm",
        metrics={"ttft": {"p50": 200.0, "p99": 800.0}, "e2e": {"p50": 2500.0}},
        rows=rows,
        out_dir=tmp_path / "out",
        ttft_slo_ms=500.0,
        e2e_slo_ms=3000.0,
    )
    assert report.exists()
    text = report.read_text()
    assert "profile-c-agent" in text
    assert "ttft_vs_turn" in text
    assert (report.parent / "ttft_vs_turn.png").exists()
