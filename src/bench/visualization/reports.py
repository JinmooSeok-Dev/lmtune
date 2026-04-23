from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from bench.analysis.aggregate import aggregate, requests_to_dataframe
from bench.analysis.metrics import summarize_requests
from bench.profiles import AnalysisSpec
from bench.runners.base import RequestRow
from bench.visualization.plots import get_plot
from bench.visualization.sinks import write as sink_write


def render_run_report(
    run_id: str,
    profile_slug: str,
    endpoint_slug: str,
    metrics: dict[str, dict[str, float]],
    rows: Iterable[RequestRow],
    out_dir: str | Path,
    analysis: AnalysisSpec | None = None,
    ttft_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> Path:
    rows = list(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_requests(rows, ttft_slo_ms=ttft_slo_ms, e2e_slo_ms=e2e_slo_ms)
    df = requests_to_dataframe(rows)

    # 1. Plot 생성 — AnalysisSpec.plots 또는 기본 2종
    plot_specs = []
    if analysis and analysis.plots:
        plot_specs = [(p.kind, p.metric, p.title, p.opts) for p in analysis.plots]
    else:
        plot_specs = [
            ("ttft_vs_turn", None, None, {}),
            ("ttft_vs_input_len", None, None, {}),
        ]

    generated_plots: list[tuple[str, Path]] = []
    for kind, metric, title, opts in plot_specs:
        fn = get_plot(kind)
        if fn is None:
            continue
        kwargs = dict(opts)
        if metric:
            kwargs["metric"] = metric
        if title:
            kwargs["title"] = title
        target = out_dir / f"{kind}.png"
        try:
            path = fn(rows, target, **kwargs)
            generated_plots.append((kind, path))
        except (ValueError, Exception):   # 데이터 부족 등
            continue

    # 2. Markdown 본문
    md: list[str] = []
    md.append(f"# Run Report — `{run_id}`")
    md.append("")
    md.append(f"- **Profile**: `{profile_slug}`")
    md.append(f"- **Endpoint**: `{endpoint_slug}`")
    md.append(f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- **Total requests (rows)**: {summary.total_requests}")
    if summary.goodput_ratio is not None:
        md.append(f"- **Goodput (SLO-내 완료율)**: {summary.goodput_ratio * 100:.2f}%")
    md.append("")
    md.append("## Metrics")
    md.append("| metric | p50 | p95 | p99 | avg |")
    md.append("|:-------|----:|----:|----:|----:|")
    for name in sorted(metrics):
        b = metrics[name]
        md.append(f"| {name} | {_f(b.get('p50'))} | {_f(b.get('p95'))} | {_f(b.get('p99'))} | {_f(b.get('avg'))} |")
    md.append("")

    if analysis and analysis.group_by:
        agg_df = aggregate(df, group_by=analysis.group_by,
                           metrics=[f"{m}_ms" if m in ("ttft", "itl", "e2e") else m for m in (analysis.metrics or ["ttft_ms"])],
                           aggs=tuple(analysis.percentiles or ["p50", "p95", "p99"]),
                           buckets=analysis.buckets)
        if not agg_df.empty:
            md.append(f"## Aggregation (group_by={analysis.group_by})")
            md.append("")
            md.append(agg_df.to_markdown(index=False))
            md.append("")

    for kind, path in generated_plots:
        md.append(f"## {kind}")
        md.append("")
        md.append(f"![{kind}]({path.name})")
        md.append("")

    if summary.total_requests and (ttft_slo_ms or e2e_slo_ms):
        md.append("## SLO Violations")
        md.append("")
        md.append(f"- TTFT > {ttft_slo_ms} ms: {summary.slo_ttft_violations}")
        md.append(f"- E2E > {e2e_slo_ms} ms: {summary.slo_e2e_violations}")
        md.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    # 3. Sinks (raw DataFrame export)
    sinks = (analysis.sinks if analysis else None) or ["markdown"]
    for s in sinks:
        if s == "markdown":
            continue            # 이미 report.md
        ext = {"csv": "csv", "parquet": "parquet", "json": "json", "html": "html", "jupyter": "ipynb"}.get(s, s)
        try:
            sink_write(s, df, out_dir / f"requests.{ext}", title=f"{profile_slug} / {run_id}")
        except Exception:
            continue

    return report_path


def _f(v):
    return f"{v:.2f}" if v is not None else "—"
