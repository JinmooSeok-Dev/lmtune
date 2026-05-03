"""Axis-importance horizontal bar (RandomForest based).

`bench report` / report templates 에서 사용. Dashboard (study.html) 는
별도로 Chart.js 로 그린다 — 본 모듈은 정적 PNG 산출용.

Source: `lmtune.search.analysis.importance.axis_importance`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmtune.visualization.plots import register_plot


@register_plot("axis_importance")
def plot_axis_importance(
    rows: list[dict],
    out_path: str | Path = "axis_importance.png",
    *,
    drop_threshold: float = 0.05,
    top_n: int | None = 20,
) -> Path:
    """Plot RandomForest feature importance per axis.

    `rows` 은 study 의 trial dict list (`status`, `score`, `params`).
    """
    from lmtune.search.analysis.importance import axis_importance

    imp = axis_importance(rows, drop_threshold=drop_threshold)
    if not imp:
        raise ValueError("not enough completed trials for importance fit")

    items = sorted(imp.items(), key=lambda kv: kv[1]["importance"], reverse=True)
    if top_n is not None:
        items = items[:top_n]
    names = [k for k, _ in items]
    vals = [v["importance"] for _, v in items]
    recs = [v["recommendation"] for _, v in items]
    colors = ["#94a3b8" if r == "drop" else "#2563eb" for r in recs]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(names) + 1)))
    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(
        drop_threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=0.8,
        label=f"drop threshold ({drop_threshold:.2f})",
    )
    ax.set_xlabel("RandomForest importance (sum over one-hot levels)")
    ax.set_title(f"Axis importance — top {len(names)}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    for b, v in zip(bars, vals, strict=False):
        ax.text(
            v + 0.005,
            b.get_y() + b.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=8,
            color="#475569",
        )
    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
