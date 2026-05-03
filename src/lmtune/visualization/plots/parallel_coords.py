"""Parallel-coordinates plot — high axis-count 의 trial path 시각화.

각 axis 가 수직선 1개. trial 1개 = axis 들을 순회하는 polyline. 점수 상위
top_k 는 진하게, 나머지는 연하게. 카테고리 축은 정수 매핑.

B5 (combined Pareto) 처럼 13+ axis 가 한 study 에 있을 때 macro 공간을
한눈에 보는 canonical view. Pure matplotlib (외부 plugin 없음).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lmtune.visualization.plots import register_plot


def _normalize_axis(values: list) -> tuple[list[float], list]:
    """Normalize values to [0, 1]. Returns (normalized, tick_labels).

    숫자형: linear min/max. 카테고리: index/(N-1).
    """
    is_num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if is_num:
        lo, hi = min(values), max(values)
        rng = (hi - lo) or 1.0
        return [(v - lo) / rng for v in values], [f"{lo:.3g}", f"{hi:.3g}"]
    # categorical (bool / str / mixed): stable order by first-seen
    order: list = []
    for v in values:
        if v not in order:
            order.append(v)
    n = max(1, len(order) - 1)
    idx = {v: i / n for i, v in enumerate(order)}
    return [idx[v] for v in values], [str(v) for v in order]


@register_plot("parallel_coords")
def plot_parallel_coords(
    rows: list[dict],
    out_path: str | Path = "parallel_coords.png",
    *,
    top_k: int = 5,
    direction: str = "maximize",
    max_axes: int = 14,
) -> Path:
    """`rows`: trial dicts with `status`, `score`, `params`."""
    completed = [r for r in rows if r.get("status") == "completed" and r.get("score") is not None]
    if len(completed) < 2:
        raise ValueError("parallel_coords needs ≥ 2 completed trials")

    completed.sort(key=lambda r: r["score"], reverse=(direction == "maximize"))
    axes_seen: list[str] = []
    for r in completed:
        for k in r.get("params") or {}:
            if k not in axes_seen:
                axes_seen.append(k)
    axes_seen = axes_seen[:max_axes]
    if len(axes_seen) < 2:
        raise ValueError("parallel_coords needs ≥ 2 axes")

    per_axis_vals: dict[str, list] = {}
    for k in axes_seen:
        col = []
        for r in completed:
            v = (r.get("params") or {}).get(k)
            col.append(v if v is not None else "_NA")
        per_axis_vals[k] = col

    norm: dict[str, list[float]] = {}
    tick_labels: dict[str, list[str]] = {}
    for k, vals in per_axis_vals.items():
        n, t = _normalize_axis(vals)
        norm[k] = n
        tick_labels[k] = t

    n_trials = len(completed)
    width = max(7.0, 0.9 * len(axes_seen))
    fig, ax = plt.subplots(figsize=(width, 5.2))
    x = np.arange(len(axes_seen))

    for i, _ in enumerate(completed):
        y = [norm[k][i] for k in axes_seen]
        if i < top_k:
            color = (
                plt.cm.viridis(1.0 - i / max(1, top_k - 1)) if top_k > 1 else plt.cm.viridis(0.95)
            )
            ax.plot(x, y, color=color, linewidth=2.0, alpha=0.95, zorder=3)
        else:
            ax.plot(x, y, color="#cbd5e1", linewidth=0.6, alpha=0.45, zorder=1)

    for xi in x:
        ax.axvline(xi, color="#94a3b8", linewidth=0.4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(axes_seen, rotation=25, ha="right", fontsize=9)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["min", "mid", "max"], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Parallel coordinates — top-{top_k} highlighted ({n_trials} completed trials)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with rank/score swatches
    handles = []
    for i in range(min(top_k, n_trials)):
        c = plt.cm.viridis(1.0 - i / max(1, top_k - 1)) if top_k > 1 else plt.cm.viridis(0.95)
        handles.append(
            plt.Line2D(
                [], [], color=c, linewidth=2, label=f"#{i + 1}  score={completed[i]['score']:.3g}"
            )
        )
    handles.append(plt.Line2D([], [], color="#cbd5e1", linewidth=1, label="rest"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
