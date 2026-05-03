"""Pareto front plot — 2D scatter of (obj1, obj2) with the non-dominated
front highlighted.

For 3+ objectives we show the first two dimensions and color points by the
third (or mark non-dominated points in the full N-dim sense).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lmtune.visualization.plots import register_plot


def _dominates(a: list[float], b: list[float], directions: list[str]) -> bool:
    better = False
    for ai, bi, d in zip(a, b, directions, strict=False):
        if d == "maximize":
            if ai < bi:
                return False
            if ai > bi:
                better = True
        else:
            if ai > bi:
                return False
            if ai < bi:
                better = True
    return better


def non_dominated(points: list[list[float]], directions: list[str]) -> list[int]:
    n = len(points)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and _dominates(points[j], points[i], directions):
                keep[i] = False
                break
    return [i for i, k in enumerate(keep) if k]


def plot_pareto(
    points: list[list[float]],
    directions: list[str],
    labels: list[str] | None = None,
    out_path: str | Path = "pareto.png",
) -> Path:
    if not points or len(points[0]) < 2:
        raise ValueError("plot_pareto needs at least 2-D objectives")
    pts = np.asarray([p[:2] for p in points], dtype=float)
    nd_idx = non_dominated(points, directions)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, label="trials")
    nd = pts[nd_idx]
    order = np.argsort(nd[:, 0])
    ax.plot(nd[order, 0], nd[order, 1], "-o", color="crimson", label="Pareto front")
    if labels:
        for i in nd_idx:
            ax.annotate(labels[i], (pts[i, 0], pts[i, 1]), fontsize=7)
    ax.set_xlabel(f"obj1 ({directions[0]})")
    ax.set_ylabel(f"obj2 ({directions[1]})")
    ax.set_title("Pareto front")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


@register_plot("pareto")
def plot_pareto_from_rows(
    rows: list[dict],
    out_path: str | Path = "pareto.png",
    *,
    obj_keys: tuple[str, str] = ("ttft_p99", "throughput_avg"),
    obj_dirs: tuple[str, str] = ("minimize", "maximize"),
    workload: str = "short",
) -> Path:
    """Registry-friendly wrapper. `rows` 는 trial dicts (metrics flat dict).

    metric 이름은 `<key>.<workload>` 또는 `<key>_<workload>.*` 형식 모두 허용.
    """
    pts: list[list[float]] = []
    labels: list[str] = []
    for r in rows:
        if r.get("status") != "completed":
            continue
        m = r.get("metrics") or {}
        x = _resolve(m, obj_keys[0], workload)
        y = _resolve(m, obj_keys[1], workload)
        if x is None or y is None:
            continue
        pts.append([x, y])
        labels.append(f"seq{r.get('seq', '?')}")
    if len(pts) < 2:
        raise ValueError("plot_pareto needs ≥ 2 trials with both objectives")
    return plot_pareto(pts, list(obj_dirs), labels=labels, out_path=out_path)


def _resolve(m: dict, key: str, workload: str) -> float | None:
    v = m.get(f"{key}.{workload}")
    if v is not None:
        return float(v)
    for k, val in m.items():
        if k.startswith(f"{key}_{workload}.") or k.startswith(f"{key}."):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None
