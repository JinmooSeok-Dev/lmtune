from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.analysis.distributions import ecdf
from bench.runners.base import RequestRow
from bench.visualization.plots import register_plot


@register_plot("cdf")
def plot_cdf(rows: Iterable[RequestRow], out_path: str | Path, metric: str = "ttft_ms", **opts) -> Path:
    values = [getattr(r, metric, None) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError(f"no values for metric {metric}")
    xs, fs = ecdf(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(xs, fs, where="post")
    for q_label, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        idx = int(q * (len(xs) - 1))
        ax.axvline(xs[idx], linestyle="--", alpha=0.4)
        ax.text(xs[idx], q, f" {q_label}", va="bottom")
    ax.set_xlabel(metric)
    ax.set_ylabel("F(x)")
    ax.set_title(opts.get("title", f"CDF of {metric}"))
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
