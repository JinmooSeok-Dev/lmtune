from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.runners.base import RequestRow
from bench.visualization.plots import register_plot


@register_plot("histogram")
def plot_histogram(rows: Iterable[RequestRow], out_path: str | Path, metric: str = "input_tokens", bins: int = 40, **opts) -> Path:
    values = [getattr(r, metric, None) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError(f"no values for metric {metric}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.set_title(opts.get("title", f"Histogram of {metric}"))
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
