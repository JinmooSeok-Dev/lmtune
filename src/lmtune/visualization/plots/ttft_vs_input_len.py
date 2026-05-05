from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmtune.runners.base import RequestRow
from lmtune.visualization.plots import register_plot


@register_plot("ttft_vs_input_len")
def plot_ttft_vs_input_len(rows: Iterable[RequestRow], out_path: str | Path, **opts) -> Path:
    xs, ys = [], []
    for r in rows:
        if r.ttft_ms is None or r.input_tokens is None:
            continue
        xs.append(r.input_tokens)
        ys.append(r.ttft_ms)
    if not xs:
        raise ValueError("no input_tokens/TTFT data")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, alpha=0.4, s=20)
    if len(xs) >= 2:
        import numpy as np

        coef = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, coef[0] * xline + coef[1], "r-", label=f"fit: {coef[0]:.3f}x+{coef[1]:.1f}")
        ax.legend()
    ax.set_xlabel("input tokens")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(opts.get("title", "TTFT vs Input Length"))
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
