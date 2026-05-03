"""Search trace — running best score over trial sequence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_search_trace(
    seqs: list[int],
    scores: list[float | None],
    direction: str = "maximize",
    out_path: str | Path = "search_trace.png",
) -> Path:
    xs = np.asarray(seqs, dtype=int)
    ys = np.asarray([(s if s is not None else np.nan) for s in scores], dtype=float)
    # running best
    if direction == "maximize":
        rb = np.maximum.accumulate(np.where(np.isnan(ys), -np.inf, ys))
    else:
        rb = np.minimum.accumulate(np.where(np.isnan(ys), np.inf, ys))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, alpha=0.5, label="trial score", s=18)
    ax.plot(xs, rb, color="crimson", label=f"running {direction[:3]}", linewidth=2)
    ax.set_xlabel("trial seq")
    ax.set_ylabel("score")
    ax.set_title("Search trace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
