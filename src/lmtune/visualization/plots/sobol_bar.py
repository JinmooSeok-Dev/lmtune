"""Sobol S1 / ST bar chart with 95% CI error bars."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sobol(results, out_path: str | Path = "sobol.png") -> Path:
    if not results:
        raise ValueError("no sobol results to plot")
    names = [r.axis for r in results]
    S1 = np.array([r.S1 for r in results])
    ST = np.array([r.ST for r in results])
    S1c = np.array([r.S1_conf for r in results])
    STc = np.array([r.ST_conf for r in results])

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 0.9), 4.5))
    ax.bar(x - w / 2, S1, w, yerr=S1c, label="S1 (first-order)", color="#3b82f6", capsize=4)
    ax.bar(x + w / 2, ST, w, yerr=STc, label="ST (total-order)", color="#f97316", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Sobol index (fraction of output variance)")
    ax.set_title("Global sensitivity — S1 vs ST")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = Path(out_path)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
