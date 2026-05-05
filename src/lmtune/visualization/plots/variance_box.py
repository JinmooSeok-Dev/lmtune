from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmtune.visualization.plots import register_plot


@register_plot("variance_box")
def plot_variance_box(
    run_values: dict[str, list[float]],
    out_path: str | Path,
    title: str = "Run-to-run variance",
    **opts,
) -> Path:
    """N-run 반복 실행 결과의 box plot.

    OpenHands (#1) 10× 편차, arXiv:2509.09853 등 variance 검증용.
    """
    if not run_values:
        raise ValueError("no run values")
    labels = list(run_values)
    data = [run_values[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel(opts.get("ylabel", "value"))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
