from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.runners.base import RequestRow
from bench.visualization.plots import register_plot


_PHASES_ORDER = ["exploration", "editing", "execution", "verification", "other"]


@register_plot("phase_breakdown")
def plot_phase_breakdown(
    rows: Iterable[RequestRow], out_path: str | Path,
    metric: str = "input_tokens", **opts,
) -> Path:
    """Agent phase(exploration/editing/execution/verification)별 토큰 분해.

    Tokenomics MSR 2026 (#20), AgentTaxo (#9), Framework 비교 (#2) 재현.
    """
    totals: dict[str, float] = defaultdict(float)
    for r in rows:
        phase = getattr(r, "phase", None) or "other"
        value = getattr(r, metric, None)
        if value is None:
            continue
        totals[phase] += float(value)
    if not totals:
        raise ValueError("no phase data to plot")
    phases = [p for p in _PHASES_ORDER if p in totals] + [p for p in totals if p not in _PHASES_ORDER]
    values = [totals[p] for p in phases]
    total_sum = sum(values) or 1.0
    percentages = [v / total_sum * 100 for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(phases, values, edgecolor="black")
    ax1.set_ylabel(metric)
    ax1.set_title(opts.get("title", f"{metric} by phase"))
    for i, (v, p) in enumerate(zip(values, percentages)):
        ax1.text(i, v, f"{p:.1f}%", ha="center", va="bottom")
    ax2.pie(values, labels=phases, autopct="%1.1f%%", startangle=90)
    ax2.set_title("share")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
