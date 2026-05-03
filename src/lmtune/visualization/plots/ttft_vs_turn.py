from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmtune.runners.base import RequestRow
from lmtune.visualization.plots import register_plot


@register_plot("ttft_vs_turn")
def plot_ttft_vs_turn(rows: Iterable[RequestRow], out_path: str | Path, **opts) -> Path:
    """턴별 TTFT 분포. Profile C (Token Snowball) 검증에 사용."""
    turn_map: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.ttft_ms is None or r.turn_idx is None:
            continue
        turn_map[int(r.turn_idx)].append(float(r.ttft_ms))
    if not turn_map:
        raise ValueError("no TTFT-per-turn data in provided rows")

    turns = sorted(turn_map)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([turn_map[t] for t in turns], tick_labels=[str(t) for t in turns], showfliers=False)
    means = [sum(turn_map[t]) / len(turn_map[t]) for t in turns]
    ax.plot(range(1, len(turns) + 1), means, "r--o", label="mean")
    ax.set_xlabel("turn index")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(opts.get("title", "TTFT vs Turn #"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
