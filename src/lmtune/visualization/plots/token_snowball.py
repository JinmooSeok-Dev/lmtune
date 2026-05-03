from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmtune.runners.base import RequestRow
from lmtune.visualization.plots import register_plot


@register_plot("token_snowball")
def plot_token_snowball(
    rows: Iterable[RequestRow], out_path: str | Path,
    **opts,
) -> Path:
    """턴별 누적 입력 토큰(Snowball Effect) + 선형 fit.

    SWE-Effi (#3, arXiv:2509.09853) "Token Snowball Effect" 재현.
    """
    per_conv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        if r.turn_idx is None or r.input_tokens is None or r.conversation_id is None:
            continue
        per_conv[r.conversation_id].append((int(r.turn_idx), int(r.input_tokens)))
    if not per_conv:
        raise ValueError("no turn/input_tokens/conversation data")

    fig, ax = plt.subplots(figsize=(9, 5))
    avg_by_turn: dict[int, list[int]] = defaultdict(list)
    for _conv, points in per_conv.items():
        points.sort()
        turns = [t for t, _ in points]
        toks = [t for _, t in points]
        ax.plot(turns, toks, alpha=0.3, linewidth=1)
        for t, k in points:
            avg_by_turn[t].append(k)
    mean_turns = sorted(avg_by_turn)
    mean_toks = [sum(avg_by_turn[t]) / len(avg_by_turn[t]) for t in mean_turns]
    ax.plot(mean_turns, mean_toks, "r-o", label="mean", linewidth=2)
    # 선형 fit
    if len(mean_turns) >= 2:
        import numpy as np
        coef = np.polyfit(mean_turns, mean_toks, 1)
        xs = np.array(mean_turns)
        ax.plot(xs, coef[0] * xs + coef[1], "k--", alpha=0.6,
                label=f"fit: {coef[0]:.0f} tok/turn")
        ax.legend()
    ax.set_xlabel("turn index")
    ax.set_ylabel("cumulative input tokens")
    ax.set_title(opts.get("title", "Token Snowball (input tokens per turn)"))
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
