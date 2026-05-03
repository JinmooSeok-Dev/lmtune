"""Plot registry — decorator 기반 플러그인 등록.

새 플롯은 다음 패턴으로 추가:
    @register_plot("my_plot")
    def plot_my_plot(rows, out_path, **opts): ...
"""

from __future__ import annotations

from typing import Callable


_REGISTRY: dict[str, Callable] = {}


def register_plot(kind: str):
    def decorator(fn: Callable):
        _REGISTRY[kind] = fn
        return fn
    return decorator


def get_plot(kind: str) -> Callable | None:
    return _REGISTRY.get(kind)


def list_plots() -> list[str]:
    return sorted(_REGISTRY)


# 하위 모듈 import 하여 registry 채우기
from bench.visualization.plots import (  # noqa: F401,E402
    cdf,
    histogram,
    phase_breakdown,
    ttft_vs_input_len,
    ttft_vs_turn,
    token_snowball,
    variance_box,
)


# Backward compat — 기존 호출자(`from bench.visualization import plot_ttft_vs_turn`)
from bench.visualization.plots.ttft_vs_input_len import plot_ttft_vs_input_len  # noqa: F401,E402
from bench.visualization.plots.ttft_vs_turn import plot_ttft_vs_turn  # noqa: F401,E402


__all__ = [
    "get_plot",
    "list_plots",
    "plot_ttft_vs_input_len",
    "plot_ttft_vs_turn",
    "register_plot",
]
