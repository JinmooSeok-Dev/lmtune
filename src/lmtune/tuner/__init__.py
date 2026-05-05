"""lmtune tuner — 탐색 알고리즘 추상화.

REFACTOR-PLAN 의 L2 layer. Sampler / Pruner ABC + Optuna 어댑터.

후속 PR 에서 native sampler (TPE/LHC/Random) 도 같은 ABC 구현체로 추가 →
``bench search start --strategy <name>`` 의 strategy 가 단일 dispatch 로 통합.
"""

from __future__ import annotations

from lmtune.tuner.base import Pruner, Sampler

__all__ = [
    "NativeMedianPruner",
    "NativePercentilePruner",
    "Pruner",
    "Sampler",
    "make_pruner",
    "make_sampler",
]


def __getattr__(name: str):
    # Optuna 어댑터 + factory 는 lazy — Optuna 미설치 환경에서도 tuner.Sampler /
    # tuner.Pruner 만 import 하면 동작.
    if name in ("OptunaSamplerAdapter", "OptunaPrunerAdapter"):
        from lmtune.tuner.optuna_adapter import OptunaPrunerAdapter, OptunaSamplerAdapter

        return {
            "OptunaSamplerAdapter": OptunaSamplerAdapter,
            "OptunaPrunerAdapter": OptunaPrunerAdapter,
        }[name]
    if name == "make_sampler":
        from lmtune.tuner.factory import make_sampler

        return make_sampler
    if name == "make_pruner":
        from lmtune.tuner.factory import make_pruner

        return make_pruner
    # Native pruner — stdlib only, 외부 SDK 0
    if name == "NativeMedianPruner":
        from lmtune.tuner.median_pruner import NativeMedianPruner

        return NativeMedianPruner
    if name == "NativePercentilePruner":
        from lmtune.tuner.percentile_pruner import NativePercentilePruner

        return NativePercentilePruner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
