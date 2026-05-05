"""lmtune tuner — 탐색 알고리즘 추상화.

REFACTOR-PLAN 의 L2 layer. Sampler / Pruner ABC + Optuna 어댑터.

후속 PR 에서 native sampler (TPE/LHC/Random) 도 같은 ABC 구현체로 추가 →
``bench search start --strategy <name>`` 의 strategy 가 단일 dispatch 로 통합.
"""

from __future__ import annotations

from lmtune.tuner.base import Pruner, Sampler

__all__ = ["Pruner", "Sampler", "make_sampler"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
