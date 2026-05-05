"""tuner.factory — strategy 문자열 → tuner.Sampler 통합 dispatch.

후속 PR (OD) 에서 ``Study`` 와 ``cli_search`` 가 본 factory 만 호출하도록 정리.
현재는 search.samplers.make_sampler (Optuna BaseSampler 반환) 와 병존.

Strategy 매핑:
- ``random_native`` / ``lhc_native`` / ``tpe_native``  →  src/lmtune/search/samplers/native.py
  (이미 tuner.Sampler 구현체, isinstance 통과 — #47 산출)
- 그 외 (``random``, ``grid``, ``lhc``, ``tpe``, ``cma_es``, ``nsga2``, ``ucb``, ``botorch``)
  →  search.samplers.make_sampler() 로 Optuna BaseSampler 받아 OptunaSamplerAdapter 로 wrap

설계 원칙:
- 본 factory 는 dispatch only — 실제 알고리즘은 기존 search/samplers/ 에 위임
- 반환 = (Sampler, prefetch_params). prefetch 는 LHC 가 미리 enqueue 할 params list
  (없으면 None). 후속 PR 에서 driver 가 prefetch 처리.
"""

from __future__ import annotations

from typing import Any

from lmtune.search.space import SearchSpace
from lmtune.tuner.base import Pruner, Sampler

_NATIVE_STRATEGIES = {"random_native", "lhc_native", "tpe_native"}
_LLM_STRATEGIES = {"llm_oracle"}

# Pruner kind 화이트리스트 — search.pruners.make_pruner 가 받는 값과 동일.
# drift 방지용 단일 진실원: 본 set 만 갱신하면 factory 가 자동 흡수.
_OPTUNA_PRUNER_KINDS = {"sh", "successive_halving", "hyperband"}

# Native Pruner — Optuna 위임 없이 stdlib 만으로 동작. PLUG slot 의 첫 합류.
_NATIVE_PRUNER_KINDS = {"median_native"}


def make_sampler(
    strategy: str,
    space: SearchSpace,
    *,
    seed: int | None = None,
    context: dict[str, Any] | None = None,
    n_samples: int | None = None,
    direction: str = "maximize",
) -> tuple[Sampler, list[dict] | None]:
    """``strategy`` 이름으로 tuner.Sampler 인스턴스 생성.

    Returns:
        (sampler, prefetch_params).
        ``prefetch_params`` 는 LHC 의 경우 미리 enqueue 할 params list, 그 외는
        None. driver 는 prefetch 가 있으면 ask() 전에 enqueue 한다.
    """
    s = strategy.lower()

    if s in _NATIVE_STRATEGIES:
        return _make_native(s, space, seed=seed, n_samples=n_samples), None

    if s in _LLM_STRATEGIES:
        return _make_llm(s, space), None

    # Optuna 경로 — 기존 search.samplers.make_sampler 재사용 후 어댑터로 wrap
    from lmtune.search.samplers import make_sampler as _make_optuna
    from lmtune.tuner.optuna_adapter import OptunaSamplerAdapter

    optuna_sampler, prefetch = _make_optuna(
        s,
        space,
        seed=seed,
        context=context,
        n_samples=n_samples,
    )
    adapter = OptunaSamplerAdapter(space, optuna_sampler, direction=direction)
    return adapter, prefetch


def _make_native(
    strategy: str,
    space: SearchSpace,
    *,
    seed: int | None,
    n_samples: int | None,
) -> Sampler:
    from lmtune.search.samplers.native import (
        NativeLHCSampler,
        NativeRandomSampler,
        NativeTPESampler,
    )

    if strategy == "random_native":
        return NativeRandomSampler(space, seed=seed)
    if strategy == "lhc_native":
        if n_samples is None:
            raise ValueError("lhc_native sampler requires n_samples")
        return NativeLHCSampler(space, n_samples=int(n_samples), seed=seed)
    if strategy == "tpe_native":
        return NativeTPESampler(space, seed=seed)
    raise ValueError(f"unknown native strategy: {strategy!r}")


def _make_llm(strategy: str, space: SearchSpace) -> Sampler:
    """LLM-guided sampler 디스패치. anthropic SDK 가 없으면 ImportError."""
    from lmtune.tuner.llm_oracle import LLMOracleSampler

    if strategy == "llm_oracle":
        return LLMOracleSampler(space)
    raise ValueError(f"unknown LLM strategy: {strategy!r}")


def make_pruner(kind: str | None = None, **kwargs) -> Pruner | None:
    """``kind`` 이름으로 tuner.Pruner ABC 인스턴스 생성.

    현재는 Optuna 위임 only — search.pruners.make_pruner 가 반환하는
    ``optuna.pruners.BasePruner`` 를 ``OptunaPrunerAdapter`` 로 wrap.
    PLUG 의 입구로 자체 ABC 구현체 (ASHA, MedianPruner 직접 구현 등) 가
    들어오면 본 함수에 분기 1줄 + ``_OPTUNA_PRUNER_KINDS`` 갱신만으로 합류.

    Args:
        kind: ``None`` 또는 ``"none"`` → ``None`` 반환 (no-op pruner).
            그 외 ``search.pruners.make_pruner`` 가 받는 모든 kind 지원
            (현재 ``sh`` / ``successive_halving`` / ``hyperband``).
        **kwargs: ``search.pruners.make_pruner`` 로 그대로 전달
            (min_resource / reduction_factor / max_resource 등).

    Returns:
        Pruner ABC 구현체 또는 None.
    """
    if kind is None or kind == "none":
        return None

    k = kind.lower()
    if k in _NATIVE_PRUNER_KINDS:
        return _make_native_pruner(k, **kwargs)
    if k in _OPTUNA_PRUNER_KINDS:
        from lmtune.search.pruners import make_pruner as _make_optuna_pruner
        from lmtune.tuner.optuna_adapter import OptunaPrunerAdapter

        optuna_pruner = _make_optuna_pruner(k, **kwargs)
        return OptunaPrunerAdapter(optuna_pruner)

    valid = sorted(_OPTUNA_PRUNER_KINDS | _NATIVE_PRUNER_KINDS) + ["none"]
    raise ValueError(f"unknown pruner kind: {kind!r}. Valid: {valid}")


def _make_native_pruner(kind: str, **kwargs) -> Pruner:
    """Native Pruner 디스패치. Optuna 미설치 환경에서도 동작."""
    from lmtune.tuner.median_pruner import NativeMedianPruner

    if kind == "median_native":
        return NativeMedianPruner(**kwargs)
    raise ValueError(f"unknown native pruner kind: {kind!r}")


__all__ = ["make_pruner", "make_sampler"]
