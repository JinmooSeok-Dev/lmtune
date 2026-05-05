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
from lmtune.tuner.base import Sampler

_NATIVE_STRATEGIES = {"random_native", "lhc_native", "tpe_native"}
_LLM_STRATEGIES = {"llm_oracle"}


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


__all__ = ["make_sampler"]
