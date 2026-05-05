"""tuner.factory.make_sampler — 통합 dispatch 검증."""

from __future__ import annotations

import pytest

from lmtune.search.samplers.native import (
    NativeLHCSampler,
    NativeRandomSampler,
    NativeTPESampler,
)
from lmtune.search.space import Axis, SearchSpace
from lmtune.tuner import OptunaSamplerAdapter, Sampler, make_sampler


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        name="t",
        axes=[
            Axis(name="lr", kind="float", low=0.001, high=0.1),
            Axis(name="batch", kind="categorical", values=[16, 32, 64]),
        ],
    )


# ─── native strategies ───────────────────────────────────────────────


def test_factory_random_native(space):
    sampler, prefetch = make_sampler("random_native", space, seed=42)
    assert isinstance(sampler, NativeRandomSampler)
    assert isinstance(sampler, Sampler)
    assert prefetch is None
    p = sampler.ask()
    assert set(p.keys()) == {"lr", "batch"}


def test_factory_tpe_native(space):
    sampler, prefetch = make_sampler("tpe_native", space, seed=42)
    assert isinstance(sampler, NativeTPESampler)
    assert prefetch is None


def test_factory_lhc_native_requires_n_samples(space):
    with pytest.raises(ValueError, match="n_samples"):
        make_sampler("lhc_native", space, seed=42)


def test_factory_lhc_native_with_n_samples(space):
    sampler, prefetch = make_sampler("lhc_native", space, seed=42, n_samples=4)
    assert isinstance(sampler, NativeLHCSampler)
    assert prefetch is None


# ─── Optuna 경로 (OptunaSamplerAdapter wrap) ─────────────────────────


def test_factory_optuna_random(space):
    sampler, prefetch = make_sampler("random", space, seed=42)
    assert isinstance(sampler, OptunaSamplerAdapter)
    assert isinstance(sampler, Sampler)
    assert prefetch is None
    p = sampler.ask()
    assert set(p.keys()) == {"lr", "batch"}


def test_factory_optuna_tpe(space):
    sampler, _ = make_sampler("tpe", space, seed=42)
    assert isinstance(sampler, OptunaSamplerAdapter)


def test_factory_optuna_lhc_returns_prefetch(space):
    sampler, prefetch = make_sampler("lhc", space, seed=42, n_samples=4)
    assert isinstance(sampler, OptunaSamplerAdapter)
    # LHC 경로는 prefetch=list[dict] 반환 (driver 가 enqueue)
    assert isinstance(prefetch, list)
    assert len(prefetch) == 4


def test_factory_unknown_native_strategy_raises(space):
    """native prefix 인데 정의 안 된 이름 → ValueError."""
    # "_native" 가 아닌 unknown 은 Optuna 경로로 감 → 거기서 ValueError.
    with pytest.raises(ValueError):
        make_sampler("definitely_not_a_real_strategy", space)


def test_factory_returns_sampler_abc(space):
    """모든 반환값이 Sampler ABC instance 임을 한 번에 확인."""
    for strategy in ("random", "tpe", "random_native", "tpe_native"):
        sampler, _ = make_sampler(strategy, space, seed=42)
        assert isinstance(sampler, Sampler), f"{strategy} did not return Sampler"
