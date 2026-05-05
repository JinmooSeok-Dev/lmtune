"""Native sampler 들이 tuner.Sampler ABC 의 구현체임을 검증."""

from __future__ import annotations

import pytest

from lmtune.search.samplers.native import (
    NativeLHCSampler,
    NativeRandomSampler,
    NativeTPESampler,
)
from lmtune.search.space import Axis, SearchSpace
from lmtune.tuner import Sampler


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        name="t",
        axes=[
            Axis(name="lr", kind="float", low=0.001, high=0.1),
            Axis(name="batch", kind="categorical", values=[16, 32, 64]),
        ],
    )


def test_native_random_is_sampler(space):
    s = NativeRandomSampler(space, seed=42)
    assert isinstance(s, Sampler)
    p = s.ask()
    assert set(p.keys()) == {"lr", "batch"}


def test_native_lhc_is_sampler(space):
    s = NativeLHCSampler(space, n_samples=4, seed=42)
    assert isinstance(s, Sampler)
    p = s.ask()
    assert set(p.keys()) == {"lr", "batch"}


def test_native_tpe_is_sampler(space):
    s = NativeTPESampler(space, seed=42, n_startup_trials=2)
    assert isinstance(s, Sampler)
    p = s.ask()
    assert set(p.keys()) == {"lr", "batch"}


def test_native_tpe_tell_accepts_metrics_kw(space):
    """tell() 시그니처가 ABC 와 호환 — metrics= kwarg 를 받아야 한다."""
    s = NativeTPESampler(space, seed=42, n_startup_trials=1)
    p = s.ask()
    # ABC 호환: metrics= 도 가능 (no-op).
    s.tell(p, 0.5, metrics={"ttft": {"p99": 200.0}})
    s.tell(p, 0.7)


def test_native_random_tell_default_noop(space):
    """Random 은 default tell (no-op) — 예외 안 나야 한다."""
    s = NativeRandomSampler(space, seed=42)
    s.tell({"lr": 0.05, "batch": 32}, 0.5)
    s.tell({"lr": 0.05, "batch": 32}, 0.5, metrics={"ttft": {"p99": 100.0}})


def test_native_lhc_tell_default_noop(space):
    """LHC 도 default tell — 예외 없어야."""
    s = NativeLHCSampler(space, n_samples=4, seed=42)
    s.tell({"lr": 0.05, "batch": 32}, 0.5)
