"""``NativePercentilePruner`` — step-wise percentile 기반 Pruner ABC 검증.

검증:
1. ABC subclass
2. invalid percentile / direction → ValueError
3. percentile=0.5 가 NativeMedianPruner 와 동일한 prune 결정 (수학적 동치)
4. percentile=0.25 (관용) vs percentile=0.75 (엄격) 결정 차이
5. minimize 방향 반전
6. warmup / startup 동작
7. tuner.factory.make_pruner('percentile_native') dispatch
8. _NATIVE_PRUNER_KINDS 두 axis 모두 포함
"""

from __future__ import annotations

import pytest

from lmtune.tuner import Pruner, make_pruner
from lmtune.tuner.median_pruner import NativeMedianPruner
from lmtune.tuner.percentile_pruner import NativePercentilePruner


def test_is_pruner_abc():
    assert issubclass(NativePercentilePruner, Pruner)
    assert isinstance(NativePercentilePruner(), Pruner)


def test_invalid_percentile_raises():
    with pytest.raises(ValueError, match="percentile must be"):
        NativePercentilePruner(percentile=0.0)
    with pytest.raises(ValueError, match="percentile must be"):
        NativePercentilePruner(percentile=1.0)
    with pytest.raises(ValueError, match="percentile must be"):
        NativePercentilePruner(percentile=1.5)


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction must be"):
        NativePercentilePruner(direction="invalid")


def test_percentile_50_matches_median():
    """percentile=0.5 일 때 NativeMedianPruner 와 동일 결정."""
    pp = NativePercentilePruner(
        percentile=0.5, n_startup_trials=3, n_warmup_steps=0, direction="maximize"
    )
    pm = NativeMedianPruner(n_startup_trials=3, n_warmup_steps=0, direction="maximize")
    # 같은 cross-trial sample 누적
    for tid, val in (("t-1", 10.0), ("t-2", 20.0), ("t-3", 30.0)):
        pp.should_prune(tid, step=0, value=val)
        pm.should_prune(tid, step=0, value=val)
    # 본 trial 의 prune 결정 — 동일해야
    for v in (5.0, 15.0, 20.0, 25.0, 35.0):
        assert pp.should_prune(f"me-{v}", step=0, value=v) == pm.should_prune(
            f"me-{v}", step=0, value=v
        )


def test_percentile_25_lenient_vs_75_strict():
    """percentile=0.25 는 관용 (하위 25% 만 prune), 0.75 는 엄격 (하위 75% prune)."""
    lenient = NativePercentilePruner(
        percentile=0.25, n_startup_trials=3, n_warmup_steps=0, direction="maximize"
    )
    strict = NativePercentilePruner(
        percentile=0.75, n_startup_trials=3, n_warmup_steps=0, direction="maximize"
    )
    for tid, val in (("t-1", 10.0), ("t-2", 20.0), ("t-3", 30.0), ("t-4", 40.0)):
        lenient.should_prune(tid, step=0, value=val)
        strict.should_prune(tid, step=0, value=val)
    # 25th percentile of [10,20,30,40] = 17.5; 75th = 32.5
    # value=25 → lenient(17.5) keep, strict(32.5) prune
    assert lenient.should_prune("me", step=0, value=25.0) is False
    assert strict.should_prune("me", step=0, value=25.0) is True


def test_minimize_inverts():
    p = NativePercentilePruner(
        percentile=0.25, n_startup_trials=3, n_warmup_steps=0, direction="minimize"
    )
    for tid, val in (("t-1", 10.0), ("t-2", 20.0), ("t-3", 30.0), ("t-4", 40.0)):
        p.should_prune(tid, step=0, value=val)
    # 25th of [10,20,30,40] = 17.5; minimize 면 value > 17.5 → prune
    assert p.should_prune("me-high", step=0, value=25.0) is True
    assert p.should_prune("me-low", step=0, value=15.0) is False


def test_warmup_and_startup_returns_false():
    p = NativePercentilePruner(percentile=0.5, n_startup_trials=10, n_warmup_steps=2)
    # startup 미만
    p.should_prune("t-1", step=5, value=1.0)
    p.should_prune("t-2", step=5, value=2.0)
    # cross-trial only 2 < n_startup=10 → False
    assert p.should_prune("me", step=5, value=-9999.0) is False
    # warmup
    assert p.should_prune("me", step=0, value=-9999.0) is False
    assert p.should_prune("me", step=1, value=-9999.0) is False


# ─── factory dispatch ─────────────────────────────────────────────────


def test_factory_dispatches_percentile_native():
    p = make_pruner("percentile_native")
    assert isinstance(p, NativePercentilePruner)
    assert isinstance(p, Pruner)


def test_factory_kwargs_pass_through():
    p = make_pruner(
        "percentile_native",
        percentile=0.3,
        n_startup_trials=2,
        n_warmup_steps=1,
        direction="minimize",
    )
    assert isinstance(p, NativePercentilePruner)
    assert p._percentile == 0.3
    assert p._n_startup == 2
    assert p._n_warmup_steps == 1
    assert p._direction == "minimize"


# ─── drift 가드 ───────────────────────────────────────────────────────


def test_native_pruner_kinds_contains_both():
    """_NATIVE_PRUNER_KINDS 가 두 axis 모두 포함 — drift 가드."""
    from lmtune.tuner.factory import _NATIVE_PRUNER_KINDS

    assert {"median_native", "percentile_native"} <= _NATIVE_PRUNER_KINDS


def test_unknown_error_lists_percentile_native():
    """ValueError 메시지에 percentile_native 가 노출."""
    with pytest.raises(ValueError) as ei:
        make_pruner("totally_unknown")
    assert "percentile_native" in str(ei.value)
    assert "median_native" in str(ei.value)
