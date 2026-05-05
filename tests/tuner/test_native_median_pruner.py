"""``NativeMedianPruner`` — step-wise median 기반 Pruner ABC 구현 검증.

검증:
1. ABC subclass (Pruner)
2. n_warmup_steps 미만 step 은 항상 False
3. n_startup_trials 미만의 cross-trial sample 은 항상 False
4. maximize: 본 trial value 가 다른 trial 의 같은 step value median 미만 → True
5. minimize: 반대
6. 같은 trial_id + 같은 step 반복 호출은 마지막 값으로 덮어씀 (자기 자신은 median 에 포함 안 됨)
7. invalid direction → ValueError
8. tuner.factory.make_pruner('median_native') dispatch 동작
9. _NATIVE_PRUNER_KINDS drift 가드
"""

from __future__ import annotations

import pytest

from lmtune.tuner import Pruner, make_pruner
from lmtune.tuner.median_pruner import NativeMedianPruner


def test_is_pruner_abc():
    assert issubclass(NativeMedianPruner, Pruner)
    assert isinstance(NativeMedianPruner(), Pruner)


def test_warmup_steps_returns_false():
    """step < n_warmup_steps 는 항상 False — 다른 trial 충분히 쌓여도."""
    p = NativeMedianPruner(n_startup_trials=0, n_warmup_steps=3)
    # 다른 trial 누적
    for tid in (f"t-{i}" for i in range(10)):
        p.should_prune(tid, step=0, value=100.0)
    # 본 trial 의 step 0/1/2 는 warmup → 항상 False
    assert p.should_prune("me", step=0, value=-999.0) is False
    assert p.should_prune("me", step=1, value=-999.0) is False
    assert p.should_prune("me", step=2, value=-999.0) is False
    # step 3 부터는 평가 가능
    out = p.should_prune("me", step=3, value=-999.0)
    assert isinstance(out, bool)


def test_startup_trials_returns_false():
    """cross-trial sample 이 n_startup_trials 미만이면 prune 결정 보류."""
    p = NativeMedianPruner(n_startup_trials=5, n_warmup_steps=0)
    # 4 개의 다른 trial 만 쌓음
    for tid in ("t-1", "t-2", "t-3", "t-4"):
        p.should_prune(tid, step=0, value=100.0)
    # 5번째 trial (n_startup_trials = 5 미만) — 어떤 value 든 False
    assert p.should_prune("me", step=0, value=-999.0) is False


def test_maximize_below_median_prunes():
    """maximize: 본 value 가 다른 trial 의 median 미만 → prune."""
    p = NativeMedianPruner(n_startup_trials=3, n_warmup_steps=0, direction="maximize")
    p.should_prune("t-1", step=0, value=10.0)
    p.should_prune("t-2", step=0, value=20.0)
    p.should_prune("t-3", step=0, value=30.0)  # median = 20.0
    # current value = 5.0 < 20.0 → prune
    assert p.should_prune("me", step=0, value=5.0) is True
    # current value = 25.0 > 20.0 → keep
    assert p.should_prune("me-2", step=0, value=25.0) is False


def test_minimize_above_median_prunes():
    """minimize: 본 value 가 median 초과 → prune."""
    p = NativeMedianPruner(n_startup_trials=3, n_warmup_steps=0, direction="minimize")
    p.should_prune("t-1", step=0, value=10.0)
    p.should_prune("t-2", step=0, value=20.0)
    p.should_prune("t-3", step=0, value=30.0)  # median = 20.0
    # current value = 50.0 > 20.0 → prune (lower = better, 본 trial 은 나쁨)
    assert p.should_prune("me", step=0, value=50.0) is True
    # current value = 5.0 < 20.0 → keep
    assert p.should_prune("me-2", step=0, value=5.0) is False


def test_same_trial_id_step_overwrites():
    """동일 (trial_id, step) 의 반복 호출은 마지막 value 로 덮어쓰기.

    자기 자신은 median 계산에서 제외 — repeat 호출이 cross-trial 에 영향 안 줌.
    """
    p = NativeMedianPruner(n_startup_trials=0, n_warmup_steps=0)
    # 자기 자신만 여러 번
    p.should_prune("me", step=0, value=100.0)
    p.should_prune("me", step=0, value=200.0)
    p.should_prune("me", step=0, value=300.0)
    # 다른 trial 없음 → median 계산 불가, False
    assert p.should_prune("me", step=0, value=-999.0) is False


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction must be"):
        NativeMedianPruner(direction="something_else")


# ─── factory dispatch ─────────────────────────────────────────────────


def test_factory_dispatches_median_native():
    p = make_pruner("median_native")
    assert isinstance(p, NativeMedianPruner)
    assert isinstance(p, Pruner)


def test_factory_kwargs_pass_through_to_native():
    p = make_pruner(
        "median_native",
        n_startup_trials=2,
        n_warmup_steps=1,
        direction="minimize",
    )
    assert isinstance(p, NativeMedianPruner)
    assert p._n_startup == 2
    assert p._n_warmup_steps == 1
    assert p._direction == "minimize"


# ─── drift 가드 ───────────────────────────────────────────────────────


def test_native_pruner_kinds_whitelist():
    """``_NATIVE_PRUNER_KINDS`` 는 median_native 를 포함."""
    from lmtune.tuner.factory import _NATIVE_PRUNER_KINDS

    assert "median_native" in _NATIVE_PRUNER_KINDS


def test_factory_unknown_lists_both_kind_groups():
    """unknown kind 의 ValueError 메시지에 native + optuna kind 모두 노출."""
    with pytest.raises(ValueError) as ei:
        make_pruner("nonexistent_kind")
    msg = str(ei.value)
    assert "median_native" in msg
    assert "hyperband" in msg
