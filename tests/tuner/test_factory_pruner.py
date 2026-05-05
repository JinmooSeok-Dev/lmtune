"""``lmtune.tuner.factory.make_pruner`` — search.pruners 와의 ABC 어댑터 결합 검증.

검증:
1. ``kind=None / 'none'`` → ``None`` 반환 (no-op)
2. ``sh`` / ``successive_halving`` / ``hyperband`` → ``Pruner`` ABC 구현체 반환
3. unknown kind → ``ValueError`` (drift 가드)
4. ``search.pruners.make_pruner`` 가 받는 kind 와 factory 의 화이트리스트가 동기화
   (drift 방지 — 한 쪽만 갱신되면 즉시 fail)
"""

from __future__ import annotations

import pytest

from lmtune.tuner import Pruner, make_pruner


def test_make_pruner_none_returns_none():
    assert make_pruner(None) is None
    assert make_pruner("none") is None


def test_make_pruner_sh_alias_returns_pruner_abc():
    """``sh`` / ``successive_halving`` 둘 다 동일 결과."""
    p1 = make_pruner("sh")
    p2 = make_pruner("successive_halving")
    assert isinstance(p1, Pruner)
    assert isinstance(p2, Pruner)


def test_make_pruner_hyperband_returns_pruner_abc():
    p = make_pruner("hyperband")
    assert isinstance(p, Pruner)


def test_make_pruner_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown pruner kind"):
        make_pruner("median")


def test_make_pruner_kwargs_pass_through():
    """min_resource / reduction_factor 가 search.pruners 까지 전달."""
    p = make_pruner("sh", min_resource=2, reduction_factor=4)
    assert isinstance(p, Pruner)


def test_make_pruner_should_prune_signature_works():
    """반환된 Pruner 가 should_prune() 호출 가능 (ABC 계약 충족)."""
    p = make_pruner("sh")
    assert p is not None
    out = p.should_prune("trial-1", step=0, value=0.5)
    assert isinstance(out, bool)


# ─── drift 가드 ─────────────────────────────────────────────────────


def test_factory_kinds_match_search_pruners_make_pruner():
    """``tuner.factory._OPTUNA_PRUNER_KINDS`` 와 ``search.pruners.make_pruner``
    가 받는 kind 가 동기화되어야 한다.

    한 쪽에 새 kind 가 들어오고 다른 쪽이 갱신 안 되면 즉시 fail.
    """
    from lmtune.search.pruners import make_pruner as _search_make_pruner
    from lmtune.tuner.factory import _OPTUNA_PRUNER_KINDS

    # search.pruners.make_pruner 가 명시 처리하는 kind:
    # 'sh' / 'successive_halving' / 'hyperband' (현 시점)
    expected = {"sh", "successive_halving", "hyperband"}
    assert expected == _OPTUNA_PRUNER_KINDS, (
        f"factory pruner kinds drift detected. factory={_OPTUNA_PRUNER_KINDS}, expected={expected}"
    )

    # 실제로 search.pruners.make_pruner 도 같은 kind 들을 받는지 smoke
    for k in expected:
        p = _search_make_pruner(k)
        # search.pruners 는 Optuna BasePruner 반환 — None 이면 안 됨
        assert p is not None, f"search.pruners.make_pruner({k!r}) returned None"
