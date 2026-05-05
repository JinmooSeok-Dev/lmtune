"""``lmtune.tuner`` 의 lazy export 검증.

검증:
1. lazy export 동작 (__getattr__) — 직접 attribute 접근
2. ``from lmtune.tuner import NativeMedianPruner`` 가능
3. lazy 로드된 클래스가 직접 import 한 클래스와 동일 (identity)
4. 기존 lazy export (Sampler, Pruner, make_sampler, make_pruner, Optuna 어댑터)
   회귀 없음
5. unknown attribute → AttributeError (drift 가드)
6. __all__ 의 모든 entry 가 실제로 import 가능
"""

from __future__ import annotations

import pytest


def test_native_median_pruner_lazy_import():
    """``from lmtune.tuner import NativeMedianPruner`` 동작."""
    from lmtune.tuner import NativeMedianPruner
    from lmtune.tuner.median_pruner import NativeMedianPruner as Direct

    assert NativeMedianPruner is Direct


def test_native_percentile_pruner_lazy_import():
    from lmtune.tuner import NativePercentilePruner
    from lmtune.tuner.percentile_pruner import NativePercentilePruner as Direct

    assert NativePercentilePruner is Direct


def test_existing_exports_still_work():
    """기존 lazy export 회귀 없음."""
    import lmtune.tuner as t

    # ABC 는 eager (top-level import)
    assert t.Sampler is not None
    assert t.Pruner is not None
    # factory 는 lazy
    assert callable(t.make_sampler)
    assert callable(t.make_pruner)
    # Optuna 어댑터 lazy
    assert t.OptunaSamplerAdapter is not None
    assert t.OptunaPrunerAdapter is not None


def test_unknown_attribute_raises():
    import lmtune.tuner as t

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = t.NonexistentClass


def test_all_attribute_consistent():
    """``__all__`` 의 모든 entry 가 실제로 import 가능."""
    import lmtune.tuner as t

    for name in t.__all__:
        attr = getattr(t, name)
        assert attr is not None, f"__all__ entry {name!r} resolved to None"


def test_native_pruners_are_pruner_abc():
    """lazy 로 import 한 native 도 Pruner ABC subclass."""
    from lmtune.tuner import NativeMedianPruner, NativePercentilePruner, Pruner

    assert issubclass(NativeMedianPruner, Pruner)
    assert issubclass(NativePercentilePruner, Pruner)
