"""Sampler / Pruner ABC contract 검증."""

from __future__ import annotations

import pytest

from lmtune.tuner import Pruner, Sampler


class _DummySampler(Sampler):
    def __init__(self, params: dict):
        self._params = params

    def ask(self, context=None):
        return dict(self._params)


class _DummyPruner(Pruner):
    def __init__(self, threshold: float):
        self._t = threshold

    def should_prune(self, trial_id, step, value, history=None):
        return value < self._t


# ─── Sampler ABC ─────────────────────────────────────────────────────


def test_sampler_is_abstract():
    with pytest.raises(TypeError):
        Sampler()  # type: ignore[abstract]


def test_sampler_subclass_must_implement_ask():
    class Incomplete(Sampler):  # noqa: D401
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_sampler_default_tell_is_noop():
    s = _DummySampler({"x": 1})
    # default tell() = no-op; 호출이 예외 안 나야 한다.
    s.tell({"x": 1}, 0.5)
    s.tell({"x": 1}, 0.5, metrics={"ttft": {"p99": 200.0}})


def test_sampler_ask_returns_dict():
    s = _DummySampler({"a": "x", "b": 7})
    out = s.ask()
    assert isinstance(out, dict)
    assert out == {"a": "x", "b": 7}


# ─── Pruner ABC ─────────────────────────────────────────────────────


def test_pruner_is_abstract():
    with pytest.raises(TypeError):
        Pruner()  # type: ignore[abstract]


def test_pruner_should_prune_threshold():
    p = _DummyPruner(threshold=0.5)
    assert p.should_prune("t-1", step=0, value=0.3) is True
    assert p.should_prune("t-1", step=1, value=0.7) is False


def test_pruner_history_kw_optional():
    """should_prune 의 history kw 는 optional — 안 줘도 동작."""
    p = _DummyPruner(threshold=0.0)
    assert p.should_prune("t-1", step=0, value=-1.0) is True
    assert p.should_prune("t-1", step=0, value=-1.0, history=[0.1, 0.2]) is True
