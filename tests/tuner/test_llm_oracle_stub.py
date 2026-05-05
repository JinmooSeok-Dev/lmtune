"""LLMOracleSampler — Tuner ABC PLUG 패턴 stub 검증.

본 stub 의 acceptance:
1. Sampler ABC subclass (type 수준 plug-in 보증).
2. ``anthropic`` SDK 미설치 시 친절 ImportError ("install with lmtune[agent]").
3. ``anthropic`` 가 있으면 인스턴스 생성 + tell() 누적까지 동작 (ask 는 stub).
4. ``tuner.factory.make_sampler('llm_oracle', space)`` 가 ImportError 또는
   LLMOracleSampler 인스턴스를 일관되게 반환 — factory dispatch 합류 보증.
"""

from __future__ import annotations

import importlib.util

import pytest

from lmtune.search.space import Axis, SearchSpace
from lmtune.tuner import Sampler, make_sampler
from lmtune.tuner.llm_oracle import LLMOracleSampler

_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace(
        name="t",
        axes=[
            Axis(name="lr", kind="float", low=0.001, high=0.1),
            Axis(name="batch", kind="categorical", values=[16, 32, 64]),
        ],
    )


def test_llm_oracle_is_sampler():
    """클래스 자체는 항상 Sampler subclass — anthropic 미설치라도 type check."""
    assert issubclass(LLMOracleSampler, Sampler)


@pytest.mark.skipif(_HAS_ANTHROPIC, reason="anthropic 설치된 환경에서는 ImportError 분기 없음")
def test_llm_oracle_import_error_when_anthropic_missing(space):
    with pytest.raises(ImportError) as ei:
        LLMOracleSampler(space)
    msg = str(ei.value)
    assert "anthropic" in msg
    assert "lmtune[agent]" in msg


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic 미설치 — instance 생성 검증 skip")
def test_llm_oracle_instance_and_tell(space):
    """anthropic 있으면 인스턴스 생성 + tell() 누적."""
    s = LLMOracleSampler(space, model="claude-opus-4-7")
    assert s.space is space
    assert s.model == "claude-opus-4-7"
    assert s._recent == []

    # tell 은 base 의 no-op 이 아니라 _recent 누적해야 함
    s.tell({"lr": 0.01, "batch": 32}, score=0.85)
    s.tell({"lr": 0.02, "batch": 64}, score=0.90)
    assert len(s._recent) == 2
    assert s._recent[0] == ({"lr": 0.01, "batch": 32}, 0.85)


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic 미설치")
def test_llm_oracle_ask_not_implemented(space):
    s = LLMOracleSampler(space)
    with pytest.raises(NotImplementedError):
        s.ask()


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="anthropic 미설치")
def test_llm_oracle_window_capped(space):
    """tell 누적이 window size (16) 를 초과하면 오래된 것이 drop."""
    s = LLMOracleSampler(space)
    for i in range(20):
        s.tell({"lr": float(i) / 100, "batch": 32}, score=float(i))
    assert len(s._recent) == 16
    # 가장 최근 16건만 남음 → 첫 entry 의 score 는 4 (0..3 drop)
    assert s._recent[0][1] == 4.0
    assert s._recent[-1][1] == 19.0


def test_factory_llm_oracle_dispatch(space):
    """factory.make_sampler('llm_oracle') 가 PLUG 매핑에 합류.

    anthropic 설치 여부에 따라 분기:
    - 설치됨 → LLMOracleSampler 인스턴스 반환
    - 미설치 → ImportError
    어느 쪽이든 'unknown strategy' 가 아닌 매핑된 path 로 들어감을 보증.
    """
    if _HAS_ANTHROPIC:
        sampler, prefetch = make_sampler("llm_oracle", space)
        assert isinstance(sampler, LLMOracleSampler)
        assert prefetch is None
    else:
        with pytest.raises(ImportError):
            make_sampler("llm_oracle", space)
