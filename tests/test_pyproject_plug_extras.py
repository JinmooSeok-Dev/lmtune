"""pyproject.toml 의 [postgres], [agent] extras ↔ ImportError 메시지 정합성.

PLUG 패턴 (#58, #59) 의 ImportError 메시지에 명시된 install command (예:
``pip install lmtune[postgres]``) 가 실제 pyproject.toml 의 extras 키와
일치해야 한다. 본 테스트가 drift 차단.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_extras() -> dict[str, list[str]]:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]["optional-dependencies"]


def test_postgres_extra_registered():
    extras = _load_extras()
    assert "postgres" in extras, "PLUG: PostgresArtifactStore 의 [postgres] extra 누락"
    pkgs = " ".join(extras["postgres"])
    assert "psycopg" in pkgs, "[postgres] extra 가 psycopg 를 포함하지 않음"


def test_agent_extra_registered():
    extras = _load_extras()
    assert "agent" in extras, "PLUG: LLMOracleSampler 의 [agent] extra 누락"
    pkgs = " ".join(extras["agent"])
    assert "anthropic" in pkgs, "[agent] extra 가 anthropic 을 포함하지 않음"


@pytest.mark.skipif(
    importlib.util.find_spec("psycopg") is not None,
    reason="psycopg 설치된 환경에서는 ImportError 분기 없음",
)
def test_postgres_import_error_mentions_extra_key():
    """PostgresArtifactStore 의 ImportError 메시지가 'lmtune[postgres]' 를 정확히 인용."""
    # store/__init__ 의 PostgresArtifactStore import 만으론 ImportError 안 남
    # — 인스턴스 생성 시점에 발생.
    from lmtune.storage.store import PostgresArtifactStore

    with pytest.raises(ImportError) as ei:
        PostgresArtifactStore("postgres://x/y")
    msg = str(ei.value)
    assert "lmtune[postgres]" in msg, "ImportError 메시지의 extra 이름이 pyproject 의 키와 drift 됨"


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None,
    reason="anthropic 설치된 환경에서는 ImportError 분기 없음",
)
def test_agent_import_error_mentions_extra_key():
    """LLMOracleSampler 의 ImportError 메시지가 'lmtune[agent]' 를 정확히 인용."""
    # 임시로 가짜 SearchSpace — 인스턴스 생성 단계에서 ImportError 가 먼저
    # 발생하므로 SearchSpace 의 실제 axis 검증은 도달하지 않음.
    from lmtune.search.space import Axis, SearchSpace
    from lmtune.tuner.llm_oracle import LLMOracleSampler

    space = SearchSpace(name="t", axes=[Axis(name="x", kind="float", low=0.0, high=1.0)])
    with pytest.raises(ImportError) as ei:
        LLMOracleSampler(space)
    msg = str(ei.value)
    assert "lmtune[agent]" in msg


def test_optional_extras_dont_pollute_default_install():
    """기본 dependencies 에 psycopg / anthropic 가 들어가면 안 됨 (PLUG 정신)."""
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
    base_deps = " ".join(cfg["dependencies"])
    assert "psycopg" not in base_deps
    assert "anthropic" not in base_deps


def test_lmtune_imports_without_optional_extras():
    """기본 환경에서 lmtune 의 PLUG 클래스들이 ImportError 없이 import 됨 (lazy)."""
    # 이미 collection 단계에서 import 가 성공했다면 본 테스트는 자명히 통과.
    # 핵심은 ImportError 없이 type 객체에 도달 가능하다는 것.
    from lmtune.storage.store import (
        ArtifactStore,
        DuckDBArtifactStore,
        InMemoryArtifactStore,
        LocalArtifactStore,
        PostgresArtifactStore,
    )
    from lmtune.tuner import Sampler
    from lmtune.tuner.llm_oracle import LLMOracleSampler

    assert all(
        cls is not None
        for cls in (
            ArtifactStore,
            DuckDBArtifactStore,
            InMemoryArtifactStore,
            LocalArtifactStore,
            PostgresArtifactStore,
            Sampler,
            LLMOracleSampler,
        )
    )
