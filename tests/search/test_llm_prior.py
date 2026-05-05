"""Phase W — LLM domain prior (hand-curated YAML) unit tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lmtune.search.llm_prior import PRIORITY_WEIGHTS, LLMDomainPrior, _matches
from lmtune.search.space import Axis, SearchSpace


def _write_priors(tmp: Path, body: str) -> Path:
    p = tmp / "priors.yaml"
    p.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return p


def test_load_minimal(tmp_path):
    path = _write_priors(
        tmp_path,
        """
        apiVersion: lmtune/autoresearch/v1alpha1
        kind: AxisPriors
        default_priorities:
          a: high
          b: medium
          c: low
    """,
    )
    p = LLMDomainPrior.from_yaml(path)
    assert p.get_priority("a") == "high"
    assert p.get_priority("b") == "medium"
    assert p.get_priority("c") == "low"
    assert p.get_priority("unknown") == "medium"  # fallback


def test_load_invalid_kind(tmp_path):
    path = _write_priors(
        tmp_path,
        """
        kind: WrongKind
        default_priorities: {a: high}
    """,
    )
    with pytest.raises(ValueError):
        LLMDomainPrior.from_yaml(path)


def test_contextual_override_wins_over_default(tmp_path):
    path = _write_priors(
        tmp_path,
        """
        kind: AxisPriors
        default_priorities:
          enable_prefix_caching: medium
        contextual_overrides:
          - applies_when: {workload_class: coding-agent}
            priorities:
              enable_prefix_caching: high
    """,
    )
    p = LLMDomainPrior.from_yaml(path)
    # default
    assert p.get_priority("enable_prefix_caching") == "medium"
    # context match → override
    assert p.get_priority("enable_prefix_caching", {"workload_class": "coding-agent"}) == "high"
    # context not match → default
    assert p.get_priority("enable_prefix_caching", {"workload_class": "summarization"}) == "medium"


def test_get_weight():
    p = LLMDomainPrior(default_priorities={"a": "high", "b": "low"})
    assert p.get_weight("a") == PRIORITY_WEIGHTS["high"]
    assert p.get_weight("b") == PRIORITY_WEIGHTS["low"]
    assert p.get_weight("unknown") == PRIORITY_WEIGHTS["medium"]


def test_warmstart_seeds_low_priority_fixed():
    """low priority axis = 첫 값으로 고정 (variation 없음)."""
    p = LLMDomainPrior(default_priorities={"a": "low"})
    space = SearchSpace(
        name="t",
        axes=[Axis(name="a", kind="categorical", values=[1, 2, 3, 4])],
    )
    seeds = p.to_warmstart_seeds(space, {}, n=5, seed=0)
    assert all(s["a"] == 1 for s in seeds)


def test_warmstart_seeds_high_priority_middle_50pct():
    """high priority + 4+ values → middle 50% 만 sample."""
    p = LLMDomainPrior(default_priorities={"a": "high"})
    space = SearchSpace(
        name="t",
        axes=[Axis(name="a", kind="categorical", values=[10, 20, 30, 40, 50, 60, 70, 80])],
    )
    seeds = p.to_warmstart_seeds(space, {}, n=20, seed=0)
    chosen = {s["a"] for s in seeds}
    # middle 50% = indices 2..6 → values {30, 40, 50, 60}
    assert chosen.issubset({30, 40, 50, 60})


def test_warmstart_seeds_int_range():
    p = LLMDomainPrior(default_priorities={"x": "medium"})
    space = SearchSpace(
        name="t",
        axes=[Axis(name="x", kind="int", low=1, high=100)],
    )
    seeds = p.to_warmstart_seeds(space, {}, n=10, seed=42)
    for s in seeds:
        assert isinstance(s["x"], int)
        assert 1 <= s["x"] <= 100


def test_warmstart_seeds_respects_active_if():
    """active_if 가 false 면 axis 가 seed 에서 빠짐."""
    p = LLMDomainPrior(default_priorities={"a": "medium", "b": "medium"})
    space = SearchSpace(
        name="t",
        axes=[
            Axis(name="a", kind="bool"),
            Axis(name="b", kind="bool", active_if={"adapter": "llmd-k8s"}),
        ],
    )
    seeds = p.to_warmstart_seeds(space, {"adapter": "local-vllm"}, n=2)
    for s in seeds:
        assert "a" in s
        assert "b" not in s  # active_if 매칭 실패 → 제외


def test_matches_helper():
    assert _matches({}, {"a": 1}) is True
    assert _matches({"a": 1}, {"a": 1}) is True
    assert _matches({"a": [1, 2]}, {"a": 1}) is True
    assert _matches({"a": [1, 2]}, {"a": 3}) is False


def test_real_axis_priors_yaml_loads():
    """Repo 의 configs/autoresearch/axis_priors.yaml 가 valid 인지 검증."""
    repo_root = Path(__file__).resolve().parents[2]
    p = LLMDomainPrior.from_yaml(repo_root / "configs/autoresearch/axis_priors.yaml")
    # Phase W 의 핵심 hints 가 high
    assert p.get_priority("enable_chunked_prefill") == "high"
    assert p.get_priority("enable_prefix_caching") == "high"
    # P/D 컨텍스트에서 NIXL_CHUNK_SIZE_MB 가 high
    assert p.get_priority("NIXL_CHUNK_SIZE_MB", {"well_lit_path": "pd-disaggregation"}) == "high"
