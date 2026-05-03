"""Phase W — cost-tier sampler unit tests.

검증 대상:
  1. SearchSpace YAML 의 cost_tier 가 Axis 에 reflected
  2. 첫 trial 에서 high-tier (> max_tier) 값이 frozen
  3. 후속 trial 에서 frozen 값이 override
  4. low-tier (<= max_tier) 는 base sampler 의 값 그대로
  5. summarize_tier_split 가 카테고리화
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lmtune.search.sampler_cost_aware import (
    CostAwareSampler,
    filter_axes_by_tier,
    summarize_tier_split,
)
from lmtune.search.space import Axis, SearchSpace, load_space


class _DeterministicBase:
    """Returns a sequence of pre-canned param dicts (one per call)."""

    def __init__(self, seq: list[dict]):
        self._seq = list(seq)
        self._idx = 0

    def sample(self, space, context):
        out = self._seq[self._idx]
        self._idx += 1
        return out


def _space_with_tiers() -> SearchSpace:
    return SearchSpace(
        name="test",
        axes=[
            Axis(name="model", kind="categorical", values=["A", "B"], cost_tier=2),
            Axis(name="tp",    kind="categorical", values=[1, 2, 4], cost_tier=3),
            Axis(name="prefix_caching", kind="bool", cost_tier=4),
            Axis(name="NCCL_BUFFSIZE", kind="categorical",
                 values=["default", "2MB"], cost_tier=5),
        ],
    )


def test_cost_tier_parsed_from_yaml(tmp_path: Path):
    p = tmp_path / "space.yaml"
    p.write_text(textwrap.dedent("""
        apiVersion: lmtune/search/v1alpha1
        kind: SearchSpace
        name: test
        axes:
          a: {type: bool, cost_tier: 5}
          b: {type: categorical, values: [1, 2]}   # default cost_tier=4
    """).strip())
    space = load_space(p)
    assert space.axis_by_name("a").cost_tier == 5
    assert space.axis_by_name("b").cost_tier == 4


def test_first_trial_freezes_high_tier():
    space = _space_with_tiers()
    base = _DeterministicBase([
        {"model": "A", "tp": 4, "prefix_caching": True, "NCCL_BUFFSIZE": "2MB"},
    ])
    sampler = CostAwareSampler(base, max_tier=4)
    params = sampler.sample(space, {})
    # max_tier=4 -> tier 5 (NCCL_BUFFSIZE) 는 frozen X (4 이하), tier 2,3 만 frozen
    assert sampler.frozen == {"model": "A", "tp": 4}
    assert params["model"] == "A"


def test_subsequent_trials_use_frozen():
    space = _space_with_tiers()
    base = _DeterministicBase([
        {"model": "A", "tp": 4, "prefix_caching": True, "NCCL_BUFFSIZE": "2MB"},
        # 두 번째 base 호출 시 model/tp 가 다른 값을 줘도 frozen 으로 덮어씀
        {"model": "B", "tp": 1, "prefix_caching": False, "NCCL_BUFFSIZE": "default"},
    ])
    sampler = CostAwareSampler(base, max_tier=4)

    t1 = sampler.sample(space, {})
    t2 = sampler.sample(space, {})

    assert t1["model"] == "A" and t1["tp"] == 4
    # high-tier override: t2 의 model/tp 은 t1 와 같아야 함
    assert t2["model"] == "A"
    assert t2["tp"] == 4
    # low-tier (4 이하) 는 base 값 그대로
    assert t2["prefix_caching"] is False
    assert t2["NCCL_BUFFSIZE"] == "default"


def test_max_tier_1_varies_everything():
    """max_tier=1 → freeze if cost_tier < 1 (None), so nothing frozen — vary 모두."""
    space = _space_with_tiers()
    base = _DeterministicBase([
        {"model": "A", "tp": 4, "prefix_caching": True, "NCCL_BUFFSIZE": "2MB"},
        {"model": "B", "tp": 1, "prefix_caching": False, "NCCL_BUFFSIZE": "default"},
    ])
    sampler = CostAwareSampler(base, max_tier=1)
    sampler.sample(space, {})
    t2 = sampler.sample(space, {})
    # Nothing frozen → base 의 값 그대로
    assert t2["model"] == "B"
    assert t2["tp"] == 1


def test_max_tier_6_freezes_everything_below():
    """max_tier=6 → freeze cost_tier < 6 (i.e., tier 1-5), vary tier 6 only."""
    space = _space_with_tiers()  # 본 space 에는 tier 6 axis 가 없음 → 모두 frozen
    base = _DeterministicBase([
        {"model": "A", "tp": 4, "prefix_caching": True, "NCCL_BUFFSIZE": "2MB"},
        {"model": "B", "tp": 1, "prefix_caching": False, "NCCL_BUFFSIZE": "default"},
    ])
    sampler = CostAwareSampler(base, max_tier=6)
    sampler.sample(space, {})
    t2 = sampler.sample(space, {})
    # 모든 axis frozen
    assert t2["model"] == "A"
    assert t2["tp"] == 4
    assert t2["prefix_caching"] is True
    assert t2["NCCL_BUFFSIZE"] == "2MB"


def test_max_tier_validated():
    base = _DeterministicBase([{}])
    with pytest.raises(ValueError):
        CostAwareSampler(base, max_tier=0)
    with pytest.raises(ValueError):
        CostAwareSampler(base, max_tier=7)


def test_filter_axes_by_tier():
    """max_tier 는 vary 하는 lowest tier. cost_tier ≥ max_tier 만 vary."""
    space = _space_with_tiers()
    # max_tier=4 → vary tier {4, 5} = {prefix_caching, NCCL_BUFFSIZE}
    a4 = filter_axes_by_tier(space, 4)
    assert {a.name for a in a4} == {"prefix_caching", "NCCL_BUFFSIZE"}
    # max_tier=3 → vary tier {3, 4, 5} = {tp, prefix_caching, NCCL_BUFFSIZE}
    a3 = filter_axes_by_tier(space, 3)
    assert {a.name for a in a3} == {"tp", "prefix_caching", "NCCL_BUFFSIZE"}
    # max_tier=2 → vary 모두 (tier 1 axis 가 없음)
    a2 = filter_axes_by_tier(space, 2)
    assert {a.name for a in a2} == {"model", "tp", "prefix_caching", "NCCL_BUFFSIZE"}


def test_summarize_tier_split():
    space = _space_with_tiers()
    summary = summarize_tier_split(space)
    assert summary[2] == ["model"]
    assert summary[3] == ["tp"]
    assert summary[4] == ["prefix_caching"]
    assert summary[5] == ["NCCL_BUFFSIZE"]


def test_reset_clears_frozen():
    """max_tier=4 → freeze tier 1-3 (model=2, tp=3)."""
    space = _space_with_tiers()
    base = _DeterministicBase([
        {"model": "A", "tp": 4, "prefix_caching": True, "NCCL_BUFFSIZE": "2MB"},
        {"model": "B", "tp": 1, "prefix_caching": False, "NCCL_BUFFSIZE": "default"},
    ])
    sampler = CostAwareSampler(base, max_tier=4)
    sampler.sample(space, {})
    assert "model" in sampler.frozen
    assert "tp" in sampler.frozen
    sampler.reset()
    assert sampler.frozen == {}
