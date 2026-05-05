"""LLM domain prior — hand-curated YAML reader, LLM-free path.

Phase W 의 Idea 3 구현. 외부 LLM 호출 없이 사전 컴파일된 priority 를
TPE/NSGA-II 의 sampling weight + warmstart seed 로 활용.

핵심 사용:
    prior = LLMDomainPrior.from_yaml(Path("configs/autoresearch/axis_priors.yaml"))
    priority = prior.get_priority("enable_chunked_prefill", ctx={"workload_class": "coding-agent"})
    # → "high" (contextual_overrides 가 default_priorities 보다 우선)
    seeds = prior.to_warmstart_seeds(space, ctx, n=3)
    # → 3 개 seed trial. high-priority axis 는 plausible value, 나머지는 random
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lmtune.search.space import Axis, SearchSpace

PRIORITY_WEIGHTS: dict[str, float] = {
    "high": 2.0,
    "medium": 1.0,
    "low": 0.3,
}


@dataclass(slots=True)
class _Override:
    applies_when: dict[str, Any]
    priorities: dict[str, str]


@dataclass(slots=True)
class LLMDomainPrior:
    default_priorities: dict[str, str] = field(default_factory=dict)
    contextual_overrides: list[_Override] = field(default_factory=list)
    api_version: str = "lmtune/autoresearch/v1alpha1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMDomainPrior:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw.get("kind") != "AxisPriors":
            raise ValueError(f"expected kind=AxisPriors, got {raw.get('kind')}")
        return cls(
            default_priorities=dict(raw.get("default_priorities") or {}),
            contextual_overrides=[
                _Override(
                    applies_when=dict(o.get("applies_when") or {}),
                    priorities=dict(o.get("priorities") or {}),
                )
                for o in (raw.get("contextual_overrides") or [])
            ],
            api_version=raw.get("apiVersion", "lmtune/autoresearch/v1alpha1"),
        )

    def get_priority(self, axis_name: str, ctx: dict[str, Any] | None = None) -> str:
        """Look up priority. Contextual override → default → 'medium' fallback."""
        ctx = ctx or {}
        for ov in self.contextual_overrides:
            if _matches(ov.applies_when, ctx) and axis_name in ov.priorities:
                return ov.priorities[axis_name]
        return self.default_priorities.get(axis_name, "medium")

    def get_weight(self, axis_name: str, ctx: dict[str, Any] | None = None) -> float:
        return PRIORITY_WEIGHTS.get(self.get_priority(axis_name, ctx), 1.0)

    def to_warmstart_seeds(
        self,
        space: SearchSpace,
        ctx: dict[str, Any] | None = None,
        n: int = 3,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Generate seed trial dicts biased toward high-priority axes' likely values.

        For each axis:
          - high: random pick from middle 50% of values (assumed "good defaults")
          - medium: random pick from any value
          - low: fixed at first value (skip variation)
        """
        rng = random.Random(seed)
        ctx = ctx or {}
        seeds: list[dict[str, Any]] = []
        for _ in range(n):
            params: dict[str, Any] = {}
            for axis in space.active_axes(ctx):
                pr = self.get_priority(axis.name, ctx)
                params[axis.name] = _suggest_value(axis, pr, rng)
            seeds.append(params)
        return seeds


def _matches(applies_when: dict[str, Any], ctx: dict[str, Any]) -> bool:
    if not applies_when:
        return True
    for k, v in applies_when.items():
        actual = ctx.get(k)
        if isinstance(v, list):
            if actual not in v:
                return False
        elif actual != v:
            return False
    return True


def _suggest_value(axis: Axis, priority: str, rng: random.Random) -> Any:
    """Suggest a value for an axis given its priority hint."""
    if axis.kind == "categorical":
        vals = axis.values or []
        if not vals:
            return None
        if priority == "low":
            return vals[0]
        if priority == "high":
            # middle 50% if there are ≥ 4 values, else random
            if len(vals) >= 4:
                lo, hi = len(vals) // 4, 3 * len(vals) // 4
                return rng.choice(vals[lo:hi] or vals)
            return rng.choice(vals)
        return rng.choice(vals)
    if axis.kind == "bool":
        return rng.choice([False, True])
    if axis.kind in ("int", "float"):
        lo, hi = float(axis.low), float(axis.high)
        if priority == "low":
            return _coerce(axis.kind, (lo + hi) / 2)
        if priority == "high":
            # bias toward middle 50%
            mid_lo = lo + (hi - lo) * 0.25
            mid_hi = lo + (hi - lo) * 0.75
            return _coerce(axis.kind, rng.uniform(mid_lo, mid_hi))
        return _coerce(axis.kind, rng.uniform(lo, hi))
    if axis.kind == "log_uniform":
        import math

        log_lo, log_hi = math.log(axis.low), math.log(axis.high)
        return math.exp(rng.uniform(log_lo, log_hi))
    return None


def _coerce(kind: str, v: float) -> float | int:
    return int(round(v)) if kind == "int" else float(v)
