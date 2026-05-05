"""Cost-aware sampler — vary only axes with cost_tier >= min_tier_to_vary.

Tier semantics (lower = more expensive to change):
  1 = topology (영구), 2 = BIOS/cmdline (~10min reboot),
  3 = helmfile apply (~3min), 4 = vllm restart (~30s),
  5 = env only (0s), 6 = runtime (<1s).

`max_tier` (a.k.a. `--max-cost-tier`) is the **lowest tier we still vary**.
Axes with cost_tier < max_tier stay frozen across trials of the study.

Examples:
  max_tier=4 → freeze tier 1-3 (topology/BIOS/helmfile), vary 4-6 (vllm/env/runtime)
  max_tier=5 → freeze tier 1-4 (also vllm), vary 5-6 (env/runtime only — fastest)
  max_tier=3 → freeze tier 1-2, vary 3-6 (allow helmfile redeploy per trial)

Wraps any underlying sampler (Optuna TPE / RandomSampler / NSGA-II / ...).
이 클래스는 Optuna 의존성 없이 sampler agnostic — `sample(space, context) -> dict`
인터페이스를 따르면 wrap 가능.
"""

from __future__ import annotations

from typing import Any, Protocol

from lmtune.search.space import Axis, SearchSpace


class _SamplerLike(Protocol):
    def sample(self, space: SearchSpace, context: dict[str, Any]) -> dict[str, Any]: ...


class CostAwareSampler:
    """Wrap a sampler so axes with cost_tier > max_tier stay frozen across trials."""

    def __init__(self, base: _SamplerLike, max_tier: int = 6):
        if max_tier < 1 or max_tier > 6:
            raise ValueError(f"max_tier must be in [1, 6], got {max_tier}")
        self._base = base
        self._max_tier = int(max_tier)
        self._frozen: dict[str, Any] = {}

    @property
    def max_tier(self) -> int:
        return self._max_tier

    @property
    def frozen(self) -> dict[str, Any]:
        return dict(self._frozen)

    def reset(self) -> None:
        self._frozen.clear()

    def sample(self, space: SearchSpace, context: dict[str, Any]) -> dict[str, Any]:
        """Sample one trial; override high-tier axes with frozen values."""
        params = self._base.sample(space, context)

        active = space.active_axes(context)
        # First trial: snapshot high-tier values for freezing.
        if not self._frozen:
            for axis in active:
                if axis.cost_tier < self._max_tier and axis.name in params:
                    self._frozen[axis.name] = params[axis.name]
            return params

        # Subsequent trials: override high-tier axes from frozen snapshot.
        for axis in active:
            if axis.cost_tier < self._max_tier and axis.name in self._frozen:
                params[axis.name] = self._frozen[axis.name]
        return params


def filter_axes_by_tier(space: SearchSpace, max_tier: int) -> list[Axis]:
    """Return axes that the cost-aware sampler will actually vary across trials."""
    return [a for a in space.axes if a.cost_tier >= max_tier]


def summarize_tier_split(space: SearchSpace) -> dict[int, list[str]]:
    """For docs/logs: which axes live at each tier."""
    out: dict[int, list[str]] = {}
    for a in space.axes:
        out.setdefault(a.cost_tier, []).append(a.name)
    return dict(sorted(out.items()))
