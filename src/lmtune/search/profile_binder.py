"""Env profile binder — macro tuple → matched profiles → merged env.

Macro axes (model, well_lit_path, node_split_strategy, tp/dp/ep, dtype) decide
which env profile YAMLs apply. Matched profiles contribute:
  - env_locked   : NCCL/UCX/LMCache env that the sampler does NOT search
  - env_tunable  : 3-5 micro axes the sampler still searches inside this profile

This collapses the 70+ flat axis problem (B6) into ~12 effective axes per study.
See plan § Autoresearch Architecture (Macro × Env Profile × Micro).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TunableAxis:
    """One micro axis exposed by a profile."""

    name: str
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    kind: str = "categorical"  # categorical | bool | int | float
    cost_tier: int = 5  # default: env-only, instant


@dataclass(slots=True)
class EnvProfile:
    name: str
    applies_when: dict[str, Any] = field(default_factory=dict)
    env_locked: dict[str, Any] = field(default_factory=dict)
    env_tunable: list[TunableAxis] = field(default_factory=list)
    priority: int = 0  # lower = applied first; later profiles override
    description: str = ""


def load_profile(path: Path) -> EnvProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EnvProfile(
        name=raw.get("name") or path.stem,
        applies_when=raw.get("applies_when") or {},
        env_locked=raw.get("env_locked") or {},
        env_tunable=[
            TunableAxis(**a) if isinstance(a, dict) else TunableAxis(name=str(a))
            for a in (raw.get("env_tunable") or [])
        ],
        priority=int(raw.get("priority", 0)),
        description=raw.get("description") or "",
    )


class EnvProfileBinder:
    """Loads all profiles in a directory; binds macro tuples to matched profiles."""

    def __init__(self, profiles_dir: str | Path):
        self._dir = Path(profiles_dir)
        self._profiles: list[EnvProfile] = []
        if self._dir.exists():
            for p in sorted(self._dir.glob("*.yaml")):
                self._profiles.append(load_profile(p))

    def all(self) -> list[EnvProfile]:
        return list(self._profiles)

    def bind(self, macro: dict[str, Any]) -> tuple[dict[str, Any], list[TunableAxis], list[str]]:
        """Match profiles against macro context.

        Returns (env_locked merged, env_tunable axes union, matched profile names).
        Later (higher-priority) profiles override earlier env_locked entries.
        """
        matched = sorted(
            (p for p in self._profiles if _matches(p.applies_when, macro)),
            key=lambda p: p.priority,
        )
        env_locked: dict[str, Any] = {}
        seen_axis_names: set[str] = set()
        env_tunable: list[TunableAxis] = []
        for p in matched:
            env_locked.update(p.env_locked)
            for axis in p.env_tunable:
                if axis.name not in seen_axis_names:
                    env_tunable.append(axis)
                    seen_axis_names.add(axis.name)
        return env_locked, env_tunable, [p.name for p in matched]


def _matches(applies_when: dict[str, Any], macro: dict[str, Any]) -> bool:
    """Empty applies_when always matches; supports list-membership matching."""
    if not applies_when:
        return True
    for k, v in applies_when.items():
        actual = macro.get(k)
        if isinstance(v, list):
            if actual not in v:
                return False
        elif actual != v:
            return False
    return True
