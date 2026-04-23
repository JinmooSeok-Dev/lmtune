"""SearchSpace — declarative parameter space, YAML-loaded.

Axis types: categorical | bool | int | float | log_uniform
Conditional: `active_if: {adapter: llmd-k8s}` gates an axis based on a context dict.

The SearchSpace is strategy-agnostic; samplers (grid/random/lhc/TPE/...)
interpret it. Optuna integration lives in `samplers/` — this module stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


AxisKind = Literal["categorical", "bool", "int", "float", "log_uniform"]


@dataclass(slots=True)
class Axis:
    name: str
    kind: AxisKind
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    step: int | float | None = None
    active_if: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind == "categorical":
            if not self.values:
                raise ValueError(f"axis {self.name}: categorical requires non-empty 'values'")
        elif self.kind == "bool":
            # Normalize to a fixed categorical representation.
            self.values = [False, True]
        elif self.kind in ("int", "float", "log_uniform"):
            if self.low is None or self.high is None:
                raise ValueError(f"axis {self.name}: {self.kind} requires low and high")
            if self.low >= self.high:
                raise ValueError(f"axis {self.name}: low must be < high")
            if self.kind == "log_uniform" and self.low <= 0:
                raise ValueError(f"axis {self.name}: log_uniform requires low > 0")
        else:
            raise ValueError(f"axis {self.name}: unknown kind '{self.kind}'")

    def is_active(self, context: dict[str, Any]) -> bool:
        """Conditional gate. Empty active_if → always active."""
        if not self.active_if:
            return True
        for k, v in self.active_if.items():
            if context.get(k) != v:
                return False
        return True


@dataclass(slots=True)
class SearchSpace:
    name: str
    axes: list[Axis]
    api_version: str = "bench/search/v1alpha1"

    def axis_by_name(self, name: str) -> Axis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(name)

    def active_axes(self, context: dict[str, Any] | None = None) -> list[Axis]:
        ctx = context or {}
        return [a for a in self.axes if a.is_active(ctx)]

    def grid_size(self, context: dict[str, Any] | None = None) -> int:
        """Total grid combinations; only defined for discrete axes."""
        n = 1
        for a in self.active_axes(context):
            if a.kind in ("categorical", "bool"):
                n *= len(a.values or [])
            elif a.kind == "int" and a.step is not None:
                span = int(a.high) - int(a.low)
                n *= max(1, span // int(a.step) + 1)
            else:
                # continuous — grid is undefined. Caller should convert.
                raise ValueError(
                    f"axis {a.name} is continuous ({a.kind}); grid requires discrete only"
                )
        return n

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "apiVersion": self.api_version,
                "kind": "SearchSpace",
                "name": self.name,
                "axes": {a.name: _axis_to_dict(a) for a in self.axes},
            },
            sort_keys=False,
        )


def _axis_to_dict(a: Axis) -> dict:
    d: dict[str, Any] = {"type": a.kind}
    if a.kind in ("categorical",):
        d["values"] = list(a.values or [])
    elif a.kind in ("int", "float", "log_uniform"):
        d["low"] = a.low
        d["high"] = a.high
        if a.step is not None:
            d["step"] = a.step
    if a.active_if:
        d["active_if"] = a.active_if
    return d


def load_space(path: str | Path) -> SearchSpace:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_space(raw)


def parse_space(raw: dict) -> SearchSpace:
    if raw.get("kind") != "SearchSpace":
        raise ValueError(f"expected kind=SearchSpace, got {raw.get('kind')}")
    axes_raw = raw.get("axes") or {}
    axes: list[Axis] = []
    for name, spec in axes_raw.items():
        kind = spec.get("type")
        axes.append(
            Axis(
                name=name,
                kind=kind,
                values=spec.get("values"),
                low=spec.get("low"),
                high=spec.get("high"),
                step=spec.get("step"),
                active_if=spec.get("active_if") or {},
            )
        )
    return SearchSpace(
        name=raw.get("name") or "unnamed",
        axes=axes,
        api_version=raw.get("apiVersion", "bench/search/v1alpha1"),
    )
