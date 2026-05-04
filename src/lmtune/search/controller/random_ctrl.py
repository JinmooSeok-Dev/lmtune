"""RandomController — Optuna-free baseline + plug-in 작동 입증용.

stateless: tell() 은 noop. seed 고정 시 deterministic.
"""
from __future__ import annotations

import logging
import math
import random as _r
from typing import Any

from lmtune.search.controller.base import Controller
from lmtune.search.space import Axis

log = logging.getLogger(__name__)


def _sample_axis(axis: Axis, rng: _r.Random) -> Any:
    if axis.kind == "categorical":
        return rng.choice(list(axis.values or []))
    if axis.kind == "bool":
        return rng.choice([False, True])
    if axis.kind == "int":
        step = int(axis.step) if axis.step else 1
        lo, hi = int(axis.low), int(axis.high)
        n_steps = (hi - lo) // step
        return lo + step * rng.randint(0, n_steps)
    if axis.kind == "float":
        return rng.uniform(float(axis.low), float(axis.high))
    if axis.kind == "log_uniform":
        lo_l = math.log(float(axis.low))
        hi_l = math.log(float(axis.high))
        return math.exp(rng.uniform(lo_l, hi_l))
    raise ValueError(f"axis {axis.name}: unsupported kind {axis.kind}")


class RandomController(Controller):
    def __init__(self, seed: int | None = None):
        self._rng = _r.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def ask(self, active_axes: list[Axis], *, context: dict | None = None) -> dict[str, Any]:
        return {a.name: _sample_axis(a, self._rng) for a in active_axes}

    def tell(self, params, *, value, status, metadata=None) -> None:
        # stateless. 학습 시그널 X.
        return None
