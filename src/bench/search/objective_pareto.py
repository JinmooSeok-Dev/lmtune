"""Multi-objective wrapper — emits tuples for Pareto samplers (NSGA-II/III, MOTPE).

A `ParetoObjective` wraps a base Objective but asks the caller to declare the
list of metric keys to optimize + their directions. Each trial returns a tuple
of scalars matching the declared directions.

Typical LLM-serving Pareto:
    objectives = [
        ("throughput_tok_avg", "short",   "maximize"),
        ("ttft_p99",            "short",   "minimize"),
        ("cost_usd",            None,      "minimize"),
    ]

The base Objective must populate these entries in `ObjectiveResult.metrics`.
If a key is missing we fill a worst-case sentinel (±inf) and mark the trial as
'crash' so the sampler treats it as dominated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bench.search.objective import Objective, ObjectiveResult


@dataclass(slots=True)
class ObjectiveKey:
    metric: str
    workload: str | None
    direction: str   # "maximize" | "minimize"


class ParetoObjective:
    """Adapts a single-objective Objective into a multi-objective one by
    extracting the declared metrics from each trial's metrics dict.

    The returned ObjectiveResult.score remains the *primary* scalar (objectives[0])
    for compatibility with Study code paths that assume a scalar — Optuna's
    multi-objective study uses `trial.values` (tuple) which Study plumbing passes
    through `tell()`.
    """

    def __init__(self, base: Objective, objectives: list[ObjectiveKey]):
        if not objectives:
            raise ValueError("ParetoObjective needs at least one ObjectiveKey")
        self._base = base
        self._keys = list(objectives)

    @property
    def directions(self) -> list[str]:
        return [k.direction for k in self._keys]

    def values(self, result: ObjectiveResult) -> list[float]:
        """Extract scalar values for each objective key from a result."""
        out: list[float] = []
        for k in self._keys:
            v = result.metrics.get((k.metric, k.workload))
            if v is None:
                out.append(-math.inf if k.direction == "maximize" else math.inf)
            else:
                out.append(float(v))
        return out

    def __call__(self, params: dict[str, Any]) -> ObjectiveResult:
        result = self._base(params)
        # Propagate the first objective onto .score (single-obj consumers).
        vals = self.values(result)
        if vals:
            result.score = vals[0]
        return result
