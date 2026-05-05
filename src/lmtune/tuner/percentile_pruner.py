"""NativePercentilePruner — step-wise percentile 기반 Pruner ABC 구현.

``NativeMedianPruner`` 의 일반화. percentile=0.5 가 median 과 동치 (수학적
검증은 ``tests/tuner/test_native_percentile_pruner.py`` 의 동치 케이스).

Pruner axis PLUG slot 의 두 번째 native impl. PLUG 패턴이 한 axis 안에서
2 회 시연됨 — _NATIVE_PRUNER_KINDS = {'median_native', 'percentile_native'}.

알고리즘:
- 모든 (trial_id, step) → value 누적
- ``should_prune(trial_id, step, value)`` 호출 시:
  - step < n_warmup_steps → 항상 False
  - 다른 trial 의 같은 step value < n_startup_trials → False
  - 그 외: ``other_values`` 의 ``percentile`` 분위수 (linear interp) 와 비교
- direction='maximize': value < threshold → prune (낮은 분위가 prune line)
- direction='minimize': value > threshold → prune

percentile 의미:
- maximize + percentile=0.25 → 하위 25% 미만이면 prune (관용적)
- maximize + percentile=0.75 → 하위 75% 미만이면 prune (엄격: top-25 만 keep)
- minimize 는 부호 반전.
"""

from __future__ import annotations

import statistics

from lmtune.tuner.base import Pruner


class NativePercentilePruner(Pruner):
    """step-wise percentile 기반 Pruner.

    Args:
        percentile: 0.0 ~ 1.0 사이. 0.5 는 ``NativeMedianPruner`` 와 동치.
            statistics.quantiles 의 linear interpolation 기준.
        n_startup_trials: cross-trial sample 이 이 수 미만이면 prune 안 함.
        n_warmup_steps: 본 trial step 이 이 수 미만이면 prune 안 함.
        direction: ``"maximize"`` (기본) 또는 ``"minimize"``.
    """

    def __init__(
        self,
        *,
        percentile: float = 0.25,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        direction: str = "maximize",
    ):
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0.0, 1.0) exclusive, got {percentile!r}")
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
        self._percentile = float(percentile)
        self._n_startup = int(n_startup_trials)
        self._n_warmup_steps = int(n_warmup_steps)
        self._direction = direction
        self._trial_history: dict[str, dict[int, float]] = {}

    def should_prune(
        self,
        trial_id: str,
        step: int,
        value: float,
        history: list[float] | None = None,
    ) -> bool:
        del history
        self._trial_history.setdefault(trial_id, {})[step] = float(value)

        if step < self._n_warmup_steps:
            return False

        other_values = [
            t_steps[step]
            for tid, t_steps in self._trial_history.items()
            if tid != trial_id and step in t_steps
        ]

        if not other_values or len(other_values) < self._n_startup:
            return False

        threshold = _percentile_value(other_values, self._percentile)

        if self._direction == "maximize":
            return value < threshold
        return value > threshold


def _percentile_value(data: list[float], p: float) -> float:
    """Linear interpolation 분위수. p=0.5 → ``statistics.median(data)``.

    statistics.quantiles 는 4분위/n분위만 직접 노출 — 임의 percentile 은 직접
    interp.
    """
    if len(data) == 1:
        return data[0]
    sorted_data = sorted(data)
    n = len(sorted_data)
    # rank in [0, n-1], with linear interpolation.
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * frac


# median 동치 검증용 helper — NativeMedianPruner 와 동일한 결정을 내야 한다.
def _median_equivalent(data: list[float]) -> float:
    """percentile=0.5 가 statistics.median 과 같은 결과인지 fallback 비교."""
    return statistics.median(data)


__all__ = ["NativePercentilePruner"]
