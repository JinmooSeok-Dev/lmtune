"""NativeMedianPruner — step-wise median 기반 Pruner ABC 구현.

PLUG 의 Pruner axis 첫 native impl (#70/#71 의 ``_OPTUNA_PRUNER_KINDS`` 옆에
신규 ``_NATIVE_PRUNER_KINDS`` 합류). Optuna 위임 없이 Python stdlib 만으로
trial 의 중간 value 가 다른 trial 의 같은 step value 의 median 미만 (maximize)
또는 초과 (minimize) 이면 prune.

알고리즘 (간단·결정적):
- 모든 (trial_id, step) → value 누적
- ``should_prune(trial_id, step, value)`` 호출 시:
  - step < n_warmup_steps → 항상 False (warmup)
  - 다른 trial 들의 같은 step value 가 n_startup_trials 미만 → False
  - 그 외: ``statistics.median(other_values)`` 와 비교
- direction='maximize': ``value < median`` 이면 prune
- direction='minimize': ``value > median`` 이면 prune

설계 의도:
- 단일 frame 안에서 cross-trial median 계산 — Optuna study 객체 불필요
- 누적 archive 를 외부 storage 로 옮길 때도 (trial_id, step, value) tuple 만 있으면 됨
- thread-safe 아님 (단일 driver loop 가정 — orchestrate/driver.py 가 owner)
"""

from __future__ import annotations

import statistics

from lmtune.tuner.base import Pruner


class NativeMedianPruner(Pruner):
    """step-wise median 기반 Pruner.

    Args:
        n_startup_trials: 다른 trial 의 같은 step value 가 이 수 미만이면
            prune 안 함 (median 신뢰 부족).
        n_warmup_steps: 본 trial 의 step 이 이 수 미만이면 prune 안 함
            (cold-start 보호).
        direction: ``"maximize"`` (기본) 또는 ``"minimize"``.
    """

    def __init__(
        self,
        *,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
        direction: str = "maximize",
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {direction!r}")
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
        del history  # 본 pruner 는 cross-trial state 만 사용
        # 누적 — 동일 (trial_id, step) 은 마지막 값으로 덮어쓰기
        self._trial_history.setdefault(trial_id, {})[step] = float(value)

        if step < self._n_warmup_steps:
            return False

        # 다른 trial 들의 같은 step value 수집 (자기 자신 제외)
        other_values = [
            t_steps[step]
            for tid, t_steps in self._trial_history.items()
            if tid != trial_id and step in t_steps
        ]

        # cross-trial sample 부족 — n_startup_trials=0 이어도 empty 면 결정 보류
        if not other_values or len(other_values) < self._n_startup:
            return False

        med = statistics.median(other_values)

        if self._direction == "maximize":
            return value < med
        return value > med


__all__ = ["NativeMedianPruner"]
