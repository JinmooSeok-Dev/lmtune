"""Optuna sampler/pruner → lmtune Sampler/Pruner ABC 어댑터.

기존 ``src/lmtune/search/samplers/`` 가 Optuna 의 ``BaseSampler`` 를 그대로
사용. S1 의 Sampler ABC 위에 동일 동작을 노출하는 thin wrapper.

설계:
- 한 study 가 1 sampler. ``OptunaSamplerAdapter`` 가 study 를 owns.
- ask() 는 ``study.ask()`` → frozen Trial → params dict.
- tell() 은 ``study.tell(trial, score, state=...)``.
- 본 어댑터는 study 가 in-memory storage 일 때만 의미 있음. persistent
  storage 는 후속 PR (`OD` orchestrate driver 분리) 에서 다룬다.
"""

from __future__ import annotations

from typing import Any

import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState

from lmtune.search.samplers import suggest_from_axis
from lmtune.search.space import SearchSpace
from lmtune.tuner.base import Pruner, Sampler


class OptunaSamplerAdapter(Sampler):
    """Optuna ``BaseSampler`` 를 lmtune ``Sampler`` ABC 로 노출.

    in-memory study 를 owns. ask() 는 ``study.ask()`` 호출, tell() 은 score 를
    그대로 전달. 직전 ask() 한 trial 을 ``_pending`` 으로 들고 있어 tell() 시
    매칭.
    """

    def __init__(
        self,
        space: SearchSpace,
        sampler: BaseSampler,
        *,
        direction: str = "maximize",
    ):
        self._space = space
        self._study = optuna.create_study(direction=direction, sampler=sampler)
        self._pending: dict[str, optuna.Trial] = {}  # params hash → Trial

    def ask(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        trial = self._study.ask()
        params: dict[str, Any] = {}
        for axis in self._space.active_axes(context):
            params[axis.name] = suggest_from_axis(trial, axis)
        # match by frozenset of items so tell() can find it later
        key = _params_key(params)
        self._pending[key] = trial
        return params

    def tell(
        self,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        del metrics
        key = _params_key(params)
        trial = self._pending.pop(key, None)
        if trial is None:
            # 외부 source (warm-start) 로부터 들어온 params — Optuna 에 enqueue
            # 후 dummy ask 로 매칭하지 않고 그대로 무시 (TPE 의 in-memory archive
            # 만 영향 받음). 외부 archive 는 storage layer 가 owner.
            return
        self._study.tell(trial, score, state=TrialState.COMPLETE)


class OptunaPrunerAdapter(Pruner):
    """Optuna ``BasePruner`` 를 lmtune ``Pruner`` ABC 로 노출.

    Optuna pruner 는 ``trial.report()`` + ``trial.should_prune()`` 패턴이라 본
    어댑터는 이를 캡슐화. 외부에서 trial 객체를 들고 있을 필요 없게 한다.
    """

    def __init__(self, pruner: optuna.pruners.BasePruner):
        self._pruner = pruner
        # study 1 개 + trial 1 개의 dummy frame — pruner 가 stat 누적할 곳.
        # 동일 trial_id 는 같은 frame 재사용.
        self._study = optuna.create_study(direction="maximize", pruner=pruner)
        self._frames: dict[str, optuna.Trial] = {}

    def should_prune(
        self,
        trial_id: str,
        step: int,
        value: float,
        history: list[float] | None = None,
    ) -> bool:
        del history
        trial = self._frames.get(trial_id)
        if trial is None:
            trial = self._study.ask()
            self._frames[trial_id] = trial
        trial.report(value, step)
        return trial.should_prune()


def _params_key(params: dict[str, Any]) -> str:
    """params dict → 결정적 hash key. tuple-of-items, sorted."""
    return repr(sorted(params.items()))


__all__ = ["OptunaPrunerAdapter", "OptunaSamplerAdapter"]
