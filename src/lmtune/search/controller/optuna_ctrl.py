"""OptunaController — 기본 controller, sampler 8종 (TPE/CMA-ES/NSGA-II/...) wrap.

기존 Study 의 _optuna_study 직접 호출 로직을 그대로 가져와 이 클래스 안에
캡슐화. Study 입장에선 Controller ABC 만 본다.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from lmtune.search.controller.base import Controller
from lmtune.search.samplers import make_sampler, suggest_from_axis
from lmtune.search.space import Axis, SearchSpace

log = logging.getLogger(__name__)


def _params_key(params: dict[str, Any]) -> tuple:
    """내부 dict 의 hashable key — pending Optuna trial 매칭용."""
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


class OptunaController(Controller):
    def __init__(
        self,
        optuna_study: optuna.Study,
        space: SearchSpace,
        *,
        prefetch: list[dict] | None = None,
        strategy_label: str = "optuna",
    ):
        self._optuna_study = optuna_study
        self._space = space
        self._prefetch = list(prefetch or [])
        self._exhausted = False
        # ask() 가 만든 Optuna trial 을 tell() 까지 보관 (params key → ot)
        self._pending: dict[tuple, optuna.Trial] = {}
        self._strategy_label = strategy_label

    # --- factory ---------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        space: SearchSpace,
        *,
        strategy: str = "tpe",
        seed: int | None = None,
        context: dict | None = None,
        n_samples: int | None = None,
        direction: str = "maximize",
        directions: list[str] | None = None,
        study_name: str | None = None,
        pruner: str | None = None,
    ) -> OptunaController:
        """기존 Study.__init__ 의 sampler/pruner/study 빌드 로직을 그대로."""
        sampler, prefetch = make_sampler(
            strategy,
            space,
            seed=seed,
            context=context,
            n_samples=n_samples,
        )
        from lmtune.search.pruners import make_pruner

        prn = make_pruner(pruner) if pruner else None
        if directions:
            ostudy = optuna.create_study(
                directions=list(directions),
                sampler=sampler,
                pruner=prn,
                study_name=study_name,
            )
        else:
            ostudy = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=prn,
                study_name=study_name,
            )
        return cls(ostudy, space, prefetch=prefetch, strategy_label=f"optuna:{strategy}")

    # --- Controller ABC --------------------------------------------------

    @property
    def name(self) -> str:
        return self._strategy_label

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def ask(self, active_axes: list[Axis], *, context: dict | None = None) -> dict[str, Any]:
        if self._prefetch:
            self._optuna_study.enqueue_trial(self._prefetch.pop(0))
        ot = self._optuna_study.ask()
        params: dict[str, Any] = {}
        for axis in active_axes:
            params[axis.name] = suggest_from_axis(ot, axis)
        self._pending[_params_key(params)] = ot
        return params

    def tell(
        self,
        params: dict[str, Any],
        *,
        value: float | list[float] | None,
        status: str,
        metadata: dict | None = None,
    ) -> None:
        ot = self._pending.pop(_params_key(params), None)
        if ot is None:
            log.debug("OptunaController.tell: no pending trial for params; skipping")
            return
        try:
            if status == "completed" and value is not None:
                self._optuna_study.tell(ot, value)
            else:
                # pruned / crash / infeasible → fail state (sampler 가 학습)
                self._optuna_study.tell(ot, state=optuna.trial.TrialState.FAIL)
        except RuntimeError as e:
            # GridSampler 가 study.stop() 호출 시 발생
            if "Study.stop" in str(e):
                self._exhausted = True
            else:
                raise

    # --- optional hooks --------------------------------------------------

    def add_trial(self, params: dict[str, Any], value: float) -> None:
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )

        d: dict = {}
        for a in self._space.active_axes(None):
            if a.name not in params:
                continue
            if a.kind == "categorical":
                d[a.name] = CategoricalDistribution(list(a.values or []))
            elif a.kind == "bool":
                d[a.name] = CategoricalDistribution([False, True])
            elif a.kind == "int":
                step = int(a.step) if a.step else 1
                d[a.name] = IntDistribution(int(a.low), int(a.high), step=step)
            elif a.kind == "float":
                d[a.name] = FloatDistribution(float(a.low), float(a.high))
            elif a.kind == "log_uniform":
                d[a.name] = FloatDistribution(float(a.low), float(a.high), log=True)
        self._optuna_study.add_trial(
            optuna.trial.create_trial(params=params, distributions=d, value=float(value))
        )

    def enqueue(self, params: dict[str, Any]) -> None:
        self._optuna_study.enqueue_trial(params)
