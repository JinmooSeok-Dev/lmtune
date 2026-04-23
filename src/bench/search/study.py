"""Study — owns the sampler + history + Optuna engine + DuckDB persistence.

S1 runs inline: Study.run() iterates up to max_trials, calling the Objective
sequentially. S3 will wrap this loop with a TrialBackend that dispatches trials
to K8s Jobs or a ProcessPool.

Persistence contract:
- `record_study()` on creation
- `record_trial(status=pending)` when ask() produces a Trial
- `record_trial(status=...)` + `record_trial_metrics(...)` on tell()
- `set_study_status()` when the loop ends
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import optuna
from ulid import ULID

from bench.search.objective import Objective, ObjectiveResult
from bench.search.samplers import make_sampler, suggest_from_axis
from bench.search.space import SearchSpace
from bench.search.trial import Trial, TrialStatus
from bench.storage.duckdb_store import DuckDBStore


log = logging.getLogger(__name__)

# Silence Optuna's per-trial info logging; we emit our own at Study level.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class StudyConfig:
    name: str
    strategy: str                      # grid | random | lhc
    space: SearchSpace
    metric_name: str = "total_score"
    direction: str = "maximize"        # maximize | minimize
    endpoint_slug: str | None = None
    profile_slugs: list[str] = field(default_factory=list)
    seed: int | None = None
    n_samples: int | None = None       # lhc only
    context: dict[str, Any] | None = None
    study_id: str | None = None
    notes: str | None = None


class Study:
    def __init__(self, config: StudyConfig, storage: DuckDBStore):
        self.cfg = config
        self.storage = storage
        self.study_id = config.study_id or f"st-{ULID()}"
        sampler, prefetch = make_sampler(
            config.strategy,
            config.space,
            seed=config.seed,
            context=config.context,
            n_samples=config.n_samples,
        )
        self._optuna_study = optuna.create_study(
            direction=config.direction,
            sampler=sampler,
            study_name=self.study_id,
        )
        self._prefetch = prefetch or []
        self._seq = 0
        self._active_axes = config.space.active_axes(config.context)
        self._exhausted = False

    # --- lifecycle ---------------------------------------------------------

    def persist_header(self):
        self.storage.record_study(
            study_id=self.study_id,
            name=self.cfg.name,
            strategy=self.cfg.strategy,
            metric_name=self.cfg.metric_name,
            direction=self.cfg.direction,
            space_yaml=self.cfg.space.to_yaml(),
            endpoint_slug=self.cfg.endpoint_slug,
            profile_slugs=self.cfg.profile_slugs,
            status="running",
            notes=self.cfg.notes,
        )

    def enqueue_warmstart(self, seeds: list[dict[str, Any]]):
        """Seed Optuna with prior (params, score) or just params dicts.
        Accepts either a bare params dict or a (params, value) tuple.
        """
        for item in seeds:
            if isinstance(item, tuple):
                params, value = item
                self._optuna_study.add_trial(
                    optuna.trial.create_trial(
                        params=params,
                        distributions=_distributions_for(self._active_axes, params),
                        value=float(value),
                    )
                )
            else:
                self._optuna_study.enqueue_trial(item)

    # --- ask / tell / run --------------------------------------------------

    def ask(self) -> Trial:
        self._seq += 1
        # Prefetched LHC samples take priority.
        if self._prefetch:
            self._optuna_study.enqueue_trial(self._prefetch.pop(0))
        ot = self._optuna_study.ask()
        params: dict[str, Any] = {}
        for axis in self._active_axes:
            params[axis.name] = suggest_from_axis(ot, axis)
        trial = Trial(
            trial_id=f"tr-{ULID()}",
            study_id=self.study_id,
            seq=self._seq,
            params=params,
            status=TrialStatus.PENDING,
            backend="inline",
            _optuna_trial=ot,
        )
        self.storage.record_trial(
            trial.trial_id, trial.study_id, trial.seq, trial.params,
            status=trial.status.value, backend=trial.backend,
        )
        return trial

    def tell(self, trial: Trial, result: ObjectiveResult):
        trial.score = result.score
        trial.metrics = dict(result.metrics)
        if result.error:
            trial.status = TrialStatus.CRASH
            trial.error = result.error
            state = optuna.trial.TrialState.FAIL
            value: float | None = None
        elif not result.accepted:
            trial.status = TrialStatus.PRUNED
            state = optuna.trial.TrialState.FAIL
            value = None
        else:
            trial.status = TrialStatus.COMPLETED
            state = optuna.trial.TrialState.COMPLETE
            value = result.score

        # Optuna's GridSampler calls study.stop() from its after_trial callback
        # when the grid is exhausted; ask/tell loops surface this as a RuntimeError.
        # We treat that as normal exhaustion (logged and swallowed).
        try:
            if state is optuna.trial.TrialState.COMPLETE:
                self._optuna_study.tell(trial._optuna_trial, value)
            else:
                self._optuna_study.tell(trial._optuna_trial, state=state)
        except RuntimeError as e:
            if "Study.stop" in str(e):
                log.info("study %s: sampler exhausted (ok)", self.study_id)
                self._exhausted = True
            else:
                raise

        self.storage.record_trial(
            trial.trial_id, trial.study_id, trial.seq, trial.params,
            status=trial.status.value, score=trial.score,
            backend=trial.backend, worker_id=trial.worker_id,
            error=trial.error, completed=True,
        )
        self.storage.record_trial_metrics(trial.trial_id, trial.metrics)

    def run(
        self,
        objective: Objective,
        *,
        max_trials: int,
        on_trial: Callable[[Trial, ObjectiveResult], None] | None = None,
    ) -> list[Trial]:
        self.persist_header()
        log.info("study %s: start strategy=%s max_trials=%d", self.study_id,
                 self.cfg.strategy, max_trials)
        out: list[Trial] = []
        for i in range(int(max_trials)):
            if self._exhausted:
                break
            try:
                trial = self.ask()
            except Exception as e:  # e.g., GridSampler exhausted mid-ask
                log.info("study %s: ask() exhausted at seq=%d: %s", self.study_id, i + 1, e)
                break
            t0 = time.time()
            try:
                result = objective(trial.params)
            except Exception as e:  # noqa: BLE001
                result = ObjectiveResult(score=0.0, error=f"objective raised: {e}", accepted=False)
            dt = time.time() - t0
            self.tell(trial, result)
            log.info(
                "study %s trial %d: status=%s score=%s dt=%.1fs",
                self.study_id, trial.seq, trial.status.value, trial.score, dt,
            )
            if on_trial:
                on_trial(trial, result)
            out.append(trial)

        self.storage.set_study_status(self.study_id, "completed", finished=True)
        return out


def _distributions_for(axes, params: dict) -> dict:
    """Build Optuna distribution objects for `create_trial(...)` warm-start."""
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    d: dict = {}
    for a in axes:
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
    return d


def load_space_from_path(path: str | Path) -> SearchSpace:
    from bench.search.space import load_space as _ls
    return _ls(path)
