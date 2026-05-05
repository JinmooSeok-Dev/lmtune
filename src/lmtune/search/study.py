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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna
from ulid import ULID

from lmtune.orchestrate.failure_handler import (
    CircuitBreaker,
    CircuitBreakerConfig,
    classify_outcome,
)
from lmtune.search.objective import Objective, ObjectiveResult
from lmtune.search.samplers import make_sampler, suggest_from_axis
from lmtune.search.space import SearchSpace
from lmtune.search.trial import Trial, TrialStatus
from lmtune.storage.duckdb_store import DuckDBStore

log = logging.getLogger(__name__)

# Silence Optuna's per-trial info logging; we emit our own at Study level.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class StudyConfig:
    name: str
    strategy: str                      # grid | random | lhc | tpe | cma_es | ucb | nsga2 | botorch
    space: SearchSpace
    metric_name: str = "total_score"
    direction: str = "maximize"        # maximize | minimize (single-objective)
    directions: list[str] | None = None  # multi-objective; e.g. ["maximize","minimize"]
    endpoint_slug: str | None = None
    profile_slugs: list[str] = field(default_factory=list)
    seed: int | None = None
    n_samples: int | None = None       # lhc only
    context: dict[str, Any] | None = None
    study_id: str | None = None
    notes: str | None = None
    pruner: str | None = None          # none | sh | hyperband
    breaker: CircuitBreakerConfig | None = None  # None → use defaults; pass disabled-config to opt out


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
        from lmtune.search.pruners import make_pruner
        pruner = make_pruner(config.pruner) if config.pruner else None
        if config.directions:
            self._optuna_study = optuna.create_study(
                directions=list(config.directions),
                sampler=sampler,
                pruner=pruner,
                study_name=self.study_id,
            )
            self._multi_obj = True
        else:
            self._optuna_study = optuna.create_study(
                direction=config.direction,
                sampler=sampler,
                pruner=pruner,
                study_name=self.study_id,
            )
            self._multi_obj = False
        self._prefetch = prefetch or []
        self._seq = 0
        self._active_axes = config.space.active_axes(config.context)
        self._exhausted = False
        # Circuit breaker — halts the loop on persistent failure (helmfile redeploy +
        # EPP/InferencePool churn 으로 study 가 silently 망가지는 것을 방지).
        self.breaker = CircuitBreaker(cfg=config.breaker or CircuitBreakerConfig())
        self._halt_reason: str | None = None
        # Feasibility — vllm-config-puzzle validation.ts 1:1 port. config.space 에
        # feasibility_constraints 가 있고 + config.context 에 environment 가 있으면
        # ask() 가 sample 후 evaluate, infeasible → tell(PRUNED) 후 retry.
        self._feasibility = _build_feasibility(config)
        self._infeasible_count = 0

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
        # Sample → feasibility check → retry up to N times.  vllm-config-puzzle
        # validation.ts 의 10 룰 위반 시 helmfile redeploy 0회로 즉시 prune.
        max_infeasible_retries = 30
        for _ in range(max_infeasible_retries + 1):
            if self._prefetch:
                self._optuna_study.enqueue_trial(self._prefetch.pop(0))
            ot = self._optuna_study.ask()
            params: dict[str, Any] = {}
            for axis in self._active_axes:
                params[axis.name] = suggest_from_axis(ot, axis)
            if self._feasibility is None or self._feasibility.is_feasible(params):
                break
            # infeasible — tell PRUNED so Optuna learns the rejection signal,
            # then loop and ask() again.
            self._infeasible_count += 1
            log.debug(
                "study %s: infeasible candidate seq~%d params=%s reason=%s",
                self.study_id, self._seq + 1, params,
                self._feasibility.last_reason(),
            )
            try:
                self._optuna_study.tell(ot, state=optuna.trial.TrialState.PRUNED)
            except RuntimeError:
                # Optuna's GridSampler may signal exhaustion here too.
                self._exhausted = True
                raise
        else:
            # Exhausted without finding a feasible candidate.
            raise RuntimeError(
                f"study {self.study_id}: no feasible candidate after "
                f"{max_infeasible_retries} retries (last reason: "
                f"{self._feasibility.last_reason() if self._feasibility else 'n/a'})"
            )

        self._seq += 1
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
            value: float | list[float] | None = None
        elif not result.accepted:
            trial.status = TrialStatus.PRUNED
            state = optuna.trial.TrialState.FAIL
            value = None
        else:
            trial.status = TrialStatus.COMPLETED
            state = optuna.trial.TrialState.COMPLETE
            if self._multi_obj:
                # multi-obj result.score carries obj1; pull the rest from result.metrics
                # via the same ordering the caller configured (directions/objectives).
                # We rely on the caller to stash the list on result.metrics under ("_values", None).
                raw = result.metrics.get(("_values", None))
                value = [float(result.score or 0.0)] if raw is None else [float(v) for v in raw]
            else:
                value = result.score

        # Optuna's GridSampler calls study.stop() from its after_trial callback
        # when the grid is exhausted; ask/tell loops surface this as a RuntimeError.
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
            # Release DuckDB file lock for the duration of the subprocess
            # (bench run → DuckDB). Reacquire to persist the result.
            self.storage.suspend()
            try:
                try:
                    result = objective(trial.params)
                except Exception as e:  # noqa: BLE001
                    result = ObjectiveResult(
                        score=0.0, error=f"objective raised: {e}", accepted=False
                    )
            finally:
                self.storage.resume()
            dt = time.time() - t0
            self.tell(trial, result)
            log.info(
                "study %s trial %d: status=%s score=%s dt=%.1fs",
                self.study_id, trial.seq, trial.status.value, trial.score, dt,
            )
            if on_trial:
                on_trial(trial, result)
            out.append(trial)

            outcome = classify_outcome(
                trial.status.value, error=trial.error, notes=trial.error,
            )
            self.breaker.record(outcome)
            halt, reason = self.breaker.should_halt()
            if halt:
                self._halt_reason = reason
                log.error(
                    "study %s: HALTED at seq=%d — %s; breaker=%s",
                    self.study_id, trial.seq, reason, self.breaker.summary(),
                )
                break

        final_status = "halted" if self._halt_reason else "completed"
        self.storage.set_study_status(self.study_id, final_status, finished=True)
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
    from lmtune.search.space import load_space as _ls
    return _ls(path)


# --- Feasibility wiring -------------------------------------------------------


class _FeasibilityChecker:
    """Wrapper around feasibility.evaluate() — caches Constraint list and the
    bound (environment, model) tuple so per-trial overhead is just one eval pass.
    """

    __slots__ = ("_constraints", "_environment", "_model_name", "_last_reason")

    def __init__(self, constraints, environment, model_name: str | None):
        self._constraints = constraints
        self._environment = environment
        self._model_name = model_name
        self._last_reason: str = "ok"

    def is_feasible(self, params: dict[str, Any]) -> bool:
        from lmtune.models import by_name
        from lmtune.search.feasibility import evaluate as _eval

        model = by_name(self._model_name) if self._model_name else None
        rep = _eval(
            params, environment=self._environment, model=model,
            constraints=self._constraints,
        )
        self._last_reason = rep.reason()
        return rep.feasible

    def last_reason(self) -> str:
        return self._last_reason


def _build_feasibility(cfg: StudyConfig) -> _FeasibilityChecker | None:
    """Construct a checker if the SearchSpace declares constraints AND the
    StudyConfig.context provides an `environment` (and optional `model_id`).

    SearchSpace 만 가지고는 environment (NPU 토폴로지) 를 알 수 없으니 caller
    (CLI 또는 외부 study driver) 가 context 로 명시 주입.
    """
    space = cfg.space
    raw = list(space.feasibility_constraints or [])
    if not raw:
        return None
    ctx = cfg.context or {}
    env = ctx.get("environment")
    if env is None:
        return None
    from lmtune.search.feasibility import Constraint
    constraints = [
        Constraint(
            id=str(e.get("id") or f"c_{i}"),
            rule=str(e["rule"]),
            message=str(e.get("message") or ""),
            severity=str(e.get("severity") or "error"),
        )
        for i, e in enumerate(raw)
        if isinstance(e, dict) and "rule" in e
    ]
    if not constraints:
        return None
    model_name = ctx.get("model_id") or ctx.get("model")
    return _FeasibilityChecker(
        constraints=constraints, environment=env, model_name=model_name,
    )
