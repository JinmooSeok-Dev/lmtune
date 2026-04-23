"""Trial — a single (params → score) candidate within a Study."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    CRASH = "crash"


@dataclass(slots=True)
class Trial:
    trial_id: str
    study_id: str
    seq: int
    params: dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    score: float | None = None
    metrics: dict[tuple[str, str | None], float] = field(default_factory=dict)
    backend: str | None = None
    worker_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    # optuna handle; not persisted (internal plumbing for ask/tell).
    _optuna_trial: Any = None
