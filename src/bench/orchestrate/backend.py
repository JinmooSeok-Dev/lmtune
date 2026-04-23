"""TrialBackend — abstract submit/poll/cancel interface for distributed execution.

A backend receives a TrialPayload (trial_id + params + endpoint + profiles) and
returns a TrialHandle the Driver polls until a TrialResult is available.

Backends are the single point of distribution variance:
- ProcessPoolBackend → local multi-GPU
- K8sJobBackend       → one Job per trial (Phase S4)
- (future) SSHBackend → remote GPU nodes

Every backend must marshal params + endpoint YAML + profile paths such that
the worker side can reconstruct the Objective. In the ProcessPool case we
pickle a small config; in the K8s case we pass JSON via env vars and read
them inside `bench.orchestrate.trial_runner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrialPayload:
    trial_id: str
    study_id: str
    seq: int
    params: dict[str, Any]
    endpoint_path: str
    profile_paths: list[str]
    repeats: int = 3
    ttft_slo_ms: float = 500.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrialResult:
    trial_id: str
    status: str                                  # completed | crash | pruned
    score: float | None
    metrics: dict[tuple[str, str | None], float] = field(default_factory=dict)
    error: str | None = None
    backend: str | None = None
    worker_id: str | None = None


@dataclass(slots=True)
class TrialHandle:
    """Opaque reference returned by submit(); poll() resolves it to a TrialResult.
    Concrete backends may store a concurrent.futures.Future, a K8s Job name, etc.
    """
    trial_id: str
    backend: str
    ref: Any = None


class TrialBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def submit(self, payload: TrialPayload) -> TrialHandle: ...

    @abstractmethod
    def poll(self, handle: TrialHandle, timeout_s: float | None = None) -> TrialResult | None:
        """Return TrialResult if done, else None. timeout_s=None → non-blocking."""

    @abstractmethod
    def cancel(self, handle: TrialHandle) -> None: ...

    def shutdown(self) -> None:
        """Optional cleanup (pool shutdown, job finalizer)."""
        return None
