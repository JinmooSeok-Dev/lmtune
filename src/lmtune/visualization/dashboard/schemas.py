"""InferenceX-app 호환 dashboard schema (Pydantic, extra='forbid').

본 schema 는 SemiAnalysisAI/InferenceX-app 의 packages/db/queries/ 와 동형.
미래에 InferenceX-app 을 fork → datasource adapter 만 우리 DuckDB 로 교체하면
라이브 대시보드 plug-and-play 가능.

field 명은 `tests/dashboard/test_inferencex_schema.py` 가 lock-in 함 — 임의 변경 금지.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrialPoint(_Strict):
    """One trial of a study, projected for dashboard plots."""
    trial_id: str
    seq: int
    score: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    """Flat dict keyed `<metric>.<workload>` (e.g. `throughput_tok_avg.short`)."""


class StudyCard(_Strict):
    """One study card on the dashboard's main matrix view (studies_index.json)."""
    study_id: str
    name: str
    strategy: str
    direction: str
    status: str
    n_trials: int
    n_completed: int
    top_score: float | None = None
    endpoint_slug: str | None = None
    profile_slugs: list[str] = Field(default_factory=list)
    created_at: str | None = None
    finished_at: str | None = None


class StudiesIndex(_Strict):
    """Top-level wrapper for studies_index.json."""
    studies: list[StudyCard] = Field(default_factory=list)


class ThroughputVsLatency(_Strict):
    """Per-study throughput-vs-latency curve (throughput_vs_latency.json)."""
    study_id: str
    model_id: str | None = None
    framework: str | None = None
    hardware_id: str | None = None
    workload: str | None = None
    points: list[TrialPoint] = Field(default_factory=list)


class PerfHistoryEntry(_Strict):
    """Single perf-changelog.yaml entry as a timeline event."""
    config_keys: list[str] = Field(default_factory=list)
    description: list[str] = Field(default_factory=list)
    pr_link: str | None = None
    evals_only: bool = False
    landed_at: str | None = None


class PerfHistory(_Strict):
    """Top-level wrapper for perf_history.json."""
    entries: list[PerfHistoryEntry] = Field(default_factory=list)
