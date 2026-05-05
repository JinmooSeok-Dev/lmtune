"""RecordSpec — store 에 들어가는 모든 record 의 envelope.

목적: WorkloadStore / ArtifactStore (lmtune#SS) 가 받는 입력 schema 정형화.
기존 schema.sql (runs/trials/metrics/requests/...) 를 Pydantic 으로 1:1 mirror.

설계 원칙:
- type-tagged union: 각 record 가 `kind` discriminator 로 분기
- timestamp 는 datetime, JSON 컬럼은 dict[str, Any], primitive 는 그대로
- 기존 schema 와 ALTER 없이 동형 (`primary_key()` 가 (run_id,) 등 PK tuple 반환)
- frozen=True (불변) — store insert 후 mutation 금지
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _RecordBase(BaseModel):
    """모든 RecordSpec 의 공통 base. frozen=True 로 불변, extra=forbid 로 stricter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_version: Literal["lmtune/record/v1alpha1"] = "lmtune/record/v1alpha1"

    def primary_key(self) -> tuple[Any, ...]:
        """Store 에서 dedup key. SS PR 의 store.upsert(record) 가 사용."""
        raise NotImplementedError


# ─── runs ────────────────────────────────────────────────────────────


class RunRecord(_RecordBase):
    """runs 테이블 — 단일 벤치마크 실행 메타."""

    kind: Literal["run"] = "run"

    run_id: str
    profile_slug: str
    endpoint_slug: str
    runner: str
    status: str  # ok | error | crash
    started_at: datetime | None = None
    finished_at: datetime | None = None
    profile_yaml: str | None = None
    endpoint_meta: str | None = None  # json string
    git_sha: str | None = None
    tool_versions: dict[str, Any] | None = None
    error: str | None = None
    trial_id: str | None = None  # nullable, search 와 link

    def primary_key(self) -> tuple[str]:
        return (self.run_id,)


# ─── metrics ─────────────────────────────────────────────────────────


class MetricRecord(_RecordBase):
    """metrics 테이블 — run 단위 percentile/avg metric."""

    kind: Literal["metric"] = "metric"

    run_id: str
    metric: str  # ttft | itl | tpot | e2e | throughput_tok | throughput_req | ...
    p: str | None = None  # avg | p50 | p95 | p99 | None for raw
    value: float

    def primary_key(self) -> tuple[str, str, str]:
        return (self.run_id, self.metric, self.p or "")


# ─── requests ────────────────────────────────────────────────────────


class RequestRecord(_RecordBase):
    """requests 테이블 — per-request agent 메타 + latency."""

    kind: Literal["request"] = "request"

    run_id: str
    req_id: str
    turn_idx: int | None = None
    conversation_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    thinking_tokens: int | None = None
    tool_call_count: int | None = None
    tool_result_tokens: int | None = None
    phase: str | None = None  # exploration | editing | execution | verification | other
    role: str | None = None  # planner | reasoner | verifier | solo
    energy_wh: float | None = None
    cost_usd: float | None = None
    ttft_ms: float | None = None
    itl_mean_ms: float | None = None
    e2e_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str | None = None
    error: str | None = None

    def primary_key(self) -> tuple[str, str]:
        return (self.run_id, self.req_id)


# ─── sessions ────────────────────────────────────────────────────────


class SessionRecord(_RecordBase):
    """sessions 테이블 — 다중 turn session 집계."""

    kind: Literal["session"] = "session"

    run_id: str
    session_id: str
    task_id: str | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cached_tokens: int | None = None
    turn_count: int | None = None
    tool_call_count: int | None = None
    duration_ms: float | None = None
    success: bool | None = None
    total_cost_usd: float | None = None
    total_energy_wh: float | None = None

    def primary_key(self) -> tuple[str, str]:
        return (self.run_id, self.session_id)


# ─── trajectory_events ───────────────────────────────────────────────


class TrajectoryEventRecord(_RecordBase):
    """trajectory_events 테이블 — agent 의 각 step (user/assistant/tool_*)."""

    kind: Literal["trajectory_event"] = "trajectory_event"

    run_id: str
    session_id: str
    seq: int
    event_type: str  # user | assistant | tool_call | tool_result | thinking
    ts: datetime | None = None
    phase: str | None = None
    tokens: int | None = None
    metadata: dict[str, Any] | None = None

    def primary_key(self) -> tuple[str, str, int]:
        return (self.run_id, self.session_id, self.seq)


# ─── prom_samples ────────────────────────────────────────────────────


class PromSampleRecord(_RecordBase):
    """prom_samples 테이블 — Prometheus scrape 한 raw 샘플."""

    kind: Literal["prom_sample"] = "prom_sample"

    run_id: str
    ts: datetime
    metric: str
    value: float
    labels: dict[str, Any] | None = None

    def primary_key(self) -> tuple[str, datetime, str, str]:
        # labels JSON 직렬화로 dedup. 같은 metric+ts 라도 labels 다르면 다른 sample.
        import json

        label_key = json.dumps(self.labels or {}, sort_keys=True) if self.labels else ""
        return (self.run_id, self.ts, self.metric, label_key)


# ─── detections ──────────────────────────────────────────────────────


class DetectionRecord(_RecordBase):
    """detections 테이블 — 룰 기반 anomaly detection 결과."""

    kind: Literal["detection"] = "detection"

    run_id: str
    detector: str
    severity: str  # info | warning | error
    metric: str | None = None
    threshold: float | None = None
    observed: float | None = None
    message: str | None = None
    created_at: datetime | None = None

    def primary_key(self) -> tuple[str, str, str]:
        # detection 은 (run_id, detector, message) 가 unique 가 아닐 수 있어 ts 도 추가
        return (self.run_id, self.detector, str(self.created_at or ""))


# ─── search: studies / trials / trial_metrics ────────────────────────


class StudyRecord(_RecordBase):
    """studies 테이블 — search session 메타."""

    kind: Literal["study"] = "study"

    study_id: str
    name: str
    strategy: str  # grid | random | lhc | tpe | cma_es | nsga2 | botorch | ...
    metric_name: str = "total_score"
    direction: str = "maximize"  # maximize | minimize
    status: str = "running"  # running | paused | completed | aborted
    space_yaml: str | None = None
    endpoint_slug: str | None = None
    profile_slugs: list[str] | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None
    notes: str | None = None

    def primary_key(self) -> tuple[str]:
        return (self.study_id,)


class TrialRecord(_RecordBase):
    """trials 테이블 — search 의 단일 trial."""

    kind: Literal["trial"] = "trial"

    trial_id: str
    study_id: str
    seq: int
    params: dict[str, Any]
    status: str  # pending | running | completed | pruned | crash
    score: float | None = None
    backend: str | None = None
    worker_id: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    def primary_key(self) -> tuple[str]:
        return (self.trial_id,)


class TrialMetricRecord(_RecordBase):
    """trial_metrics 테이블 — trial × metric × workload 의 secondary 값."""

    kind: Literal["trial_metric"] = "trial_metric"

    trial_id: str
    metric: str
    workload: str = "aggregate"  # short | medium | long | aggregate
    value: float | None = None

    def primary_key(self) -> tuple[str, str, str]:
        return (self.trial_id, self.metric, self.workload)


# ─── Discriminated Union ─────────────────────────────────────────────

# 모든 record kind 의 type-tagged union. SS PR 의 store.put(records: list[RecordSpec])
# 가 본 union 을 받으면 kind 로 자동 분기.

RecordSpec = Annotated[
    RunRecord
    | MetricRecord
    | RequestRecord
    | SessionRecord
    | TrajectoryEventRecord
    | PromSampleRecord
    | DetectionRecord
    | StudyRecord
    | TrialRecord
    | TrialMetricRecord,
    Field(discriminator="kind"),
]


RECORD_KINDS: tuple[str, ...] = (
    "run",
    "metric",
    "request",
    "session",
    "trajectory_event",
    "prom_sample",
    "detection",
    "study",
    "trial",
    "trial_metric",
)


def kind_to_class(kind: str) -> type[_RecordBase]:
    """Discriminator value → 모델 클래스. lookup helper."""
    mapping: dict[str, type[_RecordBase]] = {
        "run": RunRecord,
        "metric": MetricRecord,
        "request": RequestRecord,
        "session": SessionRecord,
        "trajectory_event": TrajectoryEventRecord,
        "prom_sample": PromSampleRecord,
        "detection": DetectionRecord,
        "study": StudyRecord,
        "trial": TrialRecord,
        "trial_metric": TrialMetricRecord,
    }
    if kind not in mapping:
        raise ValueError(f"unknown record kind: {kind!r}, valid: {list(mapping)}")
    return mapping[kind]
