"""BenchmarkResult — runner 가 emit 하는 한 번의 벤치마크 실행 결과.

설계 목적:
- 4 runner (aiperf, vllm_bench, guidellm, raw_openai) 모두 BenchmarkResult 의
  단일 형식으로 결과 emit.
- BenchmarkResult → list[RecordSpec] 변환 (`to_records()`) 으로 ArtifactStore
  와 직결. driver/output 이 store ABC 만 의존.
- 기존 RunArtifact (dataclass) 를 대체하지 않음. runner 가 RunArtifact →
  BenchmarkResult 변환 후 emit. 후속 PR (R0-rt) 에서 wire-up.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestEntry(BaseModel):
    """단일 request 의 latency + agent 메타. RequestRecord 의 Pydantic mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    req_id: str
    turn_idx: int | None = None
    conversation_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    thinking_tokens: int | None = None
    tool_call_count: int | None = None
    tool_result_tokens: int | None = None
    phase: str | None = None
    role: str | None = None
    energy_wh: float | None = None
    cost_usd: float | None = None
    ttft_ms: float | None = None
    itl_mean_ms: float | None = None
    e2e_ms: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str | None = None
    error: str | None = None


class SessionEntry(BaseModel):
    """다중 turn session 집계. SessionRecord 의 Pydantic mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class TrajectoryEntry(BaseModel):
    """Agent 의 단일 step. TrajectoryEventRecord 의 Pydantic mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    seq: int
    event_type: str  # user | assistant | tool_call | tool_result | thinking
    ts: datetime | None = None
    phase: str | None = None
    tokens: int | None = None
    metadata: dict[str, Any] | None = None


class BenchmarkResult(BaseModel):
    """단일 벤치마크 실행의 정형 결과.

    runner 의 단일 emission 단위. driver 가 BenchmarkResult 를 받아
    `to_records()` 로 list[RecordSpec] 변환 후 ArtifactStore.put() 호출.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_version: Literal["lmtune/result/v1alpha1"] = "lmtune/result/v1alpha1"
    kind: Literal["BenchmarkResult"] = "BenchmarkResult"

    # ── identity / context ────────────────────────────────────────
    run_id: str
    profile_slug: str
    endpoint_slug: str
    runner_kind: str  # aiperf | vllm_bench | guidellm | raw_openai | ...
    tool_version: str | None = None
    git_sha: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # ── outcome ────────────────────────────────────────────────────
    status: str = "ok"  # ok | error | crash
    error: str | None = None

    # ── observations ───────────────────────────────────────────────
    # metrics: {metric_name: {p_or_avg: value}}. ex: {"ttft": {"p99": 500.0, "avg": 250.0}}
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    requests: list[RequestEntry] = Field(default_factory=list)
    sessions: list[SessionEntry] = Field(default_factory=list)
    trajectory: list[TrajectoryEntry] = Field(default_factory=list)

    # ── 첨부 ────────────────────────────────────────────────────────
    profile_yaml: str | None = None  # 프로필 yaml 텍스트 (snapshot)
    endpoint_meta: str | None = None  # JSON string (resolved 후 메타)
    tool_versions: dict[str, Any] | None = None
    raw_artifact_path: str | None = None  # data/raw/<run_id>/ 등

    # ── search 연결 (선택) ─────────────────────────────────────────
    trial_id: str | None = None
