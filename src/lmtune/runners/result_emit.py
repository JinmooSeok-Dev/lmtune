"""RunArtifact → BenchmarkResult 변환.

기존 runner 의 dataclass 결과 (`RunArtifact`) 를 R0 contract (`BenchmarkResult`)
로 변환하는 단일 helper. driver 가 다음 패턴으로 호출:

    artifact = runner.run(...)
    result = runartifact_to_result(artifact, profile, endpoint, ...)
    records = to_records(result)
    store.put(records)

기존 RunArtifact 구조 변경 0. runner.run() 의 시그니처 보존. 본 helper 만 추가.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lmtune.contracts.result_spec import (
    BenchmarkResult,
    RequestEntry,
    SessionEntry,
    TrajectoryEntry,
)
from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec
from lmtune.runners.base import RequestRow, RunArtifact, SessionRow, TrajectoryEvent


def _epoch_to_dt(v: float | None) -> datetime | None:
    """epoch float → tz-aware UTC datetime."""
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=UTC)


def _request_row_to_entry(r: RequestRow) -> RequestEntry:
    return RequestEntry(
        req_id=r.req_id,
        turn_idx=r.turn_idx,
        conversation_id=r.conversation_id,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cached_tokens=r.cached_tokens,
        thinking_tokens=r.thinking_tokens,
        tool_call_count=r.tool_call_count,
        tool_result_tokens=r.tool_result_tokens,
        phase=r.phase,
        role=r.role,
        energy_wh=r.energy_wh,
        cost_usd=r.cost_usd,
        ttft_ms=r.ttft_ms,
        itl_mean_ms=r.itl_mean_ms,
        e2e_ms=r.e2e_ms,
        started_at=_epoch_to_dt(r.started_at),
        completed_at=_epoch_to_dt(r.completed_at),
        status=r.status,
        error=r.error,
    )


def _session_row_to_entry(s: SessionRow) -> SessionEntry:
    return SessionEntry(
        session_id=s.session_id,
        task_id=s.task_id,
        total_input_tokens=s.total_input_tokens,
        total_output_tokens=s.total_output_tokens,
        total_cached_tokens=s.total_cached_tokens,
        turn_count=s.turn_count,
        tool_call_count=s.tool_call_count,
        duration_ms=s.duration_ms,
        success=s.success,
        total_cost_usd=s.total_cost_usd,
        total_energy_wh=s.total_energy_wh,
    )


def _trajectory_event_to_entry(ev: TrajectoryEvent) -> TrajectoryEntry:
    return TrajectoryEntry(
        session_id=ev.session_id,
        seq=ev.seq,
        event_type=ev.event_type,
        ts=_epoch_to_dt(ev.ts),
        phase=ev.phase,
        tokens=ev.tokens,
        metadata=ev.metadata,
    )


def runartifact_to_result(
    artifact: RunArtifact,
    profile: ProfileSpec,
    endpoint: EndpointSpec,
    *,
    profile_yaml: str | None = None,
    endpoint_meta_json: str | None = None,
    git_sha: str | None = None,
    tool_versions: dict | None = None,
    trial_id: str | None = None,
) -> BenchmarkResult:
    """RunArtifact + 컨텍스트 → BenchmarkResult.

    Args:
        artifact: runner.run() 의 반환값.
        profile: 본 run 의 ProfileSpec (slug 만 사용).
        endpoint: 본 run 의 EndpointSpec (slug 만 사용).
        profile_yaml: snapshot 용 profile yaml 텍스트 (None 이면 미저장).
        endpoint_meta_json: endpoint 의 resolved meta JSON 문자열.
        git_sha: 현재 commit SHA.
        tool_versions: {tool_name: version} (e.g., {"vllm": "0.7.0"}).
        trial_id: search trial 과 연결될 때.
    """
    return BenchmarkResult(
        run_id=artifact.run_id,
        profile_slug=profile.slug,
        endpoint_slug=endpoint.slug,
        runner_kind=artifact.runner_kind,
        tool_version=artifact.tool_version,
        git_sha=git_sha,
        started_at=_epoch_to_dt(artifact.started_at),
        finished_at=_epoch_to_dt(artifact.finished_at),
        status=artifact.status,
        error=artifact.error,
        metrics=artifact.metrics,
        requests=[_request_row_to_entry(r) for r in artifact.requests],
        sessions=[_session_row_to_entry(s) for s in artifact.sessions],
        trajectory=[_trajectory_event_to_entry(e) for e in artifact.trajectory],
        profile_yaml=profile_yaml,
        endpoint_meta=endpoint_meta_json,
        tool_versions=tool_versions,
        raw_artifact_path=str(artifact.raw_dir),
        trial_id=trial_id,
    )
