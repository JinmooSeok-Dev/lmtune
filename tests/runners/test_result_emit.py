"""runartifact_to_result() — RunArtifact → BenchmarkResult 변환 helper 검증."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from lmtune.contracts import (
    BenchmarkResult,
    MetricRecord,
    RequestRecord,
    RunRecord,
    SessionRecord,
    TrajectoryEventRecord,
    to_records,
)
from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec, SyntheticWorkload
from lmtune.runners.base import RequestRow, RunArtifact, SessionRow, TrajectoryEvent
from lmtune.runners.result_emit import _epoch_to_dt, runartifact_to_result


@pytest.fixture
def profile() -> ProfileSpec:
    return ProfileSpec(
        slug="autotune-short",
        name="Autotune short",
        stage=1,
        runner="guidellm",
        mode="concurrency",
        workload=SyntheticWorkload(
            source="synthetic",
            synthetic_input_tokens_mean=256,
            output_tokens_mean=128,
            concurrency=8,
            request_count=20,
        ),
    )


@pytest.fixture
def endpoint() -> EndpointSpec:
    return EndpointSpec(
        slug="local-vllm",
        name="Local vLLM",
        url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )


# ─── _epoch_to_dt ────────────────────────────────────────────────────


def test_epoch_to_dt_none():
    assert _epoch_to_dt(None) is None


def test_epoch_to_dt_value():
    dt = _epoch_to_dt(1714000000.0)
    assert dt is not None
    assert dt.tzinfo is UTC
    assert dt == datetime(2024, 4, 24, 23, 6, 40, tzinfo=UTC)


# ─── 최소 RunArtifact 변환 ───────────────────────────────────────────


def test_minimal_artifact_to_result(tmp_path, profile, endpoint):
    raw_dir = tmp_path / "run-1"
    raw_dir.mkdir()
    artifact = RunArtifact(
        run_id="01ABC",
        runner_kind="guidellm",
        command=["guidellm", "benchmark"],
        raw_dir=raw_dir,
        stdout_path=raw_dir / "stdout.log",
        stderr_path=raw_dir / "stderr.log",
        tool_version="0.6.0",
        started_at=1714000000.0,
        finished_at=1714000060.0,
    )

    result = runartifact_to_result(artifact, profile, endpoint)

    assert isinstance(result, BenchmarkResult)
    assert result.run_id == "01ABC"
    assert result.profile_slug == "autotune-short"
    assert result.endpoint_slug == "local-vllm"
    assert result.runner_kind == "guidellm"
    assert result.tool_version == "0.6.0"
    assert result.status == "ok"
    assert result.started_at is not None
    assert result.started_at.tzinfo is UTC
    assert result.metrics == {}
    assert result.requests == []
    assert result.raw_artifact_path == str(raw_dir)


# ─── 컨텍스트 필드 pass-through ──────────────────────────────────────


def test_artifact_to_result_context_fields(tmp_path, profile, endpoint):
    artifact = RunArtifact(
        run_id="r1",
        runner_kind="guidellm",
        command=[],
        raw_dir=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
    )

    result = runartifact_to_result(
        artifact,
        profile,
        endpoint,
        profile_yaml="kind: ProfileSpec\n",
        endpoint_meta_json='{"deployment": null}',
        git_sha="abc1234",
        tool_versions={"vllm": "0.7.0", "guidellm": "0.6.0"},
        trial_id="t-99",
    )

    assert result.profile_yaml == "kind: ProfileSpec\n"
    assert result.endpoint_meta == '{"deployment": null}'
    assert result.git_sha == "abc1234"
    assert result.tool_versions == {"vllm": "0.7.0", "guidellm": "0.6.0"}
    assert result.trial_id == "t-99"


# ─── metrics + requests + sessions + trajectory 합쳐서 변환 ──────────


def test_artifact_with_full_payload(tmp_path, profile, endpoint):
    artifact = RunArtifact(
        run_id="r1",
        runner_kind="guidellm",
        command=[],
        raw_dir=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
        metrics={
            "ttft": {"p50": 26.8, "p99": 192.5},
            "throughput_tok": {"avg": 130.4},
        },
        requests=[
            RequestRow(
                req_id="req-1",
                turn_idx=0,
                conversation_id="conv-1",
                input_tokens=256,
                output_tokens=64,
                ttft_ms=42.0,
                e2e_ms=580.0,
                started_at=1714000000.0,
                completed_at=1714000000.6,
                status="ok",
            ),
            RequestRow(req_id="req-2", input_tokens=320, output_tokens=128),
        ],
        sessions=[
            SessionRow(
                session_id="s-1",
                task_id="task-A",
                turn_count=3,
                total_input_tokens=900,
                success=True,
            ),
        ],
        trajectory=[
            TrajectoryEvent(session_id="s-1", seq=0, event_type="user", ts=1714000000.0),
            TrajectoryEvent(
                session_id="s-1",
                seq=1,
                event_type="tool_call",
                metadata={"tool": "shell"},
            ),
        ],
    )

    result = runartifact_to_result(artifact, profile, endpoint)

    assert result.metrics == artifact.metrics
    assert len(result.requests) == 2
    r0 = result.requests[0]
    assert r0.req_id == "req-1"
    assert r0.turn_idx == 0
    assert r0.conversation_id == "conv-1"
    assert r0.ttft_ms == 42.0
    assert r0.started_at is not None
    assert r0.started_at.tzinfo is UTC
    assert r0.completed_at is not None

    assert len(result.sessions) == 1
    assert result.sessions[0].session_id == "s-1"
    assert result.sessions[0].success is True

    assert len(result.trajectory) == 2
    assert result.trajectory[0].event_type == "user"
    assert result.trajectory[0].ts is not None
    assert result.trajectory[1].metadata == {"tool": "shell"}


# ─── error/status 전파 ─────────────────────────────────────────────


def test_artifact_failed_status_propagates(tmp_path, profile, endpoint):
    artifact = RunArtifact(
        run_id="r1",
        runner_kind="guidellm",
        command=[],
        raw_dir=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
        status="failed",
        error="exit=1",
    )
    result = runartifact_to_result(artifact, profile, endpoint)
    assert result.status == "failed"
    assert result.error == "exit=1"


# ─── E2E: artifact → result → to_records → store.put ─────────────────


def test_artifact_to_result_into_store(tmp_path, profile, endpoint):
    """artifact → result → to_records → InMemoryArtifactStore.put 까지 round-trip."""
    from lmtune.contracts import QuerySpec
    from lmtune.storage.store import InMemoryArtifactStore

    artifact = RunArtifact(
        run_id="r1",
        runner_kind="guidellm",
        command=[],
        raw_dir=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
        metrics={"ttft": {"p99": 192.5}},
        requests=[RequestRow(req_id="req-1", ttft_ms=42.0)],
        sessions=[SessionRow(session_id="s-1", success=True)],
        trajectory=[TrajectoryEvent(session_id="s-1", seq=0, event_type="user")],
    )

    result = runartifact_to_result(artifact, profile, endpoint, trial_id="t-99", git_sha="abc")
    records = to_records(result)
    # 1 run + 1 metric + 1 request + 1 session + 1 trajectory = 5
    assert len(records) == 5

    store = InMemoryArtifactStore()
    n = store.put(records)
    assert n == 5

    runs = store.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert isinstance(runs[0], RunRecord)
    assert runs[0].trial_id == "t-99"
    assert runs[0].git_sha == "abc"

    metrics = store.query(QuerySpec(record_kind="metric"))
    assert len(metrics) == 1
    assert isinstance(metrics[0], MetricRecord)
    assert metrics[0].value == 192.5

    reqs = store.query(QuerySpec(record_kind="request"))
    assert len(reqs) == 1
    assert isinstance(reqs[0], RequestRecord)
    assert reqs[0].ttft_ms == 42.0

    sessions = store.query(QuerySpec(record_kind="session"))
    assert len(sessions) == 1
    assert isinstance(sessions[0], SessionRecord)
    assert sessions[0].success is True

    trajs = store.query(QuerySpec(record_kind="trajectory_event"))
    assert len(trajs) == 1
    assert isinstance(trajs[0], TrajectoryEventRecord)
    assert trajs[0].event_type == "user"
