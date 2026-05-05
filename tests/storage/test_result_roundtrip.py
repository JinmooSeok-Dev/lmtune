"""BenchmarkResult ↔ DuckDBArtifactStore round-trip.

R0 (#43) 의 BenchmarkResult, R0-rt (#44) 의 runartifact_to_result,
SS (#45) 의 DuckDBArtifactStore 가 통합되어 동작함을 입증.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lmtune.contracts import (
    BenchmarkResult,
    FilterCond,
    QuerySpec,
    RequestRecord,
    RunRecord,
    SessionRecord,
    SortKey,
    TrajectoryEventRecord,
    to_records,
)
from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec, SyntheticWorkload
from lmtune.runners.base import RequestRow, RunArtifact, SessionRow, TrajectoryEvent
from lmtune.runners.result_emit import runartifact_to_result
from lmtune.storage.store import DuckDBArtifactStore


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


def _make_full_artifact(tmp_path: Path) -> RunArtifact:
    return RunArtifact(
        run_id="r1",
        runner_kind="guidellm",
        command=[],
        raw_dir=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
        tool_version="0.6.0",
        status="ok",
        metrics={
            "ttft": {"p50": 26.8, "p99": 192.5},
            "throughput_tok": {"avg": 130.4},
        },
        requests=[
            RequestRow(
                req_id="req-1",
                turn_idx=0,
                input_tokens=256,
                output_tokens=64,
                ttft_ms=42.0,
                e2e_ms=580.0,
                started_at=1714000000.0,
                completed_at=1714000000.6,
                status="ok",
            ),
        ],
        sessions=[
            SessionRow(session_id="s-1", task_id="task-A", turn_count=3, success=True),
        ],
        trajectory=[
            TrajectoryEvent(session_id="s-1", seq=0, event_type="user", ts=1714000000.0),
            TrajectoryEvent(
                session_id="s-1", seq=1, event_type="tool_call", metadata={"tool": "shell"}
            ),
        ],
        started_at=1714000000.0,
        finished_at=1714000060.0,
    )


# ─── 핵심 round-trip ────────────────────────────────────────────────


def test_artifact_to_result_to_records_to_duckdb_roundtrip(tmp_path, profile, endpoint):
    """RunArtifact → BenchmarkResult → to_records → DuckDB → query → 동일 데이터."""
    artifact = _make_full_artifact(tmp_path)
    result = runartifact_to_result(
        artifact,
        profile,
        endpoint,
        git_sha="abc123",
        tool_versions={"guidellm": "0.6.0"},
        trial_id="t-99",
    )
    records = to_records(result)
    # 1 run + 3 metric (ttft p50/p99 + throughput avg) + 1 request + 1 session + 2 trajectory
    assert len(records) == 8

    db = tmp_path / "round.duckdb"
    store = DuckDBArtifactStore(db)
    n = store.put(records)
    assert n == 8

    # ── run ──────────────────────────────────────────────────────────
    runs = store.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert isinstance(runs[0], RunRecord)
    assert runs[0].run_id == "r1"
    assert runs[0].profile_slug == "autotune-short"
    assert runs[0].endpoint_slug == "local-vllm"
    assert runs[0].runner == "guidellm"
    assert runs[0].trial_id == "t-99"
    assert runs[0].git_sha == "abc123"
    assert runs[0].tool_versions == {"guidellm": "0.6.0"}

    # ── metrics ──────────────────────────────────────────────────────
    metrics = store.query(QuerySpec(record_kind="metric"))
    assert len(metrics) == 3
    by_key = {(m.metric, m.p): m.value for m in metrics}
    assert by_key[("ttft", "p50")] == 26.8
    assert by_key[("ttft", "p99")] == 192.5
    assert by_key[("throughput_tok", "avg")] == 130.4

    # filter — ttft 만
    ttft_only = store.query(
        QuerySpec(
            record_kind="metric",
            filters=[FilterCond(column="metric", op="==", value="ttft")],
        )
    )
    assert len(ttft_only) == 2
    assert all(m.metric == "ttft" for m in ttft_only)

    # ── request ──────────────────────────────────────────────────────
    reqs = store.query(QuerySpec(record_kind="request"))
    assert len(reqs) == 1
    r = reqs[0]
    assert isinstance(r, RequestRecord)
    assert r.req_id == "req-1"
    assert r.input_tokens == 256
    assert r.output_tokens == 64
    assert r.ttft_ms == 42.0
    assert r.e2e_ms == 580.0
    assert r.started_at is not None
    # DuckDB TIMESTAMP 는 naive 로 보관 — round-trip 후엔 tzinfo=None.
    # tz-aware 보관 경로는 후속 PR (TIMESTAMPTZ migration) 에서.
    assert r.started_at.replace(tzinfo=None) == datetime(2024, 4, 25, 8, 6, 40).replace(tzinfo=None)

    # ── session ──────────────────────────────────────────────────────
    sessions = store.query(QuerySpec(record_kind="session"))
    assert len(sessions) == 1
    assert isinstance(sessions[0], SessionRecord)
    assert sessions[0].session_id == "s-1"
    assert sessions[0].task_id == "task-A"
    assert sessions[0].success is True
    assert sessions[0].turn_count == 3

    # ── trajectory ───────────────────────────────────────────────────
    trajs = store.query(
        QuerySpec(
            record_kind="trajectory_event",
            sort=[SortKey(column="seq", direction="asc")],
        )
    )
    assert len(trajs) == 2
    assert isinstance(trajs[0], TrajectoryEventRecord)
    assert trajs[0].event_type == "user"
    assert trajs[0].seq == 0
    assert trajs[1].event_type == "tool_call"
    assert trajs[1].metadata == {"tool": "shell"}

    store.close()


# ─── BenchmarkResult JSON round-trip via R0 contract ──────────────────


def test_benchmark_result_json_roundtrip_into_duckdb(tmp_path, profile, endpoint):
    """raw_dir/<run_id>/result.json 시나리오 — JSON 으로 직렬화/역직렬화 후 DuckDB 적재."""
    import json

    artifact = _make_full_artifact(tmp_path)
    result = runartifact_to_result(artifact, profile, endpoint)

    # 직렬화 → 디스크 → 역직렬화 (R0-bridge 의 cli.py path 와 동등)
    payload = result.model_dump_json(exclude_none=True)
    parsed = json.loads(payload)
    rebuilt = BenchmarkResult.model_validate(parsed)

    # 결과 동등 (frozen + extra forbid → field 단위 비교)
    assert rebuilt.run_id == result.run_id
    assert rebuilt.metrics == result.metrics
    assert len(rebuilt.requests) == len(result.requests)
    assert len(rebuilt.sessions) == len(result.sessions)
    assert len(rebuilt.trajectory) == len(result.trajectory)

    # rebuilt → DuckDB 적재
    db = tmp_path / "rt.duckdb"
    store = DuckDBArtifactStore(db)
    store.put(to_records(rebuilt))
    assert store.count("run") == 1
    assert store.count("metric") == 3
    assert store.count("request") == 1
    assert store.count("session") == 1
    assert store.count("trajectory_event") == 2
    store.close()


# ─── empty / minimal artifact 도 round-trip ──────────────────────────


def test_minimal_result_roundtrip(tmp_path):
    """metrics 없는 minimal BenchmarkResult 도 정상 적재 + 조회."""
    minimal = BenchmarkResult(
        run_id="rmin",
        profile_slug="p",
        endpoint_slug="e",
        runner_kind="guidellm",
    )
    db = tmp_path / "min.duckdb"
    store = DuckDBArtifactStore(db)
    store.put(to_records(minimal))
    assert store.count("run") == 1
    assert store.count("metric") == 0
    assert store.count("request") == 0
    runs = store.query(QuerySpec(record_kind="run"))
    assert runs[0].run_id == "rmin"
    store.close()


# ─── 동일 RunRecord 재 put 시 INSERT OR REPLACE ──────────────────────


def test_run_idempotent_replace(tmp_path):
    """같은 run_id 의 BenchmarkResult 재 put 시 1 행만 존재 (INSERT OR REPLACE)."""
    initial = BenchmarkResult(
        run_id="r1",
        profile_slug="p",
        endpoint_slug="e",
        runner_kind="guidellm",
        status="ok",
    )
    updated = BenchmarkResult(
        run_id="r1",
        profile_slug="p",
        endpoint_slug="e",
        runner_kind="guidellm",
        status="failed",
        error="oops",
    )
    db = tmp_path / "idem.duckdb"
    store = DuckDBArtifactStore(db)
    store.put(to_records(initial))
    store.put(to_records(updated))
    runs = store.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].error == "oops"
    store.close()
