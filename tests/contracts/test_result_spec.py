"""BenchmarkResult contract + to_records() 변환 검증."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lmtune.contracts import (
    BenchmarkResult,
    MetricRecord,
    RequestEntry,
    RequestRecord,
    RunRecord,
    SessionEntry,
    SessionRecord,
    TrajectoryEntry,
    TrajectoryEventRecord,
    to_records,
)

# ─── 최소 BenchmarkResult ─────────────────────────────────────────────


def test_minimal_result_construction():
    r = BenchmarkResult(
        run_id="01ABC",
        profile_slug="autotune-short",
        endpoint_slug="local-vllm",
        runner_kind="guidellm",
    )
    assert r.api_version == "lmtune/result/v1alpha1"
    assert r.kind == "BenchmarkResult"
    assert r.status == "ok"
    assert r.metrics == {}
    assert r.requests == []


def test_frozen():
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g"
    )
    with pytest.raises((ValidationError, TypeError)):
        r.run_id = "x"  # type: ignore[misc]


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        BenchmarkResult(
            run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
            _bogus="x",  # type: ignore[call-arg]
        )


# ─── to_records — 다양한 조합 ─────────────────────────────────────────


def test_to_records_minimal():
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="guidellm"
    )
    recs = to_records(r)
    assert len(recs) == 1
    assert isinstance(recs[0], RunRecord)
    assert recs[0].run_id == "r1"


def test_to_records_with_metrics():
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        metrics={
            "ttft": {"p50": 100.0, "p99": 500.0, "avg": 250.0},
            "throughput_tok": {"avg": 140.5},
        },
    )
    recs = to_records(r)
    # 1 RunRecord + 4 MetricRecord (3 ttft + 1 throughput)
    assert len(recs) == 5
    metrics = [r for r in recs if isinstance(r, MetricRecord)]
    assert len(metrics) == 4
    # 모두 같은 run_id
    assert all(m.run_id == "r1" for m in metrics)
    # ttft p99 = 500.0 가 있는지
    ttft_p99 = [m for m in metrics if m.metric == "ttft" and m.p == "p99"]
    assert len(ttft_p99) == 1
    assert ttft_p99[0].value == 500.0


def test_to_records_with_requests():
    ts = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        requests=[
            RequestEntry(
                req_id="req-1", turn_idx=0, input_tokens=256, output_tokens=64,
                ttft_ms=42.0, e2e_ms=580.0, started_at=ts, status="ok",
            ),
            RequestEntry(
                req_id="req-2", turn_idx=1, input_tokens=320, output_tokens=128,
                ttft_ms=55.0, e2e_ms=720.0, started_at=ts, status="ok",
            ),
        ],
    )
    recs = to_records(r)
    # 1 RunRecord + 2 RequestRecord
    assert len(recs) == 3
    reqs = [r for r in recs if isinstance(r, RequestRecord)]
    assert len(reqs) == 2
    assert reqs[0].req_id == "req-1"
    assert reqs[0].ttft_ms == 42.0
    assert reqs[1].req_id == "req-2"


def test_to_records_with_sessions():
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        sessions=[
            SessionEntry(
                session_id="s-1", task_id="task-A", turn_count=3,
                total_input_tokens=900, success=True,
            ),
        ],
    )
    recs = to_records(r)
    sessions = [r for r in recs if isinstance(r, SessionRecord)]
    assert len(sessions) == 1
    assert sessions[0].session_id == "s-1"
    assert sessions[0].turn_count == 3
    assert sessions[0].success is True


def test_to_records_with_trajectory():
    ts = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        trajectory=[
            TrajectoryEntry(session_id="s-1", seq=0, event_type="user", ts=ts),
            TrajectoryEntry(session_id="s-1", seq=1, event_type="assistant",
                            ts=ts, tokens=42),
            TrajectoryEntry(session_id="s-1", seq=2, event_type="tool_call",
                            ts=ts, metadata={"tool": "shell"}),
        ],
    )
    recs = to_records(r)
    trajs = [r for r in recs if isinstance(r, TrajectoryEventRecord)]
    assert len(trajs) == 3
    assert trajs[0].event_type == "user"
    assert trajs[2].metadata == {"tool": "shell"}


def test_to_records_full_combo():
    """metrics + requests + sessions + trajectory 결합."""
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        metrics={"ttft": {"p50": 100.0, "p99": 500.0}},
        requests=[RequestEntry(req_id=f"req-{i}") for i in range(3)],
        sessions=[SessionEntry(session_id=f"s-{i}") for i in range(2)],
        trajectory=[TrajectoryEntry(session_id="s-0", seq=i,
                                     event_type="assistant") for i in range(4)],
    )
    recs = to_records(r)
    # 1 run + 2 metric + 3 request + 2 session + 4 trajectory = 12
    assert len(recs) == 12
    # kind 별 count
    from collections import Counter

    kind_counts = Counter(r.kind for r in recs)
    assert kind_counts["run"] == 1
    assert kind_counts["metric"] == 2
    assert kind_counts["request"] == 3
    assert kind_counts["session"] == 2
    assert kind_counts["trajectory_event"] == 4


def test_to_records_run_record_carries_context():
    """RunRecord 가 trial_id, profile_yaml, tool_versions 등 모두 보유."""
    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        trial_id="t-99", profile_yaml="kind: ProfileSpec\n",
        tool_versions={"vllm": "0.7.0", "guidellm": "0.6.0"},
        git_sha="abc1234",
    )
    recs = to_records(r)
    run = recs[0]
    assert isinstance(run, RunRecord)
    assert run.trial_id == "t-99"
    assert run.git_sha == "abc1234"
    assert run.tool_versions == {"vllm": "0.7.0", "guidellm": "0.6.0"}


# ─── BenchmarkResult ↔ ArtifactStore 통합 ────────────────────────────


def test_to_records_into_in_memory_store():
    """to_records 결과를 store.put() 으로 적재 → 단일 BenchmarkResult 의 모든
    component 가 store 에서 query 가능."""
    from lmtune.contracts import QuerySpec
    from lmtune.storage.store import InMemoryArtifactStore

    r = BenchmarkResult(
        run_id="r1", profile_slug="p", endpoint_slug="e", runner_kind="g",
        metrics={"ttft": {"p99": 500.0}},
        requests=[RequestEntry(req_id="req-1", ttft_ms=42.0)],
    )
    store = InMemoryArtifactStore()
    n = store.put(to_records(r))
    assert n == 3  # 1 run + 1 metric + 1 request

    runs = store.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert runs[0].run_id == "r1"

    metrics = store.query(QuerySpec(record_kind="metric"))
    assert len(metrics) == 1
    assert metrics[0].value == 500.0

    requests = store.query(QuerySpec(record_kind="request"))
    assert len(requests) == 1
    assert requests[0].ttft_ms == 42.0


# ─── CLI dump-schema --kind result ────────────────────────────────────


def test_cli_dump_schema_result(tmp_path):
    from typer.testing import CliRunner

    from lmtune.cli_contracts import app

    runner = CliRunner()
    out = tmp_path / "result.schema.json"
    result = runner.invoke(app, ["dump-schema", "--kind", "result", "--out", str(out)])
    assert result.exit_code == 0, result.output

    import json

    schema = json.loads(out.read_text())
    assert schema["title"] == "BenchmarkResult"
    assert "run_id" in schema["properties"]
    assert "metrics" in schema["properties"]


def test_cli_validate_result_yaml(tmp_path):
    import yaml as yamllib
    from typer.testing import CliRunner

    from lmtune.cli_contracts import app

    runner = CliRunner()
    p = tmp_path / "res.yaml"
    p.write_text(
        yamllib.safe_dump(
            {
                "run_id": "01ABC",
                "profile_slug": "p",
                "endpoint_slug": "e",
                "runner_kind": "guidellm",
                "metrics": {"ttft": {"p99": 500.0}},
            }
        )
    )
    result = runner.invoke(app, ["validate-result", str(p)])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
    assert "run_id=01ABC" in result.output
