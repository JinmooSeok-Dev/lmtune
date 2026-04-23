from __future__ import annotations

import time
from pathlib import Path

from bench.endpoints import load_endpoint
from bench.profiles import load_profile
from bench.runners.base import RequestRow, RunArtifact, SessionRow, TrajectoryEvent
from bench.storage import DuckDBStore


ROOT = Path(__file__).resolve().parents[1]


def test_schema_has_new_columns(tmp_path):
    store = DuckDBStore(tmp_path / "b.duckdb")
    tables = {r[0] for r in store.conn.execute("SHOW TABLES").fetchall()}
    assert {"sessions", "trajectory_events"} <= tables
    cols = {r[0] for r in store.conn.execute("DESCRIBE requests").fetchall()}
    for new_col in [
        "cached_tokens", "thinking_tokens", "tool_call_count", "tool_result_tokens",
        "phase", "role", "energy_wh", "cost_usd", "started_at", "completed_at",
    ]:
        assert new_col in cols, f"missing column: {new_col}"
    store.close()


def test_request_row_roundtrip_with_agent_metadata(tmp_path):
    store = DuckDBStore(tmp_path / "b.duckdb")
    profile = load_profile(ROOT / "configs/profiles/smoke.yaml")
    endpoint = load_endpoint(ROOT / "configs/endpoints/local_vllm.yaml")

    raw = tmp_path / "raw"
    raw.mkdir()
    art = RunArtifact(
        run_id="r-e1",
        runner_kind="aiperf",
        command=["echo"],
        raw_dir=raw,
        stdout_path=raw / "o.log",
        stderr_path=raw / "e.log",
        status="ok",
        started_at=time.time(),
        finished_at=time.time() + 1,
        requests=[
            RequestRow(
                req_id="req-0", turn_idx=0, conversation_id="c0",
                input_tokens=1200, output_tokens=300,
                cached_tokens=800, thinking_tokens=200, tool_call_count=3,
                tool_result_tokens=1500, phase="exploration", role="planner",
                energy_wh=0.015, cost_usd=0.0012,
                ttft_ms=220.0, itl_mean_ms=18.0, e2e_ms=2400.0,
                started_at=time.time(), completed_at=time.time() + 2,
            )
        ],
        sessions=[
            SessionRow(
                session_id="c0", task_id="swe-bench/issue-1",
                total_input_tokens=15000, total_output_tokens=3000,
                total_cached_tokens=8000, turn_count=12, tool_call_count=25,
                duration_ms=120_000, success=True,
                total_cost_usd=0.15, total_energy_wh=1.8,
            )
        ],
        trajectory=[
            TrajectoryEvent(session_id="c0", seq=0, event_type="user", phase="exploration", tokens=200),
            TrajectoryEvent(session_id="c0", seq=1, event_type="assistant", phase="exploration", tokens=150),
            TrajectoryEvent(session_id="c0", seq=2, event_type="tool_call", tokens=100, metadata={"tool": "read_file"}),
            TrajectoryEvent(session_id="c0", seq=3, event_type="tool_result", tokens=2000),
        ],
    )
    store.record_run(art, profile, endpoint, profile_yaml_text="slug: smoke\n")

    # requests 에서 신규 컬럼 읽기
    row = store.conn.execute(
        "SELECT cached_tokens, thinking_tokens, tool_call_count, phase, role, cost_usd "
        "FROM requests WHERE run_id='r-e1'"
    ).fetchone()
    assert row == (800, 200, 3, "exploration", "planner", 0.0012)

    # sessions 행 존재
    srow = store.conn.execute(
        "SELECT session_id, task_id, total_input_tokens, turn_count, success FROM sessions WHERE run_id='r-e1'"
    ).fetchone()
    assert srow == ("c0", "swe-bench/issue-1", 15000, 12, True)

    # trajectory 행
    tcount = store.conn.execute(
        "SELECT COUNT(*) FROM trajectory_events WHERE run_id='r-e1' AND session_id='c0'"
    ).fetchone()[0]
    assert tcount == 4

    # 이벤트 타입 분포
    types = {r[0] for r in store.conn.execute(
        "SELECT DISTINCT event_type FROM trajectory_events WHERE run_id='r-e1'"
    ).fetchall()}
    assert types == {"user", "assistant", "tool_call", "tool_result"}
    store.close()
