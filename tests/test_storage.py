from __future__ import annotations

import time
from pathlib import Path

import pytest

from lmtune.endpoints import load_endpoint
from lmtune.profiles import load_profile
from lmtune.runners.base import RequestRow, RunArtifact
from lmtune.storage import DuckDBStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    s = DuckDBStore(tmp_path / "lmtune.duckdb")
    yield s
    s.close()


def test_schema_created(store):
    tables = {r[0] for r in store.conn.execute("SHOW TABLES").fetchall()}
    assert {"runs", "metrics", "requests", "prom_samples", "detections"} <= tables


def test_record_run_and_read_back(store, tmp_path):
    profile = load_profile(ROOT / "configs/profiles/smoke.yaml")
    endpoint = load_endpoint(ROOT / "configs/endpoints/local_vllm.yaml")

    raw_dir = tmp_path / "raw" / "r1"
    raw_dir.mkdir(parents=True)
    artifact = RunArtifact(
        run_id="r1",
        runner_kind="aiperf",
        command=["echo"],
        raw_dir=raw_dir,
        stdout_path=raw_dir / "stdout.log",
        stderr_path=raw_dir / "stderr.log",
        status="ok",
        tool_version="test-0.0",
        started_at=time.time(),
        finished_at=time.time() + 1,
        metrics={"ttft": {"p50": 100.0, "p99": 500.0}, "e2e": {"p50": 1200.0}},
        requests=[
            RequestRow(req_id="req-0", input_tokens=200, output_tokens=50, ttft_ms=110.0, e2e_ms=1100.0),
            RequestRow(req_id="req-1", input_tokens=210, output_tokens=48, ttft_ms=125.0, e2e_ms=1300.0),
        ],
    )
    store.record_run(artifact, profile, endpoint, profile_yaml_text="slug: smoke\n", git_sha="abc123")

    rows = store.list_runs(limit=10)
    assert len(rows) == 1
    metrics = store.get_metrics("r1")
    assert metrics["ttft"]["p50"] == 100.0
    assert metrics["ttft"]["p99"] == 500.0

    req_count = store.conn.execute("SELECT COUNT(*) FROM requests WHERE run_id='r1'").fetchone()[0]
    assert req_count == 2


def test_record_detections(store):
    store.conn.execute(
        "INSERT INTO runs (run_id, profile_slug, endpoint_slug, status, runner) VALUES ('r2','x','y','ok','aiperf')"
    )
    store.record_detections(
        "r2",
        [
            {
                "detector": "slo_ttft",
                "severity": "warning",
                "metric": "ttft",
                "threshold": 300.0,
                "observed": 450.0,
                "message": "TTFT p99 exceeded SLO",
            }
        ],
    )
    count = store.conn.execute("SELECT COUNT(*) FROM detections WHERE run_id='r2'").fetchone()[0]
    assert count == 1
