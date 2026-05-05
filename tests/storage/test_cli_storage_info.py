"""``lmtune storage info`` — record kind 별 count 보고 검증."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lmtune.cli_storage import app
from lmtune.contracts import MetricRecord, RunRecord, TrialRecord
from lmtune.storage.store import DuckDBArtifactStore, LocalArtifactStore

runner = CliRunner()


def _seed_local(root: Path) -> None:
    s = LocalArtifactStore(root)
    s.put(
        [
            RunRecord(
                run_id="r1",
                profile_slug="p",
                endpoint_slug="e",
                runner="guidellm",
                status="ok",
            ),
            MetricRecord(run_id="r1", metric="ttft", p="p99", value=200.0),
            MetricRecord(run_id="r1", metric="ttft", p="p50", value=50.0),
            TrialRecord(
                trial_id="t1",
                study_id="st1",
                seq=1,
                params={"x": 1},
                status="completed",
                score=0.9,
            ),
        ]
    )
    s.close()


def test_info_local_human_output(tmp_path: Path):
    src = tmp_path / "local"
    _seed_local(src)
    result = runner.invoke(app, ["info", "--kind", "local", "--path", str(src)])
    assert result.exit_code == 0, result.output
    # human-readable: "run", "metric", "trial" 라벨 + 수치
    assert "run" in result.output
    assert "metric" in result.output
    assert "total" in result.output


def test_info_local_json_output(tmp_path: Path):
    src = tmp_path / "local"
    _seed_local(src)
    result = runner.invoke(app, ["info", "--kind", "local", "--path", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backend"] == "local"
    assert payload["total"] == 4  # 1 run + 2 metric + 1 trial
    assert payload["counts"]["run"] == 1
    assert payload["counts"]["metric"] == 2
    assert payload["counts"]["trial"] == 1
    # 빈 kind 도 0 으로 포함됨
    assert payload["counts"]["session"] == 0


def test_info_duckdb_json_output(tmp_path: Path):
    db = tmp_path / "x.duckdb"
    duck = DuckDBArtifactStore(db)
    duck.put(
        [
            RunRecord(
                run_id="r2",
                profile_slug="p",
                endpoint_slug="e",
                runner="guidellm",
                status="ok",
            ),
            MetricRecord(run_id="r2", metric="ttft", p="p99", value=180.0),
        ]
    )
    duck.close()

    result = runner.invoke(app, ["info", "--kind", "duckdb", "--path", str(db), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backend"] == "duckdb"
    assert payload["counts"]["run"] == 1
    assert payload["counts"]["metric"] == 1


def test_info_empty_local_returns_zeros(tmp_path: Path):
    src = tmp_path / "empty"
    src.mkdir()
    result = runner.invoke(app, ["info", "--kind", "local", "--path", str(src), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["total"] == 0
    assert all(v == 0 for v in payload["counts"].values())


def test_info_unknown_backend_rejected(tmp_path: Path):
    result = runner.invoke(app, ["info", "--kind", "mongodb", "--path", str(tmp_path)])
    assert result.exit_code != 0
