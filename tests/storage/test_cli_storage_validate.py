"""``lmtune storage validate`` — record schema validity 검증."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lmtune.cli_storage import app
from lmtune.contracts import MetricRecord, RunRecord, TrialRecord
from lmtune.storage.store import LocalArtifactStore

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


def test_validate_local_valid_archive(tmp_path: Path):
    src = tmp_path / "local"
    _seed_local(src)
    result = runner.invoke(app, ["validate", "--kind", "local", "--path", str(src)])
    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_validate_local_json_output(tmp_path: Path):
    src = tmp_path / "local"
    _seed_local(src)
    result = runner.invoke(app, ["validate", "--kind", "local", "--path", str(src), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["valid"] is True
    assert payload["invalid"] == {}
    # run + metric + trial 3 kinds 가 valid_counts 에서 0 보다 큼
    assert payload["valid_counts"]["run"] == 1
    assert payload["valid_counts"]["metric"] == 1
    assert payload["valid_counts"]["trial"] == 1


def test_validate_empty_local_is_valid(tmp_path: Path):
    """빈 store 도 valid (모든 kind 가 0 records)."""
    src = tmp_path / "empty"
    src.mkdir()
    result = runner.invoke(app, ["validate", "--kind", "local", "--path", str(src), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["valid"] is True


def test_validate_local_corrupted_jsonl_raises(tmp_path: Path):
    """손상된 run.jsonl line → invalid 분류 + exit 1."""
    src = tmp_path / "local"
    _seed_local(src)
    # run.jsonl 에 schema 위반 line 강제 주입 (kind 가 'run' 인데 필수 필드 누락)
    (src / "run.jsonl").write_text(
        '{"kind": "run", "run_id": "r2"}\n',  # profile_slug 등 누락
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "--kind", "local", "--path", str(src), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["valid"] is False
    assert "run" in payload["invalid"]


def test_validate_unknown_backend_rejected(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--kind", "mongodb", "--path", str(tmp_path)])
    assert result.exit_code != 0
