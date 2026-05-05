"""``lmtune contracts`` 서브커맨드 — schema dump + validate-record E2E."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from lmtune.cli_contracts import app

runner = CliRunner()


def test_dump_schema_record_full(tmp_path: Path):
    out = tmp_path / "record.schema.json"
    result = runner.invoke(app, ["dump-schema", "--kind", "record", "--out", str(out)])
    assert result.exit_code == 0, result.output
    schema = json.loads(out.read_text())
    # discriminated union → oneOf/anyOf
    assert "oneOf" in schema or "anyOf" in schema


def test_dump_schema_record_specific_kind(tmp_path: Path):
    out = tmp_path / "run.schema.json"
    result = runner.invoke(
        app, ["dump-schema", "--kind", "record", "--record-kind", "run", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    schema = json.loads(out.read_text())
    assert schema.get("title") == "RunRecord"


def test_dump_schema_query(tmp_path: Path):
    out = tmp_path / "q.schema.json"
    result = runner.invoke(app, ["dump-schema", "--kind", "query", "--out", str(out)])
    assert result.exit_code == 0, result.output
    schema = json.loads(out.read_text())
    assert schema.get("title") == "QuerySpec"


def test_dump_schema_unknown_kind_errors():
    result = runner.invoke(app, ["dump-schema", "--kind", "bogus"])
    assert result.exit_code != 0


def test_dump_schema_query_with_record_kind_errors():
    """--record-kind 는 query 에서 의미 없음."""
    result = runner.invoke(
        app, ["dump-schema", "--kind", "query", "--record-kind", "run"]
    )
    assert result.exit_code != 0


def test_validate_record_yaml_ok(tmp_path: Path):
    p = tmp_path / "rec.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "kind": "run",
                "run_id": "01ABC",
                "profile_slug": "p",
                "endpoint_slug": "e",
                "runner": "guidellm",
                "status": "ok",
            }
        )
    )
    result = runner.invoke(app, ["validate-record", str(p)])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output
    assert "kind=run" in result.output


def test_validate_record_invalid_kind(tmp_path: Path):
    p = tmp_path / "rec.yaml"
    p.write_text(yaml.safe_dump({"kind": "bogus", "run_id": "x"}))
    result = runner.invoke(app, ["validate-record", str(p)])
    assert result.exit_code == 1
    assert "invalid" in result.output


def test_validate_record_json_format(tmp_path: Path):
    p = tmp_path / "rec.json"
    p.write_text(
        json.dumps(
            {"kind": "metric", "run_id": "r1", "metric": "ttft", "p": "p99", "value": 100.0}
        )
    )
    result = runner.invoke(app, ["validate-record", str(p)])
    assert result.exit_code == 0, result.output
    assert "kind=metric" in result.output
