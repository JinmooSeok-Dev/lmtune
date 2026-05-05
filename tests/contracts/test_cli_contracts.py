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
    result = runner.invoke(app, ["dump-schema", "--kind", "query", "--record-kind", "run"])
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
        json.dumps({"kind": "metric", "run_id": "r1", "metric": "ttft", "p": "p99", "value": 100.0})
    )
    result = runner.invoke(app, ["validate-record", str(p)])
    assert result.exit_code == 0, result.output
    assert "kind=metric" in result.output


def _result_payload(run_id: str = "01RECEXP00000000000000000") -> dict:
    return {
        "api_version": "lmtune/result/v1alpha1",
        "kind": "BenchmarkResult",
        "run_id": run_id,
        "profile_slug": "p",
        "endpoint_slug": "e",
        "runner_kind": "guidellm",
        "tool_versions": {"guidellm": "0.6.0"},
        "started_at": "2024-01-01T00:00:00Z",
        "finished_at": "2024-01-01T00:01:00Z",
        "status": "ok",
        "metrics": {
            "ttft": {"p50": 10.0, "p99": 200.0},
            "throughput_tok": {"avg": 130.0},
        },
        "requests": [],
        "sessions": [],
    }


def test_records_from_result_json(tmp_path: Path):
    """``contracts records-from-result`` 가 result.json → records/<kind>.jsonl."""
    src = tmp_path / "result.json"
    src.write_text(json.dumps(_result_payload()), encoding="utf-8")
    out = tmp_path / "records"

    result = runner.invoke(app, ["records-from-result", str(src), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output

    # run.jsonl + metric.jsonl 생성
    assert (out / "run.jsonl").exists()
    assert (out / "metric.jsonl").exists()

    # LocalArtifactStore round-trip
    from lmtune.contracts import QuerySpec
    from lmtune.storage.store import LocalArtifactStore

    store = LocalArtifactStore(out)
    runs = store.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert runs[0].run_id == "01RECEXP00000000000000000"
    metrics = store.query(QuerySpec(record_kind="metric"))
    # ttft.p50, ttft.p99, throughput_tok.avg → 3 건
    assert len(metrics) == 3


def test_records_from_result_yaml(tmp_path: Path):
    """yaml 형식 입력도 지원."""
    src = tmp_path / "result.yaml"
    src.write_text(yaml.safe_dump(_result_payload("01YAMLEXP000000000000000")), encoding="utf-8")
    out = tmp_path / "records"

    result = runner.invoke(app, ["records-from-result", str(src), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "run.jsonl").exists()


def test_records_from_result_invalid_payload(tmp_path: Path):
    """schema 위반 result → exit 1."""
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"kind": "BenchmarkResult", "run_id": "x"}), encoding="utf-8")
    out = tmp_path / "records"

    result = runner.invoke(app, ["records-from-result", str(src), "--out", str(out)])
    assert result.exit_code == 1
    assert "invalid" in result.output
