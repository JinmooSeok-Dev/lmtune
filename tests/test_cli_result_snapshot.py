"""bench run 이 raw_dir/<run_id>/result.json 으로 BenchmarkResult 덤프하는지."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lmtune.cli import app
from lmtune.runners.base import RunArtifact


def _make_artifact(run_id: str, raw_dir: Path) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        runner_kind="guidellm",
        command=["guidellm"],
        raw_dir=raw_dir,
        stdout_path=raw_dir / "stdout.log",
        stderr_path=raw_dir / "stderr.log",
        tool_version="0.6.0",
        status="ok",
        metrics={"ttft": {"p99": 200.0}, "throughput_tok": {"avg": 130.0}},
        started_at=1714000000.0,
        finished_at=1714000060.0,
    )


def test_cmd_run_dumps_result_json(tmp_path):
    """`bench run` 실행 시 raw_dir/<run_id>/result.json 이 생성되어 BenchmarkResult schema 준수."""
    profile_yaml = tmp_path / "profile.yaml"
    profile_yaml.write_text(
        """\
apiVersion: lmtune/v1alpha1
slug: test-profile
name: Test
stage: 1
runner: guidellm
mode: concurrency
workload:
  source: synthetic
  synthetic_input_tokens_mean: 256
  output_tokens_mean: 128
  concurrency: 4
  request_count: 8
slo:
  ttft_p99_ms: 500.0
""",
        encoding="utf-8",
    )
    endpoint_yaml = tmp_path / "endpoint.yaml"
    endpoint_yaml.write_text(
        """\
apiVersion: lmtune/v1alpha1
slug: test-endpoint
name: Test endpoint
url: http://localhost:8000/v1
model: test-model
""",
        encoding="utf-8",
    )

    db_path = tmp_path / "lmtune.duckdb"
    raw_dir = tmp_path / "raw"

    runner = CliRunner()

    # runner 호출은 mock — 진짜 endpoint 안 띄움
    fake_run_id = "01TESTRUNULID0000000000000"

    with (
        patch("lmtune.cli.ULID", return_value=fake_run_id),
        patch("lmtune.cli.get_runner") as mock_get_runner,
        patch("lmtune.cli._git_sha", return_value="testsha"),
    ):
        mock_runner = mock_get_runner.return_value
        mock_runner.kind = "guidellm"
        mock_runner.run.return_value = _make_artifact(fake_run_id, raw_dir / fake_run_id)
        # raw_dir 미리 생성 (runner.run 이 통상 만들지만 mock 이라)
        (raw_dir / fake_run_id).mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            app,
            [
                "run",
                "-p",
                str(profile_yaml),
                "-e",
                str(endpoint_yaml),
                "--db",
                str(db_path),
                "--raw-dir",
                str(raw_dir),
            ],
        )

    assert result.exit_code == 0, result.output

    result_json_path = raw_dir / fake_run_id / "result.json"
    assert result_json_path.exists(), f"result.json missing under {result_json_path}"

    payload = json.loads(result_json_path.read_text())
    assert payload["api_version"] == "lmtune/result/v1alpha1"
    assert payload["kind"] == "BenchmarkResult"
    assert payload["run_id"] == fake_run_id
    assert payload["profile_slug"] == "test-profile"
    assert payload["endpoint_slug"] == "test-endpoint"
    assert payload["runner_kind"] == "guidellm"
    assert payload["status"] == "ok"
    assert payload["metrics"]["ttft"]["p99"] == 200.0
    assert payload["git_sha"] == "testsha"
    assert payload["tool_versions"] == {"guidellm": "0.6.0"}


def test_cmd_run_result_validates_against_schema(tmp_path):
    """result.json 이 BenchmarkResult 로 round-trip 가능."""
    from lmtune.contracts import BenchmarkResult

    profile_yaml = tmp_path / "profile.yaml"
    profile_yaml.write_text(
        """\
apiVersion: lmtune/v1alpha1
slug: test
name: T
stage: 1
runner: guidellm
mode: concurrency
workload:
  source: synthetic
  synthetic_input_tokens_mean: 64
  output_tokens_mean: 32
  concurrency: 1
  request_count: 1
""",
        encoding="utf-8",
    )
    endpoint_yaml = tmp_path / "ep.yaml"
    endpoint_yaml.write_text(
        """\
apiVersion: lmtune/v1alpha1
slug: ep
name: ep
url: http://localhost:8000/v1
model: m
""",
        encoding="utf-8",
    )
    db_path = tmp_path / "db.duckdb"
    raw_dir = tmp_path / "raw"
    fake_run_id = "01TESTROUND00000000000000"

    runner = CliRunner()
    with (
        patch("lmtune.cli.ULID", return_value=fake_run_id),
        patch("lmtune.cli.get_runner") as mock_get_runner,
        patch("lmtune.cli._git_sha", return_value=None),
    ):
        mr = mock_get_runner.return_value
        mr.kind = "guidellm"
        mr.run.return_value = _make_artifact(fake_run_id, raw_dir / fake_run_id)
        (raw_dir / fake_run_id).mkdir(parents=True, exist_ok=True)

        out = runner.invoke(
            app,
            [
                "run",
                "-p",
                str(profile_yaml),
                "-e",
                str(endpoint_yaml),
                "--db",
                str(db_path),
                "--raw-dir",
                str(raw_dir),
            ],
        )

    assert out.exit_code == 0, out.output
    payload = json.loads((raw_dir / fake_run_id / "result.json").read_text())
    # schema round-trip
    res = BenchmarkResult.model_validate(payload)
    assert res.run_id == fake_run_id
    assert res.kind == "BenchmarkResult"
