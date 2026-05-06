"""``lmtune storage describe-backend <name>`` — Storage axis metadata 표면 검증.

검증:
1. 등록된 모든 backend (`_BACKENDS`) 가 정상 introspect (exit 0)
2. JSON 모드의 안정 schema (`name / class_name / module / summary / path_kind /
   extras / transactional / concurrent_writers`)
3. ``_BACKEND_META`` ↔ ``_BACKENDS`` drift 가드 — 새 backend 추가 시 메타도
   동기화 필수
4. unknown backend → BadParameter
5. extras 가 있는 backend (`postgres`) 는 install 명령이 출력에 포함
6. transactional / concurrent_writers 가 capability 별로 다름 (drift 검증)
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_storage import _BACKEND_META, _BACKENDS, app

runner = CliRunner()


def test_describe_all_backends_exit_zero():
    for name in _BACKENDS:
        result = runner.invoke(app, ["describe-backend", name])
        assert result.exit_code == 0, f"{name}: {result.output}"


def test_describe_json_schema():
    result = runner.invoke(app, ["describe-backend", "local", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    for key in (
        "name",
        "class_name",
        "module",
        "summary",
        "path_kind",
        "transactional",
        "concurrent_writers",
    ):
        assert key in payload, f"missing key {key!r}"
    assert payload["name"] == "local"


def test_backend_meta_matches_backends_drift_guard():
    """``_BACKENDS`` 와 ``_BACKEND_META`` 의 키가 1:1 동기."""
    assert set(_BACKEND_META.keys()) == set(_BACKENDS)


def test_describe_unknown_rejected():
    result = runner.invoke(app, ["describe-backend", "totally_unknown"])
    assert result.exit_code != 0


def test_describe_postgres_shows_extras():
    """postgres backend 는 [postgres] extras 명령을 출력에 포함."""
    result = runner.invoke(app, ["describe-backend", "postgres"])
    assert result.exit_code == 0
    assert "postgres" in result.output
    assert "[postgres]" in result.output


def test_describe_capability_differentiation():
    """각 backend 의 capability 가 메타에 명시 — concurrent_writers 차별 확인."""
    for name in _BACKENDS:
        result = runner.invoke(app, ["describe-backend", name, "--json"])
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert isinstance(payload["transactional"], bool)
        assert isinstance(payload["concurrent_writers"], bool)
    # postgres 만 concurrent_writers=True
    pg_payload = json.loads(
        runner.invoke(app, ["describe-backend", "postgres", "--json"])
        .output.strip()
        .splitlines()[-1]
    )
    assert pg_payload["concurrent_writers"] is True
    local_payload = json.loads(
        runner.invoke(app, ["describe-backend", "local", "--json"]).output.strip().splitlines()[-1]
    )
    assert local_payload["concurrent_writers"] is False


def test_describe_human_readable_output_has_class_name():
    result = runner.invoke(app, ["describe-backend", "duckdb"])
    assert result.exit_code == 0
    assert "DuckDBArtifactStore" in result.output
