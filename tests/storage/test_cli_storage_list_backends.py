"""``lmtune storage list-backends`` — 가시성 표면 검증.

검증:
1. 기본 출력에 모든 ``_BACKENDS`` 노출
2. ``--json`` 출력의 안정 schema (`{"backends": [...]}`)
3. CLI 출력이 ``_BACKENDS`` 와 1:1 동기 (drift 가드)
4. JSON 출력 list 의 순서가 ``_BACKENDS`` 와 동일
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_storage import _BACKENDS, app

runner = CliRunner()


def test_list_backends_default_lists_all():
    result = runner.invoke(app, ["list-backends"])
    assert result.exit_code == 0, result.output
    for b in _BACKENDS:
        assert b in result.output


def test_list_backends_json_schema():
    result = runner.invoke(app, ["list-backends", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "backends" in payload
    assert isinstance(payload["backends"], list)


def test_list_backends_json_matches_backends_drift_guard():
    """CLI JSON 출력이 ``_BACKENDS`` 와 1:1 동기 (순서까지)."""
    result = runner.invoke(app, ["list-backends", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backends"] == list(_BACKENDS)


def test_list_backends_count_consistent():
    result = runner.invoke(app, ["list-backends", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert len(payload["backends"]) == len(_BACKENDS)
