"""``lmtune contracts list-records`` — RECORD_KINDS 가시성 표면 검증.

검증:
1. 기본 출력 (rich) 이 모든 RECORD_KINDS 노출
2. ``--json`` 모드의 안정적 schema (`{"records": [...]}`)
3. ``RECORD_KINDS`` 와 CLI 출력이 동기 (drift 가드)
4. JSON 출력의 record list 가 ``RECORD_KINDS`` 순서와 동일
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_contracts import app
from lmtune.contracts.record_spec import RECORD_KINDS

runner = CliRunner()


def test_list_records_default_lists_all_kinds():
    result = runner.invoke(app, ["list-records"])
    assert result.exit_code == 0, result.output
    for kind in RECORD_KINDS:
        assert kind in result.output, f"kind {kind!r} not in output"


def test_list_records_json_schema():
    result = runner.invoke(app, ["list-records", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "records" in payload
    assert isinstance(payload["records"], list)


def test_list_records_json_matches_record_kinds_drift_guard():
    """CLI 출력이 ``RECORD_KINDS`` 와 1:1 동기 — drift 시 즉시 실패."""
    result = runner.invoke(app, ["list-records", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["records"] == list(RECORD_KINDS)


def test_list_records_count_matches():
    result = runner.invoke(app, ["list-records"])
    assert result.exit_code == 0
    assert f"({len(RECORD_KINDS)})" in result.output
