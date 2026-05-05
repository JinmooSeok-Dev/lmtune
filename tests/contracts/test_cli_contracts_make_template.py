"""``lmtune contracts make-template <kind>`` — 빈 record 템플릿 검증.

검증:
1. 모든 RECORD_KINDS 가 template 출력
2. JSON 모드 (default)
3. YAML 모드
4. 출력 template 이 RecordSpec validate-record 통과 (round-trip)
5. unknown record kind → BadParameter
6. invalid format → BadParameter
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from lmtune.cli_contracts import app
from lmtune.contracts.record_spec import RECORD_KINDS

runner = CliRunner()


def test_make_template_all_kinds_output():
    """모든 RECORD_KINDS 가 정상 template 출력."""
    for kind in RECORD_KINDS:
        result = runner.invoke(app, ["make-template", "--record-kind", kind])
        assert result.exit_code == 0, f"{kind}: {result.output}"
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload["kind"] == kind
        assert payload["api_version"].startswith("lmtune/record/")


def test_make_template_json_default():
    """default format = json."""
    result = runner.invoke(app, ["make-template", "-k", "run"])
    assert result.exit_code == 0
    # JSON parse 가능
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["kind"] == "run"
    assert payload["run_id"] == "<run_id>"


def test_make_template_yaml():
    result = runner.invoke(app, ["make-template", "-k", "run", "-f", "yaml"])
    assert result.exit_code == 0
    parsed = yaml.safe_load(result.output)
    assert parsed["kind"] == "run"
    assert parsed["run_id"] == "<run_id>"


def test_make_template_required_fields_have_placeholder():
    """필수 string 필드는 '<name>' placeholder 로 채워짐."""
    result = runner.invoke(app, ["make-template", "-k", "run"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    # required: run_id, profile_slug, endpoint_slug, runner, status
    for f in ("run_id", "profile_slug", "endpoint_slug", "runner", "status"):
        assert payload[f] == f"<{f}>", f"{f}: {payload[f]!r}"


def test_make_template_metric_numeric_placeholder():
    """metric 의 value (float) 는 0.0 placeholder."""
    result = runner.invoke(app, ["make-template", "-k", "metric"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["value"] == 0.0


def test_make_template_round_trip_with_validate(tmp_path: Path):
    """make-template → validate-record round-trip 통과 (모든 kind)."""
    for kind in RECORD_KINDS:
        result = runner.invoke(app, ["make-template", "-k", kind])
        assert result.exit_code == 0, f"{kind} make-template: {result.output}"
        # 파일에 저장 → validate-record
        f = tmp_path / f"{kind}.json"
        f.write_text(result.output.strip().splitlines()[-1], encoding="utf-8")
        v_result = runner.invoke(app, ["validate-record", str(f)])
        assert v_result.exit_code == 0, f"{kind} validate: {v_result.output}"
        assert "ok" in v_result.output


def test_make_template_unknown_kind_rejected():
    result = runner.invoke(app, ["make-template", "-k", "totally_unknown"])
    assert result.exit_code != 0


def test_make_template_invalid_format():
    result = runner.invoke(app, ["make-template", "-k", "run", "-f", "toml"])
    assert result.exit_code != 0
