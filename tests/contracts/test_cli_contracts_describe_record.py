"""``lmtune contracts describe-record <kind>`` — 필드 introspect 검증.

검증:
1. 모든 RECORD_KINDS 가 정상 introspect (exit 0)
2. JSON 모드의 안정적 schema (kind / class_name / fields[])
3. RunRecord 의 필수 필드 (run_id, profile_slug, ...) 가 ``required=True`` 표시
4. RunRecord 의 optional 필드 (started_at 등) 가 ``required=False``
5. unknown kind → BadParameter
6. ``kind`` 자기 자신 ('kind' literal) 도 fields 에 포함
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_contracts import app
from lmtune.contracts.record_spec import RECORD_KINDS

runner = CliRunner()


def test_describe_all_kinds_exit_zero():
    """모든 RECORD_KINDS 가 정상 introspect."""
    for kind in RECORD_KINDS:
        result = runner.invoke(app, ["describe-record", kind])
        assert result.exit_code == 0, f"{kind}: {result.output}"


def test_describe_json_schema_run():
    result = runner.invoke(app, ["describe-record", "run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["kind"] == "run"
    assert payload["class_name"] == "RunRecord"
    assert "fields" in payload
    assert isinstance(payload["fields"], list)
    assert len(payload["fields"]) >= 5


def test_describe_run_required_fields():
    """RunRecord 의 필수 필드는 required=True."""
    result = runner.invoke(app, ["describe-record", "run", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    by_name = {f["name"]: f for f in payload["fields"]}
    for required_field in ("run_id", "profile_slug", "endpoint_slug", "runner", "status"):
        assert by_name[required_field]["required"] is True, f"{required_field} should be required"


def test_describe_run_optional_fields():
    """RunRecord 의 optional 필드는 required=False + default 값."""
    result = runner.invoke(app, ["describe-record", "run", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    by_name = {f["name"]: f for f in payload["fields"]}
    for opt_field in ("started_at", "finished_at", "git_sha", "trial_id"):
        assert by_name[opt_field]["required"] is False, f"{opt_field} should be optional"


def test_describe_metric_value_required():
    """MetricRecord 의 value(float) 는 필수."""
    result = runner.invoke(app, ["describe-record", "metric", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    by_name = {f["name"]: f for f in payload["fields"]}
    assert by_name["value"]["required"] is True
    assert by_name["run_id"]["required"] is True
    assert by_name["p"]["required"] is False


def test_describe_unknown_kind_rejected():
    result = runner.invoke(app, ["describe-record", "totally_unknown"])
    assert result.exit_code != 0


def test_describe_kind_literal_included():
    """``kind`` discriminator 자기 자신도 fields 에 포함."""
    result = runner.invoke(app, ["describe-record", "run", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    by_name = {f["name"]: f for f in payload["fields"]}
    assert "kind" in by_name


def test_describe_human_readable_output_has_class_name():
    """기본 (rich) 출력이 class_name 노출."""
    result = runner.invoke(app, ["describe-record", "trial"])
    assert result.exit_code == 0
    assert "TrialRecord" in result.output
    assert "trial" in result.output
