"""``lmtune tuner list-{samplers,pruners}`` — PLUG 노출 검증.

검증:
1. list-pruners 가 native + optuna 두 그룹 모두 출력
2. list-pruners 의 native 그룹에 median_native + percentile_native 포함
3. list-pruners 의 optuna 그룹에 SH / Hyperband 포함
4. list-samplers 가 native + optuna + llm 세 그룹 모두 출력
5. --json 모드 (machine-readable)
6. drift 가드: list-pruners 의 native set 이 _NATIVE_PRUNER_KINDS 와 동기
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_tuner import app

runner = CliRunner()


# ─── list-pruners ─────────────────────────────────────────────────────


def test_list_pruners_smoke():
    result = runner.invoke(app, ["list-pruners"])
    assert result.exit_code == 0, result.output
    # 그룹 헤더 + 빌트인 노출
    assert "native" in result.output
    assert "optuna" in result.output
    assert "median_native" in result.output
    assert "percentile_native" in result.output
    assert "hyperband" in result.output


def test_list_pruners_json():
    result = runner.invoke(app, ["list-pruners", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "native" in payload
    assert "optuna" in payload
    assert set(payload["native"]) == {"median_native", "percentile_native"}
    assert {"sh", "successive_halving", "hyperband"} <= set(payload["optuna"])


def test_list_pruners_drift_guard():
    """list-pruners 의 native 그룹이 tuner.factory 의 set 과 일치."""
    from lmtune.tuner.factory import _NATIVE_PRUNER_KINDS, _OPTUNA_PRUNER_KINDS

    result = runner.invoke(app, ["list-pruners", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert set(payload["native"]) == _NATIVE_PRUNER_KINDS
    assert set(payload["optuna"]) == _OPTUNA_PRUNER_KINDS


# ─── list-samplers ────────────────────────────────────────────────────


def test_list_samplers_smoke():
    result = runner.invoke(app, ["list-samplers"])
    assert result.exit_code == 0, result.output
    assert "native" in result.output
    assert "optuna" in result.output
    assert "llm" in result.output
    # 빌트인 노출
    assert "tpe" in result.output  # optuna
    assert "tpe_native" in result.output
    assert "llm_oracle" in result.output


def test_list_samplers_json():
    result = runner.invoke(app, ["list-samplers", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert {"native", "optuna", "llm"} == set(payload.keys())
    assert {"random_native", "lhc_native", "tpe_native"} == set(payload["native"])
    assert "llm_oracle" in payload["llm"]


def test_list_samplers_drift_guard():
    from lmtune.tuner.factory import _LLM_STRATEGIES, _NATIVE_STRATEGIES

    result = runner.invoke(app, ["list-samplers", "--json"])
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert set(payload["native"]) == _NATIVE_STRATEGIES
    assert set(payload["llm"]) == _LLM_STRATEGIES
