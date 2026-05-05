"""``lmtune tuner make-config <kind>`` — paste-able default kwargs 검증.

검증:
1. percentile_native 의 YAML 출력 — pruner 블록 + 4개 기본 kwargs
2. JSON 출력 (machine-readable, axis wrapper 포함)
3. --flat 모드: wrapper 없이 kwargs 만
4. Optuna 빌트인 fallback (kwargs 비어있고 comment 만)
5. unknown kind → BadParameter
6. 부정확한 format → BadParameter
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_tuner import app

runner = CliRunner()


# ─── native pruner ────────────────────────────────────────────────────


def test_make_config_percentile_yaml():
    result = runner.invoke(app, ["make-config", "percentile_native"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "pruner:" in out
    assert "kind: percentile_native" in out
    assert "percentile: 0.25" in out
    assert "n_startup_trials: 5" in out
    assert "direction: maximize" in out


def test_make_config_median_yaml():
    result = runner.invoke(app, ["make-config", "median_native"])
    assert result.exit_code == 0
    out = result.output
    assert "pruner:" in out
    assert "kind: median_native" in out
    # NativeMedianPruner 는 percentile 없음 (median 고정)
    assert "percentile:" not in out
    assert "n_startup_trials: 5" in out


# ─── JSON 모드 ─────────────────────────────────────────────────────────


def test_make_config_json():
    result = runner.invoke(app, ["make-config", "percentile_native", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "pruner" in payload
    block = payload["pruner"]
    assert block["kind"] == "percentile_native"
    assert block["percentile"] == 0.25
    assert block["n_startup_trials"] == 5


def test_make_config_json_flat():
    result = runner.invoke(app, ["make-config", "percentile_native", "--format", "json", "--flat"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    # flat 은 wrapper 없이 block 자체
    assert payload["kind"] == "percentile_native"
    assert "pruner" not in payload  # wrapper 없음


# ─── Sampler ──────────────────────────────────────────────────────────


def test_make_config_tpe_native():
    result = runner.invoke(app, ["make-config", "tpe_native", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "sampler" in payload
    assert payload["sampler"]["kind"] == "tpe_native"


# ─── Optuna 빌트인 (fallback) ──────────────────────────────────────────


def test_make_config_optuna_hyperband_fallback():
    result = runner.invoke(app, ["make-config", "hyperband"])
    assert result.exit_code == 0
    out = result.output
    assert "hyperband" in out
    assert "Optuna 빌트인" in out
    assert "kind: hyperband" in out


# ─── 에러 ──────────────────────────────────────────────────────────────


def test_make_config_unknown_rejected():
    result = runner.invoke(app, ["make-config", "totally_unknown_kind"])
    assert result.exit_code != 0


def test_make_config_invalid_format():
    result = runner.invoke(app, ["make-config", "percentile_native", "--format", "toml"])
    assert result.exit_code != 0


# ─── --flat YAML ──────────────────────────────────────────────────────


def test_make_config_yaml_flat():
    """--flat YAML 은 wrapper key 없이 kwargs 만 (top-level)."""
    result = runner.invoke(app, ["make-config", "percentile_native", "--flat"])
    assert result.exit_code == 0
    out = result.output
    # top-level 'pruner:' 없어야
    assert "pruner:" not in out
    # 하지만 'kind:' 는 있음 (top level)
    assert "kind: percentile_native" in out
