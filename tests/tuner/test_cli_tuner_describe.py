"""``lmtune tuner describe <kind>`` — sampler / pruner introspect 검증.

검증:
1. native pruner (median_native, percentile_native) 의 __init__ params 노출
2. native sampler (tpe_native 등) 의 params 노출
3. llm sampler (llm_oracle) 의 params 노출
4. Optuna 빌트인 (hyperband, tpe) 은 reference URL fallback
5. unknown kind → BadParameter (exit code != 0)
6. --json 모드 (machine-readable)
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lmtune.cli_tuner import app

runner = CliRunner()


# ─── native pruner ────────────────────────────────────────────────────


def test_describe_median_native():
    result = runner.invoke(app, ["describe", "median_native"])
    assert result.exit_code == 0, result.output
    assert "median_native" in result.output
    assert "pruner" in result.output
    assert "native" in result.output
    assert "n_startup_trials" in result.output
    assert "n_warmup_steps" in result.output
    assert "direction" in result.output


def test_describe_median_native_json():
    result = runner.invoke(app, ["describe", "median_native", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["kind"] == "median_native"
    assert payload["axis"] == "pruner"
    assert payload["group"] == "native"
    assert payload["class_name"] == "NativeMedianPruner"
    param_names = {p["name"] for p in payload["params"]}
    assert {"n_startup_trials", "n_warmup_steps", "direction"} <= param_names


def test_describe_percentile_native_includes_percentile():
    result = runner.invoke(app, ["describe", "percentile_native", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    param_names = {p["name"] for p in payload["params"]}
    assert "percentile" in param_names


# ─── native sampler ───────────────────────────────────────────────────


def test_describe_tpe_native():
    result = runner.invoke(app, ["describe", "tpe_native", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["axis"] == "sampler"
    assert payload["group"] == "native"
    assert "Native" in payload["class_name"]


def test_describe_random_native_smoke():
    result = runner.invoke(app, ["describe", "random_native"])
    assert result.exit_code == 0


# ─── llm sampler ──────────────────────────────────────────────────────


def test_describe_llm_oracle():
    result = runner.invoke(app, ["describe", "llm_oracle", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["axis"] == "sampler"
    assert payload["group"] == "llm"
    assert payload["class_name"] == "LLMOracleSampler"


# ─── Optuna 빌트인 (introspect 불가, reference fallback) ───────────────


def test_describe_hyperband_falls_back_to_reference():
    result = runner.invoke(app, ["describe", "hyperband", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["axis"] == "pruner"
    assert payload["group"] == "optuna"
    assert "reference" in payload
    assert "optuna" in payload["reference"]


def test_describe_tpe_optuna_falls_back():
    """'tpe' (without _native) 는 Optuna sampler — reference fallback."""
    result = runner.invoke(app, ["describe", "tpe", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["axis"] == "sampler"
    assert payload["group"] == "optuna"


# ─── unknown kind ─────────────────────────────────────────────────────


def test_describe_unknown_kind_rejected():
    result = runner.invoke(app, ["describe", "nonexistent_kind"])
    assert result.exit_code != 0
    assert "unknown kind" in result.output or "Invalid value" in result.output
