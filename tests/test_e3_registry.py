from __future__ import annotations

import math

import pytest

from lmtune.analysis import compare_runs
from lmtune.analysis.derived import (
    BUILTIN_FORMULAS,
    DerivedSpec,
    compute_derived,
    resolve_builtin,
    safe_eval,
)
from lmtune.analysis.registry import (
    MetricDef,
    by_category,
    direction_of,
    get,
    list_all,
    register,
)

# ---------- Registry ----------


def test_builtin_metrics_present():
    assert direction_of("ttft") == "lower_better"
    assert direction_of("throughput_tok") == "higher_better"
    assert direction_of("tool_call_count") == "neutral"
    assert direction_of("unknown_xyz") == "neutral"


def test_latency_category_contains_ttft():
    lats = {m.name for m in by_category("latency")}
    assert {"ttft", "itl", "tpot", "e2e"} <= lats


def test_register_custom_metric():
    register(MetricDef("my_metric", "ms", "lower_better", "latency"))
    assert get("my_metric").direction == "lower_better"


def test_compare_uses_registry_not_hardcoded():
    baseline = {"ttft": {"p99": 100.0}, "throughput_tok": {"avg": 1000.0}}
    candidate = {"ttft": {"p99": 150.0}, "throughput_tok": {"avg": 800.0}}
    cmp_ = compare_runs("b", "c", baseline, candidate, regression_threshold_pct=10.0)
    names = {(d.metric, d.p) for d in cmp_.regressions}
    assert ("ttft", "p99") in names
    assert ("throughput_tok", "avg") in names


# ---------- Derived formulas ----------


def test_builtin_eutb_formula_registered():
    assert "eutb" in BUILTIN_FORMULAS
    spec = resolve_builtin("eutb")
    assert spec is not None
    assert spec.formula.startswith("success_rate")


def test_safe_eval_arithmetic():
    ctx = {"a": 10, "b": 4}
    assert safe_eval("a / b", ctx) == 2.5
    assert safe_eval("(a + b) * 2", ctx) == 28.0
    assert safe_eval("max(a, b, 100)", ctx) == 100.0


def test_safe_eval_disallows_builtins():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')", {})


def test_safe_eval_missing_var():
    with pytest.raises(KeyError):
        safe_eval("x + 1", {})


def test_compute_derived_handles_zero_div():
    specs = [
        DerivedSpec(name="prefix_hit_rate", formula=BUILTIN_FORMULAS["prefix_hit_rate"]),
        DerivedSpec(name="eutb", formula=BUILTIN_FORMULAS["eutb"]),
    ]
    ctx = {"cached_tokens": 800, "input_tokens": 1000, "success_rate": 0.7, "total_input_tokens": 0}
    out = compute_derived(specs, ctx)
    assert out["prefix_hit_rate"] == 0.8
    assert math.isnan(out["eutb"])      # division by zero → NaN


def test_compute_derived_custom_formula():
    specs = [DerivedSpec(name="custom", formula="(a * 2) + b")]
    out = compute_derived(specs, {"a": 5, "b": 3})
    assert out["custom"] == 13


def test_list_all_is_sorted_by_category():
    all_defs = list_all()
    cats = [m.category for m in all_defs]
    assert cats == sorted(cats)
