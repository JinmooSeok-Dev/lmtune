"""Tests for src/lmtune/search/validate.py — search-space pre-flight validator.

R23/R25/R26/R28 결함 패턴이 study start 전에 hard block 으로 차단되는지 검증.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from lmtune.search.feasibility import Environment
from lmtune.search.validate import (
    validate_axis_allowlist,
    validate_known_regressions,
    validate_schema,
    validate_search_space,
)

# ---- schema --------------------------------------------------------------


def test_schema_valid_space_no_issues():
    text = dedent(
        """
        apiVersion: lmtune/search/v1alpha1
        kind: SearchSpace
        name: ok
        axes:
          tp: {type: categorical, values: [2, 4, 8]}
          gpu_memory_utilization: {type: float, low: 0.8, high: 0.92}
        """
    )
    issues = validate_schema(text)
    assert issues == []


def test_schema_missing_axes_block_blocks():
    issues = validate_schema("name: x\nkind: SearchSpace\n")
    assert any(i.severity == "block" and "axes" in i.msg.lower() for i in issues)


def test_schema_categorical_without_values_blocks():
    text = "kind: SearchSpace\nname: x\naxes:\n  bad: {type: categorical}\n"
    issues = validate_schema(text)
    assert any(i.axis == "bad" and i.severity == "block" for i in issues)


def test_schema_int_low_gt_high_blocks():
    text = "kind: SearchSpace\nname: x\naxes:\n  bad: {type: int, low: 8, high: 1}\n"
    issues = validate_schema(text)
    assert any(i.axis == "bad" and "low > high" in i.msg for i in issues)


# ---- axis allowlist ------------------------------------------------------


_REGISTRY_FIXTURE = {
    "tp": {"cli_flag": "--tensor-parallel-size", "type": "int"},
    "block_size": {
        "cli_flag": "--block-size",
        "type": "categorical",
        "choices": [16, 32, 64],
    },
    "prefill_context_parallel_size": {
        "cli_flag": "--prefill-context-parallel-size",
        "type": "int",
        "deprecated_or_unsupported": True,
        "gates": "R25",
    },
}


def test_allowlist_unknown_axis_blocks():
    issues = validate_axis_allowlist(
        {"tp": {"type": "categorical", "values": [2]}, "totally_made_up_axis": {"type": "bool"}},
        registry=_REGISTRY_FIXTURE,
    )
    assert any(i.axis == "totally_made_up_axis" and i.severity == "block" for i in issues)


def test_allowlist_deprecated_axis_blocks():
    issues = validate_axis_allowlist(
        {"prefill_context_parallel_size": {"type": "categorical", "values": [1, 2]}},
        registry=_REGISTRY_FIXTURE,
    )
    assert any(i.axis == "prefill_context_parallel_size" and "미지원" in i.msg for i in issues)


def test_allowlist_invalid_choice_blocks():
    issues = validate_axis_allowlist(
        {"block_size": {"type": "categorical", "values": [16, 32, 99]}},
        registry=_REGISTRY_FIXTURE,
    )
    assert any(i.axis == "block_size" and "99" in i.msg for i in issues)


def test_allowlist_lmtune_meta_axes_pass():
    """well_lit_path 같은 lmtune-internal axis 는 vllm catalog 미등재여도 통과."""
    issues = validate_axis_allowlist(
        {"well_lit_path": {"type": "categorical", "values": ["pd", "wide-ep"]}},
        registry=_REGISTRY_FIXTURE,
    )
    assert all(i.severity != "block" for i in issues)


# ---- regressions ---------------------------------------------------------


_R23 = {
    "id": "R23",
    "severity": "block",
    "match": {"has_axis": "max_num_partial_prefills"},
    "msg": "R23 msg",
}
_R25 = {
    "id": "R25",
    "severity": "block",
    "match": {"has_axis_value_gt": {"axis": "prefill_context_parallel_size", "gt": 1}},
    "msg": "R25 msg",
}
_R28 = {
    "id": "R28",
    "severity": "warn",
    "match": {
        "has_axis_combo_with_values": {
            "a": {"axis": "pp", "gt": 1},
            "b": {"axis": "ep", "equals": True},
        }
    },
    "msg": "R28 msg",
}


def test_regression_r23_axis_present_blocks():
    issues = validate_known_regressions(
        {"max_num_partial_prefills": {"type": "categorical", "values": [1, 4]}},
        registry=[_R23, _R25, _R28],
    )
    assert any(i.ref == "R23" and i.severity == "block" for i in issues)


def test_regression_r25_pcp_gt_1_blocks():
    issues = validate_known_regressions(
        {"prefill_context_parallel_size": {"type": "categorical", "values": [1, 2, 4]}},
        registry=[_R23, _R25, _R28],
    )
    assert any(i.ref == "R25" and i.severity == "block" for i in issues)


def test_regression_r25_pcp_only_1_does_not_block():
    """PCP=[1] 단일 값은 sample > 1 가 없어 매칭 안 됨."""
    issues = validate_known_regressions(
        {"prefill_context_parallel_size": {"type": "categorical", "values": [1]}},
        registry=[_R23, _R25, _R28],
    )
    assert all(i.ref != "R25" for i in issues)


def test_regression_r28_pp_ep_combo_warns():
    issues = validate_known_regressions(
        {
            "pp": {"type": "categorical", "values": [1, 2]},
            "ep": {"type": "bool"},
        },
        registry=[_R23, _R25, _R28],
    )
    assert any(i.ref == "R28" and i.severity == "warn" for i in issues)


def test_regression_r28_pp_only_1_does_not_match():
    issues = validate_known_regressions(
        {
            "pp": {"type": "categorical", "values": [1]},
            "ep": {"type": "bool"},
        },
        registry=[_R23, _R25, _R28],
    )
    assert all(i.ref != "R28" for i in issues)


# ---- top-level + feasibility ---------------------------------------------


def test_validate_b3_v3_passes(tmp_path: Path):
    """현 v3 search-space (R28 fix 적용본) 는 0 block 이어야 한다."""
    space = Path(__file__).resolve().parents[2] / "b200/search-spaces/b3_gpt_oss_120b_v3.yaml"
    if not space.exists():
        pytest.skip("v3 search-space 부재 (다른 환경)")
    report = validate_search_space(
        space_yaml_path=space,
        environment=Environment.b200_dual_node(),
        model_id="openai/gpt-oss-120b",
        n_samples=80,
    )
    assert report.n_block == 0, (
        f"expected 0 block, got: {[i.msg for i in report.issues if i.severity == 'block']}"
    )


def test_validate_blocks_bad_space(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        dedent(
            """
            apiVersion: lmtune/search/v1alpha1
            kind: SearchSpace
            name: bad
            axes:
              tp: {type: categorical, values: [2, 4]}
              prefill_context_parallel_size: {type: categorical, values: [1, 2, 4]}
              max_num_partial_prefills: {type: categorical, values: [1, 4]}
              fake_axis_xyz: {type: bool}
            """
        )
    )
    report = validate_search_space(space_yaml_path=bad, n_samples=20)
    assert report.blocked
    refs = {i.ref for i in report.issues if i.severity == "block" and i.ref}
    assert "R23" in refs
    assert "R25" in refs
    # axis_allowlist also blocks fake_axis_xyz + deprecated PCP/partial_prefills
    block_axes = {i.axis for i in report.issues if i.category == "axis_allowlist"}
    assert "fake_axis_xyz" in block_axes


def test_validate_feasibility_coverage_block_when_all_infeasible(tmp_path: Path):
    """constraint 가 항상 false 면 100% infeasible → block."""
    bad = tmp_path / "all_inf.yaml"
    bad.write_text(
        dedent(
            """
            apiVersion: lmtune/search/v1alpha1
            kind: SearchSpace
            name: all-inf
            axes:
              tp: {type: categorical, values: [2, 4]}
            feasibility_constraints:
              - id: never
                rule: "tp > 1000"
                message: "always fails"
            """
        )
    )
    report = validate_search_space(
        space_yaml_path=bad,
        environment=Environment.b200_dual_node(),
        n_samples=50,
    )
    assert report.blocked
    assert any(i.category == "feasibility" and i.severity == "block" for i in report.issues)
    assert report.feasibility_stats["ratio"] == 1.0
