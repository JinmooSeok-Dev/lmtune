"""Feasibility evaluator — b3 의 12 declarative constraints 의 단위 검증.

각 constraint 가 fail/pass 지정한 시나리오에서 정확히 분기하는지.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lmtune.models import by_name
from lmtune.search.feasibility import (
    Constraint,
    Environment,
    _safe_eval,
    evaluate,
    is_feasible,
    load_constraints,
)

B3_PATH = Path("b200/search-spaces/b3_parallelism.yaml")


@pytest.fixture
def constraints():
    return load_constraints(B3_PATH)


def _make_params(**kwargs):
    """Sane defaults for a feasible TP=8 PP=2 trial; override with kwargs."""
    base = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        expert_parallel_size=1,
        ep_strategy="standard",
        cross_node_type="ib",
        gpu_memory_utilization=0.85,
    )
    base.update(kwargs)
    return base


def test_load_constraints_count(constraints):
    # b3 has exactly 12 constraints (validation.ts 1:1 port)
    assert len(constraints) == 12
    ids = {c.id for c in constraints}
    for i in range(1, 13):
        assert any(cid.startswith(f"c{i}_") for cid in ids), f"missing c{i}_*"


def test_safe_eval_arithmetic():
    assert _safe_eval("2 + 3", {}) == 5
    assert _safe_eval("(8 % 4) == 0", {}) is True
    assert _safe_eval("a > b", {"a": 5, "b": 3}) is True


def test_safe_eval_aliases_NOT_AND_OR():
    # SQL-like operators converted to Python
    assert _safe_eval("(NOT True) OR (1 > 0)", {}) is True
    assert _safe_eval("True AND False", {}) is False


def test_safe_eval_rejects_function_call():
    from lmtune.search.feasibility import _SafeEvalError

    with pytest.raises(_SafeEvalError):
        _safe_eval("__import__('os').system('ls')", {})


def test_feasible_70b_dual_node(constraints):
    p = _make_params()
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    assert rep.feasible, rep.reason()


def test_c2_tp_exceeds_single_node(constraints):
    p = _make_params(tensor_parallel_size=16)
    env = Environment.b200_single_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    assert not rep.feasible
    assert any(f["id"] == "c2_tp_single_node" for f in rep.failures)


def test_c4_heads_not_divisible_by_tp(constraints):
    # Qwen2.5-7B has 28 attention heads. TP=8 → 28%8=4 != 0
    p = _make_params(tensor_parallel_size=8, pipeline_parallel_size=1)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Qwen2.5-7B"))
    assert not rep.feasible
    assert any(f["id"] == "c4_heads_div_tp" for f in rep.failures)


def test_c4_heads_divisible_pass(constraints):
    # Llama-70B has 64 heads. TP=8 → 64%8=0 ✓
    p = _make_params(tensor_parallel_size=8)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    # Only check c4 specifically
    assert not any(f["id"] == "c4_heads_div_tp" for f in rep.failures)


def test_c8_dcp_must_divide_tp(constraints):
    p = _make_params(tensor_parallel_size=4, decode_context_parallel_size=3)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    assert not rep.feasible
    assert any(f["id"] == "c8_dcp_tp_div" for f in rep.failures)


def test_c5_ep_skipped_for_dense_model(constraints):
    # Dense model — c5_experts_div_ep should not fire even with ep > 1
    p = _make_params(expert_parallel_size=8)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    # c5 evaluates to TRUE because (NOT model.is_moe) is true
    assert not any(f["id"] == "c5_experts_div_ep" for f in rep.failures)


def test_c5_ep_must_divide_experts_for_moe(constraints):
    # Mixtral-8x22B has 8 experts. EP=3 → 8%3 != 0
    p = _make_params(expert_parallel_size=3, tensor_parallel_size=8, pipeline_parallel_size=1)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Mixtral-8x22B"))
    assert any(f["id"] == "c5_experts_div_ep" for f in rep.failures)


def test_c12_wide_ep_requires_dp_ge_2(constraints):
    p = _make_params(ep_strategy="wide", data_parallel_size=1)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("DeepSeek-V3"))
    assert any(f["id"] == "c12_wide_ep_dp" for f in rep.failures)


def test_c11_severity_warning(constraints):
    # PCP * TP > npus_per_server (8) but other constraints ok
    p = _make_params(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=4,  # 4*4=16 > 8 → warn
        data_parallel_size=1,
    )
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("Llama-3.1-70B"))
    # Should be warning, not failure; rep.feasible may still be True
    # if no other rules fail
    assert any(w["id"] == "c11_pcp_intra_pref" for w in rep.warnings)


def test_is_feasible_convenience(constraints):
    env = Environment.b200_dual_node()
    p = _make_params()
    assert (
        is_feasible(p, environment=env, constraints=constraints, model_name="Llama-3.1-70B") is True
    )
    p_bad = _make_params(decode_context_parallel_size=3, tensor_parallel_size=4)
    assert (
        is_feasible(p_bad, environment=env, constraints=constraints, model_name="Llama-3.1-70B")
        is False
    )


def test_is_feasible_with_yaml_path(tmp_path):
    import yaml

    f = tmp_path / "small_space.yaml"
    f.write_text(
        yaml.safe_dump(
            {
                "axes": {"tensor_parallel_size": {"type": "int", "low": 1, "high": 8}},
                "feasibility_constraints": [
                    {
                        "id": "x",
                        "rule": "tensor_parallel_size <= environment.npus_per_server",
                        "message": "tp fits",
                    },
                ],
            }
        )
    )
    env = Environment.b200_single_node()
    assert is_feasible({"tensor_parallel_size": 4}, environment=env, space_yaml_path=f) is True
    assert is_feasible({"tensor_parallel_size": 16}, environment=env, space_yaml_path=f) is False


def test_evaluate_with_no_constraints():
    rep = evaluate({"x": 1}, environment=Environment.b200_dual_node(), constraints=[])
    assert rep.feasible
    assert rep.failures == []


def test_evaluate_skips_undefined_axis():
    # Constraint references a non-existent axis → treated as N/A, no failure
    c = [Constraint(id="cx", rule="undefined_axis > 0", message="m")]
    rep = evaluate({"x": 1}, environment=Environment.b200_dual_node(), constraints=c)
    assert rep.feasible


# --- gpt-oss-120b on B200 16-GPU scenarios (사용자 production 환경) ---


def test_gpt_oss_120b_default_tp8_dp2_dual_node_feasible(constraints):
    """현재 production: TP=8 × DP=2 = 16 GPU on dual-node B200 — feasible.

    chart values (decode.parallelism.tensor=8 + decode.replicas=2) 와 정렬.
    """
    p = _make_params(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=2,
        expert_parallel_size=1,
        ep_strategy="standard",
    )
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert rep.feasible, rep.reason()


def test_gpt_oss_120b_tp16_cross_node_rejected(constraints):
    """TP=16 → npus_per_server(8) 초과 → c2_tp_single_node fail. NCCL all-reduce
    가 cross-node 가는 토폴로지는 비효율 (NVLink 900Gbps → IB 400Gbps drop).
    """
    p = _make_params(tensor_parallel_size=16, pipeline_parallel_size=1, data_parallel_size=1)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert not rep.feasible
    assert any(f["id"] == "c2_tp_single_node" for f in rep.failures), rep.reason()


def test_gpt_oss_120b_tp_must_divide_64_attention_heads(constraints):
    """gpt-oss-120b 의 num_attention_heads=64. TP ∈ {1,2,4,8} 만 허용.
    TP=3 같은 비-divisor 는 c4_heads_div_tp 로 reject.
    """
    p = _make_params(tensor_parallel_size=3, pipeline_parallel_size=1)
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert not rep.feasible
    assert any(f["id"] == "c4_heads_div_tp" for f in rep.failures), rep.reason()


def test_gpt_oss_120b_ep_must_divide_128_experts(constraints):
    """gpt-oss-120b MoE 128 experts. EP ∈ {1,2,4,8,16,32,64,128} 만 허용.
    EP=3 같은 비-divisor 는 c5_experts_div_ep 로 reject.
    """
    p = _make_params(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=3,
        ep_strategy="standard",
    )
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert not rep.feasible
    assert any(f["id"] == "c5_experts_div_ep" for f in rep.failures), rep.reason()


def test_gpt_oss_120b_wide_ep_dp16_dual_node_feasible(constraints):
    """Wide-EP 시나리오: DP=16 EP=16 TP=1 — wide-ep-lws path 의 본격 활용 형태."""
    p = _make_params(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=16,
        expert_parallel_size=16,
        ep_strategy="wide",
    )
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert rep.feasible, rep.reason()


def test_gpt_oss_120b_pp2_cross_node_requires_fabric(constraints):
    """PP=2 + cross_node_type='none' → c10_multi_node fail. PP 가 노드를 가로지르려면
    IB/RoCE 가 필요.
    """
    p = _make_params(
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        data_parallel_size=1,
        cross_node_type="none",
    )
    env = Environment.b200_dual_node()
    rep = evaluate(p, environment=env, constraints=constraints, model=by_name("gpt-oss-120b"))
    assert not rep.feasible
    failure_ids = {f["id"] for f in rep.failures}
    assert "c9_pp_cross_node" in failure_ids or "c10_multi_node" in failure_ids, rep.reason()
