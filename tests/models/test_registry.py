"""Model registry — vllm-config-puzzle/models.ts 1:1 port 검증."""

from __future__ import annotations

from lmtune.models import (
    DEEPSEEK_V3,
    LLAMA_8B,
    MIXTRAL_8X22B,
    MODELS,
    QWEN3_235B_A22B,
    ModelSpec,
    by_name,
    list_models,
    normalize_model_spec,
)
from lmtune.models.registry import MoESpec


def test_models_exact_count():
    assert len(MODELS) == 11


def test_normalize_derives_head_dim():
    spec = normalize_model_spec(
        name="x",
        total_params_b=1.0,
        num_layers=4,
        hidden_size=512,
        num_attention_heads=8,
        num_kv_heads=8,
        intermediate_size=1024,
        context_length=2048,
        vocab_size=32000,
    )
    assert spec.head_dim == 64


def test_normalize_dense_active_params_equals_total():
    spec = normalize_model_spec(
        name="x", total_params_b=10.0, num_layers=2, hidden_size=128,
        num_attention_heads=4, num_kv_heads=4, intermediate_size=256,
        context_length=2048, vocab_size=1024,
    )
    assert spec.active_params_b == 10.0


def test_normalize_moe_active_params_derived():
    moe = MoESpec(num_experts=8, active_experts=2, shared_experts=0)
    spec = normalize_model_spec(
        name="x", total_params_b=141.0, num_layers=2, hidden_size=128,
        num_attention_heads=4, num_kv_heads=4, intermediate_size=256,
        context_length=2048, vocab_size=1024, moe=moe,
    )
    # 141 * (2+0)/8 = 35.25
    assert spec.active_params_b == 35.25


def test_moe_spec_flags():
    assert LLAMA_8B.is_moe is False
    assert MIXTRAL_8X22B.is_moe is True
    assert MIXTRAL_8X22B.num_experts == 8


def test_mla_spec_only_dsv3():
    assert DEEPSEEK_V3.has_mla is True
    assert DEEPSEEK_V3.mla.kv_latent_dim == 512
    assert DEEPSEEK_V3.mla.rope_head_dim == 64
    assert LLAMA_8B.has_mla is False


def test_qwen3_235b_active_params_explicit():
    # Explicit override (235B total but 22B active)
    assert QWEN3_235B_A22B.active_params_b == 22.0


def test_by_name_case_insensitive():
    assert by_name("DeepSeek-V3") is DEEPSEEK_V3
    assert by_name("deepseek-v3") is DEEPSEEK_V3
    assert by_name("DEEPSEEK-V3") is DEEPSEEK_V3
    assert by_name("Llama-3.1-8B") is LLAMA_8B
    assert by_name("nonexistent") is None
    assert by_name("") is None
    assert by_name(None) is None  # type: ignore[arg-type]


def test_list_models_returns_copy():
    assert list_models() == MODELS
    a = list_models()
    a.append("dummy")  # type: ignore[arg-type]
    assert "dummy" not in MODELS  # original untouched


def test_modelspec_is_frozen():
    import dataclasses
    assert dataclasses.fields(ModelSpec)
    try:
        LLAMA_8B.name = "changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ModelSpec should be frozen")
