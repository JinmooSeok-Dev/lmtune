"""Model meta catalog.

vllm-config-puzzle/src/engine/llm-dist-sim/models.ts 의 1:1 Python port.
TypeScript camelCase → Python snake_case 변환. 본 carryover 가 b3
feasibility_constraints 의 `model.*` 참조에 1:1 매칭.

새 모델 추가는 `_RAW` 리스트에 한 줄만 추가 — `normalize_model_spec` 이
나머지 (head_dim, kv_cache_dtype_bytes, active_params) 를 derive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoESpec:
    """Mixture-of-Experts 메타. dense 모델은 None."""

    num_experts: int
    active_experts: int  # top-k routing
    shared_experts: int = 0


@dataclass(frozen=True)
class MLASpec:
    """Multi-head Latent Attention (DeepSeek-V3) 메타. 일반 모델은 None."""

    kv_latent_dim: int  # kv_lora_rank (DSV3 = 512)
    rope_head_dim: int  # qk_rope_head_dim (DSV3 = 64)


@dataclass(frozen=True)
class ModelSpec:
    """모델 단일 사양. simulator.ts 의 ModelSpec 인터페이스와 1:1.

    feasibility constraint 의 `model.*` 가 본 dataclass 의 attribute 를 직접 참조.
    """

    name: str
    total_params_b: float  # in billions
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    context_length: int
    vocab_size: int
    dtype_bytes: int = 2  # bf16 default
    head_dim: int | None = None
    kv_cache_dtype_bytes: int = 2
    active_params_b: float | None = None  # MoE 시 활성 expert 만, dense=total
    moe: MoESpec | None = None
    mla: MLASpec | None = None

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def has_mla(self) -> bool:
        return self.mla is not None

    @property
    def num_experts(self) -> int:
        return self.moe.num_experts if self.moe else 0

    @property
    def expert_ratio(self) -> float:
        """MoE 전문가 weight 비율 (intermediate_size 비례 추정).

        memory.ts:32-45 의 dynamic ratio 와 동형. dense → 0.0.
        """
        if not self.moe:
            return 0.0
        # Coarse estimate: experts dominate when intermediate × num_experts >> attention
        attn = 4 * self.hidden_size**2 * self.num_layers
        ffn = 3 * self.hidden_size * self.intermediate_size * self.num_experts * self.num_layers
        return ffn / max(1, attn + ffn)


def normalize_model_spec(
    *,
    name: str,
    total_params_b: float,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    context_length: int,
    vocab_size: int,
    dtype_bytes: int = 2,
    head_dim: int | None = None,
    kv_cache_dtype_bytes: int | None = None,
    active_params_b: float | None = None,
    moe: MoESpec | None = None,
    mla: MLASpec | None = None,
) -> ModelSpec:
    """vllm-config-puzzle/models.ts:4 normalizeModelSpec 와 동일 로직."""
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    if kv_cache_dtype_bytes is None:
        kv_cache_dtype_bytes = 2
    if active_params_b is None:
        if moe is not None:
            active_params_b = total_params_b * (
                (moe.active_experts + moe.shared_experts) / moe.num_experts
            )
        else:
            active_params_b = total_params_b
    return ModelSpec(
        name=name,
        total_params_b=total_params_b,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        context_length=context_length,
        vocab_size=vocab_size,
        dtype_bytes=dtype_bytes,
        head_dim=head_dim,
        kv_cache_dtype_bytes=kv_cache_dtype_bytes,
        active_params_b=active_params_b,
        moe=moe,
        mla=mla,
    )


# --- 카탈로그 ---
# 1:1 port of vllm-config-puzzle/models.ts (9 models). 새 모델 추가는 본 리스트에만.

LLAMA_8B = normalize_model_spec(
    name="Llama-3.1-8B",
    total_params_b=8.0,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=8,
    intermediate_size=14336,
    context_length=131072,
    vocab_size=128256,
)
LLAMA_70B = normalize_model_spec(
    name="Llama-3.1-70B",
    total_params_b=70.0,
    num_layers=80,
    hidden_size=8192,
    num_attention_heads=64,
    num_kv_heads=8,
    intermediate_size=28672,
    context_length=131072,
    vocab_size=128256,
)
LLAMA_405B = normalize_model_spec(
    name="Llama-3.1-405B",
    total_params_b=405.0,
    num_layers=126,
    hidden_size=16384,
    num_attention_heads=128,
    num_kv_heads=8,
    intermediate_size=53248,
    context_length=131072,
    vocab_size=128256,
)
QWEN2_5_7B = normalize_model_spec(
    name="Qwen2.5-7B",
    total_params_b=7.6,
    num_layers=28,
    hidden_size=3584,
    num_attention_heads=28,
    num_kv_heads=4,
    intermediate_size=18944,
    context_length=131072,
    vocab_size=152064,
)
QWEN2_5_72B = normalize_model_spec(
    name="Qwen2.5-72B",
    total_params_b=72.0,
    num_layers=80,
    hidden_size=8192,
    num_attention_heads=64,
    num_kv_heads=8,
    intermediate_size=29568,
    context_length=131072,
    vocab_size=152064,
)
QWEN3_235B_A22B = normalize_model_spec(
    name="Qwen3-235B-A22B",
    total_params_b=235.0,
    num_layers=94,
    hidden_size=4096,
    num_attention_heads=64,
    num_kv_heads=4,
    intermediate_size=12288,
    context_length=128000,
    vocab_size=151936,
    moe=MoESpec(num_experts=128, active_experts=8, shared_experts=0),
    active_params_b=22.0,
)
GEMMA_2_27B = normalize_model_spec(
    name="Gemma-2-27B",
    total_params_b=27.0,
    num_layers=46,
    hidden_size=4608,
    num_attention_heads=32,
    num_kv_heads=16,
    intermediate_size=36864,
    context_length=8192,
    vocab_size=256128,
)
COMMAND_R_PLUS = normalize_model_spec(
    name="Command-R+ 104B",
    total_params_b=104.0,
    num_layers=64,
    hidden_size=12288,
    num_attention_heads=96,
    num_kv_heads=8,
    intermediate_size=33792,
    context_length=131072,
    vocab_size=256000,
)
MIXTRAL_8X22B = normalize_model_spec(
    name="Mixtral-8x22B",
    total_params_b=141.0,
    active_params_b=39.0,
    num_layers=56,
    hidden_size=6144,
    num_attention_heads=48,
    num_kv_heads=8,
    intermediate_size=16384,
    context_length=65536,
    vocab_size=32000,
    moe=MoESpec(num_experts=8, active_experts=2, shared_experts=0),
)
DBRX = normalize_model_spec(
    name="DBRX-132B",
    total_params_b=132.0,
    active_params_b=36.0,
    num_layers=40,
    hidden_size=6144,
    num_attention_heads=48,
    num_kv_heads=8,
    intermediate_size=10752,
    context_length=32768,
    vocab_size=100352,
    moe=MoESpec(num_experts=16, active_experts=4, shared_experts=0),
)
DEEPSEEK_V3 = normalize_model_spec(
    name="DeepSeek-V3",
    total_params_b=685.0,
    active_params_b=37.0,
    num_layers=61,
    hidden_size=7168,
    num_attention_heads=128,
    num_kv_heads=128,
    intermediate_size=18432,
    context_length=128000,
    vocab_size=129280,
    moe=MoESpec(num_experts=256, active_experts=8, shared_experts=1),
    mla=MLASpec(kv_latent_dim=512, rope_head_dim=64),
)
# OpenAI gpt-oss (2025-08 공개, open-weight, MXFP4 native).
# spec [추정] — public model card 기반 best-effort. feasibility/active_if 가 model.is_moe
# 분기 정도만 사용하므로 정밀치는 critical 하지 않음. 정확치 확인 시 갱신.
GPT_OSS_20B = normalize_model_spec(
    name="gpt-oss-20b",
    total_params_b=20.0,
    active_params_b=3.6,
    num_layers=24,
    hidden_size=2880,
    num_attention_heads=64,
    num_kv_heads=8,
    intermediate_size=2880,
    context_length=131072,
    vocab_size=201088,
    moe=MoESpec(num_experts=32, active_experts=4, shared_experts=0),
)
GPT_OSS_120B = normalize_model_spec(
    name="gpt-oss-120b",
    total_params_b=117.0,
    active_params_b=5.1,
    num_layers=36,
    hidden_size=2880,
    num_attention_heads=64,
    num_kv_heads=8,
    intermediate_size=2880,
    context_length=131072,
    vocab_size=201088,
    moe=MoESpec(num_experts=128, active_experts=4, shared_experts=0),
)
# MiniMax-M2 — HF config.json 직접 확인 (2026-05-12).
# 230B total / 10B active. MoE 256 experts × top-8, no shared expert. GQA 48:8.
# attention 은 hybrid (lightning attention + softmax 교대) — vllm 0.17.1 의
# vllm/model_executor/models/minimax_m2.py 가 1st-class 처리. R26 (GQA + DCP > 1
# 시 tp > 8) 동일 제약 적용.
MINIMAX_M2 = normalize_model_spec(
    name="MiniMax-M2",
    total_params_b=230.0,
    active_params_b=10.0,
    num_layers=62,
    hidden_size=3072,
    num_attention_heads=48,
    num_kv_heads=8,
    intermediate_size=1536,
    context_length=196608,
    vocab_size=200064,
    moe=MoESpec(num_experts=256, active_experts=8, shared_experts=0),
)


MODELS: list[ModelSpec] = [
    LLAMA_8B,
    QWEN2_5_7B,
    QWEN2_5_72B,
    GEMMA_2_27B,
    LLAMA_70B,
    COMMAND_R_PLUS,
    LLAMA_405B,
    MIXTRAL_8X22B,
    DBRX,
    QWEN3_235B_A22B,
    DEEPSEEK_V3,
    GPT_OSS_20B,
    GPT_OSS_120B,
    MINIMAX_M2,
]


_BY_NAME: dict[str, ModelSpec] = {}
for _m in MODELS:
    _BY_NAME[_m.name] = _m
    _BY_NAME[_m.name.lower()] = _m
    # huggingface-style aliases
    short = _m.name.replace(" ", "").lower()
    _BY_NAME.setdefault(short, _m)


def by_name(name: str) -> ModelSpec | None:
    """Lookup by name (case-insensitive). Returns None if unknown."""
    if not name:
        return None
    return _BY_NAME.get(name) or _BY_NAME.get(name.lower())


def list_models() -> list[ModelSpec]:
    return list(MODELS)
