"""Model meta catalog — vllm-config-puzzle simulator/models.ts 의 1:1 Python port.

Used by:
- `src/lmtune/search/feasibility.py` — 12 declarative constraints 의 `model.*` 참조
- `src/lmtune/search/surrogate_analytical.py` (future) — TTFT/ITL/TPS 공식
- dashboard — model card 의 메타 표시
"""

from lmtune.models.registry import (
    COMMAND_R_PLUS,
    DBRX,
    DEEPSEEK_V3,
    GEMMA_2_27B,
    LLAMA_8B,
    LLAMA_70B,
    LLAMA_405B,
    MIXTRAL_8X22B,
    MODELS,
    QWEN2_5_7B,
    QWEN2_5_72B,
    QWEN3_235B_A22B,
    MLASpec,
    ModelSpec,
    MoESpec,
    by_name,
    list_models,
    normalize_model_spec,
)

__all__ = [
    "COMMAND_R_PLUS",
    "DBRX",
    "DEEPSEEK_V3",
    "GEMMA_2_27B",
    "LLAMA_8B",
    "LLAMA_70B",
    "LLAMA_405B",
    "MIXTRAL_8X22B",
    "MLASpec",
    "MODELS",
    "ModelSpec",
    "MoESpec",
    "QWEN2_5_7B",
    "QWEN2_5_72B",
    "QWEN3_235B_A22B",
    "by_name",
    "list_models",
    "normalize_model_spec",
]
