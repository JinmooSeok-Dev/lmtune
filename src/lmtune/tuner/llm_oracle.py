"""LLMOracleSampler — Tuner ABC plug-in 패턴 stub (LLM-guided sampler).

REFACTOR-PLAN PLUG: PostgresArtifactStore 와 같은 정신 — Tuner 의 Sampler ABC
가 새 backend 를 1 PR 로 받아들임을 시연한다.

본 stub 의 목적:
1. ``tuner.Sampler`` ABC 의 신규 구현체가 1 파일로 추가되면 ``tuner.factory``
   의 매핑 1줄 + ``cli_search`` 의 strategy enum 1줄로 전체 driver 에 통합됨을
   증명한다.
2. 실제 LLM 호출은 ``anthropic`` SDK 가 설치된 ``[agent]`` extra 환경에서만
   동작 — 미설치 시 친절 ImportError ("install with lmtune[agent]").
3. ``ask`` 는 NotImplementedError — follow-up PR 이 prompt template + archive
   요약 + axis 도메인 지식 prior 로 채울 수 있도록 hook 만 마련.

LLM-guided 모드의 Headless 와의 양립 (README LLM Dependency Policy):
- 본 sampler 는 optional. headless (TPE/NSGA-II/random_native) 가 1st-class.
- 외부 사용자가 받아 실행하는 모든 핵심 경로는 LLM 콜 0회로 동작.
- LLMOracleSampler 는 axis_priors.yaml update tool 또는 macro 추론 보조 영역.
"""

from __future__ import annotations

from typing import Any

from lmtune.search.space import SearchSpace
from lmtune.tuner.base import Sampler


class LLMOracleSampler(Sampler):
    """LLM (Claude / GPT / local) 도메인 지식으로 다음 trial params 를 추론.

    Args:
        space: 탐색 공간.
        model: LLM 모델 식별자 (예: "claude-opus-4-7", "gpt-5o", "ollama/qwen2.5").
        api_key_env: API key 가 들어있는 환경변수명 (default ``ANTHROPIC_API_KEY``).
        max_tokens: LLM 호출당 최대 출력 토큰 (default 1024).
        archive_summary: archive 요약을 prompt 에 첨부할지 (default True).

    Raises:
        ImportError: ``anthropic`` SDK 미설치 시. ``pip install lmtune[agent]``.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        model: str = "claude-opus-4-7",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 1024,
        archive_summary: bool = True,
    ):
        try:
            import anthropic  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic SDK is required for LLMOracleSampler — "
                "install with: pip install 'lmtune[agent]'"
            ) from e

        self.space = space
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.archive_summary = archive_summary
        # archive cache — tell() 가 누적 (sampler 자체가 archive owner 는 아니지만,
        # 직전 N개 trial 결과를 prompt 에 첨부하기 위한 in-memory rolling window).
        self._recent: list[tuple[dict[str, Any], float]] = []

    # ── Sampler ABC ──────────────────────────────────────────────────

    def ask(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError(
            "LLMOracleSampler.ask — prompt template + active axis 추출 + LLM 호출이 "
            "아직 미구현. follow-up PR 에서 (1) space → axis 카탈로그 자연어 변환, "
            "(2) self._recent → archive 요약, (3) anthropic.messages.create 호출, "
            "(4) JSON 응답 → params dict parse 로 채울 예정."
        )

    def tell(
        self,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """archive rolling window 에 (params, score) 1건 누적."""
        del metrics
        self._recent.append((params, score))
        # window 크기 제한 — prompt token budget 보호 (default 16건)
        if len(self._recent) > 16:
            self._recent = self._recent[-16:]
