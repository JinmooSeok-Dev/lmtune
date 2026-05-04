"""Controller ABC — 단일 책임: (active_axes, history) → next params dict.

Study 는 이 ABC 에 ask/tell 만 위임한다. 구현체는 Optuna 일 수도, pure
random 일 수도, 외부 LLM API HTTP 호출일 수도 있다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lmtune.search.space import Axis


class Controller(ABC):
    """next params 결정자. Study 의 persistence/breaker 와 직교."""

    @abstractmethod
    def ask(
        self,
        active_axes: list[Axis],
        *,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """현재 환경에서 활성인 axis 들에 대해 next params dict 반환."""

    @abstractmethod
    def tell(
        self,
        params: dict[str, Any],
        *,
        value: float | list[float] | None,
        status: str,                        # 'completed' | 'pruned' | 'crash'
        metadata: dict | None = None,
    ) -> None:
        """trial 결과로 내부 상태 업데이트.

        Optuna controller 는 fit, RandomController 는 noop, HTTP 는 원격 학습.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """controller 식별자 (예: 'optuna:tpe', 'random', 'http', 'mock')."""

    @property
    def exhausted(self) -> bool:
        """더 이상 ask() 가 새 params 를 내놓을 수 없으면 True (예: grid 소진)."""
        return False

    # --- 선택 hook (default no-op) -----------------------------------------

    def add_trial(self, params: dict[str, Any], value: float) -> None:
        """warm-start. 과거 (params, score) 를 학습기에 주입."""
        return None

    def enqueue(self, params: dict[str, Any]) -> None:
        """다음 ask() 가 이 params 를 우선 반환하도록 hint."""
        return None
