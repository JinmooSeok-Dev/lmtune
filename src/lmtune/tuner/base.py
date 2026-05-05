"""Tuner ABC — Sampler / Pruner.

REFACTOR-PLAN 의 S1 단계. 기존 `src/lmtune/search/` 의 sampler 들 (Optuna 래핑
+ native) 을 단일 ABC 뒤로 정리하기 전 단계로, 먼저 ABC 만 정의한다. 후속 PR
이 기존 sampler 를 어댑터로 감싼다.

설계:
- `Sampler.ask(context)` → 다음 trial 의 params dict
- `Sampler.tell(params, score, metrics)` → 결과 피드백 (refit 등 sampler 별 동작)
- `Pruner.should_prune(trial_id, step, value)` → 중간 보고 (intermediate value)
  기반 조기 종료 판단

archive owner 는 Storage. Sampler 는 in-memory cache 만. tell() 은 가장 최근
1건만 받음 (필요하면 sampler 가 자체 재구축).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Sampler(ABC):
    """탐색 sampler 의 추상 인터페이스.

    구현체 예: Optuna TPE/CMA-ES/NSGAII/Random/Grid/LHC, native TPE/LHC,
    UCB bandit, BoTorch, LLMOracleSampler 등.
    """

    @abstractmethod
    def ask(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """다음 trial 의 params 를 결정해 반환.

        Args:
            context: active_if 룰 매칭에 쓰이는 dict (예: {"adapter": "llmd-k8s"}).
                None 이면 모든 axis 활성으로 간주.

        Returns:
            {axis_name: value} 형태. 비어 있으면 안 됨 (active axis 가 0개라도
            sampler 는 빈 dict 반환을 허용하되 호출자가 그 경우를 처리).
        """

    def tell(
        self,
        params: dict[str, Any],
        score: float,
        metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """trial 결과 1건을 sampler 에 피드백.

        기본 구현은 no-op. TPE/CMA-ES 같이 누적 archive 가 필요한 sampler 만
        override 한다.

        Args:
            params: ask() 가 반환한 params dict.
            score: composite objective score (higher is better).
            metrics: optional secondary metrics (sobol, importance 분석용).
        """
        del params, score, metrics
        return None


class Pruner(ABC):
    """trial 의 중간 보고 기반 조기 종료 sampler.

    구현체 예: SuccessiveHalving, Hyperband, MedianPruner, ASHA.
    repeat-N 측정 시 첫 1-2 run 결과만 보고 나머지를 skip 가능.
    """

    @abstractmethod
    def should_prune(
        self,
        trial_id: str,
        step: int,
        value: float,
        history: list[float] | None = None,
    ) -> bool:
        """현재 trial 을 중단할지 판단.

        Args:
            trial_id: 식별자 (storage 내 trial.trial_id 와 동일).
            step: 0-based 누적 보고 회차 (예: repeat-N 의 N 번째).
            value: 본 step 의 objective 값.
            history: 본 trial 의 누적 value 리스트 (optional, sampler 내부에
                cache 한 buffer 와 동등하면 생략).

        Returns:
            True 면 trial 중단. False 면 계속.
        """


__all__ = ["Sampler", "Pruner"]
