"""Trial-level exception/failure handling system.

llm-d 가 K8s 인프라 + 앱 경계가 모호한 솔루션이고, 매 trial 마다 helmfile
재배포 + Recreate strategy + EPP/InferencePool 갱신이 일어나므로 안정성
보장이 어렵다. 단순 self-heal (다음 trial 이 helmfile re-apply 로 덮어쓰기)
에 의존하지 말고, 명시적 분류 + circuit breaker 로 study 의 신뢰성을 확보.

3 컴포넌트:
1. classify_outcome — TrialResult.status + error/notes 를 FailureClass 로 분류.
2. CircuitBreaker — N consecutive failures or window failure-rate 초과 시 halt.
3. suggest_recovery — outcome 기반 param mutation 제안 (선택, retry 시).

본 모듈은 study/driver loop 에 hook 으로 들어간다 — tell() 직후 record(),
다음 ask() 전 should_halt() 체크.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class FailureClass(StrEnum):
    SUCCESS = "success"  # 정상 종료
    PRUNED = "pruned"  # SLO 위반 — 측정은 됐고 sampler 가 학습. 실패 아님.
    INFEASIBLE = "infeasible"  # 구성 자체 invalid (mxfp4×float16, unrecognized args)
    OOM = "oom"  # OutOfMemory, 더 작은 batch 로 retry 가능
    TRANSIENT = "transient"  # NCCL/Connection refused, 1회 retry 시도
    STARTUP_TIMEOUT = "startup_timeout"  # rollout deadline 초과 (crash 는 아님)
    MEASURE_FAILED = "measure_failed"  # aiperf parse 실패 / metrics 빈 결과
    HARD = "hard"  # 분류 불가, score=0 처리


# llmd_k8s.py::apply 가 ApplyResult.notes 에 packaging 하는 형식:
#   "rollout {crash_class}: {detail} | logs: ...{logs_tail}"
_ROLLOUT_NOTES_RE = re.compile(r"rollout\s+(\w+)\s*:", re.IGNORECASE)


def classify_outcome(
    trial_status: str,
    *,
    error: str | None = None,
    notes: str | None = None,
) -> FailureClass:
    """Map (status, error/notes) → FailureClass.

    trial_status: 'completed' | 'pruned' | 'crash' (Trial.TrialStatus.value)
    error/notes : free-form text (DuckDB notes column or TrialResult.error)
    """
    s = (trial_status or "").lower()
    if s == "completed":
        return FailureClass.SUCCESS
    if s == "pruned":
        return FailureClass.PRUNED

    text = " ".join(t for t in (notes, error) if t)
    if not text:
        return FailureClass.HARD

    # 1) llmd_k8s.py 가 wrap 한 "rollout <class>:" prefix 우선
    m = _ROLLOUT_NOTES_RE.search(text)
    if m:
        sub = m.group(1).lower()
        for fc in FailureClass:
            if fc.value == sub:
                return fc

    # 2) fallback: rollout_watcher.classify_crash 의 raw log 패턴 사용
    try:
        from lmtune.deploy.rollout_watcher import classify_crash

        cls = classify_crash(text)
        return FailureClass(cls)
    except (ImportError, ValueError):
        return FailureClass.HARD


@dataclass(slots=True)
class CircuitBreakerConfig:
    max_consecutive_failures: int = 5  # 연속 실패 N+ 회 → halt
    max_failure_rate: float = 0.7  # 최근 window 의 fail rate ≥ 이값 → halt
    window: int = 10  # rolling window 크기
    min_trials_before_rate_check: int = 5  # rate 검사 시작 전 최소 trial


@dataclass(slots=True)
class CircuitBreaker:
    cfg: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    consecutive_failures: int = 0
    total: int = 0
    failure_count: int = 0
    history: deque[bool] = field(default_factory=deque)
    halted: bool = False
    halt_reason: str = ""

    def record(self, outcome: FailureClass) -> None:
        # 안정성 실패에서 제외:
        #   SUCCESS — 정상 측정 완료
        #   PRUNED  — SLO 미달이지만 측정 자체는 valid
        #   INFEASIBLE — 구성 자체 invalid (axis 조합이 모델/HW 와 비호환). 인프라
        #     실패가 아니라 sampler 가 학습할 invalid region. helmfile redeploy 는
        #     성공했고 vllm 이 자기 의지로 거부한 것이므로 "안정성 신호" 가 아님.
        # 나머지 (OOM, TRANSIENT, STARTUP_TIMEOUT, MEASURE_FAILED, HARD) 만 stability
        # failure 로 카운트 → circuit breaker 가 진짜 인프라 문제만 감지.
        stable = (FailureClass.SUCCESS, FailureClass.PRUNED, FailureClass.INFEASIBLE)
        is_failure = outcome not in stable
        self.total += 1
        if is_failure:
            self.consecutive_failures += 1
            self.failure_count += 1
        else:
            self.consecutive_failures = 0
        self.history.append(is_failure)
        while len(self.history) > self.cfg.window:
            self.history.popleft()

    def should_halt(self) -> tuple[bool, str]:
        if self.halted:
            return True, self.halt_reason
        if self.consecutive_failures >= self.cfg.max_consecutive_failures:
            self.halted = True
            self.halt_reason = (
                f"{self.consecutive_failures} consecutive failures "
                f"(threshold {self.cfg.max_consecutive_failures})"
            )
            return True, self.halt_reason
        if (
            self.total >= self.cfg.min_trials_before_rate_check
            and len(self.history) >= self.cfg.min_trials_before_rate_check
        ):
            rate = sum(self.history) / len(self.history)
            if rate >= self.cfg.max_failure_rate:
                self.halted = True
                self.halt_reason = (
                    f"failure rate {rate:.0%} in last {len(self.history)} trials "
                    f"(threshold {self.cfg.max_failure_rate:.0%})"
                )
                return True, self.halt_reason
        return False, ""

    def summary(self) -> str:
        rate = sum(self.history) / len(self.history) if self.history else 0.0
        return (
            f"total={self.total} fails={self.failure_count} "
            f"consecutive={self.consecutive_failures} "
            f"window_rate={rate:.0%}"
        )


def suggest_recovery(outcome: FailureClass, params: dict[str, Any]) -> dict[str, Any] | None:
    """Param mutation suggestion for retry. None = no retry.

    Caller (driver) 가 enqueue_trial / 직접 재submit 으로 활용 가능. 본 함수는
    pure — DB/Optuna 상태에 영향 X. retry 횟수는 caller 가 관리.
    """
    if outcome == FailureClass.OOM:
        new = dict(params)
        m = new.get("max_num_seqs")
        if isinstance(m, int) and m >= 32:
            new["max_num_seqs"] = max(16, m // 2)
            return new
        gm = new.get("gpu_memory_utilization")
        if isinstance(gm, (int, float)) and gm > 0.82:
            new["gpu_memory_utilization"] = max(0.78, float(gm) - 0.05)
            return new
        return None
    if outcome == FailureClass.TRANSIENT:
        return dict(params)  # 같은 params 로 1회 retry
    # INFEASIBLE / HARD / STARTUP_TIMEOUT / MEASURE_FAILED → no retry
    return None
