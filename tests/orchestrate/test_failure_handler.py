"""Unit tests for failure_handler.

CircuitBreaker 와 classify_outcome 의 동작을 mocking 없이 검증.
실제 study/driver 와의 통합은 별도 smoke 에서 (별도 fixture 필요).
"""
from __future__ import annotations

from lmtune.orchestrate.failure_handler import (
    CircuitBreaker,
    CircuitBreakerConfig,
    FailureClass,
    classify_outcome,
    suggest_recovery,
)


# ---- classify_outcome -----------------------------------------------------

def test_classify_completed_is_success():
    assert classify_outcome("completed") == FailureClass.SUCCESS


def test_classify_pruned_is_pruned():
    assert classify_outcome("pruned") == FailureClass.PRUNED


def test_classify_crash_with_rollout_prefix_oom():
    notes = "rollout oom: pod-x: CrashLoopBackOff: cuda OOM | logs: ..."
    assert classify_outcome("crash", notes=notes) == FailureClass.OOM


def test_classify_crash_with_rollout_prefix_infeasible():
    notes = "rollout infeasible: pod-x: ValidationError | logs: ..."
    assert classify_outcome("crash", notes=notes) == FailureClass.INFEASIBLE


def test_classify_crash_with_rollout_prefix_startup_timeout():
    notes = "rollout startup_timeout: timeout 600s (1/2 ready)"
    assert classify_outcome("crash", notes=notes) == FailureClass.STARTUP_TIMEOUT


def test_classify_crash_fallback_to_log_pattern_oom():
    err = "torch.cuda.OutOfMemoryError: CUDA out of memory."
    assert classify_outcome("crash", error=err) == FailureClass.OOM


def test_classify_crash_fallback_to_log_pattern_transient():
    err = "[rank0] NCCL all-reduce timeout after 600s"
    assert classify_outcome("crash", error=err) == FailureClass.TRANSIENT


def test_classify_crash_no_text_is_hard():
    assert classify_outcome("crash") == FailureClass.HARD


def test_classify_unknown_status_is_hard():
    assert classify_outcome("foobar") == FailureClass.HARD


# ---- CircuitBreaker -------------------------------------------------------

def test_breaker_passes_through_successes():
    b = CircuitBreaker()
    for _ in range(20):
        b.record(FailureClass.SUCCESS)
    halt, reason = b.should_halt()
    assert halt is False
    assert reason == ""


def test_breaker_pruned_counts_as_success_for_stability():
    """SLO 위반 (pruned) 은 측정 자체는 valid 하므로 안정성 평가에서 success."""
    b = CircuitBreaker(cfg=CircuitBreakerConfig(max_consecutive_failures=3))
    for _ in range(10):
        b.record(FailureClass.PRUNED)
    halt, _ = b.should_halt()
    assert halt is False


def test_breaker_infeasible_counts_as_success_for_stability():
    """구성 invalid 는 인프라 실패가 아니라 sampler 학습 신호 — stability 에서 제외."""
    b = CircuitBreaker(cfg=CircuitBreakerConfig(max_consecutive_failures=3))
    for _ in range(10):
        b.record(FailureClass.INFEASIBLE)
    halt, _ = b.should_halt()
    assert halt is False


def test_breaker_consecutive_failures_trigger_halt():
    b = CircuitBreaker(cfg=CircuitBreakerConfig(max_consecutive_failures=3))
    b.record(FailureClass.OOM)
    b.record(FailureClass.OOM)
    halt, _ = b.should_halt()
    assert halt is False
    b.record(FailureClass.OOM)
    halt, reason = b.should_halt()
    assert halt is True
    assert "consecutive" in reason


def test_breaker_consecutive_resets_on_success():
    # rate-gate 비활성으로 consecutive 만 검증.
    cfg = CircuitBreakerConfig(max_consecutive_failures=3, max_failure_rate=1.01)
    b = CircuitBreaker(cfg=cfg)
    b.record(FailureClass.OOM)
    b.record(FailureClass.OOM)
    b.record(FailureClass.SUCCESS)   # reset consecutive
    b.record(FailureClass.OOM)
    b.record(FailureClass.OOM)
    halt, _ = b.should_halt()
    assert halt is False


def test_breaker_window_rate_trigger():
    cfg = CircuitBreakerConfig(
        max_consecutive_failures=99,    # 발동 X
        max_failure_rate=0.6,
        window=10,
        min_trials_before_rate_check=5,
    )
    b = CircuitBreaker(cfg=cfg)
    # 6 fails + 4 success out of 10 → rate 0.6 ≥ 0.6
    for _ in range(6):
        b.record(FailureClass.HARD)
    for _ in range(4):
        b.record(FailureClass.SUCCESS)
    halt, reason = b.should_halt()
    assert halt is True
    assert "failure rate" in reason


def test_breaker_window_rate_skipped_below_min_trials():
    cfg = CircuitBreakerConfig(
        max_consecutive_failures=99,
        max_failure_rate=0.5,
        window=10,
        min_trials_before_rate_check=8,
    )
    b = CircuitBreaker(cfg=cfg)
    # 4 fail / 6 total → rate 0.66 but only 6 trials
    for _ in range(4):
        b.record(FailureClass.HARD)
    for _ in range(2):
        b.record(FailureClass.SUCCESS)
    halt, _ = b.should_halt()
    assert halt is False


def test_breaker_remains_halted_once_tripped():
    cfg = CircuitBreakerConfig(max_consecutive_failures=2)
    b = CircuitBreaker(cfg=cfg)
    b.record(FailureClass.OOM)
    b.record(FailureClass.OOM)
    halt1, reason1 = b.should_halt()
    assert halt1 is True
    # 이후 success 가 와도 latch 유지
    b.record(FailureClass.SUCCESS)
    halt2, reason2 = b.should_halt()
    assert halt2 is True
    assert reason2 == reason1


def test_breaker_summary_format():
    b = CircuitBreaker()
    b.record(FailureClass.SUCCESS)
    b.record(FailureClass.OOM)
    s = b.summary()
    assert "total=2" in s
    assert "fails=1" in s
    assert "consecutive=1" in s


# ---- suggest_recovery -----------------------------------------------------

def test_suggest_recovery_oom_halves_max_num_seqs():
    new = suggest_recovery(FailureClass.OOM, {"max_num_seqs": 128})
    assert new == {"max_num_seqs": 64}


def test_suggest_recovery_oom_floor_at_16():
    new = suggest_recovery(FailureClass.OOM, {"max_num_seqs": 32})
    assert new == {"max_num_seqs": 16}


def test_suggest_recovery_oom_falls_back_to_gpu_mem_util():
    new = suggest_recovery(FailureClass.OOM, {"max_num_seqs": 16, "gpu_memory_utilization": 0.90})
    assert new is not None
    assert new["gpu_memory_utilization"] == 0.85


def test_suggest_recovery_oom_no_levers():
    assert suggest_recovery(FailureClass.OOM, {"max_num_seqs": 16, "gpu_memory_utilization": 0.80}) is None


def test_suggest_recovery_transient_returns_same():
    p = {"max_num_seqs": 64}
    new = suggest_recovery(FailureClass.TRANSIENT, p)
    assert new == p
    assert new is not p   # caller 가 mutate 해도 원본 안전


def test_suggest_recovery_infeasible_no_retry():
    assert suggest_recovery(FailureClass.INFEASIBLE, {"x": 1}) is None


def test_suggest_recovery_hard_no_retry():
    assert suggest_recovery(FailureClass.HARD, {"x": 1}) is None
