"""OptunaSamplerAdapter / OptunaPrunerAdapter 동작 검증."""

from __future__ import annotations

import optuna
import pytest

from lmtune.search.space import Axis, SearchSpace
from lmtune.tuner import OptunaPrunerAdapter, OptunaSamplerAdapter, Sampler


@pytest.fixture
def small_space() -> SearchSpace:
    return SearchSpace(
        name="test",
        axes=[
            Axis(name="lr", kind="float", low=0.001, high=0.1),
            Axis(name="batch", kind="categorical", values=[16, 32, 64]),
        ],
    )


# ─── OptunaSamplerAdapter ────────────────────────────────────────────


def test_adapter_is_sampler(small_space):
    adapter = OptunaSamplerAdapter(small_space, optuna.samplers.RandomSampler(seed=42))
    assert isinstance(adapter, Sampler)


def test_adapter_ask_returns_axis_dict(small_space):
    adapter = OptunaSamplerAdapter(small_space, optuna.samplers.RandomSampler(seed=42))
    p = adapter.ask()
    assert set(p.keys()) == {"lr", "batch"}
    assert 0.001 <= p["lr"] <= 0.1
    assert p["batch"] in (16, 32, 64)


def test_adapter_tell_records_score(small_space):
    """tell() 후 study.best_value 가 갱신된다."""
    adapter = OptunaSamplerAdapter(small_space, optuna.samplers.RandomSampler(seed=42))
    p1 = adapter.ask()
    adapter.tell(p1, 0.42)
    assert adapter._study.best_value == 0.42

    p2 = adapter.ask()
    adapter.tell(p2, 0.83)
    assert adapter._study.best_value == 0.83


def test_adapter_tell_unknown_params_is_silent(small_space):
    """ask() 와 매칭되지 않는 params 의 tell() 은 조용히 무시 (warm-start path)."""
    adapter = OptunaSamplerAdapter(small_space, optuna.samplers.RandomSampler(seed=42))
    # ask 한 적 없으나 tell — 예외 안 나야 함
    adapter.tell({"lr": 0.05, "batch": 32}, 0.5)


def test_adapter_tpe_workflow(small_space):
    """TPE 도 Random 과 같은 ABC 위에 동작."""
    adapter = OptunaSamplerAdapter(
        small_space, optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
    )
    for _ in range(5):
        p = adapter.ask()
        adapter.tell(p, sum(map(hash, str(p))) % 100 / 100)
    assert len(adapter._study.trials) == 5


# ─── OptunaPrunerAdapter ─────────────────────────────────────────────


def test_pruner_adapter_does_not_prune_first_step():
    """1 step 만 보고는 prune 결정 보통 안 한다 (Optuna 기본)."""
    pruner = OptunaPrunerAdapter(
        optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    )
    # 첫 step 단독으로는 pruner 가 충분한 bracket 정보 없음
    out = pruner.should_prune("t-1", step=0, value=0.5)
    assert out is False


def test_pruner_adapter_separate_trial_ids():
    """다른 trial_id 는 독립 frame — 한 쪽 prune 이 다른 쪽에 영향 없음."""
    pruner = OptunaPrunerAdapter(optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0))
    # 동일 step 다른 value 두 trial 모두 정상 동작
    pruner.should_prune("t-1", step=0, value=0.1)
    pruner.should_prune("t-2", step=0, value=0.9)
    assert "t-1" in pruner._frames
    assert "t-2" in pruner._frames
