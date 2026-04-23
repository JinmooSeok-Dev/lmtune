from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.endpoints import load_endpoint
from bench.profiles import ProfileSpec, load_profile
from bench.runners import AIPerfRunner, GuideLLMRunner, VllmBenchRunner, get_runner


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def endpoint():
    return load_endpoint(ROOT / "configs/endpoints/local_vllm.yaml")


@pytest.fixture
def smoke_profile():
    return load_profile(ROOT / "configs/profiles/smoke.yaml")


@pytest.fixture
def multiturn_profile():
    raw = {
        "slug": "multiturn",
        "name": "multiturn",
        "stage": 2,
        "runner": "aiperf",
        "mode": "user_centric",
        "workload": {
            "synthetic_input_tokens_mean": 2000,
            "output_tokens_mean": 500,
            "conversation_num": 10,
            "conversation_turn_mean": 5,
            "conversation_turn_stddev": 1,
            "num_users": 2,
            "user_centric_rate": 0.5,
        },
        "goodput_spec": "time_to_first_token:3000 request_latency:30000",
    }
    return ProfileSpec.model_validate(raw)


def test_get_runner_dispatches():
    assert isinstance(get_runner("aiperf"), AIPerfRunner)
    assert isinstance(get_runner("vllm_bench"), VllmBenchRunner)
    assert isinstance(get_runner("guidellm"), GuideLLMRunner)


def test_aiperf_concurrency_command(endpoint, smoke_profile, tmp_path):
    cmd = AIPerfRunner().build_command(smoke_profile, endpoint, "run-1", tmp_path)
    assert cmd[0:2] == ["aiperf", "profile"]
    assert "--concurrency" in cmd and "1" in cmd
    assert "--request-count" in cmd and "5" in cmd
    assert "--streaming" in cmd
    assert "--url" in cmd
    assert "--endpoint-type" in cmd
    assert "--model" in cmd
    assert "chat" in cmd
    # concurrency 모드는 user-centric / conversation 인자 없어야 함
    assert "--num-users" not in cmd
    assert "--conversation-num" not in cmd


def test_aiperf_user_centric_command(endpoint, multiturn_profile, tmp_path):
    cmd = AIPerfRunner().build_command(multiturn_profile, endpoint, "run-2", tmp_path)
    assert "--conversation-num" in cmd
    assert "--num-users" in cmd and "2" in cmd
    assert "--user-centric-rate" in cmd and "0.5" in cmd
    assert "--goodput" in cmd
    # goodput spec 값이 한 인자로 포함
    assert "time_to_first_token:3000 request_latency:30000" in cmd
    # concurrency 모드 인자 혼입 없어야 함
    assert "--concurrency" not in cmd
    assert "--request-count" not in cmd


def test_vllm_bench_requires_repo(endpoint, smoke_profile, tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_REPO", raising=False)
    with pytest.raises(Exception):
        VllmBenchRunner().build_command(smoke_profile, endpoint, "r", tmp_path)


def test_vllm_bench_concurrency_command(endpoint, smoke_profile, tmp_path, monkeypatch):
    fake_repo = tmp_path / "vllm"
    (fake_repo / "benchmarks").mkdir(parents=True)
    (fake_repo / "benchmarks/benchmark_serving.py").write_text("")
    monkeypatch.setenv("VLLM_REPO", str(fake_repo))
    cmd = VllmBenchRunner().build_command(smoke_profile, endpoint, "r", tmp_path)
    assert any(c.endswith("benchmark_serving.py") for c in cmd)
    assert "--random-input-len" in cmd and "200" in cmd
    assert "--num-prompts" in cmd and "5" in cmd
    assert "--max-concurrency" in cmd and "1" in cmd


def test_vllm_bench_multiturn_config_written(endpoint, multiturn_profile, tmp_path, monkeypatch):
    fake_repo = tmp_path / "vllm"
    (fake_repo / "benchmarks").mkdir(parents=True)
    (fake_repo / "benchmarks/benchmark_serving_multi_turn.py").write_text("")
    monkeypatch.setenv("VLLM_REPO", str(fake_repo))
    raw_dir = tmp_path / "rundir"
    raw_dir.mkdir()
    cmd = VllmBenchRunner().build_command(multiturn_profile, endpoint, "r", raw_dir)
    assert any(c.endswith("benchmark_serving_multi_turn.py") for c in cmd)
    config_path = raw_dir / "multi_turn_config.json"
    assert config_path.exists()
    import json as _json
    cfg = _json.loads(config_path.read_text())
    assert cfg["num_conversations"] == 10
    assert cfg["distributions"]["num_turns"]["params"]["mean"] == 5


def test_guidellm_command(endpoint, smoke_profile, tmp_path):
    cmd = GuideLLMRunner().build_command(smoke_profile, endpoint, "r", tmp_path)
    assert cmd[0:2] == ["guidellm", "benchmark"]
    assert "--rate-type" in cmd
    assert "--max-requests" in cmd and "5" in cmd
    assert "--output-path" in cmd
