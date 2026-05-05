from __future__ import annotations

from pathlib import Path

import pytest

from lmtune.endpoints import load_endpoint
from lmtune.profiles import ProfileSpec, load_profile
from lmtune.runners import AIPerfRunner, GuideLLMRunner, VllmBenchRunner, get_runner

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
    with pytest.raises((RuntimeError, FileNotFoundError, EnvironmentError)):
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


def test_aiperf_parse_0_7_0_schema(tmp_path):
    """0.7.0 새 파일명 (profile_export_aiperf.json) + 새 percentile 셋 호환 검증.
    NHN B200 mac 환경 실측 dump 기반 minimal fixture (sub-set of real keys).
    """
    import json

    aiperf_dir = tmp_path / "aiperf"
    aiperf_dir.mkdir()
    fixture = {
        "schema_version": "1.0",
        "aiperf_version": "0.7.0",
        "request_throughput": {"unit": "requests/sec", "avg": 5.69},
        "request_latency": {
            "unit": "ms",
            "avg": 588.94,
            "p50": 572.59,
            "p99": 625.61,
            "p1": 554.28,
            "p5": 556.34,
            "min": 553.76,
            "max": 625.64,
            "std": 29.77,
        },
        "time_to_first_token": {
            "unit": "ms",
            "avg": 61.91,
            "p50": 44.21,
            "p99": 96.86,
            "p25": 42.85,
            "min": 27.63,
            "max": 96.86,
            "std": 28.95,
        },
        "inter_token_latency": {
            "unit": "ms",
            "avg": 4.15,
            "p50": 4.15,
            "p99": 4.20,
            "min": 4.12,
            "max": 4.20,
            "std": 0.02,
        },
        "output_token_throughput": {"unit": "tokens/sec", "avg": 728.55},
        "output_token_throughput_per_user": {
            "unit": "tokens/sec/user",
            "avg": 240.98,
            "p50": 240.82,
            "p99": 242.99,
        },
    }
    (aiperf_dir / "profile_export_aiperf.json").write_text(json.dumps(fixture))

    metrics, _ = AIPerfRunner().parse(tmp_path)
    assert "ttft" in metrics and metrics["ttft"]["p99"] == 96.86
    assert "itl" in metrics and metrics["itl"]["p50"] == 4.15
    assert "e2e" in metrics and metrics["e2e"]["p99"] == 625.61
    assert "throughput_tok" in metrics and metrics["throughput_tok"]["avg"] == 728.55
    assert "throughput_req" in metrics and metrics["throughput_req"]["avg"] == 5.69
    # 새 percentile (p1, p25, std) 도 추출되어야 함
    assert metrics["e2e"]["p1"] == 554.28
    assert metrics["ttft"]["p25"] == 42.85
    assert metrics["ttft"]["std"] == 28.95
