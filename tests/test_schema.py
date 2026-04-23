from pathlib import Path

import pytest

from bench.endpoints import EndpointSpec, load_endpoint
from bench.profiles import ProfileSpec, load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_sample_endpoints_load():
    for yml in (ROOT / "configs/endpoints").glob("*.yaml"):
        ep = load_endpoint(yml)
        assert isinstance(ep, EndpointSpec)
        assert ep.base_url.endswith("/v1")


def test_smoke_profile_loads():
    p = load_profile(ROOT / "configs/profiles/smoke.yaml")
    assert p.slug == "smoke"
    assert p.runner == "aiperf"
    assert p.mode == "concurrency"
    assert p.workload.concurrency == 1
    assert p.workload.request_count == 5


def test_concurrency_mode_rejects_user_centric_fields():
    raw = {
        "slug": "bad",
        "name": "bad",
        "stage": 1,
        "runner": "aiperf",
        "mode": "concurrency",
        "workload": {
            "synthetic_input_tokens_mean": 100,
            "output_tokens_mean": 50,
            "concurrency": 1,
            "request_count": 10,
            "num_users": 2,
        },
    }
    with pytest.raises(ValueError):
        ProfileSpec.model_validate(raw)


def test_user_centric_mode_requires_core_fields():
    raw = {
        "slug": "bad",
        "name": "bad",
        "stage": 2,
        "runner": "aiperf",
        "mode": "user_centric",
        "workload": {
            "synthetic_input_tokens_mean": 1000,
            "output_tokens_mean": 200,
            "num_users": 2,
        },
    }
    with pytest.raises(ValueError):
        ProfileSpec.model_validate(raw)


def test_endpoint_api_key_env_missing_raises(monkeypatch):
    ep = EndpointSpec.model_validate(
        {
            "slug": "x",
            "name": "x",
            "url": "http://localhost:1234/v1",
            "model": "m",
            "api_key_env": "DOES_NOT_EXIST_XYZ",
        }
    )
    monkeypatch.delenv("DOES_NOT_EXIST_XYZ", raising=False)
    with pytest.raises(RuntimeError):
        ep.resolve_api_key()
