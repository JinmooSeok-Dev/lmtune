from __future__ import annotations

from pathlib import Path

import pytest

from bench.detectors import detect_slo_violations
from bench.endpoints import DeploymentSpec, EndpointSpec, load_endpoint
from bench.profiles import (
    DatasetWorkload,
    ProfileSpec,
    SLOCheck,
    SLOSpec,
    SyntheticWorkload,
    TraceWorkload,
    load_profile,
)
from bench.runners import AIPerfRunner


ROOT = Path(__file__).resolve().parents[1]


# ---------- Workload Union ----------


def test_synthetic_source_defaults():
    p = load_profile(ROOT / "configs/profiles/smoke.yaml")
    assert isinstance(p.workload, SyntheticWorkload)
    assert p.workload.source == "synthetic"
    assert p.apiVersion.startswith("bench/")


def test_dataset_workload_parses():
    raw = {
        "slug": "dsx", "name": "dsx", "stage": 2,
        "runner": "aiperf", "mode": "concurrency",
        "workload": {
            "source": "dataset",
            "dataset_id": "gonglinyuan/safim",
            "dataset_subset": "block",
            "output_tokens_mean": 50,
            "concurrency": 1, "request_count": 5,
        },
    }
    p = ProfileSpec.model_validate(raw)
    assert isinstance(p.workload, DatasetWorkload)
    assert p.workload.dataset_id.endswith("safim")


def test_trace_workload_parses():
    raw = {
        "slug": "trx", "name": "trx", "stage": 1,
        "runner": "guidellm", "mode": "concurrency",
        "workload": {
            "source": "trace",
            "trace_path": "/data/burstgpt.csv",
            "concurrency": 2, "request_count": 100,
        },
    }
    p = ProfileSpec.model_validate(raw)
    assert isinstance(p.workload, TraceWorkload)
    assert p.workload.replay_speed == 1.0


def test_unknown_source_rejected():
    raw = {
        "slug": "bad", "name": "bad", "stage": 1,
        "runner": "aiperf", "mode": "concurrency",
        "workload": {"source": "unknown-kind", "concurrency": 1, "request_count": 1},
    }
    with pytest.raises(Exception):
        ProfileSpec.model_validate(raw)


# ---------- Endpoint Deployment ----------


def test_endpoint_deployment_parses():
    ep = load_endpoint(ROOT / "configs/endpoints/llmd_k8s.yaml")
    assert isinstance(ep.deployment, DeploymentSpec)
    assert ep.deployment.engine == "vllm"
    assert ep.deployment.parallelism.tp == 4
    assert ep.deployment.parallelism.ep is True
    assert ep.deployment.engine_args["enable_prefix_caching"] is True
    assert "tp4" in ep.deployment.to_tag()
    assert "ep" in ep.deployment.to_tag()


def test_endpoint_without_deployment_ok():
    ep = EndpointSpec.model_validate(
        {"slug": "x", "name": "x", "url": "http://localhost:1/v1", "model": "m"}
    )
    assert ep.deployment is None


# ---------- Runner overrides pass-through ----------


def test_runner_overrides_injected(tmp_path):
    profile = ProfileSpec.model_validate(
        {
            "slug": "ov", "name": "ov", "stage": 1,
            "runner": "aiperf", "mode": "concurrency",
            "workload": {
                "synthetic_input_tokens_mean": 100, "output_tokens_mean": 50,
                "concurrency": 1, "request_count": 3,
            },
            "runner_overrides": {
                "aiperf": {
                    "--warmup-requests": 5,
                    "--extra-http-header": "X-Foo: bar",
                    "--debug": True,
                    "--disabled": False,  # False 는 생략되어야 함
                },
                "guidellm": {"--ignored-for-aiperf": "v"},
            },
        }
    )
    endpoint = load_endpoint(ROOT / "configs/endpoints/local_vllm.yaml")
    runner = AIPerfRunner()
    cmd = runner._apply_overrides(
        runner.build_command(profile, endpoint, "r", tmp_path), profile
    )
    assert "--warmup-requests" in cmd and "5" in cmd
    assert "--extra-http-header" in cmd
    assert "X-Foo: bar" in cmd
    assert "--debug" in cmd
    assert "--disabled" not in cmd
    # 다른 runner 의 override 는 섞이지 않아야 함
    assert "--ignored-for-aiperf" not in cmd


# ---------- SLO checks + legacy co-existence ----------


def test_slo_resolved_checks_includes_legacy_and_new():
    slo = SLOSpec(
        ttft_p99_ms=500,
        min_goodput_ratio=0.9,
        checks=[SLOCheck(metric="itl", p="p95", op="<=", value=100)],
    )
    chks = slo.resolved_checks()
    metrics = {(c.metric, c.p) for c in chks}
    assert ("ttft", "p99") in metrics
    assert ("goodput", "avg") in metrics
    assert ("itl", "p95") in metrics


def test_slo_detector_respects_op_semantics():
    slo = SLOSpec(checks=[
        SLOCheck(metric="goodput", p="avg", op=">=", value=0.9, severity="critical"),
        SLOCheck(metric="ttft", p="p99", op="<=", value=500),
    ])
    metrics = {"goodput": {"avg": 0.75}, "ttft": {"p99": 450}}
    dets = detect_slo_violations(metrics, slo)
    # goodput 만 위반이어야 함
    flags = [(d.metric, d.severity) for d in dets if d.severity != "info"]
    assert ("goodput.avg", "critical") in flags
    assert not any(m == "ttft.p99" and s != "info" for m, s in flags)
