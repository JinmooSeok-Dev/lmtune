from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from bench.profiles import SLOSpec, SLOCheck
from bench.runners.base import RunArtifact


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _make_artifact(run_id: str, status: str, metrics: dict) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        runner_kind="dummy",
        command=[],
        raw_dir=Path("."),
        stdout_path=Path("stdout"),
        stderr_path=Path("stderr"),
        status=status,
        metrics=metrics,
    )


# ----- bench CLI helpers ---------------------------------------------------

def test_build_run_summary_flatten_and_slo_pass():
    from bench.cli import _build_run_summary

    artifact = _make_artifact(
        run_id="r0",
        status="ok",
        metrics={"ttft": {"p50": 30.0, "p99": 250.0}, "throughput_tok": {"avg": 150.0}},
    )
    slo = SLOSpec(ttft_p99_ms=500.0)
    out = _build_run_summary("r0", artifact, slo)

    assert out["run_id"] == "r0"
    assert out["status"] == "ok"
    assert out["slo_pass"] is True
    assert out["metrics"]["ttft.p99"] == 250.0
    assert out["metrics"]["throughput_tok.avg"] == 150.0
    assert len(out["slo_checks"]) == 1
    assert out["slo_checks"][0]["passed"] is True


def test_build_run_summary_slo_fail_when_exceeds():
    from bench.cli import _build_run_summary

    artifact = _make_artifact(run_id="r1", status="ok", metrics={"ttft": {"p99": 800.0}})
    slo = SLOSpec(checks=[SLOCheck(metric="ttft", p="p99", op="<=", value=500.0)])
    out = _build_run_summary("r1", artifact, slo)

    assert out["slo_pass"] is False
    assert out["slo_checks"][0]["passed"] is False


def test_build_run_summary_missing_metric_counts_as_fail():
    from bench.cli import _build_run_summary

    artifact = _make_artifact(run_id="r2", status="ok", metrics={})
    slo = SLOSpec(checks=[SLOCheck(metric="ttft", p="p99", op="<=", value=500.0)])
    out = _build_run_summary("r2", artifact, slo)

    assert out["slo_pass"] is False
    assert out["slo_checks"][0]["observed"] is None


# ----- bench_score.py ------------------------------------------------------

def _load_score_module():
    spec = importlib.util.spec_from_file_location(
        "bench_score", SCRIPTS / "bench_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_score_cv_zero_for_constant():
    mod = _load_score_module()
    assert mod.cv([100.0, 100.0, 100.0]) == 0.0


def test_bench_score_cv_positive_for_spread():
    mod = _load_score_module()
    result = mod.cv([100.0, 120.0, 80.0])
    assert 0.1 < result < 0.3


def test_bench_score_aggregate_composite_formula():
    mod = _load_score_module()
    summaries = [
        {"slo_pass": True, "run_id": "a",
         "metrics": {"throughput_tok.avg": 100.0, "ttft.p99": 250.0, "e2e.p99": 500.0}},
        {"slo_pass": True, "run_id": "b",
         "metrics": {"throughput_tok.avg": 100.0, "ttft.p99": 250.0, "e2e.p99": 500.0}},
    ]
    out = mod.aggregate(summaries, ttft_slo_ms=500.0)
    # penalty = 1 - 250/(2*500) = 0.75; score = 100 * 0.75 = 75
    assert out["slo_pass"] is True
    assert out["throughput_tok_avg"] == 100.0
    assert out["ttft_p99"] == 250.0
    assert out["score"] == pytest.approx(75.0, rel=1e-6)


def test_bench_score_aggregate_zero_on_slo_fail():
    mod = _load_score_module()
    summaries = [
        {"slo_pass": True, "run_id": "a",
         "metrics": {"throughput_tok.avg": 100.0, "ttft.p99": 250.0}},
        {"slo_pass": False, "run_id": "b",
         "metrics": {"throughput_tok.avg": 100.0, "ttft.p99": 999.0}},
    ]
    out = mod.aggregate(summaries, ttft_slo_ms=500.0)
    assert out["slo_pass"] is False
    assert out["score"] == 0.0


def test_bench_score_aggregate_score_clamped_at_high_ttft():
    mod = _load_score_module()
    summaries = [
        {"slo_pass": True, "run_id": "a",
         "metrics": {"throughput_tok.avg": 100.0, "ttft.p99": 1500.0}},
    ]
    out = mod.aggregate(summaries, ttft_slo_ms=500.0)
    # ttft 1500 > 2*500 → penalty clamped to 0 → score = 0
    assert out["score"] == 0.0


def test_bench_score_config_hash_stable(tmp_path):
    mod = _load_score_module()
    p = tmp_path / "p.yaml"; p.write_text("slug: test\n")
    e = tmp_path / "e.yaml"; e.write_text("slug: x\n")
    h1 = mod.config_hash(p, e)
    h2 = mod.config_hash(p, e)
    assert h1 == h2 and len(h1) == 8
    e.write_text("slug: y\n")
    assert mod.config_hash(p, e) != h1


# ----- vllm_restart.sh dry-run --------------------------------------------

def test_vllm_restart_dry_run_converts_engine_args(tmp_path):
    endpoint = tmp_path / "ep.yaml"
    endpoint.write_text(
        "apiVersion: bench/v1alpha1\n"
        "slug: t\n"
        "url: http://localhost:8000/v1\n"
        "model: dummy/model\n"
        "api_type: openai\n"
        "deployment:\n"
        "  engine: vllm\n"
        "  parallelism:\n"
        "    tp: 1\n"
        "  engine_args:\n"
        "    enable_prefix_caching: true\n"
        "    max_num_seqs: 128\n"
        "    gpu_memory_utilization: 0.85\n"
        "    enforce_eager: false\n"
    )
    proc = subprocess.run(
        [str(SCRIPTS / "vllm_restart.sh"), str(endpoint), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    lines = proc.stdout.splitlines()
    # Script prints: <vllm> serve <model> --host <> --port <> <flags...>
    assert "serve" in lines
    assert "dummy/model" in lines
    assert "--enable-prefix-caching" in lines
    # value pair appears as two consecutive lines
    assert "--max-num-seqs" in lines
    assert "128" in lines
    assert "--gpu-memory-utilization" in lines
    assert "0.85" in lines
    # enforce_eager: false should NOT emit a flag
    assert "--enforce-eager" not in lines
    # tp=1 should NOT emit --tensor-parallel-size
    assert "--tensor-parallel-size" not in lines


def test_vllm_restart_parallel_gt1(tmp_path):
    endpoint = tmp_path / "ep.yaml"
    endpoint.write_text(
        "apiVersion: bench/v1alpha1\n"
        "slug: t\n"
        "url: http://x\n"
        "model: m\n"
        "api_type: openai\n"
        "deployment:\n"
        "  engine: vllm\n"
        "  parallelism: {tp: 2, pp: 1, dp: 1}\n"
        "  engine_args: {}\n"
    )
    proc = subprocess.run(
        [str(SCRIPTS / "vllm_restart.sh"), str(endpoint), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    lines = proc.stdout.splitlines()
    assert "--tensor-parallel-size" in lines
    assert "2" in lines
    assert "--pipeline-parallel-size" not in lines


# ----- autotune profile parsing -------------------------------------------

def test_autotune_profiles_parse():
    from bench.profiles import load_profile

    for name in ["short", "medium", "long"]:
        p = load_profile(ROOT / f"configs/profiles/autotune/{name}.yaml")
        assert p.slug == f"autotune-{name}"
        checks = p.slo.resolved_checks()
        metrics = {c.metric for c in checks}
        assert "ttft" in metrics and "e2e" in metrics
