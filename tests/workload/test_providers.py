"""WorkloadProvider ABC + 구현체 테스트.

[workloads] extra 미설치 환경에서도 본 모듈 import 가능 — provide() 호출 시점에
ImportError. 따라서 LiteralProvider 의 instantiation 자체는 lm-workloads 없이 가능.
provide() 호출 시 lm-workloads 의 Pydantic model 이 필요해 import 시도.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from lmtune.workload.providers import (
    LiteralWorkloadProvider,
    WorkloadProvider,
    build_provider,
)
from lmtune.workload.providers.base import WorkloadProvider as ABC

LM_WORKLOADS_AVAILABLE = importlib.util.find_spec("lm_workloads") is not None
needs_workloads = pytest.mark.skipif(
    not LM_WORKLOADS_AVAILABLE,
    reason="requires [workloads] extra (pip install lmtune[workloads])",
)


# ─── ABC 구조 ─────────────────────────────────────────────────────────


def test_workload_provider_is_abstract():
    """ABC 라 직접 인스턴스화 불가."""
    with pytest.raises(TypeError):
        ABC()


def test_literal_provider_is_workloadprovider():
    """LiteralWorkloadProvider 가 ABC 만족."""
    assert issubclass(LiteralWorkloadProvider, WorkloadProvider)


# ─── build_provider 분기 ──────────────────────────────────────────────


def test_build_provider_yaml_path(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("apiVersion: workloads/v1alpha1\n")  # invalid 내용이라도 인스턴스화는 OK
    prov = build_provider(spec_path=str(p), source=None)
    assert isinstance(prov, LiteralWorkloadProvider)


def test_build_provider_source(monkeypatch):
    """source 만 주면 LMWorkloadsProvider 인스턴스화 (lm-workloads import 안 함)."""
    # build_provider 가 lazy import 라 [workloads] 미설치라도 인스턴스화 가능
    prov = build_provider(spec_path=None, source="vllm-log:/tmp/x.ndjson")
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    assert isinstance(prov, LMWorkloadsProvider)


def test_build_provider_no_input_errors():
    with pytest.raises(ValueError, match="provider 입력 필요"):
        build_provider(spec_path=None, source=None)


def test_build_provider_both_inputs_error(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("")
    with pytest.raises(ValueError, match="동시 지정 불가"):
        build_provider(spec_path=str(p), source="vllm-log:/x")


# ─── LiteralWorkloadProvider — 파일 검사 ──────────────────────────────


def test_literal_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LiteralWorkloadProvider(tmp_path / "nope.yaml")


def test_literal_fingerprint_stable(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("apiVersion: workloads/v1alpha1\nkind: WorkloadSpec\n")
    fp1 = LiteralWorkloadProvider(p).fingerprint()
    fp2 = LiteralWorkloadProvider(p).fingerprint()
    assert fp1 == fp2  # 같은 파일 → 같은 fingerprint


def test_literal_fingerprint_changes_on_content(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("a")
    fp1 = LiteralWorkloadProvider(p).fingerprint()
    p.write_text("b")
    fp2 = LiteralWorkloadProvider(p).fingerprint()
    assert fp1 != fp2


# ─── LMWorkloadsProvider — URI 파싱 ───────────────────────────────────


def test_lm_workloads_provider_invalid_uri_raises():
    """':' 없으면 즉시 reject."""
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    with pytest.raises(ValueError, match="<adapter>:<path>"):
        LMWorkloadsProvider("no-colon-here")


def test_lm_workloads_provider_uri_parsed():
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    p = LMWorkloadsProvider("vllm-log:/tmp/x.ndjson")
    assert p.source_uri == "vllm-log:/tmp/x.ndjson"


def test_lm_workloads_provider_fingerprint_stable():
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    p1 = LMWorkloadsProvider("vllm-log:/x.ndjson")
    p2 = LMWorkloadsProvider("vllm-log:/x.ndjson")
    assert p1.fingerprint() == p2.fingerprint()
    p3 = LMWorkloadsProvider("vllm-log:/y.ndjson")
    assert p1.fingerprint() != p3.fingerprint()


# ─── [workloads] 미설치 환경 — 친절한 ImportError ──────────────────────


@pytest.mark.skipif(
    LM_WORKLOADS_AVAILABLE,
    reason="lm-workloads 가 설치돼있으면 ImportError path 검증 불가",
)
def test_workload_spec_import_error_message():
    """contracts/workload_spec.py import 시 친절한 에러."""
    with pytest.raises(ImportError, match=r"lmtune\[workloads\]"):
        from lmtune.contracts.workload_spec import WorkloadSpec  # noqa: F401


@pytest.mark.skipif(
    LM_WORKLOADS_AVAILABLE,
    reason="lm-workloads 가 설치돼있으면 ImportError path 검증 불가",
)
def test_lm_workloads_provider_provide_raises_without_extra(tmp_path: Path):
    """provide() 호출 시 ImportError + 설치 안내."""
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    p = LMWorkloadsProvider(f"vllm-log:{tmp_path / 'x.ndjson'}")
    with pytest.raises(ImportError, match=r"lmtune\[workloads\]"):
        p.provide()


# ─── e2e — [workloads] 설치 환경 ──────────────────────────────────────


WORKLOADS_REPO = Path("/home/jinmoo/ml_ai/workloads")
SAMPLE_NDJSON = WORKLOADS_REPO / "examples" / "vllm_request_log" / "sample.ndjson"


@needs_workloads
@pytest.mark.skipif(not SAMPLE_NDJSON.exists(), reason="lm-workloads examples fixture 필요")
def test_lm_workloads_e2e(tmp_path: Path):
    """LMWorkloadsProvider → WorkloadSpec → yaml round-trip."""
    from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

    p = LMWorkloadsProvider(
        f"vllm-log:{SAMPLE_NDJSON}", store_path=tmp_path / "store.duckdb", out_dir=tmp_path / "out"
    )
    spec = p.provide()
    assert spec.apiVersion == "workloads/v1alpha1"
    assert spec.kind == "WorkloadSpec"

    # yaml 작성 → LiteralWorkloadProvider 로 다시 읽기 (round-trip)
    out = tmp_path / "ws.yaml"
    out.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))

    spec2 = LiteralWorkloadProvider(out).provide()
    assert spec2.meta.id == spec.meta.id
    assert spec2.classification.category == spec.classification.category
