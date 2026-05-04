"""Unit tests for rollout_watcher.classify_crash.

kubectl 호출은 mocking 하지 않고 classify_crash 의 패턴 매칭만 검증.
Smart waiter 자체는 minikube 통합 테스트에서 검증 (별도).
"""
from __future__ import annotations

from lmtune.deploy.rollout_watcher import classify_crash


def test_classify_infeasible_mxfp4_dtype():
    logs = """
    pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
      Value error, torch.float16 is not supported for quantization method mxfp4.
      Supported dtypes: [torch.bfloat16]
    """
    assert classify_crash(logs) == "infeasible"


def test_classify_infeasible_unrecognized_args():
    logs = "vllm: error: unrecognized arguments: --weight-dtype --activation-dtype"
    assert classify_crash(logs) == "infeasible"


def test_classify_infeasible_duplicate_keys():
    logs = "WARNING [argparse_utils.py:353] Found duplicate keys --tensor-parallel-size"
    assert classify_crash(logs) == "infeasible"


def test_classify_oom_torch():
    logs = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ..."
    assert classify_crash(logs) == "oom"


def test_classify_oom_generic():
    logs = "RuntimeError: unable to allocate 14.5 GiB GPU memory on cuda:0"
    assert classify_crash(logs) == "oom"


def test_classify_transient_nccl_timeout():
    logs = "[rank0] NCCL all-reduce timeout after 600s"
    assert classify_crash(logs) == "transient"


def test_classify_transient_connection_refused():
    logs = "Connection refused (os error 111) at peer 10.42.0.81:8000"
    assert classify_crash(logs) == "transient"


def test_classify_hard_unknown():
    logs = "Some random fatal error not in our pattern set"
    assert classify_crash(logs) == "hard"


def test_classify_hard_empty():
    assert classify_crash("") == "hard"


def test_classify_priority_infeasible_over_transient():
    """ValidationError 우선 — connection refused 메시지가 같이 있어도 infeasible 분류."""
    logs = """
    Connection refused (...)
    pydantic ValidationError: torch.float16 is not supported for quantization method mxfp4
    """
    assert classify_crash(logs) == "infeasible"
