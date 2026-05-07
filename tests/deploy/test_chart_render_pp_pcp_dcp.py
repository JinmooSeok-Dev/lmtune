"""R24 + R25 회귀 테스트 — chart 가 PP/DCP 활성 시 nvidia.com/gpu = TP × PP 로
rendering 하는지 helm template 으로 직접 검증. DCP 는 TP group 안이라 추가 GPU
안 씀. PCP 는 R25 로 비활성 (vllm 0.17.1 backend 미지원).

본 테스트는 helmfile + helm 바이너리가 없으면 skip. CI/dev 환경엔 보통 둘 다 있음.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HELMFILE_GOTMPL = REPO_ROOT / "b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl"


@pytest.mark.skipif(
    not (shutil.which("helmfile") and shutil.which("helm")),
    reason="helmfile + helm 필요 (네트워크 chart pull 도)",
)
@pytest.mark.skipif(
    not HELMFILE_GOTMPL.exists(),
    reason="b200 helmfile 미설치",
)
def test_chart_gpu_count_with_pp_dcp() -> None:
    """TP=4 × PP=2 → GPU=8, DCP=2 는 곱셈 안 함 (TP group 내부)."""
    env = {**os.environ, "B200_MODEL_VALUES": "values-gpt-oss-120b.yaml.gotmpl"}
    cmd = [
        "helmfile",
        "--environment", "default",
        "--selector", "kind=inference-stack",
        "-f", str(HELMFILE_GOTMPL),
        "--state-values-set",
        "ms-infsch.decode.parallelism.tensor=4,"
        "ms-infsch.vllmArgs.pipeline-parallel-size=2,"
        "ms-infsch.vllmArgs.decode-context-parallel-size=2",
        "template",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        pytest.skip(f"helmfile template fail (network or chart pull): {result.stderr[:200]}")

    docs = list(yaml.safe_load_all(result.stdout))
    deployment = next(
        (d for d in docs if d and d.get("kind") == "Deployment"
         and "modelservice" in (d.get("metadata") or {}).get("name", "")),
        None,
    )
    assert deployment, "modelservice Deployment not found in rendered chart"

    container = deployment["spec"]["template"]["spec"]["containers"][0]
    gpu_request = container["resources"]["requests"]["nvidia.com/gpu"]
    gpu_limit = container["resources"]["limits"]["nvidia.com/gpu"]
    assert int(gpu_request) == 8, f"requests gpu = {gpu_request}, expected 8 (TP × PP)"
    assert int(gpu_limit) == 8, f"limits gpu = {gpu_limit}"

    args = container["args"]
    for flag, expected in [
        ("--tensor-parallel-size", "4"),
        ("--pipeline-parallel-size", "2"),
        ("--decode-context-parallel-size", "2"),
    ]:
        assert flag in args, f"{flag} missing from container args"
        idx = args.index(flag)
        assert str(args[idx + 1]) == expected, f"{flag} = {args[idx+1]}, expected {expected}"

    # PCP 는 명시 inject 안 했으니 default 1 → vllm CLI 에 emit 안 됨
    # (chart 의 mergeOverwrite 는 vllmArgs key 만 emit)
    assert "--prefill-context-parallel-size" not in args, (
        "PCP 는 R25 로 비활성 (vllm 0.17.1 backend 미지원) — search-space 에서 sample 안 됨"
    )


@pytest.mark.skipif(
    not (shutil.which("helmfile") and shutil.which("helm")),
    reason="helmfile + helm 필요",
)
@pytest.mark.skipif(
    not HELMFILE_GOTMPL.exists(),
    reason="b200 helmfile 미설치",
)
def test_chart_backward_compat_pp_pcp_default_one() -> None:
    """PP/PCP 미주입 (b3_v2 호환) → default 1 → GPU=TP."""
    env = {**os.environ, "B200_MODEL_VALUES": "values-gpt-oss-120b.yaml.gotmpl"}
    cmd = [
        "helmfile",
        "--environment", "default",
        "--selector", "kind=inference-stack",
        "-f", str(HELMFILE_GOTMPL),
        "--state-values-set", "ms-infsch.decode.parallelism.tensor=8",
        "template",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        pytest.skip(f"helmfile template fail: {result.stderr[:200]}")

    docs = list(yaml.safe_load_all(result.stdout))
    deployment = next(
        (d for d in docs if d and d.get("kind") == "Deployment"
         and "modelservice" in (d.get("metadata") or {}).get("name", "")),
        None,
    )
    assert deployment
    gpu = int(deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"])
    assert gpu == 8, f"backward-compat: TP=8, default PP/PCP=1 → GPU=8, got {gpu}"
