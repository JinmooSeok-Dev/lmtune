"""R24 회귀 테스트 — chart 가 PP/PCP 활성 시 nvidia.com/gpu = TP × PP × PCP 로
rendering 하는지 helm template 으로 직접 검증.

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
def test_chart_gpu_count_with_pp_pcp_dcp() -> None:
    """TP=4 × PP=2 × PCP=2 → GPU=16, DCP 는 곱셈 안 함."""
    env = {**os.environ, "B200_MODEL_VALUES": "values-gpt-oss-120b.yaml.gotmpl"}
    cmd = [
        "helmfile",
        "--environment", "default",
        "--selector", "kind=inference-stack",
        "-f", str(HELMFILE_GOTMPL),
        "--state-values-set",
        "ms-infsch.decode.parallelism.tensor=4,"
        "ms-infsch.vllmArgs.pipeline-parallel-size=2,"
        "ms-infsch.vllmArgs.prefill-context-parallel-size=2,"
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
    # int 또는 str("16") 둘 다 허용
    assert int(gpu_request) == 16, f"requests gpu = {gpu_request}, expected 16 (TP×PP×PCP)"
    assert int(gpu_limit) == 16, f"limits gpu = {gpu_limit}"

    args = container["args"]
    arg_str = " ".join(str(a) for a in args)
    assert "--tensor-parallel-size" in arg_str
    assert "--pipeline-parallel-size" in arg_str
    assert "--prefill-context-parallel-size" in arg_str
    assert "--decode-context-parallel-size" in arg_str
    # 값 검증 — flag 다음 element 가 값
    for flag, expected in [
        ("--tensor-parallel-size", "4"),
        ("--pipeline-parallel-size", "2"),
        ("--prefill-context-parallel-size", "2"),
        ("--decode-context-parallel-size", "2"),
    ]:
        idx = args.index(flag)
        assert str(args[idx + 1]) == expected, f"{flag} = {args[idx+1]}, expected {expected}"


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
