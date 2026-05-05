"""LLMDK8sAdapter overlay rendering tests (Phase W enabler)."""

from __future__ import annotations

from pathlib import Path

import yaml

from lmtune.deploy.llmd_k8s import LLMDK8sAdapter, render_values_overlay


def _endpoint_dict() -> dict:
    return {
        "url": "http://127.0.0.1:8011/v1",
        "model": "Qwen/Qwen2.5-1.5B",
        "deployment": {
            "engine": "vllm",
            "parallelism": {"tp": 1, "dp": 1, "ep": False},
            "engine_args": {
                "enable_prefix_caching": True,
                "max_num_seqs": 64,
                "gpu_memory_utilization": 0.85,
            },
        },
    }


def test_overlay_single_release_legacy():
    overlay = render_values_overlay(_endpoint_dict(), release_name="ms-phase1")
    assert "ms-phase1" in overlay
    payload = overlay["ms-phase1"]
    args = payload["vllmArgs"]
    # snake_case → kebab-case
    assert args["enable-prefix-caching"] is True
    assert args["max-num-seqs"] == 64
    assert args["gpu-memory-utilization"] == 0.85
    # tp/dp 는 vllmArgs 가 아니라 chart 의 decode.parallelism / decode.replicas 로
    # emit (b3 wiring + chart hardcoded path 충돌 차단). 자세한 동기는 R8 후속 메모.
    assert "tensor-parallel-size" not in args
    assert "data-parallel-size" not in args
    assert payload["decode"]["parallelism"]["tensor"] == 1
    assert payload["decode"]["replicas"] == 1
    # ep=False → no enable-expert-parallel
    assert "enable-expert-parallel" not in args


def test_overlay_multi_release_pd():
    """P/D 모드: 같은 vllmArgs 가 prefill + decode release 둘 다에 emit."""
    overlay = render_values_overlay(
        _endpoint_dict(),
        release_names=["ms-pd-prefill", "ms-pd-decode"],
    )
    assert set(overlay.keys()) == {"ms-pd-prefill", "ms-pd-decode"}
    # 둘 다 같은 payload
    assert overlay["ms-pd-prefill"] == overlay["ms-pd-decode"]
    assert overlay["ms-pd-decode"]["vllmArgs"]["max-num-seqs"] == 64


def test_overlay_default_release():
    overlay = render_values_overlay(_endpoint_dict())  # no release args
    assert list(overlay.keys()) == ["ms-phase1"]


def test_overlay_ep_true():
    ep = _endpoint_dict()
    ep["deployment"]["parallelism"]["ep"] = True
    overlay = render_values_overlay(ep, release_name="ms")
    assert overlay["ms"]["vllmArgs"]["enable-expert-parallel"] is True


def test_adapter_from_endpoint_with_overrides():
    """endpoint YAML 의 deployment.helmfile_overrides 를 읽어 adapter 구성."""
    ep = _endpoint_dict()
    ep["deployment"]["helmfile_overrides"] = {
        "helmfile_root": "/tmp/peer",
        "helmfile_file": "phase2/helmfile.yaml.gotmpl",
        "selector": "name=ms-pd",
        "namespace": "llm-d-pd-qwen25",
        "release_names": ["ms-pd-prefill", "ms-pd-decode"],
        "deployment_names": ["ms-pd-prefill", "ms-pd-decode"],
        "rollout_timeout_s": 300,
    }
    adapter = LLMDK8sAdapter.from_endpoint(ep, dry_run=True)
    assert adapter._release_names == ["ms-pd-prefill", "ms-pd-decode"]
    assert adapter._deployment_names == ["ms-pd-prefill", "ms-pd-decode"]
    assert adapter._helmfile_file == "phase2/helmfile.yaml.gotmpl"
    assert adapter._target.namespace == "llm-d-pd-qwen25"
    assert adapter._dry_run is True


def test_overlay_tp_dp_emit_to_decode_chart_path():
    """tp/dp axis sample → chart 가 읽는 위치 (decode.parallelism.tensor /
    decode.replicas) 에 들어가야 chart 의 hardcoded TP=8 default 를 override.
    vllmArgs 에는 안 들어가야 (vllm CLI duplicate flag 차단)."""
    ep = _endpoint_dict()
    ep["deployment"]["parallelism"] = {"tp": 4, "dp": 2}
    overlay = render_values_overlay(ep, release_name="ms-infsch")
    payload = overlay["ms-infsch"]
    # decode.parallelism.tensor + decode.replicas 로 emit
    assert payload["decode"]["parallelism"]["tensor"] == 4
    assert payload["decode"]["replicas"] == 2
    # vllmArgs 에는 절대 안 들어감 (chart 가 자동 inject 하는 영역과 중복 회피)
    assert "tensor-parallel-size" not in payload["vllmArgs"]
    assert "data-parallel-size" not in payload["vllmArgs"]


def test_overlay_pp_stays_in_vllm_args():
    """pp 는 vllm 이 인식하는 CLI flag (--pipeline-parallel-size) 라 vllmArgs
    경로 유지. 단 chart 가 pp 를 자동 inject 하지 않는 한정 상황."""
    ep = _endpoint_dict()
    ep["deployment"]["parallelism"] = {"pp": 2, "tp": 8}
    overlay = render_values_overlay(ep, release_name="ms")
    args = overlay["ms"]["vllmArgs"]
    assert args["pipeline-parallel-size"] == 2
    # tp 는 별도 경로
    assert "tensor-parallel-size" not in args
    assert overlay["ms"]["decode"]["parallelism"]["tensor"] == 8


def test_adapter_dry_run_writes_overlay_without_running_helmfile(tmp_path: Path):
    """dry_run=True 일 때 helmfile/kubectl 호출 없이 overlay 만 작성."""
    ep_path = tmp_path / "endpoint.yaml"
    ep_path.write_text(yaml.safe_dump(_endpoint_dict()))

    adapter = LLMDK8sAdapter(
        helmfile_root=tmp_path,  # 어차피 dry_run 이라 호출 안 함
        release_names=["ms-pd-prefill", "ms-pd-decode"],
        dry_run=True,
    )
    result = adapter.apply(ep_path, params={"engine_args": {"max_num_seqs": 128}})
    assert result.ok is True
    assert "dry-run" in result.notes
    assert result.health.ready is True
