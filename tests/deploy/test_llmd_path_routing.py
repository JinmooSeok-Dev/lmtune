"""Phase B-track prereq — well-lit-path-aware adapter routing tests.

검증:
  1. well_lit_path 메타 axis 가 engine_args 에 새지 않음
  2. resolve_well_lit_path() 가 3 ready paths (inference-scheduling,
     pd-disaggregation, wide-ep-lws) → b200/helmfile/<path>/ 로 매핑
  3. 미작성 path (tiered-prefix-cache 등) sample 시 UnsupportedWellLitPath 즉시 raise
  4. apply(dry_run=True) 가 sampled path 의 overlay 를 정상 생성
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmtune.deploy.llmd_k8s import (
    DEFAULT_LMTUNE_REPO_ROOT,
    WELL_LIT_PATHS,
    LLMDK8sAdapter,
    UnsupportedWellLitPath,
    render_values_overlay,
    resolve_well_lit_path,
)


@pytest.fixture
def endpoint_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "ep.yaml"
    p.write_text(yaml.safe_dump({
        "apiVersion": "lmtune/v1alpha1",
        "slug": "test-ep",
        "url": "http://127.0.0.1:8080/v1",
        "model": "Qwen/Qwen2.5-1.5B",
        "deployment": {
            "engine": "vllm",
            "parallelism": {"tp": 4, "dp": 2},
            "engine_args": {"max_num_seqs": 64, "enable_prefix_caching": True},
        },
    }))
    return p


def test_resolve_three_ready_paths_to_b200_helmfile():
    """All 3 paths with working helmfile.yaml.gotmpl resolve to b200/helmfile/<path>/."""
    for name in ("inference-scheduling", "pd-disaggregation", "wide-ep-lws"):
        root, file = resolve_well_lit_path(name)
        assert root == DEFAULT_LMTUNE_REPO_ROOT
        assert file == f"b200/helmfile/{name}/helmfile.yaml.gotmpl"
        # 실제 파일이 존재하는지도 확인
        assert (root / file).exists(), f"{name} helmfile missing on disk"


def test_resolve_unsupported_paths_raise():
    """README placeholder 만 있는 4 path 는 즉시 raise."""
    for name in ("tiered-prefix-cache", "precise-prefix-cache",
                 "predicted-latency-scheduling", "workload-autoscaling"):
        with pytest.raises(UnsupportedWellLitPath, match=name):
            resolve_well_lit_path(name)


def test_well_lit_paths_table_only_lists_ready_ones():
    assert set(WELL_LIT_PATHS.keys()) == {
        "inference-scheduling", "pd-disaggregation", "wide-ep-lws",
    }


def test_well_lit_path_does_not_leak_into_engine_args(endpoint_yaml, tmp_path):
    """Meta-axis is stripped before merging into endpoint.deployment.engine_args."""
    adapter = LLMDK8sAdapter(dry_run=True, helmfile_root=str(tmp_path))
    result = adapter.apply(endpoint_yaml, {
        "well_lit_path": "inference-scheduling",
        "max_num_seqs": 128,  # legitimate engine_arg override
    })
    assert result.ok, f"dry-run apply failed: {result.notes} / {result.health.detail}"
    # The overlay file path is in result.notes; parse it back
    notes = result.health.detail
    overlay_path = Path(notes.split("dry-run; overlay at ")[-1].strip())
    assert overlay_path.exists()
    overlay = yaml.safe_load(overlay_path.read_text())
    # First (and only) release block
    payload = next(iter(overlay.values()))
    args = payload["vllmArgs"]
    assert "well-lit-path" not in args
    assert "well_lit_path" not in args
    # Sampled engine_arg flowed through
    assert args["max-num-seqs"] == 128


def test_apply_dry_run_with_path_routes_to_b200(endpoint_yaml, tmp_path):
    """apply() with sampled path uses b200/helmfile/<path>/ even if helmfile_root
    points elsewhere (e.g. peer repo)."""
    # helmfile_root pointed somewhere else (peer repo path). The sampled
    # well_lit_path should override the routing to bench repo + b200/helmfile/.
    fake_peer = tmp_path / "fake_peer_repo"
    fake_peer.mkdir()
    adapter = LLMDK8sAdapter(dry_run=True, helmfile_root=str(fake_peer))
    result = adapter.apply(endpoint_yaml, {
        "well_lit_path": "pd-disaggregation",
    })
    assert result.ok
    # dry-run skips the actual helmfile/kubectl calls but writes the overlay.
    # If path routing failed, the dry_run would still pass since helmfile isn't
    # invoked — so this test mainly guards that the call doesn't raise.


def test_unsupported_path_returns_failure(endpoint_yaml, tmp_path):
    """Sampling an unimplemented path returns ApplyResult(ok=False) with the
    UnsupportedWellLitPath message — does NOT raise."""
    adapter = LLMDK8sAdapter(dry_run=True, helmfile_root=str(tmp_path))
    result = adapter.apply(endpoint_yaml, {
        "well_lit_path": "tiered-prefix-cache",
    })
    assert result.ok is False
    assert "tiered-prefix-cache" in result.health.detail
    assert "autotune-driveable" in result.health.detail
    assert result.notes == "unsupported well_lit_path"


def test_render_overlay_carries_parallelism_per_path(endpoint_yaml):
    """Sanity: render_values_overlay still emits TP/DP regardless of path routing.

    Note: tp/dp 는 vllmArgs 가 아니라 decode.parallelism.tensor / decode.replicas
    로 emit (chart hardcoded path 와 충돌 차단).
    """
    data = yaml.safe_load(endpoint_yaml.read_text())
    overlay = render_values_overlay(data, release_names=["ms-pd-prefill", "ms-pd-decode"])
    assert set(overlay.keys()) == {"ms-pd-prefill", "ms-pd-decode"}
    for payload in overlay.values():
        assert payload["decode"]["parallelism"]["tensor"] == 4
        assert payload["decode"]["replicas"] == 2
        assert "tensor-parallel-size" not in payload["vllmArgs"]
        assert "data-parallel-size" not in payload["vllmArgs"]
