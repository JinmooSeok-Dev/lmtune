"""Phase W tier-2 vLLM engine_args wiring tests.

검증:
  1. tier2 의 9 axes 가 search-space YAML 에서 valid SearchSpace 로 파싱
  2. 각 새 axis (`kv_cache_dtype`, `block_size`, `max_model_len`,
     `async_scheduling`, `enforce_eager`) 가 merge_params_into_endpoint() 후
     deployment.engine_args 에 정확히 들어감
  3. vllm_restart.sh --dry-run 이 새 axis 를 올바른 kebab-case CLI flag 로 변환
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lmtune.deploy.base import _ENGINE_ARG_KEYS, merge_params_into_endpoint
from lmtune.search.space import load_space

REPO_ROOT = Path(__file__).resolve().parents[2]
TIER2_SPACE = REPO_ROOT / "b200" / "search-spaces" / "w_local_tier2.yaml"
RESTART_SH = REPO_ROOT / "scripts" / "vllm_restart.sh"


def test_tier2_space_yaml_parses():
    space = load_space(TIER2_SPACE)
    names = {ax.name for ax in space.axes}
    assert names == {
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "max_num_seqs",
        "gpu_memory_utilization",
        "kv_cache_dtype",
        "block_size",
        "max_model_len",
        "async_scheduling",
        "enforce_eager",
    }
    # All cost_tier = 4
    for ax in space.axes:
        assert ax.cost_tier == 4, f"{ax.name} expected cost_tier=4"


def test_new_axes_recognized_as_engine_args():
    """All 5 new axes are explicit members of _ENGINE_ARG_KEYS (not falling
    through the unknown-key fallback)."""
    for k in ("kv_cache_dtype", "block_size", "max_model_len", "async_scheduling", "enforce_eager"):
        assert k in _ENGINE_ARG_KEYS, f"{k} should be explicit engine_arg"


@pytest.fixture
def endpoint_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "ep.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "lmtune/v1alpha1",
                "slug": "test",
                "url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "deployment": {
                    "engine": "vllm",
                    "parallelism": {"tp": 1, "dp": 1},
                    "engine_args": {"max_num_seqs": 64},
                },
            }
        )
    )
    return p


def test_merge_writes_all_new_axes_into_engine_args(endpoint_yaml):
    sampled = {
        "kv_cache_dtype": "fp8",
        "block_size": 32,
        "max_model_len": 8192,
        "async_scheduling": True,
        "enforce_eager": False,
        "max_num_seqs": 128,  # existing axis still works
    }
    merged = merge_params_into_endpoint(endpoint_yaml, sampled)
    args = merged["deployment"]["engine_args"]
    assert args["kv_cache_dtype"] == "fp8"
    assert args["block_size"] == 32
    assert args["max_model_len"] == 8192
    assert args["async_scheduling"] is True
    assert args["enforce_eager"] is False
    assert args["max_num_seqs"] == 128


def test_merge_does_not_create_parallelism_keys_for_engine_args(endpoint_yaml):
    """Sanity: engine_args keys must NOT leak into deployment.parallelism."""
    merge_params_into_endpoint(endpoint_yaml, {"kv_cache_dtype": "auto"})
    data = yaml.safe_load(endpoint_yaml.read_text())
    parallelism = data["deployment"]["parallelism"]
    assert "kv_cache_dtype" not in parallelism


@pytest.mark.skipif(
    not RESTART_SH.exists() or shutil.which("bash") is None,
    reason="bash or vllm_restart.sh missing",
)
def test_vllm_restart_dry_run_emits_kebab_flags(endpoint_yaml, tmp_path):
    """`vllm_restart.sh --dry-run` 가 endpoint YAML 의 새 engine_args 를
    `--kv-cache-dtype fp8`, `--block-size 32`, `--async-scheduling` 등으로 변환."""
    merge_params_into_endpoint(
        endpoint_yaml,
        {
            "kv_cache_dtype": "fp8",
            "block_size": 32,
            "max_model_len": 8192,
            "async_scheduling": True,
            "enforce_eager": False,  # boolean false → flag 미출력 (스크립트 규약)
        },
    )
    # VENV_PY 우선 (로컬 .venv), 없으면 sys.executable 로 fallback (CI 환경).
    # VENV_VLLM 은 dry-run 에서 검증 안 됨.
    py_path = str(REPO_ROOT / ".venv" / "bin" / "python")
    if not os.access(py_path, os.X_OK):
        py_path = sys.executable
    proc = subprocess.run(
        ["bash", str(RESTART_SH), str(endpoint_yaml), "--dry-run"],
        capture_output=True,
        text=True,
        env={
            "VENV_PY": py_path,
            "VENV_VLLM": str(REPO_ROOT / ".venv" / "bin" / "vllm"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
    )
    assert proc.returncode == 0, f"dry-run failed: {proc.stderr}"
    out = proc.stdout

    assert "--kv-cache-dtype" in out
    assert "fp8" in out
    assert "--block-size" in out
    assert "--max-model-len" in out
    assert "8192" in out
    # bool true → bare flag
    assert "--async-scheduling" in out
    # bool false → no flag at all
    assert "--enforce-eager" not in out
