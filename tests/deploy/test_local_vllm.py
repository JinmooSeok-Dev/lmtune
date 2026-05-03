"""LocalVLLMAdapter: mocked restart script + probe."""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import patch

import yaml

from lmtune.deploy.base import HealthReport
from lmtune.deploy.local_vllm import LocalVLLMAdapter


def _endpoint(path: Path, *, model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
    path.write_text(yaml.safe_dump({
        "apiVersion": "lmtune/v1alpha1",
        "slug": "local-vllm",
        "url": "http://localhost:8000/v1",
        "model": model,
        "deployment": {
            "engine": "vllm",
            "parallelism": {"tp": 1, "dp": 1, "pp": 1, "ep": False},
            "engine_args": {"max_num_seqs": 128},
        },
    }, sort_keys=False), encoding="utf-8")


def _fake_restart_script(path: Path) -> Path:
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_apply_merges_params_and_returns_ok_when_healthy(tmp_path: Path):
    ep = tmp_path / "ep.yaml"
    _endpoint(ep)
    script = _fake_restart_script(tmp_path / "restart.sh")

    adapter = LocalVLLMAdapter(restart_script=script)

    with patch(
        "lmtune.deploy.local_vllm.probe_openai_models",
        return_value=HealthReport(ready=True, latency_ms=3.0, detail="1 model"),
    ):
        result = adapter.apply(ep, {"max_num_seqs": 64, "enable_prefix_caching": True})

    assert result.ok is True
    assert result.adapter == "local-vllm"
    data = yaml.safe_load(ep.read_text(encoding="utf-8"))
    assert data["deployment"]["engine_args"]["max_num_seqs"] == 64
    assert data["deployment"]["engine_args"]["enable_prefix_caching"] is True


def test_apply_rejects_tp_gt_1(tmp_path: Path):
    ep = tmp_path / "ep.yaml"
    _endpoint(ep)
    script = _fake_restart_script(tmp_path / "restart.sh")
    adapter = LocalVLLMAdapter(restart_script=script)
    r = adapter.apply(ep, {"tp": 4})
    assert r.ok is False
    assert "tp" in r.health.detail.lower()


def test_apply_reports_script_failure(tmp_path: Path):
    ep = tmp_path / "ep.yaml"
    _endpoint(ep)
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/usr/bin/env bash\necho 'boom' >&2\nexit 7\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    adapter = LocalVLLMAdapter(restart_script=bad)
    r = adapter.apply(ep, {})
    assert r.ok is False
    assert "rc=7" in r.notes
