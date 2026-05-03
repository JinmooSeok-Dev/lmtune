from __future__ import annotations

from pathlib import Path

import yaml

from lmtune.deploy.base import merge_params_into_endpoint


def _baseline(path: Path) -> None:
    path.write_text(yaml.safe_dump({
        "apiVersion": "lmtune/v1alpha1",
        "slug": "x",
        "url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "deployment": {
            "engine": "vllm",
            "parallelism": {"tp": 1, "dp": 1, "pp": 1, "ep": False},
            "engine_args": {
                "enable_prefix_caching": True,
                "max_num_seqs": 128,
            },
        },
    }, sort_keys=False), encoding="utf-8")


def test_merge_routes_engine_args_vs_parallelism(tmp_path: Path):
    p = tmp_path / "ep.yaml"
    _baseline(p)

    merged = merge_params_into_endpoint(p, {
        "max_num_seqs": 64,                 # engine_args
        "kv_cache_dtype": "fp8",            # engine_args (unknown key falls here)
        "tp": 4,                            # parallelism
        "dp": 2,                            # parallelism
    })

    d = merged["deployment"]
    assert d["engine_args"]["max_num_seqs"] == 64
    assert d["engine_args"]["kv_cache_dtype"] == "fp8"
    assert d["engine_args"]["enable_prefix_caching"] is True  # preserved
    assert d["parallelism"]["tp"] == 4
    assert d["parallelism"]["dp"] == 2
    assert d["parallelism"]["pp"] == 1                         # untouched

    # File written back and round-trips
    written = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert written == merged


def test_merge_unknown_keys_land_in_engine_args(tmp_path: Path):
    p = tmp_path / "ep.yaml"
    _baseline(p)
    merged = merge_params_into_endpoint(p, {"seed": 42, "foo": "bar"})
    ea = merged["deployment"]["engine_args"]
    assert ea["seed"] == 42
    assert ea["foo"] == "bar"


def test_merge_pd_replicas_routed_to_replicas_block(tmp_path: Path):
    """prefill_replicas / decode_replicas 는 deployment.replicas.{prefill,decode} 로."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    merged = merge_params_into_endpoint(
        p, {"prefill_replicas": 2, "decode_replicas": 3, "max_num_seqs": 128}
    )
    replicas = merged["deployment"]["replicas"]
    assert replicas["prefill"] == 2
    assert replicas["decode"] == 3
    # 다른 키는 영향 없음
    assert merged["deployment"]["engine_args"]["max_num_seqs"] == 128
    # P/D 키가 engine_args 로 흘러들지 않음
    assert "prefill_replicas" not in merged["deployment"]["engine_args"]
    assert "decode_replicas" not in merged["deployment"]["engine_args"]
