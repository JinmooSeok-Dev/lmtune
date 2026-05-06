from __future__ import annotations

from pathlib import Path

import yaml

from lmtune.deploy.base import merge_params_into_endpoint, merge_params_to_dict


def _baseline(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_merge_routes_engine_args_vs_parallelism(tmp_path: Path):
    p = tmp_path / "ep.yaml"
    _baseline(p)

    merged = merge_params_into_endpoint(
        p,
        {
            "max_num_seqs": 64,  # engine_args
            "kv_cache_dtype": "fp8",  # engine_args (unknown key falls here)
            "tp": 4,  # parallelism
            "dp": 2,  # parallelism
        },
    )

    d = merged["deployment"]
    assert d["engine_args"]["max_num_seqs"] == 64
    assert d["engine_args"]["kv_cache_dtype"] == "fp8"
    assert d["engine_args"]["enable_prefix_caching"] is True  # preserved
    assert d["parallelism"]["tp"] == 4
    assert d["parallelism"]["dp"] == 2
    assert d["parallelism"]["pp"] == 1  # untouched

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


def test_merge_simulator_only_axes_dropped(tmp_path: Path):
    """R11: simulator-only axis (cross_node_type / intra_node_type / pcp / dcp 등) 는
    engine_args 로 흘러들지 않는다 — vllm 이 unrecognized argument 로 reject 하는 문제 차단."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    merged = merge_params_into_endpoint(
        p,
        {
            # simulator metadata
            "cross_node_type": "roce",
            "intra_node_type": "pcie",
            "node_split_strategy": "dual-node-pp2-tp8",
            "prefill_context_parallel_size": 2,
            "decode_context_parallel_size": 1,
            "ep_strategy": "wide",
            "sequence_parallel": True,
            # 정상 axis (control)
            "max_num_seqs": 64,
            "tp": 4,
        },
    )
    ea = merged["deployment"]["engine_args"]
    para = merged["deployment"]["parallelism"]
    # simulator metadata 는 어디에도 emit 안 됨
    for k in (
        "cross_node_type",
        "intra_node_type",
        "node_split_strategy",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "ep_strategy",
        "sequence_parallel",
    ):
        assert k not in ea, f"{k} leaked to engine_args (vllm reject)"
        assert k not in para, f"{k} leaked to parallelism"
    # 정상 axis 는 그대로
    assert ea["max_num_seqs"] == 64
    assert para["tp"] == 4


def test_merge_simulator_only_warmstart_replay(tmp_path: Path):
    """R11 시나리오: warmstart-db 가 옛 b3_parallelism trial params 를 enqueue한
    상황 — 9 axis 전부 sample 로 들어와도 vllm 미지원 7개는 모두 drop, 2개만 emit."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    warmstart_params = {
        "tp": 8,
        "dp": 2,
        "ep": False,
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.85,
        # 이하 simulator-only — 옛 study 가 sample 했던 것
        "intra_node_type": "pcie",
        "cross_node_type": "roce",
        "node_split_strategy": "dual-node-pp2-tp8",
        "prefill_context_parallel_size": 4,
        "decode_context_parallel_size": 1,
    }
    merged = merge_params_into_endpoint(p, warmstart_params)
    # engine_args 에 simulator key 없어야
    ea_keys = set(merged["deployment"]["engine_args"].keys())
    assert ea_keys.isdisjoint(
        {
            "intra_node_type",
            "cross_node_type",
            "node_split_strategy",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        }
    )
    # 정상 axis 는 emit
    assert merged["deployment"]["parallelism"]["tp"] == 8
    assert merged["deployment"]["parallelism"]["dp"] == 2
    assert merged["deployment"]["engine_args"]["max_num_seqs"] == 128


def test_merge_params_to_dict_does_not_write_file(tmp_path: Path):
    """R12: merge_params_to_dict 는 endpoint YAML 을 절대 수정하지 않는다."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    before = p.read_text(encoding="utf-8")
    merged = merge_params_to_dict(p, {"max_num_seqs": 64, "tp": 4})
    after = p.read_text(encoding="utf-8")

    # File on disk is byte-identical (not even YAML re-formatted)
    assert before == after, "merge_params_to_dict must not touch the file"
    # But the returned dict has the merged params
    assert merged["deployment"]["parallelism"]["tp"] == 4
    assert merged["deployment"]["engine_args"]["max_num_seqs"] == 64


def test_merge_params_to_dict_simulator_keys_dropped(tmp_path: Path):
    """R12 + R11: read-only merge 도 simulator-only key 는 drop."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    merged = merge_params_to_dict(
        p,
        {
            "tp": 4,
            "max_num_seqs": 64,
            "cross_node_type": "roce",
            "intra_node_type": "pcie",
            "node_split_strategy": "dual-node-pp2-tp8",
            "prefill_context_parallel_size": 4,
            "decode_context_parallel_size": 1,
        },
    )
    ea = merged["deployment"]["engine_args"]
    para = merged["deployment"]["parallelism"]
    for k in (
        "cross_node_type",
        "intra_node_type",
        "node_split_strategy",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        assert k not in ea
        assert k not in para
    assert ea["max_num_seqs"] == 64
    assert para["tp"] == 4


def test_merge_params_into_endpoint_still_writes_for_local_vllm(tmp_path: Path):
    """기존 호출자 (LocalVLLMAdapter / vllm_restart.sh) 호환성 — 여전히 file write."""
    p = tmp_path / "ep.yaml"
    _baseline(p)
    before = p.read_text(encoding="utf-8")
    merge_params_into_endpoint(p, {"max_num_seqs": 64})
    after = p.read_text(encoding="utf-8")
    # File changed (in-place mutation on for backward compat)
    assert before != after
    # And new value is written
    assert "max_num_seqs: 64" in after
