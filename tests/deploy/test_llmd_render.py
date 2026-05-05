from __future__ import annotations

import yaml

from lmtune.deploy.llmd_k8s import render_values_overlay


def test_render_kebab_case_vllm_args():
    endpoint = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "deployment": {
            "engine_args": {
                "max_num_seqs": 128,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.85,
                "kv_cache_dtype": "fp8",
            },
            "parallelism": {"tp": 4, "pp": 2, "dp": 1, "ep": True},
        },
    }
    overlay = render_values_overlay(endpoint, release_name="ms-phase1")
    assert "ms-phase1" in overlay
    ms = overlay["ms-phase1"]

    # modelArtifactUri picks up HF prefix
    assert ms["modelspec"]["modelArtifactUri"] == "hf://Qwen/Qwen2.5-1.5B-Instruct"

    # engine_args are kebab-cased
    va = ms["vllmArgs"]
    assert va["max-num-seqs"] == 128
    assert va["enable-prefix-caching"] is True
    assert va["gpu-memory-utilization"] == 0.85
    assert va["kv-cache-dtype"] == "fp8"

    # tp/dp 는 vllmArgs 가 아니라 chart 의 decode.parallelism.tensor /
    # decode.replicas 로 emit (chart 가 자동 inject 하는 영역과 중복 차단).
    assert "tensor-parallel-size" not in va
    assert "data-parallel-size" not in va
    assert ms["decode"]["parallelism"]["tensor"] == 4
    assert ms["decode"]["replicas"] == 1
    # pp/ep 는 vllm CLI flag 그대로 (chart 가 자동 emit 안 함)
    assert va["pipeline-parallel-size"] == 2
    assert va["enable-expert-parallel"] is True


def test_render_omits_ep_when_false():
    endpoint = {
        "model": "m",
        "deployment": {
            "engine_args": {},
            "parallelism": {"tp": 1, "pp": 1, "dp": 1, "ep": False},
        },
    }
    va = render_values_overlay(endpoint)["ms-phase1"]["vllmArgs"]
    assert "enable-expert-parallel" not in va


def test_render_is_yaml_safe():
    endpoint = {
        "model": "m",
        "deployment": {"engine_args": {"x": True}, "parallelism": {"tp": 2}},
    }
    overlay = render_values_overlay(endpoint)
    # Round-trip through yaml to catch un-serializable types.
    dumped = yaml.safe_dump(overlay)
    loaded = yaml.safe_load(dumped)
    assert loaded == overlay


def test_render_pd_replicas_emit_per_release():
    """deployment.replicas → overlay 의 prefill.replicas / decode.replicas 분리 emit."""
    endpoint = {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "deployment": {
            "engine_args": {"max_num_seqs": 128},
            "parallelism": {"tp": 4},
            "replicas": {"prefill": 2, "decode": 3},
        },
    }
    overlay = render_values_overlay(endpoint, release_names=["ms-pd"])
    assert "ms-pd" in overlay
    ms = overlay["ms-pd"]
    assert ms["prefill"] == {"replicas": 2}
    # tp axis (=4) → decode.parallelism.tensor; replicas.decode (=3) → decode.replicas.
    assert ms["decode"]["parallelism"]["tensor"] == 4
    assert ms["decode"]["replicas"] == 3
    # vllmArgs 는 max_num_seqs 만, tp 는 vllmArgs 에 안 들어감 (chart 영역과 중복 차단)
    assert ms["vllmArgs"]["max-num-seqs"] == 128
    assert "tensor-parallel-size" not in ms["vllmArgs"]


def test_render_omits_replicas_when_absent():
    """deployment.replicas 가 없고 axis 도 안 sample 됐을 때 overlay 에 prefill/decode
    키 자체가 없어야 한다 (chart default 그대로)."""
    endpoint = {"model": "m", "deployment": {"engine_args": {}, "parallelism": {"pp": 1}}}
    overlay = render_values_overlay(endpoint)
    ms = overlay["ms-phase1"]
    assert "prefill" not in ms
    assert "decode" not in ms
