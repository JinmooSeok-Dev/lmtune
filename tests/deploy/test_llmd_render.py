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

    # parallelism translates to vLLM CLI flag names
    assert va["tensor-parallel-size"] == 4
    assert va["pipeline-parallel-size"] == 2
    assert va["data-parallel-size"] == 1
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
    assert ms["decode"] == {"replicas": 3}
    # vllmArgs 가 prefill/decode 와 공존 (chart 가 둘을 별도로 처리)
    assert ms["vllmArgs"]["max-num-seqs"] == 128
    assert ms["vllmArgs"]["tensor-parallel-size"] == 4


def test_render_omits_replicas_when_absent():
    """deployment.replicas 가 없으면 overlay 에도 prefill/decode 키 자체가 없어야 한다."""
    endpoint = {"model": "m", "deployment": {"engine_args": {}, "parallelism": {"tp": 1}}}
    overlay = render_values_overlay(endpoint)
    ms = overlay["ms-phase1"]
    assert "prefill" not in ms
    assert "decode" not in ms
