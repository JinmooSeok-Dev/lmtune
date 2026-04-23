"""K8s Job manifest rendering — no cluster required."""

from __future__ import annotations

import json

from bench.orchestrate.backend import TrialPayload
from bench.orchestrate.backend_k8s import render_job_manifest


def test_render_manifest_fills_env_and_labels():
    payload = TrialPayload(
        trial_id="tr-abc",
        study_id="st-xyz",
        seq=7,
        params={"max_num_seqs": 128, "enable_prefix_caching": True},
        endpoint_path="/cfg/endpoint.yaml",
        profile_paths=["/cfg/short.yaml", "/cfg/medium.yaml", "/cfg/long.yaml"],
        repeats=5,
        ttft_slo_ms=400.0,
    )
    m = render_job_manifest(payload, image="bench-trial-runner:1.0", namespace="bench", gpu_count=1)

    # Name derives from trial_id (lowercased) under the given namespace
    assert m["metadata"]["name"] == "bench-trial-tr-abc"
    assert m["metadata"]["namespace"] == "bench"

    # Labels encode study / trial for kubectl selectors later
    labels = m["metadata"]["labels"]
    assert labels["bench/study-id"] == "st-xyz"
    assert labels["bench/trial-id"] == "tr-abc"

    container = m["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "bench-trial-runner:1.0"

    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["TRIAL_ID"] == "tr-abc"
    assert env["STUDY_ID"] == "st-xyz"
    assert env["TRIAL_SEQ"] == "7"
    assert json.loads(env["PARAMS_JSON"]) == {"max_num_seqs": 128, "enable_prefix_caching": True}
    assert env["PROFILE_PATHS"] == "/cfg/short.yaml:/cfg/medium.yaml:/cfg/long.yaml"
    assert env["REPEATS"] == "5"
    assert env["TTFT_SLO_MS"] == "400.0"

    # GPU request/limit propagated
    req = container["resources"]["requests"]
    lim = container["resources"]["limits"]
    assert req["nvidia.com/gpu"] == "1"
    assert lim["nvidia.com/gpu"] == "1"


def test_render_manifest_zero_gpu_omits_limit():
    payload = TrialPayload(
        trial_id="tr-cpu",
        study_id="st-1",
        seq=1,
        params={},
        endpoint_path="/e",
        profile_paths=["/p"],
    )
    m = render_job_manifest(payload, image="img:latest", gpu_count=0)
    container = m["spec"]["template"]["spec"]["containers"][0]
    # GPU keys should not be injected when gpu_count=0
    req = container["resources"].get("requests", {})
    assert "nvidia.com/gpu" not in req
