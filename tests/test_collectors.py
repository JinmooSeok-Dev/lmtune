from __future__ import annotations

from bench.collectors.prometheus import _parse_labels, scrape_metrics_endpoint  # type: ignore[attr-defined]
from bench.collectors.request_log import parse_request_log, summarize


def test_parse_labels_handles_escaping():
    labels = _parse_labels('model_name="Qwen/Qwen3-30B-A3B",quantile="0.99"')
    assert labels == {"model_name": "Qwen/Qwen3-30B-A3B", "quantile": "0.99"}


def test_scrape_metrics_endpoint(monkeypatch):
    sample = """
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3
vllm:time_to_first_token_seconds_sum{model_name="m"} 12.5
vllm:time_to_first_token_seconds_count{model_name="m"} 50
vllm:prefix_cache_hit_rate{model_name="m"} 0.82
some_other_metric{foo="bar"} 1.0
""".strip()

    class FakeResp:
        status_code = 200
        text = sample

        def raise_for_status(self):
            return None

    import bench.collectors.prometheus as mod

    monkeypatch.setattr(mod.requests, "get", lambda *a, **kw: FakeResp())
    samples = scrape_metrics_endpoint("http://fake/metrics")
    names = {s.metric for s in samples}
    assert "vllm:num_requests_running" in names
    assert "vllm:time_to_first_token_seconds_sum" in names
    assert "vllm:prefix_cache_hit_rate" in names
    # 화이트리스트 외 metric 은 걸러져야 함
    assert "some_other_metric" not in names


def test_parse_request_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "INFO Received request cmpl-abc: prompt='x', sampling_params=..., prompt_token_ids=[1, 2, 3, 4]\n"
        "INFO Finished request cmpl-abc in 1.234s\n"
        "DEBUG unrelated line\n"
    )
    events = parse_request_log(log)
    assert len(events) == 2
    summary = summarize(events)
    assert summary["unique_requests"] == 1
    assert summary["prompt_tokens_mean"] == 4
    assert summary["elapsed_mean_sec"] == 1.234
