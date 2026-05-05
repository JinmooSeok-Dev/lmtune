from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests


@dataclass
class PromSample:
    ts: datetime
    metric: str
    labels: dict[str, str]
    value: float


_VLLM_METRIC_WHITELIST = {
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:prefix_cache_hit_rate",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
}


_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)(\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)\s*(?P<ts>\d+)?$"
)


def _parse_labels(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    labels: dict[str, str] = {}
    for m in re.finditer(r'([a-zA-Z_][\w]*)="((?:[^"\\]|\\.)*)"', s):
        labels[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
    return labels


def scrape_metrics_endpoint(url: str, timeout: float = 3.0) -> list[PromSample]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    now = datetime.now(tz=UTC)
    samples: list[PromSample] = []
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if _VLLM_METRIC_WHITELIST and name not in _VLLM_METRIC_WHITELIST:
            # 자릿수 범위 히스토그램은 bucket/sum/count 만 중요 → bucket 라인도 포함
            if not any(name.endswith(s) for s in ("_bucket", "_sum", "_count")):
                continue
            core = name.rsplit("_", 1)[0]
            if core not in {n.rsplit("_", 1)[0] for n in _VLLM_METRIC_WHITELIST}:
                continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        samples.append(
            PromSample(ts=now, metric=name, labels=_parse_labels(m.group("labels")), value=value)
        )
    return samples


class PrometheusCollector:
    """백그라운드 스레드로 주기 scrape 하여 JSONL 로 저장."""

    def __init__(self, url: str, out_path: str | Path, interval_sec: float = 5.0):
        self.url = url
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_written = 0

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self):
        with self.out_path.open("a", buffering=1) as fh:
            while not self._stop.is_set():
                try:
                    samples = scrape_metrics_endpoint(self.url, timeout=self.interval)
                    for s in samples:
                        fh.write(
                            json.dumps(
                                {
                                    "ts": s.ts.isoformat(),
                                    "metric": s.metric,
                                    "labels": s.labels,
                                    "value": s.value,
                                }
                            )
                            + "\n"
                        )
                    self.samples_written += len(samples)
                except Exception as e:  # noqa: BLE001
                    fh.write(
                        json.dumps({"ts": datetime.now(tz=UTC).isoformat(), "error": str(e)}) + "\n"
                    )
                self._stop.wait(self.interval)
