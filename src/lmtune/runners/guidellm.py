from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec
from lmtune.runners.base import RequestRow, Runner

_METRIC_MAP = {
    "time_to_first_token_ms": "ttft",
    "time_per_output_token_ms": "tpot",
    "inter_token_latency_ms": "itl",
    "request_latency": "e2e",
    "output_tokens_per_second": "throughput_tok",
    "requests_per_second": "throughput_req",
}


class GuideLLMRunner(Runner):
    kind = "guidellm"

    def __init__(self, binary: str = "guidellm"):
        self.binary = binary

    def tool_version(self) -> str | None:
        if shutil.which(self.binary) is None:
            return None
        try:
            out = subprocess.run(
                [self.binary, "--version"], capture_output=True, text=True, timeout=5
            )
            return (out.stdout or out.stderr).strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def build_command(
        self, profile: ProfileSpec, endpoint: EndpointSpec, run_id: str, raw_dir: Path
    ) -> list[str]:
        w = profile.workload
        output = raw_dir / "guidellm_report.json"
        rate_type = w.rate_type or ("constant" if w.request_rate else "sweep")
        data_spec = {
            "prompt_tokens": w.synthetic_input_tokens_mean,
            "output_tokens": w.output_tokens_mean,
        }
        cmd = [
            self.binary,
            "benchmark",
            "--target",
            endpoint.base_url.removesuffix("/v1"),
            "--model",
            endpoint.model,
            "--processor",
            endpoint.tokenizer or endpoint.model,
            "--data",
            json.dumps(data_spec),
            "--rate-type",
            rate_type,
            "--output-path",
            str(output),
            "--random-seed",
            str(w.random_seed),
        ]
        if rate_type == "concurrent" and w.concurrency is not None:
            cmd += ["--rate", str(w.concurrency)]
        elif w.request_rate is not None and rate_type in {"constant", "poisson", "async"}:
            cmd += ["--rate", str(w.request_rate)]
        if w.request_count is not None:
            cmd += ["--max-requests", str(w.request_count)]
        return cmd

    def parse(self, raw_dir: Path) -> tuple[dict[str, dict[str, float]], list[RequestRow]]:
        report = raw_dir / "guidellm_report.json"
        if not report.exists():
            return {}, []
        data = json.loads(report.read_text(encoding="utf-8"))
        benchmarks = data.get("benchmarks") or []
        if not benchmarks:
            return {}, []
        last = benchmarks[-1]
        mx = last.get("metrics") or {}
        metrics: dict[str, dict[str, float]] = {}
        for guidellm_key, bench_key in _METRIC_MAP.items():
            node = mx.get(guidellm_key)
            if not isinstance(node, dict):
                continue
            stats = node.get("successful") if "successful" in node else node
            if not isinstance(stats, dict):
                continue
            bucket: dict[str, float] = {}
            if "mean" in stats:
                bucket["avg"] = float(stats["mean"])
            if "median" in stats:
                bucket["p50"] = float(stats["median"])
            pct = stats.get("percentiles") or {}
            for k in ("p50", "p95", "p99"):
                if k in pct:
                    bucket[k] = float(pct[k])
            if bucket:
                metrics[bench_key] = bucket

        rq_all = last.get("requests") or {}
        rq_list = rq_all.get("successful") if isinstance(rq_all, dict) else rq_all
        if not isinstance(rq_list, list):
            rq_list = []
        requests: list[RequestRow] = []
        for i, r in enumerate(rq_list):
            if not isinstance(r, dict):
                continue
            requests.append(
                RequestRow(
                    req_id=r.get("request_id") or r.get("id") or f"req-{i}",
                    input_tokens=r.get("prompt_tokens") or r.get("input_tokens"),
                    output_tokens=r.get("output_tokens"),
                    ttft_ms=_as_ms(r.get("time_to_first_token_ms"), already_ms=True),
                    itl_mean_ms=_as_ms(r.get("inter_token_latency_ms"), already_ms=True),
                    e2e_ms=_as_ms(r.get("request_latency"), already_ms=False),
                )
            )
        return metrics, requests


def _as_ms(v, already_ms: bool):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if already_ms else f * 1000.0
