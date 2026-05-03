from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec
from lmtune.runners.base import RequestRow, Runner, RunnerError

_PCT_MAP = {"median": "p50", "p95": "p95", "p99": "p99", "mean": "avg", "std": "std"}


class VllmBenchRunner(Runner):
    """vLLM repo 의 `benchmarks/benchmark_serving.py` / `benchmark_serving_multi_turn.py` 래퍼.

    vLLM 소스 체크아웃 경로는 환경변수 `VLLM_REPO` 또는 runner 생성자 `vllm_repo` 로 지정.
    """

    kind = "vllm_bench"

    def __init__(
        self,
        python: str | None = None,
        vllm_repo: str | Path | None = None,
    ):
        self.python = python or os.environ.get("BENCH_PYTHON") or "python3"
        repo = vllm_repo or os.environ.get("VLLM_REPO")
        self.vllm_repo = Path(repo) if repo else None

    def _script(self, name: str) -> Path:
        if self.vllm_repo is None:
            raise RunnerError("VLLM_REPO env var or vllm_repo constructor arg required")
        path = self.vllm_repo / "benchmarks" / name
        if not path.exists():
            raise RunnerError(f"vLLM benchmark script not found: {path}")
        return path

    def tool_version(self) -> str | None:
        if not self.vllm_repo:
            return None
        try:
            out = subprocess.run(
                ["git", "-C", str(self.vllm_repo), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return (out.stdout or "").strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def build_command(
        self, profile: ProfileSpec, endpoint: EndpointSpec, run_id: str, raw_dir: Path
    ) -> list[str]:
        w = profile.workload
        if profile.mode == "concurrency":
            script = self._script("benchmark_serving.py")
            result_file = raw_dir / "vllm_bench_result.json"
            cmd = [
                self.python, str(script),
                "--backend", "openai-chat" if endpoint.api_type == "openai" else "openai",
                "--base-url", endpoint.base_url.removesuffix("/v1"),
                "--endpoint", "/v1/chat/completions",
                "--model", endpoint.model,
                "--tokenizer", endpoint.tokenizer or endpoint.model,
                "--dataset-name", "random",
                "--random-input-len", str(w.synthetic_input_tokens_mean),
                "--random-output-len", str(w.output_tokens_mean),
                "--num-prompts", str(w.request_count),
                "--max-concurrency", str(w.concurrency),
                "--seed", str(w.random_seed),
                "--save-result",
                "--result-filename", str(result_file),
            ]
            if w.request_rate is not None:
                cmd += ["--request-rate", str(w.request_rate)]
            if w.random_prefix_len:
                cmd += ["--random-prefix-len", str(w.random_prefix_len)]
            if profile.goodput_spec:
                cmd += ["--goodput", profile.goodput_spec]
            return cmd

        if profile.mode == "user_centric":
            script = self._script("benchmark_serving_multi_turn.py")
            config_path = raw_dir / "multi_turn_config.json"
            config_path.write_text(json.dumps(_build_multi_turn_config(profile), indent=2))
            return [
                self.python, str(script),
                "--model", endpoint.model,
                "--tokenizer", endpoint.tokenizer or endpoint.model,
                "--url", endpoint.base_url.removesuffix("/v1"),
                "--input-file", str(config_path),
                "--num-clients", str(w.num_users or 1),
                "--max-active-conversations", str(w.num_users or 1),
                "--output-dir", str(raw_dir / "vllm_mt"),
            ]

        raise RunnerError(f"vllm_bench does not support mode={profile.mode}")

    def parse(self, raw_dir: Path) -> tuple[dict[str, dict[str, float]], list[RequestRow]]:
        result_file = raw_dir / "vllm_bench_result.json"
        if not result_file.exists():
            # multi-turn output layout: vllm_mt/... json
            candidates = list((raw_dir / "vllm_mt").rglob("*.json")) if (raw_dir / "vllm_mt").exists() else []
            if not candidates:
                return {}, []
            result_file = candidates[0]
        data = json.loads(result_file.read_text(encoding="utf-8"))

        metrics: dict[str, dict[str, float]] = {}
        for prefix, bench_key in [
            ("ttft", "ttft"),
            ("itl", "itl"),
            ("tpot", "tpot"),
            ("e2el", "e2e"),
        ]:
            bucket: dict[str, float] = {}
            for stat, canon in _PCT_MAP.items():
                key = f"{stat}_{prefix}_ms"
                if key in data:
                    bucket[canon] = float(data[key])
            if bucket:
                metrics[bench_key] = bucket

        if "request_throughput" in data:
            metrics.setdefault("throughput_req", {})["avg"] = float(data["request_throughput"])
        if "output_throughput" in data:
            metrics.setdefault("throughput_tok", {})["avg"] = float(data["output_throughput"])
        if "request_goodput" in data:
            metrics.setdefault("goodput", {})["avg"] = float(data["request_goodput"])

        requests: list[RequestRow] = []
        ttfts = data.get("ttfts") or []
        itls = data.get("itls") or []
        e2es = data.get("e2els") or data.get("e2e_latencies") or []
        input_lens = data.get("input_lens") or []
        output_lens = data.get("output_lens") or []
        cached_lens = data.get("cached_tokens") or data.get("num_cached_tokens") or []
        n = max(len(ttfts), len(e2es), len(input_lens))
        for i in range(n):
            ttft = ttfts[i] * 1000 if i < len(ttfts) else None
            itl_list = itls[i] if i < len(itls) else None
            itl_mean = (sum(itl_list) / len(itl_list) * 1000) if itl_list else None
            e2e = e2es[i] * 1000 if i < len(e2es) else None
            requests.append(
                RequestRow(
                    req_id=f"req-{i}",
                    input_tokens=input_lens[i] if i < len(input_lens) else None,
                    output_tokens=output_lens[i] if i < len(output_lens) else None,
                    cached_tokens=cached_lens[i] if i < len(cached_lens) else None,
                    ttft_ms=ttft,
                    itl_mean_ms=itl_mean,
                    e2e_ms=e2e,
                )
            )
        return metrics, requests


def _build_multi_turn_config(profile: ProfileSpec) -> dict:
    w = profile.workload
    return {
        "num_conversations": w.conversation_num,
        "common_prefix": w.shared_system_prompt_length or 0,
        "distributions": {
            "num_turns": {
                "type": "normal" if w.conversation_turn_stddev else "constant",
                "params": (
                    {"mean": w.conversation_turn_mean, "std": w.conversation_turn_stddev}
                    if w.conversation_turn_stddev
                    else {"value": w.conversation_turn_mean}
                ),
            },
            "input_tokens": {
                "type": "normal" if w.synthetic_input_tokens_stddev else "constant",
                "params": (
                    {"mean": w.synthetic_input_tokens_mean, "std": w.synthetic_input_tokens_stddev}
                    if w.synthetic_input_tokens_stddev
                    else {"value": w.synthetic_input_tokens_mean}
                ),
            },
            "output_tokens": {
                "type": "normal" if w.output_tokens_stddev else "constant",
                "params": (
                    {"mean": w.output_tokens_mean, "std": w.output_tokens_stddev}
                    if w.output_tokens_stddev
                    else {"value": w.output_tokens_mean}
                ),
            },
        },
    }
