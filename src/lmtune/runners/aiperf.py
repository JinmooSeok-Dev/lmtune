from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec
from lmtune.runners.base import RequestRow, Runner, RunnerError

_METRIC_KEY_MAP = {
    "time_to_first_token": "ttft",
    "inter_token_latency": "itl",
    "request_latency": "e2e",
    "time_per_output_token": "tpot",
    "request_throughput": "throughput_req",
    "output_token_throughput": "throughput_tok",
    "output_token_throughput_per_user": "throughput_tok_per_user",
}

_PERCENTILE_KEYS = {"p50", "p75", "p90", "p95", "p99", "p999", "avg", "min", "max"}


class AIPerfRunner(Runner):
    kind = "aiperf"

    def __init__(self, binary: str = "aiperf"):
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
        cmd: list[str] = [
            self.binary, "profile",
            "--model", endpoint.model,
            "--url", endpoint.base_url.removesuffix("/v1"),
            "--endpoint-type", "chat" if endpoint.api_type == "openai" else "completions",
            "--streaming",
            "--tokenizer", endpoint.tokenizer or endpoint.model,
            "--synthetic-input-tokens-mean", str(w.synthetic_input_tokens_mean),
            "--output-tokens-mean", str(w.output_tokens_mean),
            "--random-seed", str(w.random_seed),
            "--output-artifact-dir", str(raw_dir / "aiperf"),
        ]
        if w.synthetic_input_tokens_stddev:
            cmd += ["--synthetic-input-tokens-stddev", str(w.synthetic_input_tokens_stddev)]
        if w.output_tokens_stddev:
            cmd += ["--output-tokens-stddev", str(w.output_tokens_stddev)]
        if w.shared_system_prompt_length:
            cmd += ["--num-prefix-prompts", "1",
                    "--prefix-prompt-length", str(w.shared_system_prompt_length)]
        if profile.goodput_spec:
            cmd += ["--goodput", profile.goodput_spec]

        if profile.mode == "concurrency":
            cmd += [
                "--concurrency", str(w.concurrency),
                "--request-count", str(w.request_count),
            ]
        elif profile.mode == "user_centric":
            cmd += [
                "--conversation-num", str(w.conversation_num),
                "--conversation-turn-mean", str(w.conversation_turn_mean),
                "--conversation-turn-stddev", str(w.conversation_turn_stddev),
                "--num-users", str(w.num_users),
                "--user-centric-rate", str(w.user_centric_rate),
            ]
            if w.user_turn_delay_ms is not None:
                cmd += ["--session-turn-delay-mean", str(w.user_turn_delay_ms)]
        else:
            raise RunnerError(f"aiperf does not support mode={profile.mode}")
        return cmd

    def parse(self, raw_dir: Path) -> tuple[dict[str, dict[str, float]], list[RequestRow]]:
        artifact_dir = raw_dir / "aiperf"
        json_files = list(artifact_dir.rglob("*genai_perf*.json")) + list(
            artifact_dir.rglob("profile_export.json")
        )
        if not json_files:
            return {}, []
        data = json.loads(json_files[0].read_text(encoding="utf-8"))

        metrics: dict[str, dict[str, float]] = {}
        records = data.get("records") or data.get("statistics") or data
        for aiperf_key, bench_key in _METRIC_KEY_MAP.items():
            node = _walk(records, aiperf_key)
            if node is None:
                continue
            extracted = {k: float(v) for k, v in node.items() if k in _PERCENTILE_KEYS and _is_number(v)}
            if extracted:
                metrics[bench_key] = extracted

        requests: list[RequestRow] = []
        trace_file = next(iter(artifact_dir.rglob("*.jsonl")), None)
        if trace_file and trace_file.exists():
            for idx, line in enumerate(trace_file.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                requests.append(
                    RequestRow(
                        req_id=row.get("request_id") or f"req-{idx}",
                        turn_idx=row.get("turn_index"),
                        conversation_id=row.get("conversation_id") or row.get("session_id"),
                        input_tokens=row.get("num_input_tokens"),
                        output_tokens=row.get("num_output_tokens"),
                        ttft_ms=_ms(row.get("time_to_first_token")),
                        itl_mean_ms=_ms(row.get("inter_token_latency_mean")),
                        e2e_ms=_ms(row.get("request_latency")),
                    )
                )
        return metrics, requests


def _walk(node, target: str):
    if not isinstance(node, dict):
        return None
    if target in node and isinstance(node[target], dict):
        return node[target]
    for v in node.values():
        if isinstance(v, dict):
            hit = _walk(v, target)
            if hit is not None:
                return hit
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ms(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f * 1000.0 if f < 1 else f
