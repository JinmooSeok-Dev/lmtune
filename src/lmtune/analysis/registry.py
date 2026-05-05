"""Metric Registry — direction·unit·category 메타데이터 중앙화.

compare, detectors, plots 가 하드코딩된 set 대신 이 registry 를 조회한다.
새 metric 은 `register()` 또는 `@metric_def` 로 등록.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Direction = Literal["lower_better", "higher_better", "neutral"]
Category = Literal[
    "latency", "throughput", "cost", "energy", "quality", "resource", "agent", "other"
]
Source = Literal["request", "session", "prom", "derived"]


@dataclass
class MetricDef:
    name: str
    unit: str
    direction: Direction
    category: Category
    default_aggs: list[str] = field(default_factory=lambda: ["p50", "p95", "p99", "avg"])
    source: Source = "request"
    description: str = ""


_REGISTRY: dict[str, MetricDef] = {}


def register(md: MetricDef) -> MetricDef:
    _REGISTRY[md.name] = md
    return md


def get(name: str) -> MetricDef | None:
    return _REGISTRY.get(name)


def list_all() -> list[MetricDef]:
    return sorted(_REGISTRY.values(), key=lambda m: (m.category, m.name))


def by_category(cat: Category) -> list[MetricDef]:
    return [m for m in _REGISTRY.values() if m.category == cat]


def direction_of(name: str) -> Direction:
    md = get(name)
    return md.direction if md else "neutral"


# ---------- Built-in catalogue ----------

# Latency (낮을수록 좋음)
register(MetricDef("ttft", "ms", "lower_better", "latency", description="Time to First Token"))
register(MetricDef("itl", "ms", "lower_better", "latency", description="Inter-Token Latency"))
register(MetricDef("tpot", "ms", "lower_better", "latency", description="Time Per Output Token"))
register(
    MetricDef("e2e", "ms", "lower_better", "latency", description="End-to-End request latency")
)

# Throughput (높을수록 좋음)
register(MetricDef("throughput_req", "req/s", "higher_better", "throughput"))
register(MetricDef("throughput_tok", "tok/s", "higher_better", "throughput"))
register(MetricDef("throughput_tok_per_user", "tok/s/user", "higher_better", "throughput"))
register(MetricDef("goodput", "ratio", "higher_better", "throughput"))

# Cost / Energy (낮을수록 좋음)
register(MetricDef("cost_usd", "USD", "lower_better", "cost", source="request"))
register(MetricDef("cost_per_task", "USD/task", "lower_better", "cost", source="derived"))
register(MetricDef("energy_wh", "Wh", "lower_better", "energy", source="request"))
register(MetricDef("energy_per_token", "Wh/tok", "lower_better", "energy", source="derived"))
register(MetricDef("tokens_per_usd", "tok/USD", "higher_better", "cost", source="derived"))

# Agent 메타
register(MetricDef("tool_call_count", "count", "neutral", "agent"))
register(MetricDef("tool_call_ratio", "ratio", "neutral", "agent", source="derived"))
register(MetricDef("cached_tokens", "tok", "higher_better", "agent"))
register(MetricDef("thinking_tokens", "tok", "neutral", "agent"))
register(MetricDef("prefix_hit_rate", "ratio", "higher_better", "agent", source="derived"))
register(MetricDef("input_output_ratio", "ratio", "neutral", "agent", source="derived"))
register(
    MetricDef(
        "eutb",
        "ratio",
        "higher_better",
        "quality",
        source="derived",
        description="Effectiveness under Token Budget (SWE-Effi arXiv:2509.09853)",
    )
)
register(
    MetricDef(
        "variance_cv",
        "ratio",
        "lower_better",
        "quality",
        source="derived",
        description="Coefficient of Variation across repeat runs",
    )
)

# Resource (Prometheus)
register(MetricDef("vllm:gpu_cache_usage_perc", "ratio", "neutral", "resource", source="prom"))
register(
    MetricDef("vllm:prefix_cache_hit_rate", "ratio", "higher_better", "resource", source="prom")
)
register(MetricDef("vllm:num_requests_running", "count", "neutral", "resource", source="prom"))
register(MetricDef("vllm:num_requests_waiting", "count", "lower_better", "resource", source="prom"))
