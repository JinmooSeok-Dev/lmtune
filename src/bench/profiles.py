from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator


RunnerKind = Literal["aiperf", "vllm_bench", "guidellm", "raw_openai"]
RunnerMode = Literal["concurrency", "user_centric", "rate_sweep"]

CmpOp = Literal["<=", "<", ">=", ">", "==", "!="]


# ---------- Workload (source-discriminated Union) ----------


ArrivalKind = Literal["constant", "poisson", "diurnal", "burst", "replay"]


class ArrivalSpec(BaseModel):
    kind: ArrivalKind = "constant"
    rate: float | None = None             # req/s (constant/poisson)
    duration_sec: float | None = None
    # diurnal
    peak_rate: float | None = None
    valley_rate: float | None = None
    period_sec: float | None = None
    # burst
    burst_rate: float | None = None
    burst_sec: float | None = None
    idle_sec: float | None = None


class DistSpec(BaseModel):
    kind: Literal["constant", "uniform", "normal", "zipf", "bimodal", "lognormal"] = "constant"
    mean: float | None = None
    stddev: float | None = None
    low: float | None = None
    high: float | None = None
    zipf_s: float | None = None
    zipf_clip: int | None = None
    modes: list[list[float]] | None = None
    mode_weight: float | None = None
    ln_mu: float | None = None
    ln_sigma: float | None = None


class _WorkloadBase(BaseModel):
    source: str
    random_seed: int = 42

    # 부하 발생 축 (runner 가 자기 모드에 맞는 필드만 사용)
    concurrency: int | None = None
    request_count: int | None = None
    request_rate: float | None = None
    rate_type: Literal["constant", "poisson", "sweep", "throughput"] | None = None

    num_users: int | None = None
    user_centric_rate: float | None = None
    user_turn_delay_ms: int | None = None

    conversation_num: int | None = None
    conversation_turn_mean: int | None = None
    conversation_turn_stddev: int = 0

    # 고급 워크로드 패턴 (raw_openai runner 에서 사용, aiperf/guidellm 은 부분 매핑)
    arrival: ArrivalSpec | None = None
    input_dist: DistSpec | None = None
    output_dist: DistSpec | None = None


class SyntheticWorkload(_WorkloadBase):
    """runner 가 직접 랜덤/합성 데이터를 생성."""

    source: Literal["synthetic"] = "synthetic"
    synthetic_input_tokens_mean: int = Field(gt=0)
    synthetic_input_tokens_stddev: int = 0
    output_tokens_mean: int = Field(gt=0)
    output_tokens_stddev: int = 0
    random_prefix_len: int = 0
    shared_system_prompt_length: int = 0


class DatasetWorkload(_WorkloadBase):
    """HuggingFace/로컬 데이터셋 기반. 로더는 추후 구현."""

    source: Literal["dataset"] = "dataset"
    dataset_id: str
    dataset_split: str = "test"
    dataset_subset: str | None = None
    sample_count: int | None = None
    prompt_field: str = "prompt"
    response_field: str | None = None
    output_tokens_mean: int = Field(gt=0)


class TraceWorkload(_WorkloadBase):
    """프로덕션 trace replay (예: BurstGPT, ServeGen)."""

    source: Literal["trace"] = "trace"
    trace_path: str
    trace_format: Literal["auto", "burstgpt", "servegen"] = "auto"
    replay_speed: float = 1.0
    sample_count: int | None = None


Workload = Annotated[
    Union[SyntheticWorkload, DatasetWorkload, TraceWorkload],
    Field(discriminator="source"),
]


# ---------- SLO (flat legacy 필드 + 일반화된 checks) ----------


class SLOCheck(BaseModel):
    metric: str                    # ttft | itl | e2e | throughput_tok | goodput ...
    p: str = "p99"                 # p50 | p95 | p99 | avg ...
    op: CmpOp = "<="
    value: float
    severity: Literal["warning", "critical"] = "warning"
    label: str | None = None


class SLOSpec(BaseModel):
    # Legacy flat 필드 (하위호환 유지)
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    itl_p99_ms: float | None = None
    e2e_p99_ms: float | None = None
    min_goodput_ratio: float | None = None

    # 일반화된 SLO assertion 목록
    checks: list[SLOCheck] = Field(default_factory=list)

    def resolved_checks(self) -> list[SLOCheck]:
        """Legacy 필드를 SLOCheck 로 정규화 + explicit checks 합본."""
        out: list[SLOCheck] = []
        legacy_map = [
            ("ttft", "p50", self.ttft_p50_ms),
            ("ttft", "p99", self.ttft_p99_ms),
            ("itl", "p99", self.itl_p99_ms),
            ("e2e", "p99", self.e2e_p99_ms),
        ]
        for metric, p, v in legacy_map:
            if v is not None:
                out.append(SLOCheck(metric=metric, p=p, op="<=", value=v))
        if self.min_goodput_ratio is not None:
            out.append(SLOCheck(metric="goodput", p="avg", op=">=", value=self.min_goodput_ratio))
        out.extend(self.checks)
        return out


# ---------- Profile ----------


class PlotRequest(BaseModel):
    kind: str
    metric: str | None = None
    title: str | None = None
    opts: dict[str, Any] = Field(default_factory=dict)


class DerivedMetricSpec(BaseModel):
    name: str
    formula: str | None = None
    description: str = ""


class AnalysisSpec(BaseModel):
    group_by: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    percentiles: list[str] = Field(default_factory=lambda: ["p50", "p95", "p99"])
    derived: list[DerivedMetricSpec] = Field(default_factory=list)
    plots: list[PlotRequest] = Field(default_factory=list)
    sinks: list[str] = Field(default_factory=lambda: ["markdown"])
    buckets: dict[str, list[float]] = Field(default_factory=dict)


class ProfileSpec(BaseModel):
    apiVersion: str = "bench/v1alpha1"
    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9\-_]*$")
    name: str
    stage: int = Field(ge=1, le=3)
    description: str = ""
    references: list[str] = Field(default_factory=list)

    runner: RunnerKind
    mode: RunnerMode

    workload: Workload
    slo: SLOSpec = SLOSpec()
    goodput_spec: str | None = None

    # Runner 가 모르는 임의 CLI flag 주입 (escape hatch).
    # 예: {"aiperf": {"--session-turn-delay-mean": "1500"}, "guidellm": {"--warmup-iters": "5"}}
    runner_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    expected_kv_usage_tok: int | None = None
    analysis: AnalysisSpec | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if isinstance(data, dict):
            wl = data.get("workload")
            if isinstance(wl, dict) and "source" not in wl:
                wl["source"] = "synthetic"
            ov = data.get("runner_overrides")
            if isinstance(ov, dict):
                for k, v in list(ov.items()):
                    if v is None:
                        ov[k] = {}
        return data

    @model_validator(mode="after")
    def _validate_mode_consistency(self) -> "ProfileSpec":
        w = self.workload
        if self.mode == "concurrency":
            missing = [k for k in ("concurrency", "request_count") if getattr(w, k) is None]
            if missing:
                raise ValueError(
                    f"concurrency mode requires workload.{missing} to be set (profile={self.slug})"
                )
            if any([w.num_users, w.user_centric_rate, w.conversation_num]):
                raise ValueError(
                    f"concurrency mode must not set user_centric / conversation fields (profile={self.slug})"
                )
        elif self.mode == "user_centric":
            missing = [
                k
                for k in (
                    "num_users",
                    "user_centric_rate",
                    "conversation_num",
                    "conversation_turn_mean",
                )
                if getattr(w, k) is None
            ]
            if missing:
                raise ValueError(
                    f"user_centric mode requires workload.{missing} (profile={self.slug})"
                )
            if w.concurrency is not None:
                raise ValueError(
                    f"user_centric mode must not set concurrency (profile={self.slug})"
                )
        elif self.mode == "rate_sweep":
            if w.rate_type is None:
                raise ValueError(
                    f"rate_sweep mode requires workload.rate_type (profile={self.slug})"
                )
        return self


def load_profile(path: str | Path) -> ProfileSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "workload" in raw and isinstance(raw["workload"], dict):
        raw["workload"].setdefault("source", "synthetic")
    return ProfileSpec.model_validate(raw)


def discover_profiles(directory: str | Path) -> list[ProfileSpec]:
    directory = Path(directory)
    profiles: list[ProfileSpec] = []
    for yml in sorted(directory.rglob("*.yaml")):
        profiles.append(load_profile(yml))
    return profiles
