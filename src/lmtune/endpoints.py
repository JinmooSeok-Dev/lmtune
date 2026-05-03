from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, field_validator


class ParallelismSpec(BaseModel):
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: bool = False
    rsd: bool = False


class DeploymentSpec(BaseModel):
    """서빙 구성 축. 같은 URL·모델에 대해 병렬화/엔진 플래그만 다른 실험을 식별·비교.

    자동화 로직은 여기 필드를 읽어 동작을 바꾸지 않음. runs 테이블에 메타로 저장하여
    '구성별 비교' 축으로만 사용.
    """

    engine: Literal["vllm", "llm-d", "sglang", "trtllm", "anthropic", "other"] = "vllm"
    version: str | None = None
    parallelism: ParallelismSpec = ParallelismSpec()
    engine_args: dict[str, Any] = Field(default_factory=dict)

    def to_tag(self) -> str:
        p = self.parallelism
        bits = [f"tp{p.tp}"]
        if p.dp > 1:
            bits.append(f"dp{p.dp}")
        if p.pp > 1:
            bits.append(f"pp{p.pp}")
        if p.ep:
            bits.append("ep")
        if p.rsd:
            bits.append("rsd")
        return "-".join(bits)


class EndpointSpec(BaseModel):
    apiVersion: str = "bench/v1alpha1"
    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9\-_]*$")
    name: str
    url: HttpUrl
    model: str
    tokenizer: str | None = None
    api_type: Literal["openai", "anthropic"] = "openai"
    api_key_env: str | None = None
    metrics_url: HttpUrl | None = None
    request_log_path: Path | None = None
    deployment: DeploymentSpec | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("tokenizer", mode="before")
    @classmethod
    def default_tokenizer(cls, v: str | None, info):
        return v if v else info.data.get("model")

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None:
            raise RuntimeError(
                f"endpoint '{self.slug}' requires env var {self.api_key_env}, which is not set"
            )
        return value

    @property
    def base_url(self) -> str:
        return str(self.url).rstrip("/")


def load_endpoint(path: str | Path) -> EndpointSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EndpointSpec.model_validate(raw)
