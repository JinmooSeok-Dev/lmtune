"""QuerySpec — store 조회 DSL.

목적: WorkloadStore / ArtifactStore (lmtune#SS) 의 query() 인터페이스 정형화.
이 spec 만 만족하면 어떤 backend (DuckDB, Postgres, Parquet, S3) 든 동일하게 호출.

설계 원칙:
- DuckDB SQL 로 자명하게 변환 가능한 단순 schema (별도 SQL parser X)
- Aggregate 는 GROUP BY + 1 함수만 (복잡 쿼리는 raw_sql escape hatch)
- Pydantic frozen=True 로 query 객체 자체 불변 (cache key 로 그대로 사용)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ─── Filter ──────────────────────────────────────────────────────────

CompareOp = Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "like", "is_null", "is_not_null"]


class FilterCond(BaseModel):
    """단일 컬럼 비교 조건."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    op: CompareOp
    value: Any | None = None  # in/not_in 시 list, is_null/is_not_null 시 무시


# ─── Sort ────────────────────────────────────────────────────────────


class SortKey(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    direction: Literal["asc", "desc"] = "asc"


# ─── Aggregate ───────────────────────────────────────────────────────


class AggregateSpec(BaseModel):
    """단순 GROUP BY + 1 aggregation. 복잡 쿼리는 raw_sql 사용."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_by: list[str] = []  # 빈 리스트면 전체 1 row
    function: Literal["count", "sum", "avg", "min", "max", "stddev", "median"] = "count"
    column: str | None = None  # count 외에는 column 필수
    alias: str = "agg_value"


# ─── QuerySpec ───────────────────────────────────────────────────────


class QuerySpec(BaseModel):
    """Store 의 단일 조회 요청.

    record_kind 가 어느 테이블/컬렉션을 가리키는지 디스패치. RecordSpec 의
    discriminator 와 동일 namespace 사용 (run, metric, request, ..., trial_metric).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_version: Literal["lmtune/query/v1alpha1"] = "lmtune/query/v1alpha1"

    record_kind: str  # RECORD_KINDS 중 하나, 또는 raw_sql 시 빈 값
    filters: list[FilterCond] = []
    sort: list[SortKey] = []
    limit: int | None = None
    offset: int = 0
    select: list[str] | None = None  # None = 전체 컬럼
    aggregate: AggregateSpec | None = None
    raw_sql: str | None = None  # escape hatch: 위 필드 무시하고 직접 SQL 실행

    def is_raw(self) -> bool:
        return self.raw_sql is not None
