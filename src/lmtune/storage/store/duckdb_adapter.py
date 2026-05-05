"""DuckDBArtifactStore — RecordSpec ↔ DuckDB schema.sql 변환 layer.

기존 schema.sql 그대로 재활용. DuckDBStore 의 풍부한 메서드는 유지하되
ArtifactStore ABC 만 노출 (record-level put/query). Runner / driver / output
은 이 ABC 만 의존 → backend 교체 시 client 코드 변경 0.

설계:
- 신규 connection 또는 기존 DuckDBStore 의 conn 위에 동작
- put() = INSERT OR REPLACE per kind
- query() = SELECT WHERE ORDER BY LIMIT OFFSET
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import duckdb

from lmtune.contracts.query_spec import FilterCond, QuerySpec
from lmtune.contracts.record_spec import RecordSpec, kind_to_class
from lmtune.storage.duckdb_store import _SCHEMA_PATH
from lmtune.storage.store.base import ArtifactStore

# kind → 실제 table 이름. RecordSpec 의 discriminator 와 schema.sql 의
# 테이블 이름이 일대일이지만 일부 (request → requests) 는 plural 차이.
_KIND_TO_TABLE: dict[str, str] = {
    "run": "runs",
    "metric": "metrics",
    "request": "requests",
    "session": "sessions",
    "trajectory_event": "trajectory_events",
    "prom_sample": "prom_samples",
    "detection": "detections",
    "study": "studies",
    "trial": "trials",
    "trial_metric": "trial_metrics",
}

# JSON 으로 직렬화 저장하는 컬럼 (schema.sql 의 JSON 타입).
_JSON_COLUMNS: dict[str, set[str]] = {
    "run": {"tool_versions"},
    "trajectory_event": {"metadata"},
    "prom_sample": {"labels"},
    "study": {"profile_slugs"},
    "trial": {"params"},
}

# Append-only 테이블 (schema.sql 에 PRIMARY KEY 없음 → INSERT OR REPLACE 불가).
# 이들은 단순 INSERT 로 처리하며 같은 record 가 여러 번 put 되면 중복 행 허용.
# put() 사용자는 dedup 책임을 진다.
_APPEND_ONLY_KINDS: set[str] = {"prom_sample", "detection"}


def _serialize_row(record: RecordSpec) -> dict[str, Any]:
    """RecordSpec → dict (JSON 컬럼은 str 직렬화, kind/api_version 제외)."""
    data = record.model_dump()
    kind = data.pop("kind")
    data.pop("api_version", None)
    json_cols = _JSON_COLUMNS.get(kind, set())
    for col in json_cols:
        v = data.get(col)
        if v is not None and not isinstance(v, str):
            data[col] = json.dumps(v)
    return data


def _deserialize_row(kind: str, row: dict[str, Any]) -> RecordSpec:
    """dict (DuckDB row) → RecordSpec."""
    cls = kind_to_class(kind)
    data = dict(row)
    data["kind"] = kind
    json_cols = _JSON_COLUMNS.get(kind, set())
    for col in json_cols:
        v = data.get(col)
        if isinstance(v, str):
            with contextlib.suppress(json.JSONDecodeError):
                data[col] = json.loads(v)
    return cls.model_validate(data)


def _filter_to_sql(cond: FilterCond) -> tuple[str, list[Any]]:
    """FilterCond → (SQL fragment, params list)."""
    col = cond.column
    op = cond.op
    if op == "is_null":
        return f"{col} IS NULL", []
    if op == "is_not_null":
        return f"{col} IS NOT NULL", []
    if op == "in":
        vals = list(cond.value or [])
        if not vals:
            return "1=0", []
        placeholders = ",".join("?" for _ in vals)
        return f"{col} IN ({placeholders})", vals
    if op == "not_in":
        vals = list(cond.value or [])
        if not vals:
            return "1=1", []
        placeholders = ",".join("?" for _ in vals)
        return f"{col} NOT IN ({placeholders})", vals
    if op == "like":
        return f"{col} LIKE ?", [cond.value]
    sql_op = {"==": "=", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[op]
    return f"{col} {sql_op} ?", [cond.value]


class DuckDBArtifactStore(ArtifactStore):
    """DuckDB-backed ArtifactStore.

    기존 DuckDBStore 와 동일 schema.sql 사용 — 같은 DB 파일을 양쪽에서 열어도
    동일 데이터 보임. record-level put/query 만 ABC 통해 노출.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA_PATH.read_text(encoding="utf-8").split(";"):
            s = stmt.strip()
            if s:
                self.conn.execute(s)
        # Phase S1 compat: older DBs have runs without trial_id column.
        existing_cols = {r[1] for r in self.conn.execute("PRAGMA table_info('runs')").fetchall()}
        if "trial_id" not in existing_cols:
            self.conn.execute("ALTER TABLE runs ADD COLUMN trial_id TEXT")

    def put(self, records: list[RecordSpec]) -> int:
        # kind 별로 그룹핑 → 테이블별 batched INSERT OR REPLACE.
        by_kind: dict[str, list[RecordSpec]] = {}
        for rec in records:
            by_kind.setdefault(rec.kind, []).append(rec)  # type: ignore[attr-defined]

        total = 0
        for kind, recs in by_kind.items():
            table = _KIND_TO_TABLE[kind]
            verb = "INSERT" if kind in _APPEND_ONLY_KINDS else "INSERT OR REPLACE"
            for rec in recs:
                data = _serialize_row(rec)
                cols = list(data.keys())
                placeholders = ",".join("?" for _ in cols)
                col_list = ",".join(cols)
                self.conn.execute(
                    f"{verb} INTO {table} ({col_list}) VALUES ({placeholders})",
                    [data[c] for c in cols],
                )
                total += 1
        return total

    def query(self, spec: QuerySpec) -> list[RecordSpec]:
        if spec.is_raw():
            assert spec.raw_sql is not None
            cur = self.conn.execute(spec.raw_sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
            # raw_sql 결과는 record kind 가 명시 안 되면 dict 그대로 반환 불가 →
            # spec.record_kind 가 있으면 그 kind 로 변환 시도, 없으면 빈 리스트.
            if not spec.record_kind:
                return []
            return [_deserialize_row(spec.record_kind, r) for r in rows]

        table = _KIND_TO_TABLE[spec.record_kind]
        select_cols = ",".join(spec.select) if spec.select else "*"
        sql = f"SELECT {select_cols} FROM {table}"
        params: list[Any] = []
        if spec.filters:
            wheres = []
            for cond in spec.filters:
                frag, p = _filter_to_sql(cond)
                wheres.append(frag)
                params.extend(p)
            sql += " WHERE " + " AND ".join(wheres)
        if spec.sort:
            order = ",".join(f"{sk.column} {sk.direction.upper()}" for sk in spec.sort)
            sql += f" ORDER BY {order}"
        if spec.limit is not None:
            sql += f" LIMIT {int(spec.limit)}"
        if spec.offset:
            sql += f" OFFSET {int(spec.offset)}"

        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
        out: list[RecordSpec] = []
        for r in rows:
            try:
                out.append(_deserialize_row(spec.record_kind, r))
            except Exception:
                # 부분 select 시 deserialize 실패 가능 — 무시 (spec.select 시).
                if spec.select:
                    continue
                raise
        return out

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]
