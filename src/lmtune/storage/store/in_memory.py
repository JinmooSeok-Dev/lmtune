"""InMemoryArtifactStore — 테스트/dry-run 용 dict-backed store.

backend 종속성 없는 reference 구현. ArtifactStore semantics 의 mirror —
backend 별 구현이 본 클래스의 동작을 만족해야 한다.
"""

from __future__ import annotations

from typing import Any

from lmtune.contracts.query_spec import FilterCond, QuerySpec
from lmtune.contracts.record_spec import RecordSpec, kind_to_class
from lmtune.storage.store.base import ArtifactStore


def _eval_filter(record_dict: dict[str, Any], cond: FilterCond) -> bool:
    """단일 FilterCond 를 dict 위에서 evaluate."""
    val = record_dict.get(cond.column)
    op = cond.op
    if op == "is_null":
        return val is None
    if op == "is_not_null":
        return val is not None
    if op == "in":
        return val in (cond.value or [])
    if op == "not_in":
        return val not in (cond.value or [])
    if op == "like":
        if val is None or cond.value is None:
            return False
        # SQL LIKE 의 % → python str contains. 단순 구현.
        pattern = str(cond.value).replace("%", "")
        return pattern in str(val)
    if val is None or cond.value is None:
        return False
    if op == "==":
        return val == cond.value
    if op == "!=":
        return val != cond.value
    if op == "<":
        return val < cond.value
    if op == "<=":
        return val <= cond.value
    if op == ">":
        return val > cond.value
    if op == ">=":
        return val >= cond.value
    return False


class InMemoryArtifactStore(ArtifactStore):
    """Dict 기반 ArtifactStore. record_kind → {primary_key: record}."""

    def __init__(self) -> None:
        self._data: dict[str, dict[tuple[Any, ...], RecordSpec]] = {}

    def put(self, records: list[RecordSpec]) -> int:
        for rec in records:
            kind = rec.kind  # type: ignore[attr-defined]
            self._data.setdefault(kind, {})[rec.primary_key()] = rec
        return len(records)

    def query(self, spec: QuerySpec) -> list[RecordSpec]:
        if spec.is_raw():
            raise NotImplementedError("InMemoryArtifactStore 는 raw_sql 미지원")

        bucket = self._data.get(spec.record_kind, {})
        rows: list[RecordSpec] = list(bucket.values())

        # filters
        if spec.filters:
            filtered: list[RecordSpec] = []
            for r in rows:
                d = r.model_dump()
                if all(_eval_filter(d, c) for c in spec.filters):
                    filtered.append(r)
            rows = filtered

        # sort
        for sk in reversed(spec.sort):  # stable multi-key
            rows.sort(
                key=lambda r, col=sk.column: (
                    r.model_dump().get(col) is None,
                    r.model_dump().get(col) or 0,
                ),
                reverse=(sk.direction == "desc"),
            )

        # offset + limit
        if spec.offset:
            rows = rows[spec.offset :]
        if spec.limit is not None:
            rows = rows[: spec.limit]

        return rows

    def close(self) -> None:
        self._data.clear()

    # ── 진단 helper (테스트 친화) ──────────────────────────────────

    def kinds(self) -> list[str]:
        return sorted(self._data.keys())

    def get_class(self, kind: str) -> type:
        return kind_to_class(kind)
