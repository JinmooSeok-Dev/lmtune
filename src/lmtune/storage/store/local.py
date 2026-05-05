"""LocalArtifactStore — record kind 별 JSONL 파일에 적재.

DuckDBArtifactStore 가 필요 없는 환경 (CI smoke, dry-run, archive sync, S3
sync) 에서 같은 ABC 를 그대로 사용. backend 가 아니라 file-based mirror —
git 으로 commit 가능, grep 가능, 외부 도구 (jq) 호환.

레이아웃:
    <root>/
        run.jsonl
        metric.jsonl
        request.jsonl
        ...

각 라인이 RecordSpec 1 건 (model_dump_json). 같은 primary_key 의 record 가 들어
오면 마지막 값을 채택 (read 시 dedup).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RECORD_KINDS, RecordSpec, kind_to_class
from lmtune.storage.store.base import ArtifactStore
from lmtune.storage.store.in_memory import _eval_filter


class LocalArtifactStore(ArtifactStore):
    """File-based ArtifactStore. 각 kind 가 ``<root>/<kind>.jsonl``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _file_for(self, kind: str) -> Path:
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown record kind: {kind!r}")
        return self.root / f"{kind}.jsonl"

    def put(self, records: list[RecordSpec]) -> int:
        # kind 별 그룹핑 → 각 파일에 append. dedup 은 read 시.
        by_kind: dict[str, list[RecordSpec]] = {}
        for rec in records:
            by_kind.setdefault(rec.kind, []).append(rec)  # type: ignore[attr-defined]
        total = 0
        for kind, recs in by_kind.items():
            path = self._file_for(kind)
            with path.open("a", encoding="utf-8") as f:
                for rec in recs:
                    f.write(rec.model_dump_json())
                    f.write("\n")
                    total += 1
        return total

    def query(self, spec: QuerySpec) -> list[RecordSpec]:
        if spec.is_raw():
            raise NotImplementedError("LocalArtifactStore 는 raw_sql 미지원")

        path = self._file_for(spec.record_kind)
        if not path.exists():
            return []

        cls = kind_to_class(spec.record_kind)
        # primary_key dedup — last write wins (append-only file 의 자연스러운 의미)
        latest: dict[tuple[Any, ...], RecordSpec] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rec = cls.model_validate(data)
                latest[rec.primary_key()] = rec  # type: ignore[attr-defined]

        rows: list[RecordSpec] = list(latest.values())

        if spec.filters:
            rows = [r for r in rows if all(_eval_filter(r.model_dump(), c) for c in spec.filters)]

        for sk in reversed(spec.sort):
            rows.sort(
                key=lambda r, col=sk.column: (
                    r.model_dump().get(col) is None,
                    r.model_dump().get(col) or 0,
                ),
                reverse=(sk.direction == "desc"),
            )

        if spec.offset:
            rows = rows[spec.offset :]
        if spec.limit is not None:
            rows = rows[: spec.limit]
        return rows

    def close(self) -> None:
        # 파일 기반 — close 시 별도 자원 없음
        return None
