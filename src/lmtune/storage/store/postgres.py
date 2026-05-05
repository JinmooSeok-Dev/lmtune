"""PostgresArtifactStore — PLUG 패턴 실증을 위한 backend stub.

REFACTOR-PLAN PLUG: "새 backend 추가 = 1 PR" 의 정신을 코드로 보여주는 entry.

본 stub 의 목적:
1. ArtifactStore ABC 의 구현체가 1 파일로 추가되면 driver/CLI 에 자동 plug-in 됨을
   증명 (lmtune storage migrate 에 자동 합류, _open_store 디스패치만 1줄).
2. 실제 Postgres 연결은 ``psycopg`` (v3) 가 설치된 환경에서만 동작 — optional
   ``[postgres]`` extra 로 분리. 미설치 시 명확한 ImportError 메시지.
3. 본 stub 은 ``put`` / ``query`` 의 SQL 골격만 두고 미구현 부분은
   ``NotImplementedError`` 던짐 — 향후 PR 이 단계적으로 채울 수 있게.

설계 노트 (vs DuckDB):
- DuckDB 는 file-based, embedded → process 안에서 단일 writer queue 정착.
- Postgres 는 server-based → multi-writer 가능, lock/concurrency 가 backend 측.
- 따라서 Postgres 구현 시 driver 의 writer_queue 우회 (직접 INSERT) 하는 옵션이
  생김 — 본 stub 은 그 옵션의 hook 만 마련.
"""

from __future__ import annotations

from typing import Any

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RecordSpec
from lmtune.storage.store.base import ArtifactStore


class PostgresArtifactStore(ArtifactStore):
    """ArtifactStore over Postgres (psycopg v3).

    Args:
        dsn: ``postgres://user:pass@host:port/db`` 형식 connection string.
        schema: 테이블 namespace (default ``lmtune``).
        autocommit: 매 put 단위 commit (default True). False 시 ``commit()`` 명시.

    Raises:
        ImportError: ``psycopg`` 미설치 시. ``pip install lmtune[postgres]`` 안내.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "lmtune",
        autocommit: bool = True,
    ):
        try:
            import psycopg  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:  # pragma: no cover — extra 미설치 환경 분기
            raise ImportError(
                "psycopg is required for PostgresArtifactStore — "
                "install with: pip install 'lmtune[postgres]'"
            ) from e

        self.dsn = dsn
        self.schema = schema
        self.autocommit = autocommit
        # 실제 connection 은 lazy — 첫 put/query 호출 시.
        self._conn: Any | None = None

    # ── ArtifactStore ABC ─────────────────────────────────────────────

    def put(self, records: list[RecordSpec]) -> int:
        raise NotImplementedError(
            "PostgresArtifactStore.put — schema migration + INSERT 가 아직 미구현. "
            "follow-up PR 에서 lmtune.storage.schema.sql 의 Postgres 변환 + "
            "kind 별 upsert 로 채워질 예정."
        )

    def query(self, spec: QuerySpec) -> list[RecordSpec]:
        raise NotImplementedError(
            "PostgresArtifactStore.query — SQL 빌더가 아직 미구현. "
            "follow-up PR 에서 QuerySpec.is_raw() 분기 + filter/sort/limit 변환."
        )

    def close(self) -> None:
        """Connection 정리 — 미수립 상태면 no-op."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
