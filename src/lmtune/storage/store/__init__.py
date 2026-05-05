"""ArtifactStore — RecordSpec / QuerySpec 위에서 동작하는 store ABC + 구현체.

L1 (Storage) 의 정문. backend (DuckDB, in-memory, Postgres, S3, Parquet) 는
모두 ArtifactStore 인터페이스만 구현하면 모든 client (driver, output, CLI) 에
플러그.
"""

from __future__ import annotations

from lmtune.storage.store.base import ArtifactStore
from lmtune.storage.store.duckdb_adapter import DuckDBArtifactStore
from lmtune.storage.store.in_memory import InMemoryArtifactStore

__all__ = ["ArtifactStore", "InMemoryArtifactStore", "DuckDBArtifactStore"]
