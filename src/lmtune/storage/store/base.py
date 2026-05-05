"""ArtifactStore ABC — 모든 record backend 의 정문.

설계 원칙:
- put(): list[RecordSpec] 을 받아 backend 에 적재 (idempotent — same primary_key
  은 upsert, 즉 마지막 값이 살아남음)
- query(): QuerySpec → list[RecordSpec] (동형 round-trip)
- backend 는 본 ABC 만 구현하면 driver / output / CLI 모두에 plug-in
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lmtune.contracts.query_spec import QuerySpec
from lmtune.contracts.record_spec import RecordSpec


class ArtifactStore(ABC):
    """Record-level abstraction. RecordSpec / QuerySpec 만 의존."""

    @abstractmethod
    def put(self, records: list[RecordSpec]) -> int:
        """Upsert records. 같은 primary_key 의 record 는 덮어쓰기.

        Returns:
            적재된 record 수 (입력 길이와 동일이 일반적, dedup 로직 시 다를 수 있음).
        """

    @abstractmethod
    def query(self, spec: QuerySpec) -> list[RecordSpec]:
        """Query records.

        Returns:
            spec.record_kind 에 해당하는 RecordSpec 인스턴스 리스트.
            aggregate 가 활성이면 빈 리스트 + side-channel (구현체별) 가능.
        """

    @abstractmethod
    def close(self) -> None:
        """Backend 자원 정리."""

    # ── 편의 메서드 (default 구현) ──────────────────────────────────

    def __enter__(self) -> ArtifactStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def count(self, record_kind: str) -> int:
        """간이 카운트. backend 별 더 효율적 구현 가능."""
        from lmtune.contracts.query_spec import QuerySpec as _Q

        return len(self.query(_Q(record_kind=record_kind)))
