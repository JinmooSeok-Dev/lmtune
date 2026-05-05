"""WorkloadProvider ABC — WorkloadSpec 을 어디서 가져오든 동일 인터페이스.

설계 메모:
  - provide() 는 WorkloadSpec Pydantic obj 반환 (lm_workloads.spec.workload_spec)
  - fingerprint() 는 cache key 용 — 같은 입력은 같은 fingerprint
  - subclass 는 provide()/fingerprint() 만 구현; 다른 lmtune 코드는 본 ABC 만 의존
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmtune.contracts.workload_spec import WorkloadSpec


class WorkloadProvider(ABC):
    """WorkloadSpec 을 어디서 가져오든 동일 인터페이스."""

    @abstractmethod
    def provide(self, *, refresh: bool = False) -> WorkloadSpec:
        """WorkloadSpec 반환. refresh=True 면 cache 무시하고 재실행."""

    def fingerprint(self) -> str:
        """Cache key — 같은 입력은 같은 fingerprint. default 는 클래스+repr."""
        h = hashlib.sha256()
        h.update(type(self).__name__.encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(self).encode("utf-8"))
        return h.hexdigest()[:16]
