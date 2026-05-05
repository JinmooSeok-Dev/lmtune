"""lmtune workload layer — WorkloadSpec 의 lmtune 측 Provider 호스팅.

WorkloadProvider ABC 와 구현체:
  - LiteralWorkloadProvider  (yaml path → WorkloadSpec)
  - LMWorkloadsProvider      (lm-workloads 호출 → WorkloadSpec)

다른 외부 프로젝트나 사용자는 entry_points "lmtune.workload_providers" 로
자기 Provider 를 추가 가능.
"""

from __future__ import annotations

from lmtune.workload.providers.base import WorkloadProvider

__all__ = ["WorkloadProvider"]
