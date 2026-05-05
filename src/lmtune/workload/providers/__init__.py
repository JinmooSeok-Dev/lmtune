"""WorkloadProvider 구현체 모음."""

from __future__ import annotations

from lmtune.workload.providers.base import WorkloadProvider
from lmtune.workload.providers.literal import LiteralWorkloadProvider

__all__ = [
    "LiteralWorkloadProvider",
    "WorkloadProvider",
    "build_provider",
]


def build_provider(spec_path: str | None, source: str | None) -> WorkloadProvider:
    """CLI 인자에서 적절한 Provider 인스턴스화.

    우선순위:
      --workload-spec <path>   → LiteralWorkloadProvider
      --workload-source <uri>  → LMWorkloadsProvider (requires [workloads] extra)
    """
    if spec_path and source:
        raise ValueError("--workload-spec 과 --workload-source 는 동시 지정 불가")
    if spec_path:
        return LiteralWorkloadProvider(spec_path)
    if source:
        # Lazy import — [workloads] extra 미설치 환경에서도 LiteralProvider 동작
        from lmtune.workload.providers.lm_workloads import LMWorkloadsProvider

        return LMWorkloadsProvider(source)
    raise ValueError("provider 입력 필요 — --workload-spec 또는 --workload-source")
