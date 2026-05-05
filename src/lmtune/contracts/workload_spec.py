"""WorkloadSpec — re-export from lm-workloads (master).

lm-workloads 의 ``apiVersion: workloads/v1alpha1`` 가 정본. 본 모듈은 직접
re-export 만 — Pydantic 모델 자체를 lm-workloads 에서 그대로 가져와 drift 0.

Install via ``pip install lmtune[workloads]`` (또는 file:// editable in dev).
"""

from __future__ import annotations

try:
    from lm_workloads.spec.workload_spec import (
        AnomalyFlags,
        ArrivalSpec,
        BurstWindow,
        Classification,
        DistSpec,
        PayloadDistributions,
        ReproductionMode,
        SLOObservation,
        SourceEndpoint,
        SourceWindow,
        StatSummary,
        TokenSnowballSpec,
        TraceArtifact,
        TrafficPattern,
        TurnDistribution,
        WorkloadMeta,
        WorkloadSpec,
    )
except ImportError as e:  # pragma: no cover - exercised via integration
    raise ImportError(
        "WorkloadSpec contract requires lm-workloads. "
        "Install with: pip install 'lmtune[workloads]' "
        "(or pip install -e /path/to/workloads)"
    ) from e


__all__ = [
    "AnomalyFlags",
    "ArrivalSpec",
    "BurstWindow",
    "Classification",
    "DistSpec",
    "PayloadDistributions",
    "ReproductionMode",
    "SLOObservation",
    "SourceEndpoint",
    "SourceWindow",
    "StatSummary",
    "TokenSnowballSpec",
    "TraceArtifact",
    "TrafficPattern",
    "TurnDistribution",
    "WorkloadMeta",
    "WorkloadSpec",
]
