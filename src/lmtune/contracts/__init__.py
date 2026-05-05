"""lmtune contracts — input/output spec 정본 패키지.

본 패키지는 lmtune 의 input/output spec 모음. 6 contract 중 일부는 외부 master
(lm-workloads, ariadne) 에서 mirror, 나머지는 lmtune own.

| Contract        | apiVersion                    | Master         |
|-----------------|-------------------------------|----------------|
| WorkloadSpec    | workloads/v1alpha1            | lm-workloads   |
| ClusterSpec     | ariadne/cluster/v1alpha1      | ariadne        |
| EndpointSpec    | lmtune/endpoint/v1alpha1      | lmtune         |
| ProfileSpec     | lmtune/profile/v1alpha1       | lmtune         |
| SearchSpace     | lmtune/search/v1alpha1        | lmtune         |
| BenchmarkResult | lmtune/result/v1alpha1        | lmtune         |

진행 상황은 docs/architecture/REFACTOR-PLAN.md 참조.
"""

from __future__ import annotations

from lmtune.contracts.query_spec import (
    AggregateSpec,
    CompareOp,
    FilterCond,
    QuerySpec,
    SortKey,
)
from lmtune.contracts.record_spec import (
    RECORD_KINDS,
    DetectionRecord,
    MetricRecord,
    PromSampleRecord,
    RecordSpec,
    RequestRecord,
    RunRecord,
    SessionRecord,
    StudyRecord,
    TrajectoryEventRecord,
    TrialMetricRecord,
    TrialRecord,
    kind_to_class,
)

__all__ = [
    # record
    "RECORD_KINDS",
    "RecordSpec",
    "RunRecord",
    "MetricRecord",
    "RequestRecord",
    "SessionRecord",
    "TrajectoryEventRecord",
    "PromSampleRecord",
    "DetectionRecord",
    "StudyRecord",
    "TrialRecord",
    "TrialMetricRecord",
    "kind_to_class",
    # query
    "QuerySpec",
    "FilterCond",
    "SortKey",
    "AggregateSpec",
    "CompareOp",
    # workload (lazy-imported from external master lm-workloads)
    "WorkloadSpec",
]


def __getattr__(name: str):
    if name == "WorkloadSpec":
        from lmtune.contracts.workload_spec import WorkloadSpec

        return WorkloadSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
