"""lmtune contracts — input/output spec 정본 패키지.

본 패키지의 각 모듈은 lmtune 의 contract 1개씩을 호스팅. 외부 master 가 있는
contract (WorkloadSpec=lm-workloads, ClusterSpec=ariadne) 는 lazy import.
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
]
