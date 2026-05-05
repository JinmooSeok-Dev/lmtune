"""RecordSpec — 모든 record kind 의 round-trip + discriminator 검증."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

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

# ─── 단일 모델 round-trip ─────────────────────────────────────────────


def test_run_record_basic():
    rec = RunRecord(
        run_id="01ABC",
        profile_slug="autotune-short",
        endpoint_slug="local-vllm",
        runner="guidellm",
        status="ok",
    )
    assert rec.kind == "run"
    assert rec.api_version == "lmtune/record/v1alpha1"
    assert rec.primary_key() == ("01ABC",)


def test_metric_record_pk_with_p_none():
    """p=None 시 PK 의 p 자리는 빈 문자열로."""
    rec = MetricRecord(run_id="r1", metric="ttft", value=42.0)
    assert rec.primary_key() == ("r1", "ttft", "")


def test_metric_record_pk_with_p():
    rec = MetricRecord(run_id="r1", metric="ttft", p="p99", value=500.0)
    assert rec.primary_key() == ("r1", "ttft", "p99")


def test_request_record_optional_fields_default_none():
    rec = RequestRecord(run_id="r1", req_id="req-1")
    assert rec.input_tokens is None
    assert rec.phase is None


def test_session_record_pk():
    rec = SessionRecord(run_id="r1", session_id="s1")
    assert rec.primary_key() == ("r1", "s1")


def test_trajectory_event_record_pk():
    rec = TrajectoryEventRecord(run_id="r1", session_id="s1", seq=3, event_type="user")
    assert rec.primary_key() == ("r1", "s1", 3)


def test_prom_sample_record_labels_in_pk():
    """Same metric+ts but different labels → different PK."""
    ts = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    a = PromSampleRecord(run_id="r1", ts=ts, metric="m", value=1.0, labels={"k": "a"})
    b = PromSampleRecord(run_id="r1", ts=ts, metric="m", value=2.0, labels={"k": "b"})
    assert a.primary_key() != b.primary_key()


def test_detection_record_pk():
    ts = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    rec = DetectionRecord(run_id="r1", detector="d1", severity="warning", created_at=ts)
    pk = rec.primary_key()
    assert pk[0] == "r1" and pk[1] == "d1"
    assert "2026-05-06" in pk[2]


def test_study_record_defaults():
    rec = StudyRecord(study_id="st1", name="hello", strategy="tpe")
    assert rec.metric_name == "total_score"
    assert rec.direction == "maximize"
    assert rec.status == "running"


def test_trial_record_with_params_dict():
    rec = TrialRecord(
        trial_id="t1",
        study_id="st1",
        seq=1,
        params={"max_num_seqs": 64, "tp": 1},
        status="completed",
        score=42.5,
    )
    assert rec.params["max_num_seqs"] == 64
    assert rec.score == 42.5


def test_trial_metric_record_workload_default():
    rec = TrialMetricRecord(trial_id="t1", metric="ttft.p99", value=100.0)
    assert rec.workload == "aggregate"


# ─── frozen=True (불변) ──────────────────────────────────────────────


def test_record_is_frozen():
    rec = RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="x", status="ok")
    with pytest.raises((ValidationError, TypeError)):
        rec.run_id = "different"  # type: ignore[misc]


# ─── extra=forbid ────────────────────────────────────────────────────


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="r1",
            profile_slug="p",
            endpoint_slug="e",
            runner="x",
            status="ok",
            _bogus="x",  # type: ignore[call-arg]
        )


# ─── Discriminated Union ─────────────────────────────────────────────


def test_recordspec_discriminator_round_trip():
    """dict → RecordSpec → kind 별 정확한 클래스 인스턴스."""
    adapter = TypeAdapter(RecordSpec)
    payload = {
        "kind": "run",
        "run_id": "r1",
        "profile_slug": "p",
        "endpoint_slug": "e",
        "runner": "guidellm",
        "status": "ok",
    }
    rec = adapter.validate_python(payload)
    assert isinstance(rec, RunRecord)
    assert rec.run_id == "r1"


def test_recordspec_discriminator_metric():
    adapter = TypeAdapter(RecordSpec)
    rec = adapter.validate_python(
        {"kind": "metric", "run_id": "r1", "metric": "ttft", "p": "p99", "value": 100.0}
    )
    assert isinstance(rec, MetricRecord)


def test_recordspec_unknown_kind_rejected():
    adapter = TypeAdapter(RecordSpec)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "bogus", "run_id": "r1"})


# ─── kind_to_class lookup ────────────────────────────────────────────


def test_kind_to_class_all_kinds_resolve():
    """RECORD_KINDS 의 모든 kind 가 클래스 lookup 가능."""
    expected = {
        "run": RunRecord,
        "metric": MetricRecord,
        "request": RequestRecord,
        "session": SessionRecord,
        "trajectory_event": TrajectoryEventRecord,
        "prom_sample": PromSampleRecord,
        "detection": DetectionRecord,
        "study": StudyRecord,
        "trial": TrialRecord,
        "trial_metric": TrialMetricRecord,
    }
    assert set(RECORD_KINDS) == set(expected)
    for kind, cls in expected.items():
        assert kind_to_class(kind) is cls


def test_kind_to_class_unknown_raises():
    with pytest.raises(ValueError, match="unknown record kind"):
        kind_to_class("nope")


# ─── JSON Schema dump ────────────────────────────────────────────────


def test_recordspec_json_schema_has_discriminator():
    adapter = TypeAdapter(RecordSpec)
    schema = adapter.json_schema()
    # Pydantic 의 union schema 는 'oneOf' 또는 'anyOf' 로 표현 — 둘 다 허용
    assert "oneOf" in schema or "anyOf" in schema
