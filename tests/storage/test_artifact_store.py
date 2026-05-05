"""ArtifactStore ABC + 2 구현체의 동형 동작 검증.

InMemory 와 DuckDB 가 같은 (put, query) 결과를 주는지 parametrized 테스트.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lmtune.contracts.query_spec import FilterCond, QuerySpec, SortKey
from lmtune.contracts.record_spec import (
    DetectionRecord,
    MetricRecord,
    PromSampleRecord,
    RunRecord,
    StudyRecord,
    TrialRecord,
)
from lmtune.storage.store import (
    ArtifactStore,
    DuckDBArtifactStore,
    InMemoryArtifactStore,
)

# ─── store factory ────────────────────────────────────────────────────


@pytest.fixture(params=["in_memory", "duckdb"])
def store(request, tmp_path: Path):
    if request.param == "in_memory":
        s = InMemoryArtifactStore()
    else:
        s = DuckDBArtifactStore(tmp_path / "test.duckdb")
    yield s
    s.close()


# ─── put / query 기본 동형 ─────────────────────────────────────────────


def test_put_then_query_run(store: ArtifactStore):
    rec = RunRecord(
        run_id="01ABC",
        profile_slug="p",
        endpoint_slug="e",
        runner="guidellm",
        status="ok",
    )
    n = store.put([rec])
    assert n == 1
    out = store.query(QuerySpec(record_kind="run"))
    assert len(out) == 1
    assert out[0].run_id == "01ABC"
    assert out[0].kind == "run"


def test_put_upsert_same_pk(store: ArtifactStore):
    """같은 primary_key 재 put → 마지막 값이 살아남음."""
    a = RunRecord(run_id="r1", profile_slug="p1", endpoint_slug="e", runner="x", status="ok")
    b = RunRecord(run_id="r1", profile_slug="p2", endpoint_slug="e", runner="x", status="error")
    store.put([a])
    store.put([b])
    out = store.query(QuerySpec(record_kind="run"))
    assert len(out) == 1
    assert out[0].profile_slug == "p2"
    assert out[0].status == "error"


def test_filter_eq(store: ArtifactStore):
    store.put(
        [
            RunRecord(run_id="r1", profile_slug="a", endpoint_slug="e", runner="g", status="ok"),
            RunRecord(run_id="r2", profile_slug="b", endpoint_slug="e", runner="g", status="ok"),
            RunRecord(run_id="r3", profile_slug="a", endpoint_slug="e", runner="g", status="error"),
        ]
    )
    rows = store.query(
        QuerySpec(
            record_kind="run",
            filters=[FilterCond(column="profile_slug", op="==", value="a")],
        )
    )
    assert len(rows) == 2
    assert {r.run_id for r in rows} == {"r1", "r3"}


def test_filter_in(store: ArtifactStore):
    store.put(
        [
            RunRecord(run_id=f"r{i}", profile_slug="p", endpoint_slug="e", runner="g", status=s)
            for i, s in enumerate(["ok", "ok", "error", "crash"])
        ]
    )
    rows = store.query(
        QuerySpec(
            record_kind="run",
            filters=[FilterCond(column="status", op="in", value=["error", "crash"])],
        )
    )
    assert {r.run_id for r in rows} == {"r2", "r3"}


def test_sort_desc_with_limit(store: ArtifactStore):
    store.put(
        [
            TrialRecord(
                trial_id=f"t{i}",
                study_id="st1",
                seq=i,
                params={"x": i},
                status="completed",
                score=float(i),
            )
            for i in range(5)
        ]
    )
    rows = store.query(
        QuerySpec(
            record_kind="trial",
            sort=[SortKey(column="score", direction="desc")],
            limit=2,
        )
    )
    assert len(rows) == 2
    assert [r.score for r in rows] == [4.0, 3.0]


def test_count(store: ArtifactStore):
    for i in range(3):
        store.put(
            [
                RunRecord(
                    run_id=f"r{i}", profile_slug="p", endpoint_slug="e", runner="g", status="ok"
                )
            ]
        )
    assert store.count("run") == 3
    assert store.count("trial") == 0


# ─── 다양한 kind 의 round-trip ────────────────────────────────────────


def test_metric_record_round_trip(store: ArtifactStore):
    store.put(
        [
            MetricRecord(run_id="r1", metric="ttft", p="p99", value=500.0),
            MetricRecord(run_id="r1", metric="ttft", p="p50", value=100.0),
            MetricRecord(run_id="r1", metric="throughput_tok", p="avg", value=140.5),
        ]
    )
    rows = store.query(
        QuerySpec(
            record_kind="metric",
            filters=[
                FilterCond(column="run_id", op="==", value="r1"),
                FilterCond(column="metric", op="==", value="ttft"),
            ],
            sort=[SortKey(column="value", direction="asc")],
        )
    )
    assert [(r.p, r.value) for r in rows] == [("p50", 100.0), ("p99", 500.0)]


def test_study_with_list_field(store: ArtifactStore):
    """study.profile_slugs 가 list[str] (JSON 컬럼) round-trip."""
    rec = StudyRecord(
        study_id="st1",
        name="hello",
        strategy="tpe",
        profile_slugs=["short", "medium", "long"],
    )
    store.put([rec])
    out = store.query(QuerySpec(record_kind="study"))
    assert len(out) == 1
    assert out[0].profile_slugs == ["short", "medium", "long"]


def test_trial_with_dict_field(store: ArtifactStore):
    """trial.params dict round-trip via JSON column."""
    rec = TrialRecord(
        trial_id="t1",
        study_id="st1",
        seq=1,
        params={"max_num_seqs": 64, "tp": 1, "kv_cache_dtype": "fp8"},
        status="completed",
        score=42.5,
    )
    store.put([rec])
    out = store.query(QuerySpec(record_kind="trial"))
    assert out[0].params["max_num_seqs"] == 64
    assert out[0].params["kv_cache_dtype"] == "fp8"


def test_prom_sample_with_labels(store: ArtifactStore):
    ts = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    store.put(
        [
            PromSampleRecord(run_id="r1", ts=ts, metric="cpu", value=0.5, labels={"host": "h1"}),
            PromSampleRecord(run_id="r1", ts=ts, metric="cpu", value=0.7, labels={"host": "h2"}),
        ]
    )
    out = store.query(QuerySpec(record_kind="prom_sample"))
    assert len(out) == 2
    hosts = {r.labels["host"] for r in out if r.labels}  # type: ignore[index]
    assert hosts == {"h1", "h2"}


def test_detection_record(store: ArtifactStore):
    store.put(
        [
            DetectionRecord(
                run_id="r1",
                detector="ttft_p99",
                severity="warning",
                metric="ttft",
                threshold=500.0,
                observed=620.0,
                message="exceeded",
            ),
        ]
    )
    out = store.query(QuerySpec(record_kind="detection"))
    assert out[0].observed == 620.0


# ─── empty / not-found ────────────────────────────────────────────────


def test_query_empty_kind_returns_empty(store: ArtifactStore):
    rows = store.query(QuerySpec(record_kind="metric"))
    assert rows == []


def test_offset_with_limit(store: ArtifactStore):
    for i in range(5):
        store.put(
            [
                TrialRecord(
                    trial_id=f"t{i}",
                    study_id="st1",
                    seq=i,
                    params={"i": i},
                    status="completed",
                    score=float(i),
                )
            ]
        )
    rows = store.query(
        QuerySpec(
            record_kind="trial",
            sort=[SortKey(column="seq", direction="asc")],
            offset=1,
            limit=2,
        )
    )
    assert [r.trial_id for r in rows] == ["t1", "t2"]


# ─── ABC 동형 ────────────────────────────────────────────────────────


def test_both_stores_same_query_result(tmp_path: Path):
    """두 store backend 가 같은 (put, query) sequence 에 같은 결과."""
    in_mem = InMemoryArtifactStore()
    duck = DuckDBArtifactStore(tmp_path / "x.duckdb")

    recs = [
        TrialRecord(
            trial_id=f"t{i}",
            study_id="st1",
            seq=i,
            params={"i": i},
            status="completed",
            score=float(i),
        )
        for i in range(5)
    ]
    in_mem.put(recs)
    duck.put(recs)

    q = QuerySpec(
        record_kind="trial",
        filters=[FilterCond(column="status", op="==", value="completed")],
        sort=[SortKey(column="score", direction="desc")],
        limit=3,
    )
    a = in_mem.query(q)
    b = duck.query(q)
    assert [r.trial_id for r in a] == [r.trial_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]

    in_mem.close()
    duck.close()


# ─── context manager ─────────────────────────────────────────────────


def test_context_manager(tmp_path: Path):
    db = tmp_path / "ctx.duckdb"
    with DuckDBArtifactStore(db) as store:
        store.put(
            [RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="ok")]
        )
        assert store.count("run") == 1
    # exit close 됨 — 새 연결도 가능
    with DuckDBArtifactStore(db) as store2:
        assert store2.count("run") == 1
