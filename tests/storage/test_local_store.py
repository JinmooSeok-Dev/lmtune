"""LocalArtifactStore — file-based ArtifactStore 검증."""

from __future__ import annotations

from pathlib import Path

from lmtune.contracts import (
    FilterCond,
    MetricRecord,
    QuerySpec,
    RequestRecord,
    RunRecord,
    SortKey,
    TrialRecord,
)
from lmtune.storage.store import (
    ArtifactStore,
    DuckDBArtifactStore,
    InMemoryArtifactStore,
    LocalArtifactStore,
)


def test_local_is_artifact_store(tmp_path: Path):
    s = LocalArtifactStore(tmp_path)
    assert isinstance(s, ArtifactStore)


def test_local_put_creates_jsonl(tmp_path: Path):
    s = LocalArtifactStore(tmp_path)
    n = s.put(
        [
            RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="ok"),
        ]
    )
    assert n == 1
    assert (tmp_path / "run.jsonl").exists()
    text = (tmp_path / "run.jsonl").read_text()
    assert "r1" in text


def test_local_put_multiple_kinds(tmp_path: Path):
    s = LocalArtifactStore(tmp_path)
    s.put(
        [
            RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="ok"),
            MetricRecord(run_id="r1", metric="ttft", p="p99", value=200.0),
            MetricRecord(run_id="r1", metric="ttft", p="p50", value=50.0),
            RequestRecord(run_id="r1", req_id="req-1", ttft_ms=42.0),
        ]
    )
    assert (tmp_path / "run.jsonl").exists()
    assert (tmp_path / "metric.jsonl").exists()
    assert (tmp_path / "request.jsonl").exists()
    assert s.count("metric") == 2


def test_local_query_filter_sort_limit(tmp_path: Path):
    s = LocalArtifactStore(tmp_path)
    s.put(
        [
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
    )
    q = QuerySpec(
        record_kind="trial",
        filters=[FilterCond(column="status", op="==", value="completed")],
        sort=[SortKey(column="score", direction="desc")],
        limit=3,
    )
    out = s.query(q)
    assert len(out) == 3
    assert [r.score for r in out] == [4.0, 3.0, 2.0]


def test_local_dedup_by_primary_key(tmp_path: Path):
    """같은 primary_key 의 record 를 두 번 put → query 결과는 1 건 (last wins)."""
    s = LocalArtifactStore(tmp_path)
    s.put([RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="ok")])
    s.put(
        [RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="failed")]
    )
    runs = s.query(QuerySpec(record_kind="run"))
    assert len(runs) == 1
    assert runs[0].status == "failed"


def test_local_query_empty_kind(tmp_path: Path):
    """파일 없는 kind 의 query 는 빈 리스트."""
    s = LocalArtifactStore(tmp_path)
    assert s.query(QuerySpec(record_kind="run")) == []


def test_local_same_query_result_as_in_memory(tmp_path: Path):
    """LocalArtifactStore 와 InMemoryArtifactStore 가 같은 (put, query) 결과."""
    in_mem = InMemoryArtifactStore()
    local = LocalArtifactStore(tmp_path)

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
    local.put(recs)

    q = QuerySpec(
        record_kind="trial",
        filters=[FilterCond(column="status", op="==", value="completed")],
        sort=[SortKey(column="score", direction="desc")],
        limit=3,
    )
    a = in_mem.query(q)
    b = local.query(q)
    assert [r.trial_id for r in a] == [r.trial_id for r in b]
    assert [r.score for r in a] == [r.score for r in b]


def test_local_same_query_result_as_duckdb(tmp_path: Path):
    """LocalArtifactStore 와 DuckDBArtifactStore 의 query 결과 동일."""
    duck = DuckDBArtifactStore(tmp_path / "x.duckdb")
    local = LocalArtifactStore(tmp_path / "local")

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
    duck.put(recs)
    local.put(recs)

    q = QuerySpec(record_kind="trial", sort=[SortKey(column="score", direction="asc")])
    a = duck.query(q)
    b = local.query(q)
    assert [r.trial_id for r in a] == [r.trial_id for r in b]
    duck.close()
    local.close()


def test_local_context_manager(tmp_path: Path):
    with LocalArtifactStore(tmp_path) as s:
        s.put(
            [RunRecord(run_id="r1", profile_slug="p", endpoint_slug="e", runner="g", status="ok")]
        )
    # close 후에도 파일은 남아 있다 — 같은 root 로 다시 열면 데이터 보임
    with LocalArtifactStore(tmp_path) as s2:
        assert s2.count("run") == 1
