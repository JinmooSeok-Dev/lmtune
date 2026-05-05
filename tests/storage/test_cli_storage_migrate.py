"""``lmtune storage migrate`` — backend 간 변환 E2E."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from lmtune.cli_storage import app
from lmtune.contracts import (
    MetricRecord,
    QuerySpec,
    RunRecord,
    TrialRecord,
)
from lmtune.storage.store import (
    DuckDBArtifactStore,
    LocalArtifactStore,
)

runner = CliRunner()


def _seed_local(root: Path) -> None:
    s = LocalArtifactStore(root)
    s.put(
        [
            RunRecord(
                run_id="r1",
                profile_slug="p",
                endpoint_slug="e",
                runner="guidellm",
                status="ok",
            ),
            MetricRecord(run_id="r1", metric="ttft", p="p99", value=200.0),
            MetricRecord(run_id="r1", metric="ttft", p="p50", value=50.0),
            TrialRecord(
                trial_id="t1",
                study_id="st1",
                seq=1,
                params={"x": 1},
                status="completed",
                score=0.9,
            ),
        ]
    )
    s.close()


def test_migrate_local_to_duckdb(tmp_path: Path):
    src = tmp_path / "local"
    dst = tmp_path / "out.duckdb"
    _seed_local(src)

    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "local",
            "--src",
            str(src),
            "--dst-kind",
            "duckdb",
            "--dst",
            str(dst),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "migrated" in result.output

    duck = DuckDBArtifactStore(dst)
    assert duck.count("run") == 1
    assert duck.count("metric") == 2
    assert duck.count("trial") == 1
    duck.close()


def test_migrate_duckdb_to_local(tmp_path: Path):
    src = tmp_path / "src.duckdb"
    dst = tmp_path / "dst-local"
    duck = DuckDBArtifactStore(src)
    duck.put(
        [
            RunRecord(
                run_id="r2",
                profile_slug="p",
                endpoint_slug="e",
                runner="guidellm",
                status="ok",
            ),
            MetricRecord(run_id="r2", metric="ttft", p="p99", value=180.0),
        ]
    )
    duck.close()

    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "duckdb",
            "--src",
            str(src),
            "--dst-kind",
            "local",
            "--dst",
            str(dst),
        ],
    )
    assert result.exit_code == 0, result.output

    # local 파일 확인
    assert (dst / "run.jsonl").exists()
    assert (dst / "metric.jsonl").exists()
    local = LocalArtifactStore(dst)
    assert local.count("run") == 1
    assert local.count("metric") == 1


def test_migrate_kinds_filter(tmp_path: Path):
    """--kinds metric 만 → run/trial 은 dest 에 가지 않음."""
    src = tmp_path / "local"
    dst = tmp_path / "out.duckdb"
    _seed_local(src)

    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "local",
            "--src",
            str(src),
            "--dst-kind",
            "duckdb",
            "--dst",
            str(dst),
            "--kinds",
            "metric",
        ],
    )
    assert result.exit_code == 0, result.output

    duck = DuckDBArtifactStore(dst)
    assert duck.count("metric") == 2
    assert duck.count("run") == 0
    assert duck.count("trial") == 0
    duck.close()


def test_migrate_unknown_kinds_rejected(tmp_path: Path):
    src = tmp_path / "local"
    dst = tmp_path / "out.duckdb"
    _seed_local(src)

    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "local",
            "--src",
            str(src),
            "--dst-kind",
            "duckdb",
            "--dst",
            str(dst),
            "--kinds",
            "metric,bogus",
        ],
    )
    assert result.exit_code != 0


def test_migrate_unknown_backend_rejected(tmp_path: Path):
    """등록되지 않은 backend kind → exit != 0."""
    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "mongodb",  # 등록되지 않은 가상의 backend
            "--src",
            str(tmp_path / "x"),
            "--dst-kind",
            "local",
            "--dst",
            str(tmp_path / "y"),
        ],
    )
    assert result.exit_code != 0


def test_migrate_empty_src(tmp_path: Path):
    """빈 src → 'no records' 메시지, exit 0."""
    src = tmp_path / "empty"
    src.mkdir()
    dst = tmp_path / "out.duckdb"

    result = runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "local",
            "--src",
            str(src),
            "--dst-kind",
            "duckdb",
            "--dst",
            str(dst),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no records" in result.output


def test_list_backends():
    """list-backends 가 등록된 backend 출력."""
    result = runner.invoke(app, ["list-backends"])
    assert result.exit_code == 0
    assert "local" in result.output
    assert "duckdb" in result.output


def test_migrate_round_trip_query_equivalence(tmp_path: Path):
    """src.query == dst.query (after migrate). ABC 의 핵심 보증."""
    src = tmp_path / "local"
    dst = tmp_path / "out.duckdb"
    _seed_local(src)

    runner.invoke(
        app,
        [
            "migrate",
            "--src-kind",
            "local",
            "--src",
            str(src),
            "--dst-kind",
            "duckdb",
            "--dst",
            str(dst),
        ],
    )

    src_store = LocalArtifactStore(src)
    dst_store = DuckDBArtifactStore(dst)

    a = sorted(
        src_store.query(QuerySpec(record_kind="metric")),
        key=lambda r: (r.metric, r.p),
    )
    b = sorted(
        dst_store.query(QuerySpec(record_kind="metric")),
        key=lambda r: (r.metric, r.p),
    )
    assert [(r.metric, r.p, r.value) for r in a] == [(r.metric, r.p, r.value) for r in b]

    src_store.close()
    dst_store.close()
