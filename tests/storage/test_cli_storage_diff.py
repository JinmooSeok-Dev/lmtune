"""``lmtune storage diff`` — 두 store 의 record 차이 보고."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lmtune.cli_storage import app
from lmtune.contracts import MetricRecord, RunRecord
from lmtune.storage.store import DuckDBArtifactStore, LocalArtifactStore

runner = CliRunner()


def _seed(store, ids: tuple[str, ...], with_status: str = "ok") -> None:
    store.put(
        [
            RunRecord(
                run_id=rid,
                profile_slug="p",
                endpoint_slug="e",
                runner="guidellm",
                status=with_status,
            )
            for rid in ids
        ]
    )


def test_diff_equal_stores(tmp_path: Path):
    left = LocalArtifactStore(tmp_path / "left")
    right = LocalArtifactStore(tmp_path / "right")
    _seed(left, ("r1", "r2"))
    _seed(right, ("r1", "r2"))
    left.close()
    right.close()

    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "local",
            "--left",
            str(tmp_path / "left"),
            "--right-kind",
            "local",
            "--right",
            str(tmp_path / "right"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "equal" in result.output


def test_diff_only_left_only_right(tmp_path: Path):
    left = LocalArtifactStore(tmp_path / "left")
    right = LocalArtifactStore(tmp_path / "right")
    _seed(left, ("r1", "r2", "r3"))  # left 에만 r3
    _seed(right, ("r1", "r2", "r4"))  # right 에만 r4
    left.close()
    right.close()

    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "local",
            "--left",
            str(tmp_path / "left"),
            "--right-kind",
            "local",
            "--right",
            str(tmp_path / "right"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["equal"] is False
    assert payload["by_kind"]["run"]["only_left"] == 1
    assert payload["by_kind"]["run"]["only_right"] == 1
    assert payload["by_kind"]["run"]["mismatched"] == 0


def test_diff_mismatched_records(tmp_path: Path):
    """같은 primary_key 인데 다른 값 → mismatched."""
    left = LocalArtifactStore(tmp_path / "left")
    right = LocalArtifactStore(tmp_path / "right")
    _seed(left, ("r1",), with_status="ok")
    _seed(right, ("r1",), with_status="failed")  # 같은 r1, 다른 status
    left.close()
    right.close()

    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "local",
            "--left",
            str(tmp_path / "left"),
            "--right-kind",
            "local",
            "--right",
            str(tmp_path / "right"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["equal"] is False
    assert payload["by_kind"]["run"]["mismatched"] == 1
    assert payload["by_kind"]["run"]["only_left"] == 0
    assert payload["by_kind"]["run"]["only_right"] == 0


def test_diff_cross_backend_local_vs_duckdb(tmp_path: Path):
    """local↔duckdb 간 비교 — migrate 후 round-trip 검증의 핵심."""
    local = LocalArtifactStore(tmp_path / "left")
    duck = DuckDBArtifactStore(tmp_path / "right.duckdb")

    records = [
        RunRecord(
            run_id="r1",
            profile_slug="p",
            endpoint_slug="e",
            runner="guidellm",
            status="ok",
        ),
        MetricRecord(run_id="r1", metric="ttft", p="p99", value=200.0),
    ]
    local.put(records)
    duck.put(records)
    local.close()
    duck.close()

    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "local",
            "--left",
            str(tmp_path / "left"),
            "--right-kind",
            "duckdb",
            "--right",
            str(tmp_path / "right.duckdb"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["equal"] is True


def test_diff_both_empty(tmp_path: Path):
    """양쪽 모두 비어있어도 equal."""
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "local",
            "--left",
            str(tmp_path / "left"),
            "--right-kind",
            "local",
            "--right",
            str(tmp_path / "right"),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["equal"] is True
    assert payload["by_kind"] == {}


def test_diff_unknown_backend_rejected(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "diff",
            "--left-kind",
            "mongodb",
            "--left",
            str(tmp_path),
            "--right-kind",
            "local",
            "--right",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
