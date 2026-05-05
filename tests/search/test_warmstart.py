from __future__ import annotations

import json
from pathlib import Path

import duckdb

from lmtune.search.space import Axis, SearchSpace
from lmtune.search.warmstart import warmstart_from_archive


def _seed_fixture_db(path: Path) -> None:
    """Build a tiny archive DB matching the runs+metrics schema."""
    c = duckdb.connect(str(path))
    c.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            profile_slug TEXT,
            endpoint_slug TEXT,
            endpoint_meta TEXT,
            status TEXT
        );
        CREATE TABLE metrics (
            run_id TEXT,
            metric TEXT,
            p TEXT,
            value DOUBLE,
            PRIMARY KEY (run_id, metric, p)
        );
        """
    )
    good_ea = json.dumps(
        {
            "deployment": {
                "engine_args": {
                    "max_num_seqs": 128,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": 0.85,
                }
            }
        }
    )
    bad_ea = json.dumps(
        {
            "deployment": {
                "engine_args": {
                    "max_num_seqs": 999,  # out of range for current space
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": 0.85,
                }
            }
        }
    )
    c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("r1", "short", "ep1", good_ea, "ok"))
    c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("r2", "medium", "ep1", good_ea, "ok"))
    c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("r3", "short", "ep1", bad_ea, "ok"))
    c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("r4", "short", "ep2", good_ea, "ok"))
    for run_id, thr, ttft, e2e in [
        ("r1", 800.0, 200.0, 2.0),
        ("r2", 600.0, 300.0, 5.0),
        ("r3", 200.0, 600.0, 10.0),  # SLO fail (ttft > 500)
        ("r4", 400.0, 100.0, 1.0),
    ]:
        c.execute("INSERT INTO metrics VALUES (?,?,?,?)", (run_id, "throughput_tok", "avg", thr))
        c.execute("INSERT INTO metrics VALUES (?,?,?,?)", (run_id, "ttft", "p99", ttft))
        c.execute("INSERT INTO metrics VALUES (?,?,?,?)", (run_id, "e2e", "p99", e2e))
    c.close()


def _space() -> SearchSpace:
    return SearchSpace(
        name="t1",
        axes=[
            Axis("max_num_seqs", "categorical", values=[32, 64, 128, 256]),
            Axis("enable_prefix_caching", "bool"),
            Axis("gpu_memory_utilization", "float", low=0.80, high=0.92),
        ],
    )


def test_warmstart_returns_top_composite(tmp_path: Path):
    db = tmp_path / "arch.duckdb"
    _seed_fixture_db(db)
    seeds = warmstart_from_archive(
        db,
        _space(),
        endpoint_slug="ep1",
        top_k=5,
    )
    # r1+r2 pass SLO; their per-workload scores sum as a single group
    assert len(seeds) == 1
    params, score = seeds[0]
    assert params["max_num_seqs"] == 128
    assert params["enable_prefix_caching"] is True
    assert abs(params["gpu_memory_utilization"] - 0.85) < 1e-9
    assert score > 0


def test_warmstart_rejects_out_of_range_params(tmp_path: Path):
    db = tmp_path / "arch.duckdb"
    _seed_fixture_db(db)
    # Space only allows max_num_seqs ∈ {32,64,128,256}; r3's 999 is rejected.
    seeds = warmstart_from_archive(
        db,
        _space(),
        endpoint_slug="ep1",
        top_k=5,
    )
    for params, _ in seeds:
        assert params["max_num_seqs"] != 999


def test_warmstart_endpoint_filter(tmp_path: Path):
    db = tmp_path / "arch.duckdb"
    _seed_fixture_db(db)
    seeds_ep1 = warmstart_from_archive(db, _space(), endpoint_slug="ep1", top_k=5)
    seeds_ep2 = warmstart_from_archive(db, _space(), endpoint_slug="ep2", top_k=5)
    # ep1 has 2 good runs; ep2 has 1. Both should return one aggregated group
    assert len(seeds_ep1) == 1
    assert len(seeds_ep2) == 1


def test_warmstart_missing_db_returns_empty(tmp_path: Path):
    assert warmstart_from_archive(tmp_path / "nope.duckdb", _space()) == []
