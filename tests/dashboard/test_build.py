"""Phase W — bench dashboard build unit tests.

검증:
  1. fixture DuckDB → 1 study + 2 trial 시드
  2. build_dashboard() 가 기대 파일 (index/study/compare + 3 JSON) 생성
  3. studies_index.json / throughput_vs_latency.json / perf_history.json 의 strict schema
  4. study card top_score, model_id 등 inference 결과 정확
  5. perf-changelog.yaml 의 date 객체가 JSON 으로 직렬화 (landed_at)
  6. perf-changelog.yaml 가 list 또는 {entries: [...]} 두 형식 모두 허용
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lmtune.storage import DuckDBStore
from lmtune.visualization.dashboard import build_dashboard, dump_inferencex_json
from lmtune.visualization.dashboard.schemas import (
    PerfHistory,
    PerfHistoryEntry,
    StudiesIndex,
    StudyCard,
    ThroughputVsLatency,
    TrialPoint,
)


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lmtune.duckdb"
    store = DuckDBStore(db_path)
    space_yaml = yaml.safe_dump({"name": "test", "axes": {"x": {"type": "int", "low": 0, "high": 10}}})
    store.record_study(
        study_id="st-DASH",
        name="dashboard-test",
        strategy="tpe",
        metric_name="total_score",
        direction="maximize",
        space_yaml=space_yaml,
        endpoint_slug="minikube-pd-qwen25-vllm",
        profile_slugs=["short"],
        notes="",
    )
    for i, (score, params) in enumerate([
        (100.0, {"x": 1}),
        (250.0, {"x": 9}),  # winner
    ], start=1):
        tid = f"tr-{i:03d}"
        store.record_trial(
            trial_id=tid, study_id="st-DASH", seq=i, params=params,
            status="completed", score=score, backend="inline", completed=True,
        )
        store.record_trial_metrics(tid, {
            ("throughput_avg", "short"): score * 0.4,
            ("ttft_p99", "short"): 100.0 + i * 10,
        })
    return db_path


def test_build_dashboard_creates_expected_files(seeded_db, tmp_path):
    out = tmp_path / "dash"
    written = build_dashboard(db_path=seeded_db, out_dir=out)
    assert (out / "index.html").exists()
    assert (out / "compare.html").exists()
    assert (out / "studies" / "st-DASH.html").exists()
    # 3 JSON files (InferenceX-compat)
    assert (out / "data" / "studies_index.json").exists()
    assert (out / "data" / "throughput_vs_latency.json").exists()
    assert (out / "data" / "perf_history.json").exists()
    keys = set(written.keys())
    assert "index.html" in keys
    assert "compare.html" in keys
    assert "studies/st-DASH.html" in keys
    assert "data/studies_index.json" in keys
    assert "data/throughput_vs_latency.json" in keys
    assert "data/perf_history.json" in keys


def test_studies_index_json_strict_schema(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out)
    data = json.loads((out / "data" / "studies_index.json").read_text())
    # StudiesIndex round-trip: model_validate must accept the dump
    StudiesIndex.model_validate(data)
    assert len(data["studies"]) == 1
    s = data["studies"][0]
    assert s["study_id"] == "st-DASH"
    assert s["top_score"] == 250.0
    assert s["n_completed"] == 2
    assert s["profile_slugs"] == ["short"]
    # strict: only the locked field set
    assert set(s.keys()) == {
        "study_id", "name", "strategy", "direction", "status",
        "endpoint_slug", "profile_slugs", "n_trials", "n_completed",
        "top_score", "created_at", "finished_at",
    }


def test_throughput_vs_latency_inferred_metadata(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out)
    data = json.loads((out / "data" / "throughput_vs_latency.json").read_text())
    assert len(data) == 1
    tvl = data[0]
    # InferenceX-compat strict keys
    assert set(tvl.keys()) == {
        "study_id", "model_id", "framework", "hardware_id", "workload", "points",
    }
    assert tvl["study_id"] == "st-DASH"
    assert tvl["model_id"] == "Qwen2.5"      # inferred from "qwen25" in slug
    assert tvl["framework"] == "vllm"        # inferred from "vllm" in slug
    assert tvl["hardware_id"] == "minikube"  # inferred from "minikube" in slug
    assert tvl["workload"] == "short"
    # winner trial: throughput = 250 * 0.4 = 100.0
    winner = next(p for p in tvl["points"] if p["seq"] == 2)
    assert winner["metrics"]["throughput_avg.short"] == 100.0
    assert winner["metrics"]["ttft_p99.short"] == 120.0
    # TrialPoint strict keys
    assert set(winner.keys()) == {"trial_id", "seq", "score", "params", "metrics"}


def test_study_html_contains_winner_score(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out)
    html = (out / "studies" / "st-DASH.html").read_text()
    assert "st-DASH" in html
    assert "dashboard-test" in html
    assert "250.0000" in html


def test_compare_html_lists_all_studies(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out)
    html = (out / "compare.html").read_text()
    assert "dashboard-test" in html
    assert "Cross-study compare" in html


def test_perf_changelog_legacy_list_format(seeded_db, tmp_path):
    """Legacy list-of-entries format with `timestamp`/`config-keys`."""
    out = tmp_path / "dash"
    changelog = tmp_path / "perf-changelog.yaml"
    changelog.write_text(yaml.safe_dump([{
        "timestamp": "2026-05-03",
        "config-keys": ["test-*"],
        "description": ["entry"],
        "pr-link": None,
        "evals-only": False,
    }]))
    build_dashboard(db_path=seeded_db, out_dir=out, perf_changelog=changelog)
    perf = json.loads((out / "data" / "perf_history.json").read_text())
    PerfHistory.model_validate(perf)
    assert len(perf["entries"]) == 1
    assert perf["entries"][0]["landed_at"] == "2026-05-03"
    assert perf["entries"][0]["config_keys"] == ["test-*"]


def test_perf_changelog_new_envelope_format(seeded_db, tmp_path):
    """New `{entries: [...]}` envelope with `landed_at`/`config_keys`."""
    out = tmp_path / "dash"
    changelog = tmp_path / "perf-changelog.yaml"
    changelog.write_text(yaml.safe_dump({"entries": [{
        "landed_at": "2026-05-04",
        "config_keys": ["x-*"],
        "description": ["e"],
        "pr_link": "https://example.com/pr/1",
        "evals_only": True,
    }]}))
    build_dashboard(db_path=seeded_db, out_dir=out, perf_changelog=changelog)
    perf = json.loads((out / "data" / "perf_history.json").read_text())
    assert len(perf["entries"]) == 1
    e = perf["entries"][0]
    assert e["landed_at"] == "2026-05-04"
    assert e["pr_link"] == "https://example.com/pr/1"
    assert e["evals_only"] is True


def test_dump_inferencex_json_standalone(tmp_path):
    """dump_inferencex_json() 가 build_dashboard 없이도 단독 사용 가능."""
    si = StudiesIndex(studies=[StudyCard(
        study_id="x", name="x", strategy="random", direction="maximize",
        status="completed", n_trials=0, n_completed=0,
    )])
    tvl = [ThroughputVsLatency(study_id="x", points=[
        TrialPoint(trial_id="t1", seq=1, score=1.0,
                   params={"a": 1}, metrics={"throughput_avg.short": 9.0}),
    ])]
    perf = PerfHistory(entries=[PerfHistoryEntry(
        config_keys=["c"], description=["d"], landed_at="2026-05-03",
    )])
    out = tmp_path / "data"
    dump_inferencex_json(
        studies_index=si, throughput_vs_latency=tvl, perf_history=perf, out_dir=out,
    )
    assert (out / "studies_index.json").exists()
    assert (out / "throughput_vs_latency.json").exists()
    assert (out / "perf_history.json").exists()
    parsed_si = json.loads((out / "studies_index.json").read_text())
    StudiesIndex.model_validate(parsed_si)
    assert parsed_si["studies"][0]["study_id"] == "x"


def test_build_filtered_by_study_ids(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out, study_ids=["st-DASH"])
    data = json.loads((out / "data" / "studies_index.json").read_text())
    assert {s["study_id"] for s in data["studies"]} == {"st-DASH"}


def test_build_unknown_study_id_ignored(seeded_db, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db, out_dir=out, study_ids=["st-MISSING", "st-DASH"])
    data = json.loads((out / "data" / "studies_index.json").read_text())
    assert {s["study_id"] for s in data["studies"]} == {"st-DASH"}
