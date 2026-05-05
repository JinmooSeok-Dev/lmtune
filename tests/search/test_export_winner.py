"""Phase W — bench search export winner unit tests.

검증:
  1. fixture DuckDB 에 study + 3 trial 시드
  2. export_winner top-1 호출 → 4 파일 생성
  3. params.json 가 winner trial 의 params 와 일치
  4. values-overlay.yaml 가 valid YAML + vllmArgs 포함
  5. apply.sh 가 executable 권한
  6. README.md 가 score + trial_id 포함
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml

from lmtune.search.export_winner import export_winner
from lmtune.storage import DuckDBStore


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lmtune.duckdb"
    store = DuckDBStore(db_path)
    # study
    space_yaml = yaml.safe_dump(
        {
            "apiVersion": "lmtune/search/v1alpha1",
            "kind": "SearchSpace",
            "name": "test-space",
            "axes": {"max_num_seqs": {"type": "categorical", "values": [16, 32, 64]}},
        }
    )
    store.record_study(
        study_id="st-TEST",
        name="export-test-study",
        strategy="random",
        metric_name="total_score",
        direction="maximize",
        space_yaml=space_yaml,
        endpoint_slug="local-vllm-test",
        profile_slugs=["short"],
        notes="",
    )
    # 3 trials
    for i, (score, params) in enumerate(
        [
            (100.0, {"max_num_seqs": 16}),
            (200.0, {"max_num_seqs": 64}),  # winner
            (150.0, {"max_num_seqs": 32}),
        ],
        start=1,
    ):
        tid = f"tr-{i:03d}"
        store.record_trial(
            trial_id=tid,
            study_id="st-TEST",
            seq=i,
            params=params,
            status="completed",
            score=score,
            backend="inline",
            completed=True,
        )
        store.record_trial_metrics(
            tid,
            {
                ("throughput_avg", "short"): score * 0.5,
                ("ttft_p99", "short"): 50.0,
            },
        )
    return db_path


def test_export_winner_creates_four_files(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=1)
    assert (out / "winner").is_dir()
    names = {f.name for f in result.files}
    assert names == {"params.json", "values-overlay.yaml", "apply.sh", "README.md"}
    assert result.trial_id == "tr-002"
    assert result.score == 200.0


def test_winner_params_json_matches_top_trial(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=1)
    params = json.loads((result.out_dir / "params.json").read_text())
    assert params == {"max_num_seqs": 64}


def test_winner_overlay_is_valid_yaml(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=1)
    data = yaml.safe_load((result.out_dir / "values-overlay.yaml").read_text())
    # at least one release with vllmArgs
    assert data, "overlay must have at least one release"
    first = next(iter(data.values()))
    assert "vllmArgs" in first
    # max_num_seqs → max-num-seqs
    assert first["vllmArgs"]["max-num-seqs"] == 64


def test_apply_sh_executable(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=1)
    apply_sh = result.out_dir / "apply.sh"
    mode = apply_sh.stat().st_mode
    assert mode & stat.S_IXUSR, "apply.sh must be executable"
    text = apply_sh.read_text()
    assert "#!/usr/bin/env bash" in text
    assert "st-TEST" in text
    assert "tr-002" in text


def test_readme_includes_winner_metadata(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=1)
    readme = (result.out_dir / "README.md").read_text()
    assert "st-TEST" in readme
    assert "tr-002" in readme
    assert "200.00" in readme  # score
    assert "max_num_seqs" in readme
    # workload-level metrics surfaced
    assert "throughput_avg" in readme
    assert "ttft_p99" in readme


def test_export_top_2_picks_second_best(seeded_db, tmp_path):
    out = tmp_path / "results"
    result = export_winner("st-TEST", db_path=seeded_db, out_dir=out, rank=2)
    assert result.trial_id == "tr-003"
    assert result.score == 150.0


def test_export_unknown_study_raises(seeded_db, tmp_path):
    with pytest.raises(ValueError, match="study not found"):
        export_winner("st-DOES-NOT-EXIST", db_path=seeded_db, out_dir=tmp_path, rank=1)


def test_export_rank_too_high_raises(seeded_db, tmp_path):
    with pytest.raises(ValueError, match="cannot export rank=5"):
        export_winner("st-TEST", db_path=seeded_db, out_dir=tmp_path, rank=5)
