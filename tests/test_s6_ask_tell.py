"""Phase S6 — `bench search ask` / `tell` 외부 통합 단위 테스트.

flow:
  1. bench search start --max-trials 0 으로 study 만 생성
  2. bench search ask 가 trial_id + params JSON 반환
  3. bench search tell 이 metrics-json 받아 trial 을 completed 로 update
  4. bench search status 가 top-K 에 반영
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def isolated_db(tmp_path: Path) -> Path:
    return tmp_path / "lmtune.duckdb"


@pytest.fixture
def smoke_space(tmp_path: Path) -> Path:
    p = tmp_path / "space.yaml"
    p.write_text(
        """
apiVersion: lmtune/search/v1alpha1
kind: SearchSpace
name: s6-ask-tell-smoke
axes:
  max_num_seqs: {type: categorical, values: [32, 64, 128]}
  enable_prefix_caching: {type: bool}
""".strip(),
    )
    return p


_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_LMTUNE = _REPO_ROOT / ".venv" / "bin" / "lmtune"
_LMTUNE_BIN = str(_VENV_LMTUNE) if _VENV_LMTUNE.exists() else "lmtune"


def _bench(args: list[str], db: Path, **kwargs):
    import os
    base_path = os.environ.get("PATH", "")
    venv_bin = _VENV_LMTUNE.parent
    if venv_bin.exists():
        base_path = f"{venv_bin}:{base_path}"
    env = {"LMTUNE_DB": str(db), "PATH": base_path}
    return subprocess.run(
        [_LMTUNE_BIN, *args],
        capture_output=True,
        text=True,
        env=env,
        **kwargs,
    )


def _start_empty_study(db: Path, space: Path, name: str = "s6-test") -> str:
    r = _bench(
        ["search", "start", "--space", str(space), "--strategy", "random",
         "--max-trials", "0", "--name", name, "--dry-run"],
        db=db,
    )
    assert r.returncode == 0, r.stderr
    # parse "study_id: st-XXXX" from stdout
    import re
    m = re.search(r"study_id:\s*(st-[A-Z0-9]+)", r.stdout)
    assert m, r.stdout
    return m.group(1)


def test_ask_returns_valid_json(isolated_db, smoke_space):
    sid = _start_empty_study(isolated_db, smoke_space)

    r = _bench(["search", "ask", sid], db=isolated_db)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["study_id"] == sid
    assert payload["trial_id"].startswith("tr-")
    assert isinstance(payload["seq"], int) and payload["seq"] >= 1
    assert "max_num_seqs" in payload["params"]
    assert payload["params"]["max_num_seqs"] in [32, 64, 128]
    assert isinstance(payload["params"]["enable_prefix_caching"], bool)


def test_tell_records_completed_trial(isolated_db, smoke_space, tmp_path):
    sid = _start_empty_study(isolated_db, smoke_space)

    # ask
    r_ask = _bench(["search", "ask", sid], db=isolated_db)
    payload = json.loads(r_ask.stdout)
    trial_id = payload["trial_id"]

    # tell with sample metrics-json
    metrics_file = tmp_path / "result.json"
    metrics_file.write_text(json.dumps({
        "total_score": 1289.5,
        "metrics": {
            "throughput_avg_short": 145.2,
            "ttft_p99_short": 192.5,
        },
        "accepted": True,
    }))
    r_tell = _bench(
        ["search", "tell", sid, "--trial", trial_id, "--metrics-json", str(metrics_file)],
        db=isolated_db,
    )
    assert r_tell.returncode == 0, r_tell.stderr
    assert "completed" in r_tell.stdout

    # status — top trial 에 우리 결과가 보여야 함
    r_status = _bench(["search", "status", sid], db=isolated_db)
    assert r_status.returncode == 0, r_status.stderr
    assert "1289" in r_status.stdout, r_status.stdout
    assert "completed=1" in r_status.stdout


def test_tell_pruned_on_not_accepted(isolated_db, smoke_space, tmp_path):
    sid = _start_empty_study(isolated_db, smoke_space)

    r_ask = _bench(["search", "ask", sid], db=isolated_db)
    trial_id = json.loads(r_ask.stdout)["trial_id"]

    metrics_file = tmp_path / "rejected.json"
    metrics_file.write_text(json.dumps({
        "total_score": 0.0,
        "metrics": {"throughput_avg_short": 0.0},
        "accepted": False,
    }))
    r_tell = _bench(
        ["search", "tell", sid, "--trial", trial_id, "--metrics-json", str(metrics_file)],
        db=isolated_db,
    )
    assert r_tell.returncode == 0, r_tell.stderr
    # status 의 pruned 카운트 확인
    r_status = _bench(["search", "status", sid], db=isolated_db)
    assert r_status.returncode == 0
    assert "pruned=1" in r_status.stdout, r_status.stdout


def test_three_iterations_warmstart(isolated_db, smoke_space, tmp_path):
    """ask → tell → ask 가 정상 누적. 두 번째 ask 가 첫 결과를 warmstart 로 활용."""
    sid = _start_empty_study(isolated_db, smoke_space)

    seen_trials: list[str] = []
    for i, score in enumerate([1100.0, 1300.0, 1200.0]):
        r_ask = _bench(["search", "ask", sid], db=isolated_db)
        assert r_ask.returncode == 0
        tid = json.loads(r_ask.stdout)["trial_id"]
        assert tid not in seen_trials, "duplicate trial_id from ask"
        seen_trials.append(tid)

        rf = tmp_path / f"r{i}.json"
        rf.write_text(json.dumps({
            "total_score": score,
            "metrics": {"throughput_avg_short": score / 10},
            "accepted": True,
        }))
        r_tell = _bench(
            ["search", "tell", sid, "--trial", tid, "--metrics-json", str(rf)],
            db=isolated_db,
        )
        assert r_tell.returncode == 0

    # 3 trial 모두 completed
    r_status = _bench(["search", "status", sid], db=isolated_db)
    assert "completed=3" in r_status.stdout, r_status.stdout
    # top-1 은 score=1300 (최고치)
    assert "1300" in r_status.stdout
