"""Dashboard 신규 기능 단위 테스트.

- Axis importance: sklearn-기반, 부족한 trial 시 None
- Pareto front: 비지배 set 계산 정확성
- n_axes: space_yaml 에서 axis 개수 카운트
- spaces.html: search-space + env-profile 카탈로그 렌더
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lmtune.storage import DuckDBStore
from lmtune.visualization.dashboard import build_dashboard
from lmtune.visualization.dashboard.build import (
    _compute_pareto,
    _count_axes_in_space_yaml,
    _safe_axis_importance,
    _summarize_env_profile,
    _summarize_search_space,
)
from lmtune.visualization.dashboard.schemas import TrialPoint


def _pt(seq, score, params, metrics):
    return TrialPoint(trial_id=f"t{seq}", seq=seq, score=score, params=params, metrics=metrics)


def test_count_axes_dict_form():
    space_yaml = yaml.safe_dump({"axes": {"a": {"type": "int"}, "b": {"type": "bool"}}})
    assert _count_axes_in_space_yaml(space_yaml) == 2


def test_count_axes_list_form():
    space_yaml = yaml.safe_dump({"axes": [{"name": "x"}, {"name": "y"}, {"name": "z"}]})
    assert _count_axes_in_space_yaml(space_yaml) == 3


def test_count_axes_none_for_empty_or_invalid():
    assert _count_axes_in_space_yaml(None) is None
    assert _count_axes_in_space_yaml("") is None
    assert _count_axes_in_space_yaml("not yaml: [unclosed") is None
    assert _count_axes_in_space_yaml(yaml.safe_dump({"name": "x"})) is None


def test_pareto_front_picks_nondominated():
    # x=ttft (lower=better), y=throughput (higher=better)
    pts = [
        _pt(1, 1.0, {}, {"ttft_p99.short": 100.0, "throughput_tok_avg.short": 50.0}),
        _pt(
            2, 1.0, {}, {"ttft_p99.short": 200.0, "throughput_tok_avg.short": 30.0}
        ),  # dominated by 1
        _pt(3, 1.0, {}, {"ttft_p99.short": 80.0, "throughput_tok_avg.short": 70.0}),  # dominates 1
        _pt(
            4, 1.0, {}, {"ttft_p99.short": 60.0, "throughput_tok_avg.short": 40.0}
        ),  # extreme low ttft
    ]
    front = _compute_pareto(pts)
    seqs = sorted(p["seq"] for p in front)
    # 3 dominates 1 and 2; 4 is extreme low ttft. So front = {3, 4}
    assert seqs == [3, 4]
    # sorted by x ascending
    assert front[0]["x"] <= front[1]["x"]


def test_pareto_empty_for_no_metrics():
    pts = [_pt(1, 1.0, {}, {})]
    assert _compute_pareto(pts) == []


def test_axis_importance_returns_none_for_few_trials():
    pts = [_pt(i, float(i), {"a": i}, {}) for i in range(3)]
    out = _safe_axis_importance(pts)
    # axis_importance() needs ≥ 5 completed
    assert out is None


def test_axis_importance_picks_signal_over_noise():
    # axis `a` correlates strongly with score, axis `b` is constant noise
    pts: list[TrialPoint] = []
    for i in range(20):
        pts.append(_pt(i, float(i * 5), {"a": i, "b": 7}, {}))
    out = _safe_axis_importance(pts)
    assert out is not None
    by_axis = {r["axis"]: r["importance"] for r in out}
    assert by_axis["a"] > by_axis["b"]


def test_summarize_search_space(tmp_path: Path):
    f = tmp_path / "test_space.yaml"
    f.write_text(
        yaml.safe_dump(
            {
                "name": "test-space",
                "description": "demo",
                "axes": {
                    "x": {
                        "type": "int",
                        "low": 0,
                        "high": 10,
                        "cost_tier": 2,
                        "apply_via": "container_env",
                    },
                    "y": {"type": "categorical", "values": ["a", "b", "c"]},
                },
                "feasibility_constraints": ["x > 0", "x < 10"],
                "default_pruner": "anova",
            }
        )
    )
    s = _summarize_search_space(f)
    assert s is not None
    assert s["n_axes"] == 2
    assert s["n_constraints"] == 2
    assert s["default_pruner"] == "anova"
    axis_names = [a["name"] for a in s["axes"]]
    assert "x" in axis_names and "y" in axis_names
    x_axis = next(a for a in s["axes"] if a["name"] == "x")
    assert x_axis["values"] == "[0, 10]"
    assert x_axis["apply_via"] == "container_env"


def test_summarize_env_profile(tmp_path: Path):
    f = tmp_path / "test_profile.yaml"
    f.write_text(
        yaml.safe_dump(
            {
                "name": "tp-nvl",
                "priority": 1,
                "description": "nvl",
                "applies_when": {"intra_node_type": "nvlink"},
                "env_locked": {"NCCL_P2P_LEVEL": "NVL", "NCCL_IB_DISABLE": "1"},
                "env_tunable": [
                    {"name": "NCCL_BUFFSIZE", "kind": "categorical", "values": [1, 2, 4]},
                ],
            }
        )
    )
    p = _summarize_env_profile(f)
    assert p is not None
    assert p["name"] == "tp-nvl"
    assert p["priority"] == 1
    assert p["n_locked"] == 2
    assert p["n_tunable"] == 1
    assert p["env_tunable"][0]["name"] == "NCCL_BUFFSIZE"


@pytest.fixture
def seeded_db_with_axes(tmp_path: Path) -> Path:
    """5+ trial 로 importance 계산 가능한 fixture."""
    db_path = tmp_path / "lmtune.duckdb"
    store = DuckDBStore(db_path)
    space_yaml = yaml.safe_dump(
        {
            "name": "test",
            "axes": {
                "x": {"type": "int", "low": 0, "high": 10},
                "y": {"type": "bool"},
            },
        }
    )
    store.record_study(
        study_id="st-AXES",
        name="axes-test",
        strategy="tpe",
        metric_name="total_score",
        direction="maximize",
        space_yaml=space_yaml,
        endpoint_slug="b200-vllm",
        profile_slugs=["short"],
        notes="",
    )
    for i in range(8):
        tid = f"tr-{i:03d}"
        score = float(i * 10 + 1)  # x correlates with score
        store.record_trial(
            trial_id=tid,
            study_id="st-AXES",
            seq=i,
            params={"x": i, "y": (i % 2 == 0)},
            status="completed",
            score=score,
            backend="inline",
            completed=True,
        )
        store.record_trial_metrics(
            tid,
            {
                ("throughput_tok_avg", "short"): score * 2.0,
                ("ttft_p99", "short"): 100.0 - i * 5,
            },
        )
    return db_path


def test_build_dashboard_includes_axis_extras_in_html(seeded_db_with_axes, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db_with_axes, out_dir=out)
    html = (out / "studies" / "st-AXES.html").read_text()
    # axis importance section
    assert "Axis importance" in html
    assert "axis-importance-bar" in html
    # parallel coords
    assert "parallel-coords" in html
    assert "Parallel coordinates" in html
    # n_axes badge in header
    assert '<span class="text-slate-400">axes</span>' in html


def test_build_dashboard_with_search_spaces_dir_renders_spaces_page(seeded_db_with_axes, tmp_path):
    ss_dir = tmp_path / "spaces"
    ss_dir.mkdir()
    (ss_dir / "demo.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "demo-space",
                "axes": {"a": {"type": "int", "low": 0, "high": 5}},
            }
        )
    )
    ep_dir = tmp_path / "profiles"
    ep_dir.mkdir()
    (ep_dir / "host.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "host-default",
                "priority": 0,
                "applies_when": {"always": True},
                "env_locked": {"X": "1"},
            }
        )
    )
    out = tmp_path / "dash"
    written = build_dashboard(
        db_path=seeded_db_with_axes,
        out_dir=out,
        search_spaces_dir=ss_dir,
        env_profiles_dir=ep_dir,
    )
    assert "spaces.html" in written
    spaces_html = (out / "spaces.html").read_text()
    assert "Search-space catalog" in spaces_html
    assert "demo-space" in spaces_html
    assert "host-default" in spaces_html
    # index.html should link to spaces.html
    index_html = (out / "index.html").read_text()
    assert "spaces.html" in index_html
    assert "Search-space catalog" in index_html


def test_matrix_page_built(seeded_db_with_axes, tmp_path):
    out = tmp_path / "dash"
    written = build_dashboard(db_path=seeded_db_with_axes, out_dir=out)
    assert "matrix.html" in written
    html = (out / "matrix.html").read_text()
    assert "Model × HW matrix" in html
    # cell metric toggles
    assert "metric-toggle" in html
    assert 'data-metric="score"' in html
    assert 'data-metric="throughput"' in html
    assert 'data-metric="ttft_p99"' in html
    # the seeded study has model_id inferred from "b200-vllm" → unknown,
    # but the table itself must render at least once
    assert '<table id="matrix-table"' in html


def test_compare_page_includes_convergence_and_pareto_blocks(seeded_db_with_axes, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db_with_axes, out_dir=out)
    html = (out / "compare.html").read_text()
    assert "Cross-study running-best" in html
    assert "convergence-line" in html
    assert "Cross-study Pareto" in html
    assert "pareto-cross" in html


def test_study_page_includes_axis_pair_heatmap(seeded_db_with_axes, tmp_path):
    out = tmp_path / "dash"
    build_dashboard(db_path=seeded_db_with_axes, out_dir=out)
    html = (out / "studies" / "st-AXES.html").read_text()
    assert "Axis-pair score heatmap" in html
    assert "axis-pair-heatmap" in html
    assert "hm-x" in html
    assert "hm-y" in html


def test_study_page_shows_infeasible_count(tmp_path):
    """trial.error_msg containing FAIL: c* should be counted as infeasible."""
    db = tmp_path / "lmtune.duckdb"
    store = DuckDBStore(db)
    import yaml as _y

    space_yaml = _y.safe_dump({"axes": {"x": {"type": "int", "low": 0, "high": 5}}})
    store.record_study(
        study_id="st-INF",
        name="infeas-test",
        strategy="random",
        metric_name="score",
        direction="maximize",
        space_yaml=space_yaml,
        endpoint_slug="b200-vllm",
        profile_slugs=["short"],
        notes="",
    )
    # 5 completed + 3 pruned (2 infeasible, 1 generic)
    for i, (status, score, err) in enumerate(
        [
            ("completed", 10.0, None),
            ("completed", 20.0, None),
            ("completed", 15.0, None),
            ("completed", 30.0, None),
            ("completed", 25.0, None),
            ("pruned", None, "FAIL: c4_heads_div_tp"),
            ("pruned", None, "FAIL: c2_tp_single_node | WARN: c11_pcp_intra_pref"),
            ("pruned", None, "duplicate trial"),
        ]
    ):
        tid = f"tr-{i:03d}"
        store.record_trial(
            trial_id=tid,
            study_id="st-INF",
            seq=i,
            params={"x": i},
            status=status,
            score=score,
            backend="inline",
            completed=True,
            error=err,
        )
    out = tmp_path / "dash"
    build_dashboard(db_path=db, out_dir=out)
    html = (out / "studies" / "st-INF.html").read_text()
    assert "2 infeasible" in html


def test_build_dashboard_no_spaces_dir_no_spaces_page(seeded_db_with_axes, tmp_path):
    out = tmp_path / "dash"
    empty_ss = tmp_path / "empty_spaces"
    empty_ep = tmp_path / "empty_profiles"
    empty_ss.mkdir()
    empty_ep.mkdir()
    written = build_dashboard(
        db_path=seeded_db_with_axes,
        out_dir=out,
        search_spaces_dir=empty_ss,
        env_profiles_dir=empty_ep,
    )
    assert "spaces.html" not in written
    assert not (out / "spaces.html").exists()
