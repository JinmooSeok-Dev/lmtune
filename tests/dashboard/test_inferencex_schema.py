"""Phase W — InferenceX-app schema compatibility lock.

These names must not change without bumping a version envelope around
the JSON. The names mirror SemiAnalysisAI/InferenceX-app's
`packages/db/queries/` shapes; renaming them means any downstream fork
breaks until field aliases land.
"""

from __future__ import annotations

from lmtune.visualization.dashboard.schemas import (
    PerfHistory,
    PerfHistoryEntry,
    StudiesIndex,
    StudyCard,
    ThroughputVsLatency,
    TrialPoint,
)


def test_study_card_field_names_locked():
    s = StudyCard(
        study_id="st-x", name="x", strategy="tpe", direction="maximize",
        status="completed", n_trials=10, n_completed=8, top_score=123.4,
        endpoint_slug="ep-1", profile_slugs=["short"], created_at="2026-05-02",
        finished_at="2026-05-02",
    )
    dump = s.model_dump()
    assert set(dump.keys()) == {
        "study_id", "name", "strategy", "direction", "status",
        "endpoint_slug", "profile_slugs", "n_trials", "n_completed",
        "top_score", "created_at", "finished_at",
    }


def test_throughput_vs_latency_field_names_locked():
    t = ThroughputVsLatency(
        study_id="st-x",
        points=[TrialPoint(
            trial_id="tr-1", seq=1, score=99.0,
            params={"max_num_seqs": 64},
            metrics={"throughput_tok_avg.short": 200.0,
                     "ttft_p99.short": 230.0},
        )],
    )
    dump = t.model_dump()
    assert set(dump.keys()) == {
        "study_id", "model_id", "framework", "hardware_id",
        "workload", "points",
    }
    pt = dump["points"][0]
    assert set(pt.keys()) == {"trial_id", "seq", "score", "params", "metrics"}


def test_perf_history_entry_field_names_locked():
    e = PerfHistoryEntry(
        config_keys=["70b-fp8-*"],
        description=["+12% throughput"],
        pr_link="https://example.com",
        evals_only=False,
    )
    dump = e.model_dump()
    assert set(dump.keys()) == {
        "config_keys", "description", "pr_link", "evals_only", "landed_at",
    }


def test_extra_fields_forbidden():
    """`extra='forbid'` is the InferenceX AGENTS.md rule we adopt."""
    import pydantic
    try:
        StudyCard(study_id="x", name="y", strategy="z", direction="maximize",
                  status="ok", n_trials=0, n_completed=0,
                  unknown_field="oops")  # type: ignore[call-arg]
    except pydantic.ValidationError:
        return
    raise AssertionError("expected ValidationError on extra field")


def test_studies_index_round_trip():
    idx = StudiesIndex(studies=[StudyCard(
        study_id="x", name="x", strategy="random", direction="maximize",
        status="running", n_trials=0, n_completed=0,
    )])
    dump = idx.model_dump()
    assert "studies" in dump and len(dump["studies"]) == 1
    re = StudiesIndex.model_validate(dump)
    assert re.studies[0].study_id == "x"


def test_perf_history_default_empty_list():
    h = PerfHistory()
    assert h.entries == []
    assert h.model_dump()["entries"] == []
