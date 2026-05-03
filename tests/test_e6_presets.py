from __future__ import annotations

from pathlib import Path

from lmtune.profiles import discover_profiles, load_profile

ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "configs/profiles/research"

EXPECTED_PRESETS = {
    "research-tokenomics-sdlc",
    "research-variance-10x",
    "research-agent-phase-breakdown",
    "research-token-snowball",
    "research-burstgpt-replay",
    "research-diurnal-34x",
    "research-prefix-cache-effect",
    "research-agents-md-impact",
    "research-eutb-evaluation",
}


def test_all_9_presets_exist():
    yamls = list(PRESETS_DIR.glob("*.yaml"))
    assert len(yamls) == 9


def test_presets_parse_and_have_expected_slugs():
    profiles = discover_profiles(PRESETS_DIR)
    slugs = {p.slug for p in profiles}
    assert slugs >= EXPECTED_PRESETS


def test_presets_carry_analysis_block():
    for p in discover_profiles(PRESETS_DIR):
        assert p.analysis is not None, f"{p.slug} missing analysis block"
        assert p.analysis.plots, f"{p.slug} has no plots"
        assert p.analysis.sinks, f"{p.slug} has no sinks"


def test_burstgpt_uses_arrival_and_distributions():
    p = load_profile(PRESETS_DIR / "burstgpt_replay.yaml")
    assert p.workload.arrival is not None
    assert p.workload.arrival.kind == "poisson"
    assert p.workload.input_dist.kind == "zipf"
    assert p.workload.output_dist.kind == "bimodal"


def test_diurnal_preset_has_peak_valley():
    p = load_profile(PRESETS_DIR / "diurnal_34x.yaml")
    assert p.workload.arrival.kind == "diurnal"
    assert p.workload.arrival.peak_rate == 35
    assert p.workload.arrival.valley_rate == 1


def test_eutb_preset_declares_derived_and_slo_check():
    p = load_profile(PRESETS_DIR / "eutb_evaluation.yaml")
    derived_names = {d.name for d in p.analysis.derived}
    assert "eutb" in derived_names
    slo_metrics = {c.metric for c in p.slo.resolved_checks()}
    assert "eutb" in slo_metrics


def test_variance_preset_runs_short():
    p = load_profile(PRESETS_DIR / "variance_10x.yaml")
    # repeat 로 돌리기 쉬운 작은 규모 유지
    assert p.workload.conversation_num <= 5
