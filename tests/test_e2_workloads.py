from __future__ import annotations

import random

import pytest

from lmtune.profiles import ProfileSpec, TraceWorkload
from lmtune.workloads import (
    ArrivalPattern,
    ArrivalScheduler,
    DistributionSampler,
    TraceReplay,
    sample_zipf,
)
from lmtune.workloads.arrival import empirical_rate
from lmtune.workloads.datasets import KNOWN_DATASETS, DatasetLoader

# ---------- Distributions ----------


def test_zipf_positive_integer():
    rng = random.Random(0)
    xs = [sample_zipf(1.5, rng) for _ in range(200)]
    assert all(x >= 1 for x in xs)
    # heavy-tailed: 드물지만 큰 값 존재
    assert max(xs) > 10


def test_distribution_sampler_bimodal_bimodal():
    rng = random.Random(0)
    s = DistributionSampler(kind="bimodal", modes=((50, 5), (500, 20)), mode_weight=0.5)
    vals = s.sample_n(1000, rng)
    # 두 peak 모두 샘플링되어야 함
    assert sum(1 for v in vals if v < 100) > 200
    assert sum(1 for v in vals if v > 400) > 200


def test_distribution_sampler_constant():
    s = DistributionSampler(kind="constant", mean=42.0)
    assert all(v == 42.0 for v in s.sample_n(5))


# ---------- Arrival patterns ----------


def test_arrival_constant_rate():
    pat = ArrivalPattern(kind="constant", rate=10, duration_sec=2)
    times = list(ArrivalScheduler(pat))
    # 10 req/s × 2s ≈ 20개, 간격 0.1
    assert 18 <= len(times) <= 22
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(abs(g - 0.1) < 1e-6 for g in gaps)


def test_arrival_poisson_count_range():
    pat = ArrivalPattern(kind="poisson", rate=20, duration_sec=3)
    times = list(ArrivalScheduler(pat, seed=1))
    # 평균 60, 여유 있게 30~100 사이
    assert 30 <= len(times) <= 100
    assert all(b >= a for a, b in zip(times, times[1:], strict=False))


def test_arrival_diurnal_peak_gt_valley():
    pat = ArrivalPattern(
        kind="diurnal", peak_rate=20, valley_rate=1, period_sec=10, duration_sec=10
    )
    times = list(ArrivalScheduler(pat, seed=0))
    rates = empirical_rate(times, window_sec=1.0)
    assert rates
    peak = max(r for _, r in rates)
    valley = min(r for _, r in rates)
    assert peak > valley * 3  # 34.6x 까진 안 가도 차이 뚜렷


def test_arrival_burst_alternates():
    pat = ArrivalPattern(kind="burst", burst_rate=50, burst_sec=1.0, idle_sec=2.0, duration_sec=6)
    times = list(ArrivalScheduler(pat, seed=0))
    # burst 단계에만 dense → 전체 간격이 bimodal 근사
    assert len(times) > 20


# ---------- Trace replay ----------


def test_trace_burstgpt_csv_roundtrip(tmp_path):
    csv_file = tmp_path / "trace.csv"
    csv_file.write_text(
        "Timestamp,Model,Request tokens,Response tokens,Total tokens\n"
        "1000,gpt-4,200,50,250\n"
        "1001,gpt-4,400,30,430\n"
        "1005,gpt-4,100,120,220\n"
    )
    records = list(TraceReplay(csv_file, fmt="burstgpt", replay_speed=1.0))
    assert [r.offset_sec for r in records] == [0.0, 1.0, 5.0]
    assert records[1].input_tokens == 400
    assert records[1].output_tokens == 30


def test_trace_servegen_jsonl(tmp_path):
    jsonl = tmp_path / "trace.jsonl"
    jsonl.write_text('{"t": 100, "in": 50, "out": 20}\n{"t": 100.5, "in": 200, "out": 80}\n')
    records = list(TraceReplay(jsonl, fmt="servegen"))
    assert records[0].offset_sec == 0.0
    assert records[1].offset_sec == 0.5
    assert records[1].input_tokens == 200


def test_trace_auto_detect(tmp_path):
    csv_file = tmp_path / "x.csv"
    csv_file.write_text("Timestamp,Request tokens,Response tokens\n10,1,2\n")
    replay = TraceReplay(csv_file)
    assert replay.fmt == "burstgpt"


# ---------- Dataset catalogue ----------


def test_known_datasets_exist():
    assert "safim" in KNOWN_DATASETS
    assert "swe_bench_verified" in KNOWN_DATASETS
    assert "burstgpt" not in KNOWN_DATASETS  # trace, not HF dataset


def test_dataset_loader_from_slug_sets_fields():
    loader = DatasetLoader.from_slug("safim")
    assert loader.dataset_id.endswith("safim")
    assert loader.subset == "block"


def test_dataset_loader_graceful_missing_datasets(monkeypatch):
    # `datasets` 패키지 미설치 시 NotImplementedError 가 분명한 메시지로 올라와야 함
    loader = DatasetLoader(dataset_id="x/y")
    import lmtune.workloads.datasets as m

    def fake_load(*args, **kwargs):
        raise NotImplementedError("HF datasets missing")

    monkeypatch.setattr(m, "load_hf_dataset", fake_load)
    with pytest.raises(NotImplementedError):
        list(loader.iter_samples(limit=1))


# ---------- Profile parsing w/ arrival + trace ----------


def test_profile_accepts_arrival_and_distributions():
    p = ProfileSpec.model_validate(
        {
            "slug": "diurnal",
            "name": "diurnal",
            "stage": 1,
            "runner": "raw_openai",
            "mode": "concurrency",
            "workload": {
                "synthetic_input_tokens_mean": 2000,
                "output_tokens_mean": 100,
                "concurrency": 8,
                "request_count": 200,
                "arrival": {
                    "kind": "diurnal",
                    "peak_rate": 50,
                    "valley_rate": 5,
                    "period_sec": 600,
                    "duration_sec": 1800,
                },
                "input_dist": {"kind": "zipf", "zipf_s": 1.3, "zipf_clip": 16000},
                "output_dist": {
                    "kind": "bimodal",
                    "modes": [[100, 20], [1000, 100]],
                    "mode_weight": 0.6,
                },
            },
        }
    )
    assert p.workload.arrival.kind == "diurnal"
    assert p.workload.input_dist.kind == "zipf"
    assert p.workload.output_dist.modes == [[100, 20], [1000, 100]]


def test_trace_workload_fields():
    p = ProfileSpec.model_validate(
        {
            "slug": "rep",
            "name": "rep",
            "stage": 1,
            "runner": "raw_openai",
            "mode": "concurrency",
            "workload": {
                "source": "trace",
                "trace_path": "/data/burstgpt.csv",
                "trace_format": "burstgpt",
                "replay_speed": 5.0,
                "sample_count": 100,
                "concurrency": 4,
                "request_count": 100,
            },
        }
    )
    assert isinstance(p.workload, TraceWorkload)
    assert p.workload.replay_speed == 5.0
    assert p.workload.trace_format == "burstgpt"
