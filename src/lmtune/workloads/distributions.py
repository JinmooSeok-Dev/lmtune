"""토큰 길이/요청 간격 샘플링용 분포 생성기.

BurstGPT (arXiv:2401.17644) 는 입력 Zipf·출력 bimodal 을 보고했고, 여기서는
외부 라이브러리 의존 없이 `random` 만으로 근사 구현한다. 정밀 fit 이 필요하면
`scipy.stats` 로 교체한다.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

DistKind = Literal["constant", "uniform", "normal", "zipf", "bimodal", "lognormal"]


@dataclass
class DistributionSampler:
    kind: DistKind = "constant"
    # constant / normal 평균
    mean: float = 0.0
    stddev: float = 0.0
    # uniform 범위
    low: float | None = None
    high: float | None = None
    # zipf: s (shape) + N (clip max)
    zipf_s: float = 1.2
    zipf_clip: int = 16384
    # bimodal: two-mode GMM (50:50 default)
    modes: tuple[tuple[float, float], tuple[float, float]] | None = None
    mode_weight: float = 0.5
    # lognormal
    ln_mu: float = 0.0
    ln_sigma: float = 1.0

    def sample(self, rng: random.Random | None = None) -> float:
        r = rng or random
        if self.kind == "constant":
            return self.mean
        if self.kind == "uniform":
            return r.uniform(self.low or 0.0, self.high or 1.0)
        if self.kind == "normal":
            return r.gauss(self.mean, self.stddev)
        if self.kind == "zipf":
            return min(sample_zipf(self.zipf_s, r), self.zipf_clip)
        if self.kind == "bimodal":
            modes = self.modes or ((self.mean, self.stddev or 1.0), (self.mean * 3, self.stddev or 1.0))
            (m1, s1), (m2, s2) = modes
            return r.gauss(m1, s1) if r.random() < self.mode_weight else r.gauss(m2, s2)
        if self.kind == "lognormal":
            return r.lognormvariate(self.ln_mu, self.ln_sigma)
        raise ValueError(f"unknown distribution: {self.kind}")

    def sample_n(self, n: int, rng: random.Random | None = None) -> list[float]:
        return [self.sample(rng) for _ in range(n)]


def sample_zipf(s: float, rng: random.Random | None = None) -> int:
    """Zipf 분포 샘플 (rejection 방식). 최소 1 반환."""
    r = rng or random
    while True:
        u = r.random()
        x = int(math.floor((1 - u) ** (-1 / (s - 1)))) if s > 1 else 1
        if x >= 1:
            return x


def sample_bimodal(mean_small: float, mean_large: float, weight_small: float = 0.5,
                   rng: random.Random | None = None) -> float:
    r = rng or random
    return r.gauss(mean_small, mean_small * 0.2) if r.random() < weight_small else r.gauss(mean_large, mean_large * 0.2)


def fit_summary(values: Iterable[float]) -> dict[str, float]:
    xs = [float(v) for v in values]
    if not xs:
        return {}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = math.sqrt(var)
    xs_sorted = sorted(xs)
    q1 = xs_sorted[n // 4]
    q3 = xs_sorted[(3 * n) // 4]
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean else 0.0,
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
        "p50": xs_sorted[n // 2],
        "iqr": q3 - q1,
    }
