"""요청 도착 시각 생성기.

DynamoLLM (HPCA 2025) 는 peak:valley 34.6× diurnal 파형을 보고했다.
여기서는 순수 Python 으로 도착 시각 시퀀스를 생성하는 pattern generator 만 제공하고,
실제 dispatch 는 runner 쪽에서 asyncio sleep-until 으로 처리한다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Literal


ArrivalKind = Literal["constant", "poisson", "diurnal", "burst", "replay"]


@dataclass
class ArrivalPattern:
    kind: ArrivalKind = "constant"
    rate: float = 1.0                     # req/s (constant / poisson 평균)
    duration_sec: float = 60.0

    # diurnal
    peak_rate: float | None = None
    valley_rate: float | None = None
    period_sec: float = 3600.0            # 1h

    # burst
    burst_rate: float | None = None
    burst_sec: float | None = None
    idle_sec: float | None = None

    # replay
    replay_offsets_sec: list[float] = field(default_factory=list)


class ArrivalScheduler:
    """주어진 ArrivalPattern 으로 도착 시각(seconds since t0) iterator 생성."""

    def __init__(self, pattern: ArrivalPattern, seed: int = 42):
        self.p = pattern
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[float]:
        kind = self.p.kind
        if kind == "constant":
            yield from self._constant()
        elif kind == "poisson":
            yield from self._poisson()
        elif kind == "diurnal":
            yield from self._diurnal()
        elif kind == "burst":
            yield from self._burst()
        elif kind == "replay":
            for off in self.p.replay_offsets_sec:
                yield float(off)
        else:
            raise ValueError(f"unknown arrival kind: {kind}")

    def _constant(self):
        if self.p.rate <= 0:
            return
        dt = 1.0 / self.p.rate
        t = 0.0
        while t < self.p.duration_sec:
            yield t
            t += dt

    def _poisson(self):
        if self.p.rate <= 0:
            return
        t = 0.0
        while t < self.p.duration_sec:
            # 지수 간격
            gap = -math.log(1 - self.rng.random()) / self.p.rate
            t += gap
            if t < self.p.duration_sec:
                yield t

    def _diurnal(self):
        peak = self.p.peak_rate or self.p.rate * 2
        valley = self.p.valley_rate or self.p.rate / 2
        period = max(self.p.period_sec, 1.0)
        t = 0.0
        while t < self.p.duration_sec:
            phase = (t % period) / period
            # sine wave: valley..peak..valley
            rate = valley + (peak - valley) * (0.5 - 0.5 * math.cos(2 * math.pi * phase))
            if rate <= 0:
                t += 1.0
                continue
            gap = -math.log(1 - self.rng.random()) / rate
            t += gap
            if t < self.p.duration_sec:
                yield t

    def _burst(self):
        brate = self.p.burst_rate or self.p.rate * 10
        bsec = self.p.burst_sec or 10.0
        isec = self.p.idle_sec or 30.0
        t = 0.0
        in_burst = True
        phase_end = bsec
        while t < self.p.duration_sec:
            if in_burst:
                gap = -math.log(1 - self.rng.random()) / max(brate, 1e-6)
                t += gap
                if t >= phase_end:
                    in_burst = False
                    phase_end = t + isec
                    continue
                if t < self.p.duration_sec:
                    yield t
            else:
                t = phase_end
                in_burst = True
                phase_end = t + bsec


def empirical_rate(arrivals: list[float], window_sec: float = 1.0) -> list[tuple[float, float]]:
    """도착 시각 리스트 → (window 중심 시각, 평균 req/s) — 검증·플롯용."""
    if not arrivals:
        return []
    out: list[tuple[float, float]] = []
    t0 = arrivals[0]
    t_end = arrivals[-1]
    w = window_sec
    t = t0
    i = 0
    while t < t_end:
        j = i
        while j < len(arrivals) and arrivals[j] < t + w:
            j += 1
        out.append((t + w / 2, (j - i) / w))
        t += w
        i = j
    return out
