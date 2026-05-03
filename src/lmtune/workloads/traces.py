"""프로덕션 trace replay.

대상:
- BurstGPT (arXiv:2401.17644) — CSV: `Timestamp,Model,Request tokens,Response tokens,Total tokens,...`
- ServeGen (arXiv:2505.09999) — JSONL: `{t,in,out,...}`

파일 형식을 자동 감지 (확장자) 하며, 불명확하면 `format=` 으로 강제.
replay_speed > 1 이면 빠르게 재생 (timestamp / speed).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TraceFormat = Literal["burstgpt", "servegen", "auto"]


@dataclass
class TraceRecord:
    offset_sec: float                     # trace 시작 기준 상대 시각
    input_tokens: int
    output_tokens: int
    meta: dict | None = None


class TraceReplay:
    def __init__(
        self,
        path: str | Path,
        fmt: TraceFormat = "auto",
        replay_speed: float = 1.0,
    ):
        self.path = Path(path)
        self.replay_speed = max(replay_speed, 1e-6)
        self.fmt = fmt if fmt != "auto" else self._detect_format()

    def _detect_format(self) -> TraceFormat:
        if self.path.suffix.lower() == ".csv":
            return "burstgpt"
        if self.path.suffix.lower() in {".jsonl", ".ndjson"}:
            return "servegen"
        raise ValueError(f"cannot auto-detect trace format for {self.path}")

    def __iter__(self) -> Iterable[TraceRecord]:
        if self.fmt == "burstgpt":
            yield from self._iter_burstgpt()
        elif self.fmt == "servegen":
            yield from self._iter_servegen()
        else:
            raise ValueError(f"unsupported trace format: {self.fmt}")

    def _iter_burstgpt(self):
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            t0: float | None = None
            for row in reader:
                try:
                    ts = float(row.get("Timestamp") or row.get("timestamp") or 0)
                    inp = int(row.get("Request tokens") or row.get("request_tokens") or 0)
                    out = int(row.get("Response tokens") or row.get("response_tokens") or 0)
                except (TypeError, ValueError):
                    continue
                if t0 is None:
                    t0 = ts
                yield TraceRecord(
                    offset_sec=(ts - t0) / self.replay_speed,
                    input_tokens=inp,
                    output_tokens=out,
                    meta={k: v for k, v in row.items() if k not in {"Timestamp", "Request tokens", "Response tokens"}},
                )

    def _iter_servegen(self):
        with self.path.open("r", encoding="utf-8") as f:
            t0: float | None = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = float(obj.get("t") or obj.get("timestamp") or 0)
                if t0 is None:
                    t0 = ts
                yield TraceRecord(
                    offset_sec=(ts - t0) / self.replay_speed,
                    input_tokens=int(obj.get("in") or obj.get("input_tokens") or 0),
                    output_tokens=int(obj.get("out") or obj.get("output_tokens") or 0),
                    meta={k: v for k, v in obj.items() if k not in {"t", "in", "out"}},
                )


def load_trace(path: str | Path, **kwargs) -> list[TraceRecord]:
    return list(TraceReplay(path, **kwargs))
