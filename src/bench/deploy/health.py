"""Shared health + warmup probes (OpenAI-compatible endpoints)."""

from __future__ import annotations

import time
from typing import Any

import requests

from bench.deploy.base import HealthReport


def probe_openai_models(url: str, timeout_s: float = 5.0) -> HealthReport:
    """Query /v1/models; success if the body contains a non-empty data list."""
    base = url.rstrip("/")
    target = base if base.endswith("/v1/models") else f"{base.rsplit('/v1', 1)[0]}/v1/models"
    t0 = time.time()
    try:
        r = requests.get(target, timeout=timeout_s)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
        ok = len(data) > 0
        return HealthReport(
            ready=ok,
            latency_ms=(time.time() - t0) * 1000.0,
            detail=f"{len(data)} model(s)" if ok else "empty data",
        )
    except Exception as e:  # noqa: BLE001
        return HealthReport(ready=False, latency_ms=(time.time() - t0) * 1000.0, detail=str(e))


def warmup_one_token(url: str, model: str, *, timeout_s: float = 30.0) -> HealthReport:
    """Fire one max_tokens=1 completion so the next timed request is post-warmup."""
    base = url.rstrip("/")
    target = f"{base.rsplit('/v1', 1)[0]}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    t0 = time.time()
    try:
        r = requests.post(target, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return HealthReport(
            ready=True,
            latency_ms=(time.time() - t0) * 1000.0,
            detail="warmup ok",
        )
    except Exception as e:  # noqa: BLE001
        return HealthReport(ready=False, latency_ms=(time.time() - t0) * 1000.0, detail=str(e))
