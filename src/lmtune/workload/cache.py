"""WorkloadProvider 결과 cache — fingerprint + TTL.

사용자가 같은 source 를 여러번 호출할 때 매번 lm-workloads 가 운영 trace 를
다시 분석하지 않도록 결과 yaml 을 cache. ``--refresh-cache`` 로 강제 재실행.

기본 위치: ``~/.lmtune/cache/workload/<fingerprint>.yaml``.
TTL 기본 1h (운영 트래픽 변동 빠르므로 보수적).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from lmtune.workload.providers.base import WorkloadProvider

DEFAULT_TTL_SECONDS = 3600  # 1h
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("LMTUNE_CACHE_ROOT", str(Path.home() / ".lmtune" / "cache"))
)


def cache_path(provider: WorkloadProvider, *, root: Path | None = None) -> Path:
    root = root or DEFAULT_CACHE_ROOT
    return root / "workload" / f"{provider.fingerprint()}.yaml"


def load_cached(
    provider: WorkloadProvider, *, ttl_sec: int = DEFAULT_TTL_SECONDS, root: Path | None = None
):
    """Cached WorkloadSpec 반환. 없거나 stale 이면 None."""
    p = cache_path(provider, root=root)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl_sec:
        return None
    from lmtune.contracts.workload_spec import WorkloadSpec

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return WorkloadSpec.model_validate(data)


def save_cache(provider: WorkloadProvider, spec, *, root: Path | None = None) -> Path:
    """WorkloadSpec → yaml file. 반환: 저장 경로."""
    p = cache_path(provider, root=root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return p


def provide_with_cache(
    provider: WorkloadProvider,
    *,
    refresh: bool = False,
    ttl_sec: int = DEFAULT_TTL_SECONDS,
    root: Path | None = None,
):
    """Cache lookup → fallback to provider.provide() → cache 갱신."""
    if not refresh:
        cached = load_cached(provider, ttl_sec=ttl_sec, root=root)
        if cached is not None:
            return cached, True  # (spec, from_cache)
    spec = provider.provide(refresh=refresh)
    save_cache(provider, spec, root=root)
    return spec, False
