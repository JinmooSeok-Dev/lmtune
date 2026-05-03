"""Warm-start — read past (params, score) from archived DuckDB, seed the Study.

For each distinct `deployment.engine_args` ever recorded against a given
(endpoint_slug, profile_slug), we compute a composite score analogous to
`bench_score.py`:

    penalty = max(0, 1 - ttft_p99 / (2 * ttft_slo_ms))
    score   = throughput_tok_avg * penalty
    (ttft_p99 > ttft_slo_ms OR e2e_p99 > e2e_slo_ms → score = 0)

Per-workload scores are then summed per engine_args group. Only engine_args
keys that appear in the given `SearchSpace.axes` are retained — the rest are
dropped (history contains axes now frozen or removed from the current space).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb

from bench.search.space import Axis, SearchSpace


log = logging.getLogger(__name__)


def _compute_score(
    throughput_avg: float | None,
    ttft_p99_ms: float | None,
    e2e_p99_s: float | None,
    ttft_slo_ms: float = 500.0,
    e2e_slo_s: float = 30.0,
) -> float:
    if throughput_avg is None or ttft_p99_ms is None:
        return 0.0
    if ttft_p99_ms > ttft_slo_ms:
        return 0.0
    if e2e_p99_s is not None and e2e_p99_s > e2e_slo_s:
        return 0.0
    penalty = max(0.0, 1.0 - ttft_p99_ms / (2.0 * ttft_slo_ms))
    return float(throughput_avg) * penalty


def _coerce_param(axis: Axis, value: Any) -> Any:
    """Coerce a raw history value into the axis' expected type."""
    if value is None:
        return None
    if axis.kind == "bool":
        return bool(value)
    if axis.kind == "int":
        return int(value)
    if axis.kind in ("float", "log_uniform"):
        return float(value)
    return value  # categorical


def _project_params(axes: list[Axis], engine_args: dict[str, Any]) -> dict[str, Any] | None:
    """Project history onto the current space. Missing axes are skipped (Optuna
    will sample them); out-of-range values on PRESENT axes reject the whole row.

    Returns None if every axis is missing (nothing to seed with).
    """
    out: dict[str, Any] = {}
    for axis in axes:
        if axis.name not in engine_args:
            continue  # partial params — Optuna will complete
        v = _coerce_param(axis, engine_args[axis.name])
        if axis.kind == "categorical":
            if v not in (axis.values or []):
                return None
        elif axis.kind == "bool":
            if v not in (False, True):
                return None
        elif axis.kind in ("int", "float", "log_uniform"):
            if axis.low is None or axis.high is None:
                return None
            if v < axis.low or v > axis.high:
                return None
        out[axis.name] = v
    return out if out else None


def warmstart_from_archive(
    archive_db: str | Path,
    space: SearchSpace,
    *,
    endpoint_slug: str | None = None,
    profile_slugs: list[str] | None = None,
    top_k: int = 5,
    ttft_slo_ms: float = 500.0,
    e2e_slo_s: float = 30.0,
    context: dict | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Return up to top_k (params, score) pairs aggregated from archive history."""
    db_path = Path(archive_db)
    if not db_path.exists():
        log.warning("archive DB not found: %s — warmstart skipped", db_path)
        return []

    axes = space.active_axes(context)
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # One row per run with engine_args JSON text + key avg metrics.
        sql = """
        WITH by_run AS (
            SELECT
                r.run_id, r.profile_slug,
                json_extract(r.endpoint_meta, '$.deployment.engine_args') AS ea,
                MAX(CASE WHEN m.metric='throughput_tok' AND m.p='avg' THEN m.value END) AS thr,
                MAX(CASE WHEN m.metric='ttft'           AND m.p='p99' THEN m.value END) AS ttft,
                MAX(CASE WHEN m.metric='e2e'            AND m.p='p99' THEN m.value END) AS e2e
            FROM runs r LEFT JOIN metrics m USING (run_id)
            WHERE r.status = 'ok'
              AND ({endpoint_filter})
              AND ({profile_filter})
            GROUP BY r.run_id, r.profile_slug, ea
        )
        SELECT ea, profile_slug, AVG(thr) AS thr, AVG(ttft) AS ttft, AVG(e2e) AS e2e, COUNT(*) AS n
        FROM by_run
        WHERE ea IS NOT NULL
        GROUP BY ea, profile_slug
        """
        ep_clause = "r.endpoint_slug = ?" if endpoint_slug else "TRUE"
        if profile_slugs:
            ph = ",".join("?" for _ in profile_slugs)
            prof_clause = f"r.profile_slug IN ({ph})"
        else:
            prof_clause = "TRUE"
        sql = sql.format(endpoint_filter=ep_clause, profile_filter=prof_clause)

        args: list[Any] = []
        if endpoint_slug:
            args.append(endpoint_slug)
        if profile_slugs:
            args.extend(profile_slugs)

        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()

    # Aggregate per engine_args: sum per-workload composite scores.
    bucket: dict[str, dict] = {}
    for ea_text, prof, thr, ttft, e2e, n in rows:
        if not ea_text:
            continue
        ea = json.loads(ea_text) if isinstance(ea_text, str) else ea_text
        params = _project_params(axes, ea)
        if params is None:
            continue
        key = json.dumps(params, sort_keys=True)
        score = _compute_score(thr, ttft, e2e, ttft_slo_ms, e2e_slo_s)
        entry = bucket.setdefault(key, {"params": params, "score": 0.0, "workloads": 0, "runs": 0})
        entry["score"] += score
        entry["workloads"] += 1
        entry["runs"] += int(n or 0)

    ranked = sorted(bucket.values(), key=lambda e: e["score"], reverse=True)
    top = ranked[:top_k]
    return [(e["params"], float(e["score"])) for e in top if e["score"] > 0]
