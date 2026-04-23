from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from bench.endpoints import EndpointSpec
from bench.profiles import ProfileSpec
from bench.runners.base import RunArtifact


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DuckDBStore:
    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        for stmt in _SCHEMA_PATH.read_text(encoding="utf-8").split(";"):
            s = stmt.strip()
            if s:
                self.conn.execute(s)

    def record_run(
        self,
        artifact: RunArtifact,
        profile: ProfileSpec,
        endpoint: EndpointSpec,
        profile_yaml_text: str,
        git_sha: str | None = None,
    ):
        started = datetime.fromtimestamp(artifact.started_at, tz=timezone.utc) if artifact.started_at else None
        finished = datetime.fromtimestamp(artifact.finished_at, tz=timezone.utc) if artifact.finished_at else None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, profile_slug, endpoint_slug, started_at, finished_at, status,
             runner, profile_yaml, endpoint_meta, git_sha, tool_versions, error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                artifact.run_id,
                profile.slug,
                endpoint.slug,
                started,
                finished,
                artifact.status,
                artifact.runner_kind,
                profile_yaml_text,
                json.dumps(_endpoint_meta(endpoint)),
                git_sha,
                json.dumps({artifact.runner_kind: artifact.tool_version}),
                artifact.error,
            ],
        )

        for metric, bucket in artifact.metrics.items():
            for p, v in bucket.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO metrics (run_id, metric, p, value) VALUES (?,?,?,?)",
                    [artifact.run_id, metric, p, float(v)],
                )

        if artifact.requests:
            rows = [
                (
                    artifact.run_id,
                    r.req_id, r.turn_idx, r.conversation_id,
                    r.input_tokens, r.output_tokens,
                    r.cached_tokens, r.thinking_tokens,
                    r.tool_call_count, r.tool_result_tokens,
                    r.phase, r.role,
                    r.energy_wh, r.cost_usd,
                    r.ttft_ms, r.itl_mean_ms, r.e2e_ms,
                    _epoch_to_ts(r.started_at), _epoch_to_ts(r.completed_at),
                    r.status, r.error,
                )
                for r in artifact.requests
            ]
            self.conn.executemany(
                """
                INSERT INTO requests
                (run_id, req_id, turn_idx, conversation_id,
                 input_tokens, output_tokens,
                 cached_tokens, thinking_tokens,
                 tool_call_count, tool_result_tokens,
                 phase, role, energy_wh, cost_usd,
                 ttft_ms, itl_mean_ms, e2e_ms,
                 started_at, completed_at,
                 status, error)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

        if artifact.sessions:
            srows = [
                (
                    artifact.run_id, s.session_id, s.task_id,
                    s.total_input_tokens, s.total_output_tokens, s.total_cached_tokens,
                    s.turn_count, s.tool_call_count, s.duration_ms,
                    s.success, s.total_cost_usd, s.total_energy_wh,
                )
                for s in artifact.sessions
            ]
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO sessions
                (run_id, session_id, task_id,
                 total_input_tokens, total_output_tokens, total_cached_tokens,
                 turn_count, tool_call_count, duration_ms,
                 success, total_cost_usd, total_energy_wh)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                srows,
            )

        if artifact.trajectory:
            trows = [
                (
                    artifact.run_id, e.session_id, e.seq,
                    _epoch_to_ts(e.ts), e.event_type, e.phase, e.tokens,
                    json.dumps(e.metadata) if e.metadata else None,
                )
                for e in artifact.trajectory
            ]
            self.conn.executemany(
                """
                INSERT INTO trajectory_events
                (run_id, session_id, seq, ts, event_type, phase, tokens, metadata)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                trows,
            )

    def record_detections(self, run_id: str, detections: list[dict]):
        for d in detections:
            self.conn.execute(
                """
                INSERT INTO detections (run_id, detector, severity, metric, threshold, observed, message)
                VALUES (?,?,?,?,?,?,?)
                """,
                [
                    run_id,
                    d["detector"],
                    d["severity"],
                    d.get("metric"),
                    d.get("threshold"),
                    d.get("observed"),
                    d.get("message"),
                ],
            )

    def list_runs(
        self,
        endpoint_slug: str | None = None,
        profile_slug: str | None = None,
        limit: int = 20,
    ):
        where = []
        args: list = []
        if endpoint_slug:
            where.append("endpoint_slug = ?")
            args.append(endpoint_slug)
        if profile_slug:
            where.append("profile_slug = ?")
            args.append(profile_slug)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        args.append(int(limit))
        return self.conn.execute(
            f"""
            SELECT run_id, profile_slug, endpoint_slug, started_at, status, runner
            FROM runs {clause}
            ORDER BY started_at DESC NULLS LAST
            LIMIT ?
            """,
            args,
        ).fetchall()

    def get_run(self, run_id: str):
        return self.conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchone()

    def get_metrics(self, run_id: str) -> dict[str, dict[str, float]]:
        rows = self.conn.execute(
            "SELECT metric, p, value FROM metrics WHERE run_id = ? ORDER BY metric, p",
            [run_id],
        ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for metric, p, value in rows:
            out.setdefault(metric, {})[p] = float(value)
        return out

    def close(self):
        self.conn.close()


def _endpoint_meta(ep: EndpointSpec) -> dict:
    meta = ep.model_dump(mode="json")
    meta["url"] = ep.base_url
    return meta


def _epoch_to_ts(v: float | None):
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=timezone.utc)
