from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from lmtune.endpoints import EndpointSpec
from lmtune.profiles import ProfileSpec
from lmtune.runners.base import RunArtifact

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
        # Phase S1 compat: older DBs have runs without trial_id column.
        # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk).
        existing_cols = {r[1] for r in self.conn.execute("PRAGMA table_info('runs')").fetchall()}
        if "trial_id" not in existing_cols:
            self.conn.execute("ALTER TABLE runs ADD COLUMN trial_id TEXT")

    def record_run(
        self,
        artifact: RunArtifact,
        profile: ProfileSpec,
        endpoint: EndpointSpec,
        profile_yaml_text: str,
        git_sha: str | None = None,
    ):
        started = (
            datetime.fromtimestamp(artifact.started_at, tz=UTC) if artifact.started_at else None
        )
        finished = (
            datetime.fromtimestamp(artifact.finished_at, tz=UTC) if artifact.finished_at else None
        )
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
                    r.req_id,
                    r.turn_idx,
                    r.conversation_id,
                    r.input_tokens,
                    r.output_tokens,
                    r.cached_tokens,
                    r.thinking_tokens,
                    r.tool_call_count,
                    r.tool_result_tokens,
                    r.phase,
                    r.role,
                    r.energy_wh,
                    r.cost_usd,
                    r.ttft_ms,
                    r.itl_mean_ms,
                    r.e2e_ms,
                    _epoch_to_ts(r.started_at),
                    _epoch_to_ts(r.completed_at),
                    r.status,
                    r.error,
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
                    artifact.run_id,
                    s.session_id,
                    s.task_id,
                    s.total_input_tokens,
                    s.total_output_tokens,
                    s.total_cached_tokens,
                    s.turn_count,
                    s.tool_call_count,
                    s.duration_ms,
                    s.success,
                    s.total_cost_usd,
                    s.total_energy_wh,
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
                    artifact.run_id,
                    e.session_id,
                    e.seq,
                    _epoch_to_ts(e.ts),
                    e.event_type,
                    e.phase,
                    e.tokens,
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

    # ---- Phase S1: studies / trials / trial_metrics ------------------------

    def record_study(
        self,
        study_id: str,
        name: str,
        strategy: str,
        metric_name: str,
        direction: str,
        space_yaml: str | None = None,
        endpoint_slug: str | None = None,
        profile_slugs: list[str] | None = None,
        status: str = "running",
        notes: str | None = None,
    ):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO studies
              (study_id, name, strategy, space_yaml, endpoint_slug, profile_slugs,
               metric_name, direction, status, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                study_id,
                name,
                strategy,
                space_yaml,
                endpoint_slug,
                json.dumps(profile_slugs) if profile_slugs else None,
                metric_name,
                direction,
                status,
                notes,
            ],
        )

    def set_study_status(self, study_id: str, status: str, finished: bool = False):
        if finished:
            self.conn.execute(
                "UPDATE studies SET status=?, finished_at=CURRENT_TIMESTAMP WHERE study_id=?",
                [status, study_id],
            )
        else:
            self.conn.execute(
                "UPDATE studies SET status=? WHERE study_id=?",
                [status, study_id],
            )

    def get_study(self, study_id: str):
        return self.conn.execute("SELECT * FROM studies WHERE study_id = ?", [study_id]).fetchone()

    def list_studies(self, limit: int = 20):
        return self.conn.execute(
            """
            SELECT study_id, name, strategy, endpoint_slug, status, created_at, finished_at
            FROM studies ORDER BY created_at DESC NULLS LAST LIMIT ?
            """,
            [int(limit)],
        ).fetchall()

    def record_trial(
        self,
        trial_id: str,
        study_id: str,
        seq: int,
        params: dict,
        status: str,
        score: float | None = None,
        backend: str | None = None,
        worker_id: str | None = None,
        error: str | None = None,
        completed: bool = False,
    ):
        completed_at = "CURRENT_TIMESTAMP" if completed else "NULL"
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO trials
              (trial_id, study_id, seq, params, status, score,
               backend, worker_id, error, completed_at)
            VALUES (?,?,?,?,?,?,?,?,?, {completed_at})
            """,
            [
                trial_id,
                study_id,
                int(seq),
                json.dumps(params, sort_keys=True),
                status,
                score,
                backend,
                worker_id,
                error,
            ],
        )

    def record_trial_metrics(self, trial_id: str, metrics: dict[tuple[str, str | None], float]):
        """metrics keyed by (metric_name, workload). None workload → 'aggregate'.

        Non-scalar values (e.g. the multi-obj '_values' tuple ParetoObjective
        stashes for Study.tell) are skipped; only scalar entries persist to DB.
        """
        rows = []
        for (m, wl), v in metrics.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict)):
                continue  # multi-obj sentinel
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            rows.append((trial_id, m, wl or "aggregate", fv))
        if rows:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO trial_metrics (trial_id, metric, workload, value)
                VALUES (?,?,?,?)
                """,
                rows,
            )

    def list_trials(self, study_id: str, limit: int | None = None):
        sql = (
            "SELECT trial_id, seq, params, status, score, completed_at, backend, error "
            "FROM trials WHERE study_id = ? ORDER BY seq ASC"
        )
        args: list = [study_id]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        return self.conn.execute(sql, args).fetchall()

    def top_trials(self, study_id: str, direction: str = "maximize", k: int = 5):
        order = "DESC" if direction == "maximize" else "ASC"
        return self.conn.execute(
            f"""
            SELECT trial_id, seq, params, score, status
            FROM trials WHERE study_id = ? AND score IS NOT NULL AND status = 'completed'
            ORDER BY score {order} LIMIT ?
            """,
            [study_id, int(k)],
        ).fetchall()

    def get_trial_metrics(self, trial_id: str) -> dict[str, dict[str | None, float]]:
        rows = self.conn.execute(
            "SELECT metric, workload, value FROM trial_metrics WHERE trial_id = ?",
            [trial_id],
        ).fetchall()
        out: dict[str, dict[str | None, float]] = {}
        for m, wl, v in rows:
            out.setdefault(m, {})[wl] = float(v)
        return out

    # ---- detections (기존) -------------------------------------------------

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

    # S1 shim until S3 writer_queue: release DuckDB file lock so a child
    # process (e.g. subprocess bench run) can open the same DB briefly.
    def suspend(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]

    def resume(self):
        if self.conn is None:
            self.conn = duckdb.connect(str(self.db_path))


def _endpoint_meta(ep: EndpointSpec) -> dict:
    meta = ep.model_dump(mode="json")
    meta["url"] = ep.base_url
    return meta


def _epoch_to_ts(v: float | None):
    if v is None:
        return None
    return datetime.fromtimestamp(v, tz=UTC)
