CREATE TABLE IF NOT EXISTS runs (
  run_id        TEXT PRIMARY KEY,
  profile_slug  TEXT NOT NULL,
  endpoint_slug TEXT NOT NULL,
  started_at    TIMESTAMP,
  finished_at   TIMESTAMP,
  status        TEXT NOT NULL,
  runner        TEXT NOT NULL,
  profile_yaml  TEXT,
  endpoint_meta TEXT,
  git_sha       TEXT,
  tool_versions JSON,
  error         TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  p      TEXT,
  value  DOUBLE NOT NULL,
  PRIMARY KEY (run_id, metric, p)
);

CREATE TABLE IF NOT EXISTS requests (
  run_id              TEXT NOT NULL,
  req_id              TEXT NOT NULL,
  turn_idx            INTEGER,
  conversation_id     TEXT,
  input_tokens        INTEGER,
  output_tokens       INTEGER,
  cached_tokens       INTEGER,
  thinking_tokens     INTEGER,
  tool_call_count     INTEGER,
  tool_result_tokens  INTEGER,
  phase               TEXT,     -- exploration | editing | execution | verification | other
  role                TEXT,     -- planner | reasoner | verifier | solo
  energy_wh           DOUBLE,
  cost_usd            DOUBLE,
  ttft_ms             DOUBLE,
  itl_mean_ms         DOUBLE,
  e2e_ms              DOUBLE,
  started_at          TIMESTAMP,
  completed_at        TIMESTAMP,
  status              TEXT,
  error               TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS idx_requests_conv ON requests(run_id, conversation_id);
CREATE INDEX IF NOT EXISTS idx_requests_phase ON requests(run_id, phase);

CREATE TABLE IF NOT EXISTS sessions (
  run_id              TEXT NOT NULL,
  session_id          TEXT NOT NULL,
  task_id             TEXT,     -- 선택: 외부 dataset task identifier
  total_input_tokens  BIGINT,
  total_output_tokens BIGINT,
  total_cached_tokens BIGINT,
  turn_count          INTEGER,
  tool_call_count     INTEGER,
  duration_ms         DOUBLE,
  success             BOOLEAN,
  total_cost_usd      DOUBLE,
  total_energy_wh     DOUBLE,
  PRIMARY KEY (run_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_task ON sessions(task_id);

CREATE TABLE IF NOT EXISTS trajectory_events (
  run_id       TEXT NOT NULL,
  session_id   TEXT NOT NULL,
  seq          INTEGER NOT NULL,
  ts           TIMESTAMP,
  event_type   TEXT NOT NULL,   -- user | assistant | tool_call | tool_result | thinking
  phase        TEXT,
  tokens       INTEGER,
  metadata     JSON
);
CREATE INDEX IF NOT EXISTS idx_traj_sess ON trajectory_events(run_id, session_id, seq);

CREATE TABLE IF NOT EXISTS prom_samples (
  run_id TEXT NOT NULL,
  ts     TIMESTAMP NOT NULL,
  metric TEXT NOT NULL,
  labels JSON,
  value  DOUBLE NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prom_run_ts ON prom_samples(run_id, ts);

CREATE TABLE IF NOT EXISTS detections (
  run_id     TEXT NOT NULL,
  detector   TEXT NOT NULL,
  severity   TEXT NOT NULL,
  metric     TEXT,
  threshold  DOUBLE,
  observed   DOUBLE,
  message    TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id);

-- Phase S1 search framework. Existing runs table keeps its schema;
-- runs.trial_id is added via ALTER in duckdb_store._init_schema() for
-- backward-compat with earlier DBs.

CREATE TABLE IF NOT EXISTS studies (
  study_id      TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  strategy      TEXT NOT NULL,          -- grid | random | lhc | tpe | cma_es | nsga2
  space_yaml    TEXT,                   -- frozen snapshot at start
  endpoint_slug TEXT,
  profile_slugs JSON,                   -- e.g. ["autotune-short","autotune-medium","autotune-long"]
  metric_name   TEXT NOT NULL,          -- default: total_score
  direction     TEXT NOT NULL,          -- maximize | minimize
  status        TEXT NOT NULL DEFAULT 'running',  -- running | paused | completed | aborted
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  finished_at   TIMESTAMP,
  notes         TEXT
);

CREATE TABLE IF NOT EXISTS trials (
  trial_id      TEXT PRIMARY KEY,
  study_id      TEXT NOT NULL,
  seq           INTEGER NOT NULL,       -- 1-indexed within study
  params        JSON NOT NULL,
  status        TEXT NOT NULL,          -- pending | running | completed | pruned | crash
  score         DOUBLE,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at  TIMESTAMP,
  backend       TEXT,                   -- inline | process-pool | k8s-job
  worker_id     TEXT,
  error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_study ON trials(study_id, seq);

CREATE TABLE IF NOT EXISTS trial_metrics (
  trial_id TEXT NOT NULL,
  metric   TEXT NOT NULL,
  workload TEXT NOT NULL DEFAULT 'aggregate',  -- short | medium | long | aggregate
  value    DOUBLE,
  PRIMARY KEY (trial_id, metric, workload)
);
