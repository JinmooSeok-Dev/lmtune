#!/usr/bin/env bash
# Autoresearch benchmark wrapper.
#
# Pipeline: (optional) restart endpoint → bench_score.py × 3 workloads →
#           emit `METRIC name=value` lines on stdout for autoresearch.
#
# Env vars:
#   ENDPOINT       default: configs/endpoints/local_vllm_autotune.yaml
#   PROFILES_DIR   default: configs/profiles/autotune
#   WORKLOADS      default: "short medium long" (space-separated, ordered)
#   REPEATS        default: 3   (bench_score --count)
#   SKIP_RESTART   default: 0   (set 1 to reuse already-running server)
#   RESTART_CMD    default: scripts/vllm_restart.sh (skipped if endpoint not vllm)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ENDPOINT="${ENDPOINT:-configs/endpoints/local_vllm_autotune.yaml}"
PROFILES_DIR="${PROFILES_DIR:-configs/profiles/autotune}"
WORKLOADS="${WORKLOADS:-short medium long}"
REPEATS="${REPEATS:-3}"
SKIP_RESTART="${SKIP_RESTART:-0}"
RESTART_CMD="${RESTART_CMD:-scripts/vllm_restart.sh}"
PY="${PY:-./.venv/bin/python}"

if [ ! -x "$PY" ]; then PY="python3"; fi

# bench CLI: prefer venv, fall back to PATH
VENV_BENCH="$ROOT/.venv/bin/bench"
if [ -x "$VENV_BENCH" ]; then
    export BENCH_BIN="${BENCH_BIN:-$VENV_BENCH}"
fi

BENCH_SCORE="$PY scripts/bench_score.py"

log() { echo "[autoresearch.sh] $*" >&2; }

# --- Pre-check: endpoint file exists, syntax valid YAML ---
[ -f "$ENDPOINT" ] || { log "ERROR: endpoint not found: $ENDPOINT"; exit 2; }
$PY -c "import yaml,sys; yaml.safe_load(open('$ENDPOINT'))" >/dev/null 2>&1 \
    || { log "ERROR: invalid YAML: $ENDPOINT"; exit 2; }

# Determine engine from deployment.engine (skip restart for non-vllm)
ENGINE=$($PY -c "import yaml; d=yaml.safe_load(open('$ENDPOINT')); print(((d.get('deployment') or {}).get('engine') or '').lower())")
log "endpoint=$ENDPOINT engine=$ENGINE workloads='$WORKLOADS' repeats=$REPEATS"

# --- Optional restart (vllm only) ---
if [ "$SKIP_RESTART" != "1" ] && [ "$ENGINE" = "vllm" ]; then
    log "restarting vllm via $RESTART_CMD …"
    if ! bash "$RESTART_CMD" "$ENDPOINT"; then
        log "ERROR: vllm restart failed; see /tmp/vllm_server.log"
        # Emit METRIC with zero so autoresearch can treat as crash.
        echo "METRIC total_score=0"
        echo "METRIC slo_pass_all=0"
        exit 1
    fi
else
    log "skipping restart (SKIP_RESTART=$SKIP_RESTART, engine=$ENGINE)"
fi

# --- Run each workload, collect JSON ---
TOTAL_SCORE=0
ANY_SLO_FAIL=0
declare -A TTFT_P99 THR_AVG E2E_P99 WORKLOAD_SCORE

for W in $WORKLOADS; do
    PROFILE="$PROFILES_DIR/$W.yaml"
    if [ ! -f "$PROFILE" ]; then
        log "WARN: profile missing, skipping: $PROFILE"
        TTFT_P99[$W]=0; THR_AVG[$W]=0; E2E_P99[$W]=0; WORKLOAD_SCORE[$W]=0
        ANY_SLO_FAIL=1
        continue
    fi

    log "[$W] running bench_score (n=$REPEATS) …"
    JSON=$($BENCH_SCORE -p "$PROFILE" -e "$ENDPOINT" -n "$REPEATS" | tail -n1)
    if [ -z "$JSON" ]; then
        log "[$W] ERROR: no JSON from bench_score"
        TTFT_P99[$W]=0; THR_AVG[$W]=0; E2E_P99[$W]=0; WORKLOAD_SCORE[$W]=0
        ANY_SLO_FAIL=1
        continue
    fi

    SCORE=$(echo "$JSON"    | jq -r '.score       // 0')
    TTFT=$(echo "$JSON"     | jq -r '.ttft_p99    // 0')
    THR=$(echo "$JSON"      | jq -r '.throughput_tok_avg // 0')
    E2E=$(echo "$JSON"      | jq -r '.e2e_p99     // 0')
    PASS=$(echo "$JSON"     | jq -r '.slo_pass    // false')
    ACCEPT=$(echo "$JSON"   | jq -r '.accepted    // false')

    WORKLOAD_SCORE[$W]="$SCORE"
    TTFT_P99[$W]="$TTFT"
    THR_AVG[$W]="$THR"
    E2E_P99[$W]="$E2E"

    log "[$W] score=$SCORE ttft_p99=$TTFT throughput=$THR e2e_p99=$E2E slo=$PASS accepted=$ACCEPT"

    if [ "$PASS" != "true" ]; then ANY_SLO_FAIL=1; fi
    TOTAL_SCORE=$(echo "$TOTAL_SCORE + $SCORE" | bc -l)
done

# --- Emit METRIC lines (last lines of stdout = what autoresearch parses) ---
SLO_PASS_ALL=$([ "$ANY_SLO_FAIL" = "0" ] && echo 1 || echo 0)

echo "METRIC total_score=${TOTAL_SCORE}"
for W in $WORKLOADS; do
    echo "METRIC score_${W}=${WORKLOAD_SCORE[$W]}"
    echo "METRIC ttft_p99_${W}=${TTFT_P99[$W]}"
    echo "METRIC throughput_avg_${W}=${THR_AVG[$W]}"
    echo "METRIC e2e_p99_${W}=${E2E_P99[$W]}"
done
echo "METRIC slo_pass_all=${SLO_PASS_ALL}"

log "total_score=$TOTAL_SCORE slo_pass_all=$SLO_PASS_ALL"
exit 0
