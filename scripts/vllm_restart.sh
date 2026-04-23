#!/usr/bin/env bash
#
# vllm_restart.sh <endpoint.yaml> [--dry-run]
#
# Reads `deployment.engine_args` from the endpoint YAML, kills any running
# `vllm serve` on the host, then restarts vllm with flags derived from the YAML.
# Polls /v1/models until ready (max 180s) or exits non-zero on timeout.
#
# Environment overrides:
#   VLLM_CUDA_DEVICES   — default "1"
#   VLLM_PORT           — default 8000
#   VLLM_HOST           — default 0.0.0.0
#   VLLM_READY_TIMEOUT  — default 180 (seconds)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <endpoint.yaml> [--dry-run]" >&2
  exit 64
fi

ENDPOINT_YAML="$1"
DRY_RUN=0
if [ "${2:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if [ ! -f "$ENDPOINT_YAML" ]; then
  echo "endpoint yaml not found: $ENDPOINT_YAML" >&2
  exit 65
fi

VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-180}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
VENV_VLLM="${VENV_VLLM:-$REPO_ROOT/.venv/bin/vllm}"

if [ ! -x "$VENV_PY" ]; then
  echo "python not found at $VENV_PY (set VENV_PY or activate venv first)" >&2
  exit 66
fi

# ----- YAML -> CLI flags via python/PyYAML -----
FLAGS_BLOB="$("$VENV_PY" - "$ENDPOINT_YAML" <<'PY'
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
dep = (cfg or {}).get("deployment") or {}
eng = dep.get("engine_args") or {}
parallel = dep.get("parallelism") or {}
model = cfg.get("model")

flags = []
# parallelism (if > 1)
pmap = {"tp": "tensor-parallel-size", "pp": "pipeline-parallel-size",
        "dp": "data-parallel-size",   "ep": "expert-parallel-size"}
for key, cli in pmap.items():
    v = parallel.get(key)
    if v and int(v) > 1:
        flags.append(f"--{cli}"); flags.append(str(int(v)))

# engine_args: booleans -> bare flag when true; otherwise skip.
# value args -> --k v (kebab-case the key).
for k, v in eng.items():
    cli = k.replace("_", "-")
    if isinstance(v, bool):
        if v:
            flags.append(f"--{cli}")
    elif v is None:
        continue
    else:
        flags.append(f"--{cli}"); flags.append(str(v))

print(model or "")
for f in flags:
    print(f)
PY
)"

MODEL="$(echo "$FLAGS_BLOB" | head -n1)"
CLI_FLAGS=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  CLI_FLAGS+=("$line")
done <<< "$(echo "$FLAGS_BLOB" | tail -n +2)"

if [ -z "$MODEL" ]; then
  echo "model not found in endpoint yaml" >&2
  exit 67
fi

CMD=("$VENV_VLLM" serve "$MODEL"
     "--host" "$VLLM_HOST"
     "--port" "$VLLM_PORT"
     "${CLI_FLAGS[@]}")

if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' "${CMD[@]}"
  exit 0
fi

# ----- kill previous vllm -----
PIDS="$(pgrep -f "vllm serve" || true)"
if [ -n "$PIDS" ]; then
  echo "killing existing vllm pids: $PIDS"
  # shellcheck disable=SC2086
  kill -TERM $PIDS 2>/dev/null || true
  for _ in $(seq 1 20); do
    sleep 0.5
    pgrep -f "vllm serve" >/dev/null || break
  done
  # shellcheck disable=SC2086
  kill -KILL $PIDS 2>/dev/null || true
fi

# ----- start vllm -----
HASH="$(echo "${CMD[*]}" | sha1sum | cut -c1-8)"
LOG="/tmp/vllm_autotune_${HASH}.log"
echo "starting: ${CMD[*]}"
echo "log: $LOG"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVICES" \
VLLM_LOGGING_LEVEL=INFO \
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
VLLM_PID=$!
echo "pid: $VLLM_PID"

# ----- wait until /v1/models responds -----
READY_URL="http://localhost:${VLLM_PORT}/v1/models"
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  # if process died, fail fast
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vllm process exited prematurely (pid $VLLM_PID)" >&2
    tail -n 50 "$LOG" >&2 || true
    exit 1
  fi
  if curl -sf -m 2 "$READY_URL" | grep -q '"data"' 2>/dev/null; then
    echo "ready: $READY_URL"
    exit 0
  fi
  sleep 2
done

echo "timeout after ${VLLM_READY_TIMEOUT}s" >&2
tail -n 50 "$LOG" >&2 || true
exit 1
