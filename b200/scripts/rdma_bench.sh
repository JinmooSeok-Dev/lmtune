#!/usr/bin/env bash
# RDMA Perftest baseline — ib_write_bw / ib_read_bw / ib_send_bw
#
# Phase B6 의 host-level RDMA fabric 측정. B0 fabric_test.yaml (k8s pod 기반) 의
# 보조 baseline 으로 host 에서 직접 실행한다.
#
# 사용:
#   server (node-1):  bash b200/scripts/rdma_bench.sh server
#   client (node-2):  bash b200/scripts/rdma_bench.sh client <SERVER_IP>
#
# 결과:
#   data/raw/rdma/<timestamp>/{ib_write_bw,ib_read_bw,ib_send_bw}.txt
#   data/raw/rdma/<timestamp>/summary.json
#
# 참고: NHN Cloud B200 환경에서 363.98 Gbps RDMA Write (Non-priv Pod) 재현.
set -euo pipefail

ROLE="${1:-server}"
SERVER_IP="${2:-}"
DURATION="${DURATION:-30}"
MSG_SIZE="${MSG_SIZE:-65536}"
QP_COUNT="${QP_COUNT:-2}"
PORT_BASE="${PORT_BASE:-18515}"
DEVICE="${RDMA_DEVICE:-}"
GID_INDEX="${GID_INDEX:-3}"

OUT_ROOT="${OUT_ROOT:-data/raw/rdma}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT}/${TS}"
mkdir -p "${OUT_DIR}"

log() { printf '[rdma_bench] %s\n' "$*" >&2; }

require() {
  command -v "$1" >/dev/null 2>&1 || { log "missing tool: $1 (install perftest pkg)"; exit 1; }
}

require ib_write_bw
require ib_read_bw
require ib_send_bw

if [[ -z "${DEVICE}" ]]; then
  if command -v ibv_devices >/dev/null 2>&1; then
    DEVICE="$(ibv_devices | awk 'NR>2 {print $1; exit}')"
  fi
  [[ -z "${DEVICE}" ]] && { log "RDMA device not detected. set RDMA_DEVICE=mlx5_0 etc."; exit 1; }
fi

COMMON_OPTS=(-F -d "${DEVICE}" -x "${GID_INDEX}" -s "${MSG_SIZE}" -q "${QP_COUNT}" -D "${DURATION}" --report_gbits --output bandwidth)

run_test() {
  local op="$1" port="$2" out="$3"
  local cmd
  case "$op" in
    write) cmd=(ib_write_bw "${COMMON_OPTS[@]}" -p "${port}") ;;
    read)  cmd=(ib_read_bw  "${COMMON_OPTS[@]}" -p "${port}") ;;
    send)  cmd=(ib_send_bw  "${COMMON_OPTS[@]}" -p "${port}") ;;
    *) log "unknown op: $op"; return 1 ;;
  esac
  if [[ "${ROLE}" == "client" ]]; then
    cmd+=("${SERVER_IP}")
  fi
  log "running: ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee "${out}"
}

case "${ROLE}" in
  server)
    log "device=${DEVICE} gid=${GID_INDEX} qp=${QP_COUNT} msg=${MSG_SIZE} dur=${DURATION}"
    log "ports: write=$((PORT_BASE)) read=$((PORT_BASE+1)) send=$((PORT_BASE+2))"
    log "-- run client on the OTHER node now: bash b200/scripts/rdma_bench.sh client <THIS_HOST_IP>"
    run_test write "${PORT_BASE}"     "${OUT_DIR}/ib_write_bw.server.txt" &
    PID_W=$!
    run_test read  "$((PORT_BASE+1))" "${OUT_DIR}/ib_read_bw.server.txt"  &
    PID_R=$!
    run_test send  "$((PORT_BASE+2))" "${OUT_DIR}/ib_send_bw.server.txt"  &
    PID_S=$!
    wait "${PID_W}" "${PID_R}" "${PID_S}" || true
    ;;
  client)
    [[ -z "${SERVER_IP}" ]] && { log "client mode requires SERVER_IP"; exit 1; }
    log "device=${DEVICE} server=${SERVER_IP}"
    run_test write "${PORT_BASE}"     "${OUT_DIR}/ib_write_bw.client.txt"
    run_test read  "$((PORT_BASE+1))" "${OUT_DIR}/ib_read_bw.client.txt"
    run_test send  "$((PORT_BASE+2))" "${OUT_DIR}/ib_send_bw.client.txt"
    ;;
  *)
    log "usage: $0 {server|client} [SERVER_IP]"
    exit 1
    ;;
esac

if [[ "${ROLE}" == "client" ]]; then
  python3 - "${OUT_DIR}" <<'PY'
import json, re, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary = {"ts": out_dir.name, "tests": {}}

bw_re = re.compile(r"^\s*\d+\s+\d+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", re.MULTILINE)

for op in ("ib_write_bw", "ib_read_bw", "ib_send_bw"):
    f = out_dir / f"{op}.client.txt"
    if not f.exists():
        continue
    text = f.read_text()
    rows = bw_re.findall(text)
    if not rows:
        summary["tests"][op] = {"status": "no_data"}
        continue
    avg = float(rows[-1][2])
    peak = max(float(r[1]) for r in rows)
    summary["tests"][op] = {
        "avg_gbps": avg,
        "peak_gbps": peak,
        "samples": len(rows),
    }

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
fi

log "done. results in ${OUT_DIR}/"
