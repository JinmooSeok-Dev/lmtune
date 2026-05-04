# util/pf.sh — kubectl port-forward 함수 라이브러리.
#
# 사용: `source b200/scripts/util/pf.sh` 후 `pf::<func>` 호출.
# 단독 실행 의도 없음 (함수만 정의).
#
# 명세:
#   pf::list                     — 현재 살아있는 port-forward 프로세스
#   pf::stop_all                 — 모든 kubectl port-forward 정리 + PID 파일 제거
#   pf::stop_local <local-port>  — 특정 로컬 포트 점유 wrapper 만 정리
#   pf::start <ns> <svc> <local> <remote>
#                                — 재시도 wrapper 데몬으로 띄움. PID=/tmp/pf_<local>.pid
#                                  log=/tmp/pf_<local>.log
#   pf::probe <local> [path]     — http://127.0.0.1:<local><path> 200 polling (max 5분)
#   pf::status                   — PID, listener, 마지막 응답 한 화면

# shellcheck shell=bash

pf::_pidfile()  { printf '/tmp/pf_%s.pid' "$1"; }
pf::_logfile()  { printf '/tmp/pf_%s.log' "$1"; }

pf::list() {
  local found=0
  if pgrep -af "kubectl.*port-forward" >/dev/null 2>&1; then
    pgrep -af "kubectl.*port-forward"
    found=1
  fi
  if pgrep -af "_pf_loop_" >/dev/null 2>&1; then
    pgrep -af "_pf_loop_"
    found=1
  fi
  if [[ "$found" -eq 0 ]]; then
    echo "[pf::list] (none)"
  fi
  return 0
}

pf::stop_all() {
  pkill -f "kubectl.*port-forward" 2>/dev/null || true
  pkill -f "_pf_loop_" 2>/dev/null || true
  rm -f /tmp/pf_*.pid 2>/dev/null || true
  echo "[pf::stop_all] cleaned"
}

pf::stop_local() {
  local local_port="$1"
  local pidfile; pidfile=$(pf::_pidfile "$local_port")
  if [[ -f "$pidfile" ]]; then
    local pid; pid=$(cat "$pidfile" 2>/dev/null || true)
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
    rm -f "$pidfile"
  fi
  pkill -f "_pf_loop_${local_port}_" 2>/dev/null || true
  pkill -f "kubectl.*port-forward.*${local_port}:" 2>/dev/null || true
  echo "[pf::stop_local] ${local_port} cleaned"
}

pf::start() {
  local ns="$1" svc="$2" local_port="$3" remote_port="$4"
  local pidfile; pidfile=$(pf::_pidfile "$local_port")
  local logfile; logfile=$(pf::_logfile "$local_port")

  pf::stop_local "$local_port" >/dev/null 2>&1 || true
  sleep 1

  # 재시도 wrapper — helmfile rolling update 마다 끊겨도 자동 재연결.
  # _pf_loop_<local>_<svc> 라는 함수명 패턴으로 pgrep 식별 가능.
  (
    exec -a "_pf_loop_${local_port}_${svc}" bash -c '
      set -u
      ns="$1"; svc="$2"; lp="$3"; rp="$4"
      while true; do
        kubectl -n "$ns" port-forward "svc/$svc" "$lp:$rp" 2>&1
        echo "[pf] disconnected rc=$?, retry 3s" >&2
        sleep 3
      done
    ' _pf_loop "$ns" "$svc" "$local_port" "$remote_port"
  ) >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  echo "[pf::start] ${ns}/${svc} :${remote_port} → 127.0.0.1:${local_port}  pid=$(cat "$pidfile")  log=${logfile}"
}

pf::probe() {
  local local_port="$1" path="${2:-/v1/models}" max_attempts="${3:-60}" sleep_s="${4:-5}"
  local url="http://127.0.0.1:${local_port}${path}"
  echo "[pf::probe] ${url}"
  for i in $(seq 1 "$max_attempts"); do
    if curl -s --max-time 3 "$url" 2>/dev/null | grep -q '"data"\|"object"'; then
      echo "[pf::probe] OK (${i}×${sleep_s}s)"
      curl -s --max-time 3 "$url" | head -c 400; echo
      return 0
    fi
    sleep "$sleep_s"
  done
  echo "[pf::probe] WARN: no 200 within $((max_attempts * sleep_s))s" >&2
  return 1
}

# /v1/models 응답에서 첫 번째 model id 추출.
# 호출 전 pf::probe 가 200 받았어야 함.
pf::current_model() {
  local local_port="${1:-8011}"
  curl -s --max-time 3 "http://127.0.0.1:${local_port}/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d["data"][0]["id"])
except Exception:
    sys.exit(1)' 2>/dev/null
}

pf::status() {
  echo "── pf::status ──"
  echo "[processes]"
  pf::list
  echo "[listeners]"
  ss -tlnp 2>/dev/null | grep -E ":(80[0-9][0-9]|9[0-9]{3}) " || echo "  (no local listener on 8000-8099, 9000-9999)"
  echo "[pidfiles]"
  ls /tmp/pf_*.pid 2>/dev/null || echo "  (no /tmp/pf_*.pid)"
  echo "[probe 8011]"
  curl -s --max-time 2 http://127.0.0.1:8011/v1/models 2>&1 | head -c 200 || true
  echo
}
