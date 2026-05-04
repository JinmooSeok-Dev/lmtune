#!/usr/bin/env bash
# kubectl port-forward keepalive — pod cycle 후 자동 재기동.
#
# `kubectl port-forward svc/X` 는 시작 시점에 svc 의 backend pod 한 개를 골라 터널을
# 만든다. pod 가 사라지면 (rolling update / autotune cycle) 터널이 죽고 다시 살아
# 나지 않는다. autotune cycle 중 endpoint url 안정성을 보장하려면 외부 keepalive
# 가 필요.
#
# 사용:
#   NS=mini-infsch SVC=vllm-decode LOCAL=9100 REMOTE=8000 \
#     bash b200/scripts/pf_keepalive.sh
#
# 또는 nohup 백그라운드:
#   nohup bash b200/scripts/pf_keepalive.sh > /tmp/pf_keepalive.log 2>&1 &
#
# 종료: PID 파일 (/tmp/pf_keepalive_<svc>_<local>.pid) 의 PID 를 kill.

set -uo pipefail

NS="${NS:-mini-infsch}"
SVC="${SVC:-vllm-decode}"
LOCAL="${LOCAL:-9100}"
REMOTE="${REMOTE:-8000}"
PIDFILE="/tmp/pf_keepalive_${SVC}_${LOCAL}.pid"

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"; exit 0' INT TERM

log() { printf '[pf_keepalive %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

while true; do
  log "starting kubectl port-forward -n $NS svc/$SVC $LOCAL:$REMOTE"
  kubectl -n "$NS" port-forward "svc/$SVC" "$LOCAL:$REMOTE" \
    --address 127.0.0.1 || true
  log "port-forward exited (pod cycle?). retrying in 2s ..."
  sleep 2
done
