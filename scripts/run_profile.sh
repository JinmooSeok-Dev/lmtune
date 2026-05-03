#!/usr/bin/env bash
# 단일 profile 실행 + 리포트 생성 단축 스크립트.
#
# 사용: ./scripts/run_profile.sh <profile.yaml> <endpoint.yaml>

set -euo pipefail

PROFILE="${1:?profile yaml required}"
ENDPOINT="${2:?endpoint yaml required}"

# lmtune run 은 stdout 에 run_id 를 포함한 행을 남김: "run_id=<ULID>..."
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

lmtune run --profile "$PROFILE" --endpoint "$ENDPOINT" 2>&1 | tee "$LOG"
RUN_ID=$(grep -oE 'run_id=[A-Z0-9]+' "$LOG" | head -1 | cut -d= -f2)
if [[ -n "$RUN_ID" ]]; then
    lmtune report "$RUN_ID"
fi
