#!/usr/bin/env bash
# tests/run_all.sh — bash util / ops 함수 회귀 테스트.
#
# 영속화 정책: bash util 또는 ops 의 함수가 추가/변경/픽스될 때마다 본 디렉토리에
# 시나리오 테스트가 1개 이상 추가되어야 한다 (CLAUDE.md 의 § PR 게이트 참조).
#
# 한계 — 본 테스트가 "잡지 못하는" 결함:
#   - K8s pod scheduling (GPU 리소스 충돌, taint/toleration)
#   - Deployment strategy 의 GPU deadlock (RollingUpdate vs Recreate)
#   - Chart 가 만드는 manifest 의 라벨/필드 부재
#   - vLLM 모델 로딩 시간 / 응답 형태
# 이런 것은 b200/docs/regressions.md 의 catalog 를 통해 영속화한다.
#
# 사용:
#   bash b200/scripts/tests/run_all.sh
#   bash b200/scripts/tests/run_all.sh --only env       # env.sh 만
set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

if [[ -t 2 ]]; then
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RESET=$'\033[0m'
else
  C_RED=''; C_GREEN=''; C_YELLOW=''; C_RESET=''
fi

PASS=0; FAIL=0; SKIP=0
FAILED_TESTS=()

run_test() {
  local script="$1"
  local name; name=$(basename "$script" .sh)
  printf '── %s ' "$name"
  if bash "$script" >/tmp/test_${name}.log 2>&1; then
    printf '%s✓ PASS%s\n' "$C_GREEN" "$C_RESET"
    PASS=$((PASS + 1))
  else
    printf '%s✗ FAIL%s (log: /tmp/test_%s.log)\n' "$C_RED" "$C_RESET" "$name"
    FAIL=$((FAIL + 1))
    FAILED_TESTS+=("$name")
    sed 's/^/    /' /tmp/test_${name}.log | tail -20 >&2
  fi
}

ONLY="${2:-}"
if [[ "${1:-}" == "--only" ]]; then ONLY="$2"; fi

for f in "${SCRIPT_DIR}"/test_*.sh; do
  [[ -e "$f" ]] || continue
  name=$(basename "$f" .sh)
  if [[ -n "$ONLY" && "$name" != *"$ONLY"* ]]; then
    SKIP=$((SKIP + 1))
    continue
  fi
  run_test "$f"
done

echo
echo "── summary ──"
printf '  %sPASS%s %d  %sFAIL%s %d  SKIP %d\n' \
  "$C_GREEN" "$C_RESET" "$PASS" "$C_RED" "$C_RESET" "$FAIL" "$SKIP"

if [[ "$FAIL" -gt 0 ]]; then
  printf 'failed: %s\n' "${FAILED_TESTS[*]}" >&2
  exit 1
fi
