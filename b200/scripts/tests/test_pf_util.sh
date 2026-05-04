#!/usr/bin/env bash
# pf util 함수 회귀.
set -euo pipefail
IFS=$'\n\t'

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source b200/scripts/util/pf.sh

# pf::list — unbound 안 내고 정상 종료 (호스트에 PF 가 떠있을 수도, 없을 수도)
pf::list >/dev/null 2>&1

# pf::stop_all 이 idempotent (반복 호출 안전). 단 본 테스트는 호스트의 PF 데몬을
# 실제로 죽이지 않는다 — fake_dry 모드 아닌 한.
# (호스트 격리 책임은 사용자가 ops/reset.sh 로. 여기선 함수 흐름만 검증.)

# pf::stop_local 이 PID 파일 없어도 정상 종료
pf::stop_local 99991 >/dev/null

# pf::current_model — model id 추출 (fake curl)
# shellcheck disable=SC1091
source b200/scripts/tests/fakes/setup.sh
trap 'rm -rf "$FAKE_BIN"' EXIT
PATH="$FAKE_BIN:$PATH"

export FAKE_CURL_MODEL="openai/gpt-oss-120b"
got=$(pf::current_model 8011)
[[ "$got" == "openai/gpt-oss-120b" ]] || { echo "expected gpt-oss-120b, got: '$got'"; exit 1; }

unset FAKE_CURL_MODEL
got=$(pf::current_model 8011 2>/dev/null || true)
[[ -z "$got" ]] || { echo "expected empty (refused), got: '$got'"; exit 1; }
