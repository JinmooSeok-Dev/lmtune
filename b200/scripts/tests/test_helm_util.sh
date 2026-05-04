#!/usr/bin/env bash
# helm util 함수 회귀 — 특히 R2 (PR #25 set -u + 한 줄 다중 local) 와
# R1 (PR #26 wait_decode_ready 의 라벨 부재 우회) 보호.
set -euo pipefail
IFS=$'\n\t'

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source b200/scripts/tests/fakes/setup.sh
# shellcheck disable=SC1091
source b200/scripts/util/helm.sh
trap 'rm -rf "$FAKE_BIN"' EXIT
PATH="$FAKE_BIN:$PATH"

# R2 회귀 — 'local rn="$1" ns="b200-${rn}"' 한 줄 다중 선언이 unbound 폭주
#   set -u 환경에서 cluster_check / list / releases_check 호출 시 unbound 없어야 함
set -u

# helmd::file_for 매핑
[[ "$(helmd::file_for infsch)" == "b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl" ]]
[[ "$(helmd::file_for pd)"     == "b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl" ]]
[[ "$(helmd::file_for wideep)" == "b200/helmfile/wide-ep-lws/helmfile.yaml.gotmpl" ]]
helmd::file_for bogus 2>/dev/null && { echo "expected fail on bogus"; exit 1; } || true

# helmd::list — fake helm 가 release 3종 반환
FAKE_HELM_LIST=all helmd::list infsch >/dev/null

# helmd::releases_check — 모두 deployed
FAKE_HELM_LIST=all helmd::releases_check infsch >/dev/null
# 일부 missing
FAKE_HELM_LIST=partial helmd::releases_check infsch && { echo "expected fail"; exit 1; } || true
# 모두 missing
FAKE_HELM_LIST=none helmd::releases_check infsch && { echo "expected fail"; exit 1; } || true

# R1 회귀 — wait_decode_ready 가 deployment metadata 라벨 셀렉터로 찾으면
# llm-d-modelservice chart 가 그 라벨 안 붙여서 'no matching resources' 발생.
# 이름 패턴 매칭으로 우회한 동작 검증.
FAKE_K8S=has_decode helmd::wait_decode_ready infsch 1m >/dev/null
FAKE_K8S=no_decode  helmd::wait_decode_ready infsch 1m 2>/dev/null && {
  echo "expected fail (no decode)"; exit 1;
} || true
