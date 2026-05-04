#!/usr/bin/env bash
# pf_gateway.sh — llm-d infra gateway port-forward 데몬 thin entrypoint.
# 본 파일은 util/pf.sh + util/helm.sh 의 함수를 호출하기만 하는 진입점이다.
#
# decode service 직접 forward 는 InferencePool/EPP smart routing 우회로 금지
# (b200/docs/port_forward_runbook.md).
#
# 사용:
#   bash b200/scripts/pf_gateway.sh                # rn=infsch
#   bash b200/scripts/pf_gateway.sh pd             # pd-disaggregation
#   bash b200/scripts/pf_gateway.sh wideep         # wide-ep-lws
#   bash b200/scripts/pf_gateway.sh infsch 8012    # local port override
#   bash b200/scripts/pf_gateway.sh --status
#   bash b200/scripts/pf_gateway.sh --stop [rn]    # rn 생략 시 전체 정리
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/util/pf.sh"
# shellcheck disable=SC1091
source "${HERE}/util/helm.sh"

case "${1:-}" in
  --status|status)
    pf::status
    exit 0
    ;;
  --stop|-s|stop)
    if [[ -n "${2:-}" ]]; then
      pf::stop_local 8011
    else
      pf::stop_all
    fi
    exit 0
    ;;
  --help|-h|help)
    sed -n '1,18p' "$0"; exit 0
    ;;
esac

RN="${1:-infsch}"
LOCAL="${2:-8011}"
NS="b200-${RN}"
SVC="infra-${RN}-inference-gateway"

echo "[pf_gateway] rn=${RN} ns=${NS} svc=${SVC} :80 → 127.0.0.1:${LOCAL}"

# 1. release 3종 검증 (없으면 명확한 메시지 + helmfile apply 안내)
if ! helmd::releases_check "$RN"; then
  echo "  → 다음 명령으로 install:" >&2
  echo "    bash b200/scripts/ops/prepare.sh ${RN}" >&2
  echo "    (또는: helmfile -f $(helmd::file_for "$RN") apply)" >&2
  exit 2
fi

# 2. service 실재 검증
if ! kubectl -n "$NS" get svc "$SVC" >/dev/null 2>&1; then
  echo "[pf_gateway] FAIL: svc/${SVC} not found in ns=${NS}" >&2
  kubectl -n "$NS" get svc >&2
  exit 3
fi

# 3. 데몬 + probe
pf::start "$NS" "$SVC" "$LOCAL" 80
pf::probe "$LOCAL" /v1/models || {
  echo "[pf_gateway] decode pod 가 아직 모델 로딩 중일 수 있음. 데몬은 살아 있음." >&2
  echo "  kubectl -n ${NS} get pods -l llm-d.ai/role=decode" >&2
  echo "  kubectl -n ${NS} logs -l llm-d.ai/role=decode --tail=80" >&2
}
