#!/usr/bin/env bash
# ops/prepare.sh — 다음 lmtune 실험 진입 직전, 사전 조건을 한 번에 정비.
#
# 검증·복구 항목 (idempotent, 안전한 것만 자동, 나머지는 명확한 안내):
#   1. kubectl 가 클러스터에 닿고 namespace 존재
#   2. helm release 3종 (infra-<rn> / gaie-<rn> / ms-<rn>) deployed
#      → 빠지면 helmfile apply (사용자 승인 옵션 --apply)
#   3. decode Deployment Available
#   4. stale port-forward 정리
#   5. infra gateway port-forward 데몬 (재시도 wrapper) 띄움
#   6. /v1/models 200 응답 검증
#   7. B200_MODEL_VALUES env 안내 (helmfile apply 시 필요)
#
# 사용:
#   bash b200/scripts/ops/prepare.sh                 # rn=infsch, helmfile apply 안 함 (검증만)
#   bash b200/scripts/ops/prepare.sh infsch          # 동일
#   bash b200/scripts/ops/prepare.sh infsch --apply  # release 빠지면 helmfile apply 자동 실행
#   bash b200/scripts/ops/prepare.sh pd              # pd-disaggregation
#
# 종료 코드: 0 = 모든 사전 조건 OK, 2 = 검증 실패 (사용자 조치 필요), 3 = 부분 성공 (probe 실패)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../util/pf.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/helm.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/env.sh"

RN="${1:-infsch}"
APPLY=0
[[ "${2:-}" == "--apply" ]] && APPLY=1

NS="b200-${RN}"
LOCAL=8011

step() { echo; echo "── [$1] $2"; }

step 1 "cluster check (rn=${RN}, ns=${NS})"
bench_env::cluster_check "$RN" || exit 2

step 2 "helm release 3종"
if ! helmd::releases_check "$RN"; then
  if [[ "$APPLY" -eq 1 ]]; then
    echo "  → --apply 지정, helmfile apply 실행"
    if [[ -z "${B200_MODEL_VALUES:-}" ]]; then
      echo "  WARN: B200_MODEL_VALUES 미설정 — chart default 모델로 install 됩니다." >&2
      echo "  의도된 모델로 띄우려면 먼저:  export B200_MODEL_VALUES=values-<model>.yaml.gotmpl" >&2
    fi
    cd "$ROOT"
    helmd::apply "$RN"
    helmd::releases_check "$RN" || { echo "[prepare] release 여전히 missing" >&2; exit 2; }
  else
    echo "  --apply 미지정 — 자동 install 안 함." >&2
    echo "  install 하려면:  bash $0 ${RN} --apply  (또는 helmfile apply 직접)" >&2
    exit 2
  fi
fi

step 3 "decode Deployment Available 대기"
helmd::wait_decode_ready "$RN" 20m || {
  echo "[prepare] decode 가 20m 안에 Available 안 됨" >&2
  kubectl -n "$NS" get pods -l llm-d.ai/role=decode
  exit 2
}

step 4 "stale port-forward 정리"
pf::stop_local "$LOCAL"

step 5 "gateway port-forward 데몬"
SVC="infra-${RN}-inference-gateway"
if ! kubectl -n "$NS" get svc "$SVC" >/dev/null 2>&1; then
  echo "[prepare] svc/${SVC} 없음" >&2
  kubectl -n "$NS" get svc
  exit 2
fi
pf::start "$NS" "$SVC" "$LOCAL" 80

step 6 "probe /v1/models (최대 5분)"
if pf::probe "$LOCAL" /v1/models; then
  echo
  echo "[prepare] OK — 다음 실행 가능:"
  echo "  lmtune search start --adapter llmd-k8s ..."
  exit 0
fi

echo "[prepare] WARN: probe 실패 — 데몬은 살아있음. 모델 로딩이 더 길 수 있음." >&2
exit 3
