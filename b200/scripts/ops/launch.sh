#!/usr/bin/env bash
# ops/launch.sh — endpoint YAML 한 줄로 launcher 진입 가능 상태까지 끌어올린다.
#
# "처음 시작" 과 "설정 변경 후 재실행" 은 vLLM 의 본성상 본질적으로 같은 비용
# (config change = engine restart = weight reload). 따라서 두 시나리오를
# 동일 진입점으로 통합. idempotent — 여러 번 실행해도 안전.
#
# 처리 단계 (모두 자동, 사용자 손작업 0):
#   1. endpoint YAML 파싱 → 의도한 model 추출
#   2. model → values 파일 매핑 → B200_MODEL_VALUES 자동 export
#   3. helm release 3종 deployed 검증
#   4. 현 vLLM 의 model id 와 endpoint 의 model 비교
#      → 일치: skip helmfile apply
#      → 불일치 또는 release 미설치: helmfile apply (의도한 모델로)
#   5. decode Deployment Available 대기
#   6. stale port-forward 정리 + 재시도 wrapper 데몬
#   7. /v1/models 200 polling
#   8. 응답 model id 와 endpoint 의 model 최종 일치 검증
#
# 사용:
#   bash b200/scripts/ops/launch.sh b200/endpoints/b200_gpt-oss-120b.yaml
#   bash b200/scripts/ops/launch.sh b200/endpoints/b200_gpt-oss-120b.yaml infsch
#
# 종료 코드: 0 = launcher 진입 가능, 2 = 사용자 조치 필요, 3 = redeploy 시도 후 mismatch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../util/pf.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/helm.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/env.sh"

ENDPOINT_YAML="${1:?endpoint YAML 경로 필요}"
RN="${2:-infsch}"
LOCAL=8011
NS="b200-${RN}"

step() { echo; echo "── [launch:${RN}] $1"; }

[[ -f "$ENDPOINT_YAML" ]] || { echo "[launch] endpoint not found: $ENDPOINT_YAML" >&2; exit 2; }

step "1. endpoint 파싱"
EXPECTED_MODEL=$(bench_env::model_from_endpoint "$ENDPOINT_YAML")
[[ -n "$EXPECTED_MODEL" ]] || { echo "[launch] endpoint 의 model 필드 추출 실패" >&2; exit 2; }
echo "  expected_model: ${EXPECTED_MODEL}"

step "2. model → values 매핑 → B200_MODEL_VALUES export"
VALUES_FILE=$(bench_env::values_for_model "$EXPECTED_MODEL") || exit $?
export B200_MODEL_VALUES="$VALUES_FILE"
echo "  B200_MODEL_VALUES=${B200_MODEL_VALUES}"

step "3. cluster + ns"
bench_env::cluster_check "$RN" || exit 2

step "4. helm release 3종 + 모델 일치"
NEED_APPLY=0
if ! helmd::releases_check "$RN" >/dev/null 2>&1; then
  echo "  releases missing → helmfile apply 필요"
  NEED_APPLY=1
else
  # 살아있을 때만 현 모델 빠르게 확인 (port-forward 가 이미 떠있을 때 의미 있음).
  # 안 떠 있으면 step 6 이후 다시 검증되므로 여기서 false-negative 무시.
  CUR_MODEL=$(pf::current_model "$LOCAL" || true)
  if [[ -n "${CUR_MODEL:-}" && "$CUR_MODEL" != "$EXPECTED_MODEL" ]]; then
    echo "  현재 vLLM model='${CUR_MODEL}' ≠ 기대 '${EXPECTED_MODEL}' → redeploy"
    NEED_APPLY=1
  fi
fi

if [[ "$NEED_APPLY" -eq 1 ]]; then
  echo "  → helmfile apply (B200_MODEL_VALUES=${B200_MODEL_VALUES})"
  cd "$ROOT"
  helmd::apply "$RN"
fi

step "5. decode Deployment Available 대기"
helmd::wait_decode_ready "$RN" 20m || {
  echo "[launch] decode 가 20m 안에 Available 안 됨" >&2
  kubectl -n "$NS" get pods -l llm-d.ai/role=decode
  exit 2
}

step "6. port-forward (stale 정리 + 데몬)"
pf::stop_local "$LOCAL"
SVC="infra-${RN}-inference-gateway"
kubectl -n "$NS" get svc "$SVC" >/dev/null 2>&1 \
  || { echo "[launch] svc/${SVC} 없음" >&2; exit 2; }
pf::start "$NS" "$SVC" "$LOCAL" 80

step "7. /v1/models 200 polling"
pf::probe "$LOCAL" /v1/models || {
  echo "[launch] /v1/models 미응답 — vLLM 모델 로딩 더 길 수 있음. 데몬은 살아있음." >&2
  echo "  kubectl -n ${NS} logs -l llm-d.ai/role=decode --tail=80" >&2
  exit 2
}

step "8. model id 일치 검증"
ACTUAL_MODEL=$(pf::current_model "$LOCAL")
echo "  actual_model:   ${ACTUAL_MODEL}"
echo "  expected_model: ${EXPECTED_MODEL}"
if [[ "$ACTUAL_MODEL" != "$EXPECTED_MODEL" ]]; then
  echo "[launch] model mismatch — helmfile apply 가 의도대로 적용되지 않음" >&2
  echo "  현재 떠있는 모델이 endpoint 와 다릅니다. 수동 helmfile destroy 후 재시도 권장." >&2
  exit 3
fi

echo
echo "[launch] OK — launcher 진입 가능"
echo "  curl http://127.0.0.1:${LOCAL}/v1/models"
echo "  lmtune search start --endpoint ${ENDPOINT_YAML} --adapter llmd-k8s ..."
exit 0
