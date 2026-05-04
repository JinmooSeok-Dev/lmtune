#!/usr/bin/env bash
# ops/reset.sh — 이전 실험 잔재 정리. idempotent.
#
# 정리 항목 (단계별 옵트인):
#   기본 (소프트):
#     - 모든 kubectl port-forward 프로세스 종료
#     - /tmp/pf_*.{pid,log} 정리
#   --pods: decode pods rolling restart (helm release 는 유지, weight cache 도 유지)
#   --hard: helmfile destroy (release 3종 모두 uninstall) — 신중
#
# 사용:
#   bash b200/scripts/ops/reset.sh                  # 소프트
#   bash b200/scripts/ops/reset.sh infsch --pods    # decode pod restart
#   bash b200/scripts/ops/reset.sh infsch --hard    # helmfile destroy (확인 prompt)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../util/pf.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/helm.sh"

RN="${1:-infsch}"
MODE="${2:-soft}"
NS="b200-${RN}"

step() { echo; echo "── [reset:${MODE}] $1"; }

step "soft: port-forward 정리"
pf::stop_all
rm -f /tmp/pf_*.log 2>/dev/null || true

case "$MODE" in
  soft|"")
    echo "[reset] done (soft)"
    ;;
  --pods|pods)
    step "pods: decode rollout restart (release 유지)"
    if kubectl get ns "$NS" >/dev/null 2>&1; then
      kubectl -n "$NS" rollout restart deploy -l llm-d.ai/role=decode || true
      kubectl -n "$NS" rollout status  deploy -l llm-d.ai/role=decode --timeout=10m || true
    else
      echo "  ns=${NS} 없음, skip"
    fi
    ;;
  --hard|hard)
    step "hard: helmfile destroy (release 3종 uninstall)"
    echo "  ns=${NS} 의 infra-${RN} / gaie-${RN} / ms-${RN} 가 모두 삭제됩니다."
    read -r -p "  진행하려면 'yes' 입력: " ans
    [[ "$ans" == "yes" ]] || { echo "  취소"; exit 0; }
    helmd::destroy "$RN" || true
    echo "[reset] done (hard)"
    ;;
  *)
    echo "[reset] unknown mode: $MODE (expected: soft|--pods|--hard)" >&2
    exit 2
    ;;
esac
