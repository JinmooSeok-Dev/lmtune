#!/usr/bin/env bash
# ops/status.sh — 현재 환경 상태 한 화면.
#
# 사용:
#   bash b200/scripts/ops/status.sh           # rn=infsch
#   bash b200/scripts/ops/status.sh pd        # pd-disaggregation
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../util/pf.sh"
# shellcheck disable=SC1091
source "${HERE}/../util/helm.sh"

RN="${1:-infsch}"
NS="b200-${RN}"

hr() { printf '── %-72s ──\n' "$1"; }

hr "1. cluster"
kubectl version --request-timeout=5s --short 2>/dev/null | head -3 || echo "  kubectl unreachable"
kubectl get ns "$NS" -o jsonpath='  ns={.metadata.name} status={.status.phase}' 2>/dev/null && echo \
  || echo "  ns=${NS} not found"

hr "2. helm releases (rn=${RN})"
helmd::list "$RN" || echo "  (none)"
helmd::releases_check "$RN" >/dev/null 2>&1 && echo "  releases OK" || echo "  ⚠ some releases missing"

hr "3. decode deployment / pods"
kubectl -n "$NS" get deploy -l llm-d.ai/role=decode 2>/dev/null || echo "  (none)"
kubectl -n "$NS" get pods   -l llm-d.ai/role=decode -o wide 2>/dev/null || true

hr "4. gateway / service"
kubectl -n "$NS" get svc 2>/dev/null | head -10 || echo "  (none)"

hr "5. port-forward"
pf::status

hr "6. env"
echo "  B200_MODEL_VALUES=${B200_MODEL_VALUES:-(unset)}"
echo "  KUBECONFIG=${KUBECONFIG:-~/.kube/config}"
