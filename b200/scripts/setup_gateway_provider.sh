#!/usr/bin/env bash
# b200/scripts/setup_gateway_provider.sh — gateway provider prereq (cluster-once)
#
# llm-d 정본 가이드를 wrap:
#   https://github.com/llm-d/llm-d/tree/main/guides/prereq/gateway-provider
#
# 정본은 inference helmfile (modelservice + GAIE inferencepool) 이 돌기 전에
# 별도 prereq 단계로 (1) Gateway API + GAIE CRDs, (2) gateway controller 자체를
# 클러스터에 한 번 깔아두라고 안내한다. 본 스크립트는 그 두 단계를 한 줄로 합친다.
#
# 사용:
#   bash b200/scripts/setup_gateway_provider.sh                       # agentgateway, apply
#   bash b200/scripts/setup_gateway_provider.sh kgateway              # kgateway, apply
#   bash b200/scripts/setup_gateway_provider.sh istio                 # istio, apply
#   bash b200/scripts/setup_gateway_provider.sh agentgateway delete   # uninstall
#
# 환경변수:
#   UPSTREAM_REF   참조할 llm-d/llm-d 브랜치/태그 (default main)

set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_NAME="$(basename "$0")"
readonly UPSTREAM_REF="${UPSTREAM_REF:-main}"
readonly UPSTREAM_BASE="https://raw.githubusercontent.com/llm-d/llm-d/${UPSTREAM_REF}/guides/prereq/gateway-provider"

PROVIDER="${1:-agentgateway}"
MODE="${2:-apply}"

case "${PROVIDER}" in
  agentgateway|kgateway|istio) ;;
  *) echo "${SCRIPT_NAME}: unknown provider '${PROVIDER}' (agentgateway|kgateway|istio)" >&2; exit 2 ;;
esac
case "${MODE}" in
  apply|delete) ;;
  *) echo "${SCRIPT_NAME}: unknown mode '${MODE}' (apply|delete)" >&2; exit 2 ;;
esac

if [[ -t 2 ]]; then
  readonly C_GREEN=$'\033[32m' C_YELLOW=$'\033[33m' C_RED=$'\033[31m' C_RESET=$'\033[0m'
else
  readonly C_GREEN='' C_YELLOW='' C_RED='' C_RESET=''
fi
log()  { printf '%s[setup-gw]%s %s\n' "${C_GREEN}"  "${C_RESET}" "$*" >&2; }
warn() { printf '%s[setup-gw]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
die()  { printf '%s[setup-gw]%s %s\n' "${C_RED}"    "${C_RESET}" "$*" >&2; exit 1; }

command -v kubectl  >/dev/null 2>&1 || die "kubectl required (clientside)"
command -v helmfile >/dev/null 2>&1 || die "helmfile required (clientside; see setup_host.sh)"
command -v curl     >/dev/null 2>&1 || die "curl required"

cleanup_files=()
cleanup() {
  local f
  for f in "${cleanup_files[@]:-}"; do
    [[ -n "${f}" && -e "${f}" ]] && rm -f "${f}"
  done
}
trap cleanup EXIT

log "step 1/2: Gateway API + GAIE CRDs (${MODE})"
log "  source: ${UPSTREAM_BASE}/install-gateway-provider-dependencies.sh"
curl -fsSL "${UPSTREAM_BASE}/install-gateway-provider-dependencies.sh" \
  | bash -s "${MODE}" \
  || die "CRD install/uninstall failed"

log "step 2/2: ${PROVIDER} controller (${MODE})"
helmfile_yaml="$(mktemp -t llm-d-gw.XXXXXX.yaml)"
cleanup_files+=("${helmfile_yaml}")

helmfile_url="${UPSTREAM_BASE}/${PROVIDER}.helmfile.yaml"
log "  source: ${helmfile_url}"
curl -fsSL "${helmfile_url}" -o "${helmfile_yaml}" \
  || die "fetch ${PROVIDER}.helmfile.yaml failed"

helmfile -f "${helmfile_yaml}" "${MODE}" \
  || die "helmfile ${MODE} failed"

log "done. verify:"
case "${MODE}" in
  apply)
    log "  kubectl get crd | grep -iE 'gateway|inferencepool'"
    log "  kubectl get pods -n agentgateway-system 2>/dev/null \\"
    log "    | grep -E 'agentgateway|controller'"
    log "  next: helmfile -f b200/helmfile/<path>/helmfile.yaml.gotmpl apply"
    ;;
  delete)
    log "  cluster gateway-provider artifacts removed"
    ;;
esac
