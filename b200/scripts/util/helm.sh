# util/helm.sh — llm-d helm release 검증/조작 함수.
#
# 사용: `source b200/scripts/util/helm.sh` 후 `helmd::<func>` 호출.
#
# llm-d well-lit-path 의 release 명명 규칙: infra-<rn> / gaie-<rn> / ms-<rn>
# (release suffix `<rn>` 가 path 구분자 = infsch | pd | wideep)

# shellcheck shell=bash

helmd::list() {
  local rn="$1" ns="b200-${rn}"
  helm -n "$ns" list 2>/dev/null
}

# 3종 release 가 모두 deployed 인지 0/N 으로 반환. 빠진 release 이름 stdout.
# return 0 = 모두 OK, 1 = 일부 missing
helmd::releases_check() {
  local rn="$1" ns="b200-${rn}"
  local missing=()
  for r in infra gaie ms; do
    if ! helm -n "$ns" list -q 2>/dev/null | grep -qE "^${r}-${rn}$"; then
      missing+=("${r}-${rn}")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "[helmd] ${ns}: all 3 releases deployed (infra-${rn}, gaie-${rn}, ms-${rn})"
    return 0
  fi
  echo "[helmd] ${ns}: missing ${missing[*]}"
  return 1
}

# helmfile path 매핑 — well-lit-path key → repo 안의 helmfile 파일 경로
helmd::file_for() {
  local rn="$1"
  case "$rn" in
    infsch) echo "b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl" ;;
    pd)     echo "b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl" ;;
    wideep) echo "b200/helmfile/wide-ep-lws/helmfile.yaml.gotmpl" ;;
    *) echo "[helmd] unknown rn: $rn (expected: infsch|pd|wideep)" >&2; return 2 ;;
  esac
}

# helmfile diff (dry-run) — 실 변경 없이 plan 만
helmd::diff() {
  local rn="$1"
  local file; file=$(helmd::file_for "$rn") || return $?
  helmfile --environment default --selector kind=inference-stack -f "$file" diff
}

# helmfile apply — release 3종 install/upgrade. timeout 길게 (large model rollout 수 분).
helmd::apply() {
  local rn="$1"
  local file; file=$(helmd::file_for "$rn") || return $?
  echo "[helmd::apply] ${file}  (env=default, selector=kind=inference-stack)"
  helmfile --environment default --selector kind=inference-stack -f "$file" apply
}

# decode pod ready 대기 — gpt-oss-120b 같은 큰 모델은 weight 다운로드/로딩 5-15분
helmd::wait_decode_ready() {
  local rn="$1" ns="b200-${rn}" timeout="${2:-20m}"
  echo "[helmd::wait] ns=${ns}  timeout=${timeout}"
  kubectl -n "$ns" wait --for=condition=Available deploy \
    -l llm-d.ai/role=decode --timeout="$timeout"
}

# release 모두 삭제 (cleanup) — 신중. 호출 전 확인 필요.
helmd::destroy() {
  local rn="$1"
  local file; file=$(helmd::file_for "$rn") || return $?
  echo "[helmd::destroy] WARNING — uninstall all 3 releases of rn=${rn}"
  helmfile --environment default --selector kind=inference-stack -f "$file" destroy
}
