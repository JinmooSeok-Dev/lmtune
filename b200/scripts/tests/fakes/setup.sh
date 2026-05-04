# tests/fakes/setup.sh — fake kubectl/helm/helmfile/curl 를 PATH 앞에 설치.
# 각 fake binary 의 응답은 환경변수 (FAKE_K8S, FAKE_CURL_MODEL, FAKE_HELM_LIST,
# FAKE_HELMFILE_RC) 로 제어한다.
#
# source 후:
#   PATH="$FAKE_BIN:$PATH"
# 처럼 사용. test 끝나면 trap 으로 cleanup 자동.

# shellcheck shell=bash

if [[ "${FAKES_SETUP:-}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
FAKES_SETUP=1

FAKE_BIN="$(mktemp -d)"
export FAKE_BIN

# kubectl —----------------------------------------------------------------
cat > "$FAKE_BIN/kubectl" <<'KUBECTL'
#!/bin/bash
ARGS="$*"
case "${FAKE_K8S:-default}" in
  unreachable)
    exit 1 ;;
  no_ns)
    case "$ARGS" in
      *"version"*) exit 0 ;;
      *"get ns"*)  exit 1 ;;
      *) exit 1 ;;
    esac ;;
  has_decode)
    case "$ARGS" in
      *"version"*) exit 0 ;;
      *"get ns "*) echo "Active"; exit 0 ;;
      *"get deploy"*"jsonpath"*)
        echo "infra-infsch-inference-gateway"
        echo "gaie-infsch-epp"
        echo "ms-infsch-llm-d-modelservice-decode"
        exit 0 ;;
      *"wait deploy/ms-infsch"*)
        echo "deployment.apps/ms-infsch-llm-d-modelservice-decode condition met"
        exit 0 ;;
      *"get svc infra-infsch-inference-gateway"*) exit 0 ;;
      *) exit 0 ;;
    esac ;;
  no_decode)
    case "$ARGS" in
      *"version"*) exit 0 ;;
      *"get ns "*) echo "Active"; exit 0 ;;
      *"get deploy"*"jsonpath"*)
        echo "infra-infsch-inference-gateway"
        echo "gaie-infsch-epp"
        exit 0 ;;
      *"get deploy"*) echo "infra-infsch-inference-gateway 1/1"; exit 0 ;;
      *) exit 0 ;;
    esac ;;
  default|*) exit 0 ;;
esac
KUBECTL
chmod +x "$FAKE_BIN/kubectl"

# helm —-------------------------------------------------------------------
cat > "$FAKE_BIN/helm" <<'HELM'
#!/bin/bash
ARGS="$*"
case "${FAKE_HELM_LIST:-all}" in
  all)
    if [[ "$ARGS" == *"list -q"* ]]; then
      printf 'infra-infsch\ngaie-infsch\nms-infsch\n'
      exit 0
    fi ;;
  partial)
    if [[ "$ARGS" == *"list -q"* ]]; then
      printf 'infra-infsch\ngaie-infsch\n'
      exit 0
    fi ;;
  none)
    if [[ "$ARGS" == *"list -q"* ]]; then
      exit 0
    fi ;;
esac
[[ "$ARGS" == *"list"* ]] && { echo "NAME    NAMESPACE"; exit 0; }
exit 0
HELM
chmod +x "$FAKE_BIN/helm"

# helmfile —---------------------------------------------------------------
cat > "$FAKE_BIN/helmfile" <<'HELMFILE'
#!/bin/bash
echo "[fake helmfile] $*" >&2
exit "${FAKE_HELMFILE_RC:-0}"
HELMFILE
chmod +x "$FAKE_BIN/helmfile"

# curl —-------------------------------------------------------------------
cat > "$FAKE_BIN/curl" <<'CURL'
#!/bin/bash
url=""
for a in "$@"; do
  case "$a" in http://*|https://*) url="$a" ;; esac
done
if [[ -z "${FAKE_CURL_MODEL:-}" ]]; then
  exit 7
fi
if [[ "$url" == *"/v1/models"* ]]; then
  printf '{"object":"list","data":[{"id":"%s","object":"model"}]}' "$FAKE_CURL_MODEL"
  exit 0
fi
exit 0
CURL
chmod +x "$FAKE_BIN/curl"

# python3 (model id 추출) — system 그대로 통과시키되 exec 경로만
# 실제 system python3 사용 (fake 안 만듦). pf::current_model 은 python3 -c 호출.
