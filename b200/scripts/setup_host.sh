#!/usr/bin/env bash
# b200/scripts/setup_host.sh — B200-1 컨테이너 안에서 helmfile / peer repo 설치
#
# 목적: probe.sh --mode host 가 잡아낸 두 FAIL (helmfile 미설치 + peer_repo 부재) 을
#       컨테이너 안 사용자 권한만으로 한 번에 해결한다. sudo / gcsudo 불필요.
#
# 사용:
#   bash b200/scripts/setup_host.sh                                 # helmfile 만 설치
#   PEER_REPO_URL=<git url> bash b200/scripts/setup_host.sh         # + peer repo clone
#
# 환경변수:
#   HELMFILE_VERSION  helmfile 버전 (default 1.1.3, 클라이언트 PC 와 동일 버전 매칭)
#   BIN_DIR           binary 설치 위치 (default $HOME/bin — sudo 불필요)
#   PEER_REPO_URL     peer repo git URL. 미지정 시 clone 단계 skip 후 안내 출력.
#   PEER_REPO_DIR     peer repo clone 경로 (default $HOME/ml_ai/agentic/llm-distributed-inference)

set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_NAME="$(basename "$0")"
readonly HELMFILE_VERSION="${HELMFILE_VERSION:-1.1.3}"
readonly BIN_DIR="${BIN_DIR:-$HOME/bin}"
readonly PEER_REPO_DIR="${PEER_REPO_DIR:-$HOME/ml_ai/agentic/llm-distributed-inference}"

if [[ -t 2 ]]; then
  readonly C_RED=$'\033[31m' C_YELLOW=$'\033[33m' C_GREEN=$'\033[32m' C_RESET=$'\033[0m'
else
  readonly C_RED='' C_YELLOW='' C_GREEN='' C_RESET=''
fi

log()  { printf '%s[INFO]%s  %s\n' "${C_GREEN}"  "${C_RESET}" "$*" >&2; }
warn() { printf '%s[WARN]%s  %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}"    "${C_RESET}" "$*" >&2; exit 1; }

cleanup_files=()
cleanup() {
  local f
  for f in "${cleanup_files[@]:-}"; do
    [[ -n "${f}" && -e "${f}" ]] && rm -f "${f}"
  done
}
trap cleanup EXIT
trap 'die "interrupted"' INT TERM

install_helmfile() {
  if command -v helmfile >/dev/null 2>&1; then
    local current
    current="$(helmfile --version 2>/dev/null | head -1 || true)"
    log "helmfile already installed: ${current:-unknown}"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || die "curl required"
  command -v tar >/dev/null 2>&1 || die "tar required"

  local arch
  case "$(uname -m)" in
    x86_64)  arch="linux_amd64" ;;
    aarch64) arch="linux_arm64" ;;
    *) die "unsupported arch: $(uname -m)" ;;
  esac

  mkdir -p "${BIN_DIR}"

  local tarball
  tarball="$(mktemp -t helmfile.XXXXXX.tar.gz)"
  cleanup_files+=("${tarball}")

  local url
  url="https://github.com/helmfile/helmfile/releases/download/v${HELMFILE_VERSION}/helmfile_${HELMFILE_VERSION}_${arch}.tar.gz"
  log "downloading ${url}"
  curl -fsSL "${url}" -o "${tarball}" || die "download failed (no network or version typo?)"

  tar -xzf "${tarball}" -C "${BIN_DIR}" helmfile || die "extract failed"
  chmod +x "${BIN_DIR}/helmfile"

  if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
    warn "${BIN_DIR} is not in PATH — append this line to ~/.bashrc, then reopen shell:"
    warn "  export PATH=\"${BIN_DIR}:\$PATH\""
  fi

  log "helmfile installed: $("${BIN_DIR}/helmfile" --version 2>/dev/null | head -1)"
}

clone_peer_repo() {
  if [[ -z "${PEER_REPO_URL:-}" ]]; then
    log "PEER_REPO_URL not set — skipping peer repo clone."
    log "  to install later:"
    log "    PEER_REPO_URL=<git url> bash b200/scripts/setup_host.sh"
    return 0
  fi

  if [[ -d "${PEER_REPO_DIR}/.git" ]]; then
    local sha
    sha="$(git -C "${PEER_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    log "peer repo already at ${PEER_REPO_DIR} (HEAD=${sha}) — skip clone"
    return 0
  fi

  command -v git >/dev/null 2>&1 || die "git required"
  mkdir -p "$(dirname "${PEER_REPO_DIR}")"
  log "cloning ${PEER_REPO_URL} → ${PEER_REPO_DIR}"
  git clone "${PEER_REPO_URL}" "${PEER_REPO_DIR}" || die "clone failed"
  log "peer repo cloned: $(git -C "${PEER_REPO_DIR}" rev-parse --short HEAD)"
}

main() {
  log "running ${SCRIPT_NAME}"
  log "  HELMFILE_VERSION = ${HELMFILE_VERSION}"
  log "  BIN_DIR          = ${BIN_DIR}"
  log "  PEER_REPO_DIR    = ${PEER_REPO_DIR}"
  log "  PEER_REPO_URL    = ${PEER_REPO_URL:-<unset, will skip>}"

  install_helmfile
  clone_peer_repo

  log "done — re-run probe to verify:"
  log "  bash b200/scripts/probe.sh --mode host"
}

main "$@"
