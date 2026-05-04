# util/env.sh — 다음 실험 사전 조건 환경 변수 검증/세팅.
#
# 사용: `source b200/scripts/util/env.sh` 후 `bench_env::<func>` 호출.

# shellcheck shell=bash

# B200_MODEL_VALUES — helmfile 의 model values 파일 (env override 없으면 default
# 가 들어가 모델이 갈아끼워짐). endpoint YAML 의 모델과 일치해야 한다.
bench_env::require_model_values() {
  local expected="$1"  # 예: values-gpt-oss-120b.yaml.gotmpl
  if [[ -z "${B200_MODEL_VALUES:-}" ]]; then
    echo "[env] B200_MODEL_VALUES 미설정 — export 필요:" >&2
    echo "  export B200_MODEL_VALUES=${expected}" >&2
    return 2
  fi
  if [[ -n "$expected" && "$B200_MODEL_VALUES" != "$expected" ]]; then
    echo "[env] B200_MODEL_VALUES=${B200_MODEL_VALUES}  (예상=${expected})" >&2
    echo "  → endpoint 와 모델이 다를 수 있음. 의도적이면 무시." >&2
    return 1
  fi
  echo "[env] B200_MODEL_VALUES=${B200_MODEL_VALUES}"
  return 0
}

# endpoint YAML 의 deployment.helmfile_overrides 블록에서 release_name suffix
# (= rn) 를 추출. 예: ms-infsch → infsch
bench_env::rn_from_endpoint() {
  local endpoint_yaml="$1"
  local rname
  rname=$(awk '/^[[:space:]]+release_name:/ {print $2; exit}' "$endpoint_yaml" \
    | tr -d '"'"'"'')
  if [[ -z "$rname" || "$rname" != ms-* ]]; then
    echo "[env] cannot derive rn from $endpoint_yaml (release_name=$rname)" >&2
    return 2
  fi
  echo "${rname#ms-}"
}

# endpoint YAML 의 url 에서 expected local port 추출 (default 8011)
bench_env::local_port_from_endpoint() {
  local endpoint_yaml="$1"
  local port
  port=$(awk '/^url:/ {print $2; exit}' "$endpoint_yaml" \
    | sed -nE 's|.*://[^:]+:([0-9]+).*|\1|p')
  echo "${port:-8011}"
}

# RUNTIME 검증 — kubectl 가 닿는 클러스터, namespace 존재
bench_env::cluster_check() {
  local rn="$1"
  local ns="b200-${rn}"
  if ! kubectl version --request-timeout=5s >/dev/null 2>&1; then
    echo "[env] kubectl 가 클러스터에 닿지 않음 (kubeconfig?)" >&2
    return 2
  fi
  if ! kubectl get ns "$ns" >/dev/null 2>&1; then
    echo "[env] namespace '${ns}' 없음 (rn=${rn})" >&2
    return 2
  fi
  echo "[env] cluster OK, ns=${ns} exists"
  return 0
}
