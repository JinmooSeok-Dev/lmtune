#!/usr/bin/env bash
# b200/helmfile/_postrender/postrender.sh — helm --post-renderer hook.
#
# 목적: llm-d-modelservice v0.4.12 chart 가 spec.template.spec.runtimeClassName 를
# values 로 expose 하지 않는다. NHN Cloud k3s 처럼 default runtime 이 runc 이고
# RuntimeClass 'nvidia' 로 GPU pod 를 격리하는 multi-RuntimeClass 환경에서는,
# chart 가 만든 Deployment 가 nvidia runtime 으로 안 떠서 vLLM 이 libcuda.so.1
# 를 못 찾고 죽는다. 본 post-renderer 가 모든 Deployment 에 runtimeClassName
# 'nvidia' 를 주입해 그 격차를 메운다.
#
# helmfile.yaml.gotmpl 의 release 에서:
#   postRenderer: ../_postrender/postrender.sh
#
# 동작: stdin 의 multi-doc YAML manifest 를 받아 kustomize 로 Deployment 만
# patch 한 뒤 stdout 으로 반환. Service / HTTPRoute / Gateway 등 다른 리소스는
# 손대지 않는다.
#
# 외부 의존: `kubectl` (kustomize 빌트인). 클러스터 접근 가능한 클라이언트엔 항상 있음.

set -euo pipefail
IFS=$'\n\t'

readonly SCRIPT_NAME="$(basename "$0")"

command -v kubectl >/dev/null 2>&1 || {
  printf '%s: kubectl required (kustomize 빌트인)\n' "${SCRIPT_NAME}" >&2
  exit 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# helm 이 stdin 으로 multi-doc manifest 전달
cat > "${TMP}/all.yaml"

cat > "${TMP}/kustomization.yaml" <<'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - all.yaml
patches:
  - target:
      kind: Deployment
    patch: |-
      - op: add
        path: /spec/template/spec/runtimeClassName
        value: nvidia
  # GPU deadlock 차단 — chart default RollingUpdate (maxSurge=25%) 가
  # 16 GPU 클러스터에서 8-GPU pod 를 redeploy 시 deadlock (Insufficient
  # nvidia.com/gpu) 을 일으킴. Recreate 가 LLM 서빙의 정합 strategy:
  # 기존 pod 모두 삭제 후 새 pod 띄움. surge 가 의미 없는 영역이라 손실 없음.
  # 자세한 진단/재현은 b200/docs/regressions.md 의 R3 참조.
  - target:
      kind: Deployment
    patch: |-
      - op: replace
        path: /spec/strategy
        value:
          type: Recreate
EOF

exec kubectl kustomize "${TMP}"
