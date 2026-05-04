#!/usr/bin/env bash
# postrender 회귀 — R3 (RollingUpdate GPU deadlock) 차단 패치 보존.
# Deployment manifest 가 stdin 으로 들어왔을 때 strategy.type=Recreate 로
# 변환되는지 검증. kubectl kustomize 가 있어야 동작 — 없으면 SKIP.
set -euo pipefail
IFS=$'\n\t'

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not in PATH — SKIP"
  exit 0
fi

INPUT="$(cat <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ms-infsch-llm-d-modelservice-decode
spec:
  replicas: 2
  selector:
    matchLabels:
      app: x
  template:
    metadata:
      labels:
        app: x
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.17.1
EOF
)"

OUT=$(printf '%s\n' "$INPUT" | bash b200/helmfile/_postrender/postrender.sh)

# runtimeClassName 주입 검증 (기존 패치)
[[ "$OUT" == *"runtimeClassName: nvidia"* ]] || { echo "FAIL: runtimeClassName missing"; exit 1; }

# strategy.type=Recreate 주입 검증 (R3 차단)
[[ "$OUT" == *"strategy:"* ]] && [[ "$OUT" == *"type: Recreate"* ]] || {
  echo "FAIL: strategy.type=Recreate missing"
  echo "--- output ---"
  echo "$OUT"
  exit 1
}

# RollingUpdate 가 남아있지 않아야 (혹시 다른 deployment 가 default 로 들어가도 OK 단,
# 우리 패치 대상 manifest 에 한해)
if echo "$OUT" | grep -A2 "type: RollingUpdate" >/dev/null 2>&1; then
  echo "FAIL: RollingUpdate residual found"
  exit 1
fi
