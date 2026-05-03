#!/usr/bin/env bash
# b200/scripts/probe.sh — 클러스터 진단 (B0)
#
# 목적: B200 k3s 클러스터에서 lmtune search 가 동작할 환경인지 점검.
# 사용: bash b200/scripts/probe.sh [--json]
#
# 점검 항목:
#   1. kubectl 컨텍스트와 노드 상태
#   2. 노드별 GPU 자원 (nvidia.com/gpu, 모델, 드라이버)
#   3. nvidia-device-plugin / Multus / CNI 상태
#   4. peer repo 경로 + helmfile / helm 버전
#   5. ghcr.io 도달 여부 + 이미지 캐시 상태
#   6. RDMA fabric (rdma 디바이스, ibstat / mlx5)
#   7. 인터노드 nccl-tests + iperf3 (옵션, --skip-fabric 으로 생략 가능)
#
# 결과: stdout 에 사람이 읽는 표 + (옵션) JSON. 실패 시 비0 exit.

set -u

JSON_MODE=false
SKIP_FABRIC=false
PEER_REPO_DEFAULT="/home/jinmoo/ml_ai/agentic/llm-distributed-inference"
PEER_REPO="${PEER_REPO:-$PEER_REPO_DEFAULT}"

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
    --skip-fabric) SKIP_FABRIC=true ;;
    --help|-h)
      sed -n '2,18p' "$0" | sed 's/^# *//'
      exit 0 ;;
  esac
done

PASS=0
FAIL=0
WARN=0
declare -A RESULTS

record() {
  local key="$1" status="$2" msg="$3"
  RESULTS["$key"]="$status|$msg"
  case "$status" in
    PASS) PASS=$((PASS+1)) ;;
    FAIL) FAIL=$((FAIL+1)) ;;
    WARN) WARN=$((WARN+1)) ;;
  esac
  if ! $JSON_MODE; then
    case "$status" in
      PASS) printf '  [\033[32mPASS\033[0m] %-40s %s\n' "$key" "$msg" ;;
      WARN) printf '  [\033[33mWARN\033[0m] %-40s %s\n' "$key" "$msg" ;;
      FAIL) printf '  [\033[31mFAIL\033[0m] %-40s %s\n' "$key" "$msg" ;;
    esac
  fi
}

section() {
  $JSON_MODE && return
  printf '\n=== %s ===\n' "$1"
}

# ----------------------------------------------------------------------
# 1. kubectl + 노드
# ----------------------------------------------------------------------
section "1. Kubernetes context"

if ! command -v kubectl >/dev/null 2>&1; then
  record "kubectl.installed" FAIL "not found in PATH"
else
  CTX=$(kubectl config current-context 2>/dev/null || echo "<none>")
  record "kubectl.context" PASS "$CTX"
  if kubectl cluster-info >/dev/null 2>&1; then
    record "kubectl.reachable" PASS "$(kubectl cluster-info 2>/dev/null | head -1 | sed 's/\x1b\[[0-9;]*m//g' | tr -s ' ')"
  else
    record "kubectl.reachable" FAIL "cluster-info failed"
  fi

  NODES=$(kubectl get nodes --no-headers 2>/dev/null)
  NODE_COUNT=$(echo "$NODES" | wc -l | tr -d ' ')
  READY=$(echo "$NODES" | awk '$2=="Ready"' | wc -l | tr -d ' ')
  if [ "$NODE_COUNT" -eq 0 ]; then
    record "k8s.node_count" FAIL "0 nodes visible"
  elif [ "$READY" -eq "$NODE_COUNT" ]; then
    record "k8s.node_count" PASS "$READY/$NODE_COUNT Ready"
  else
    record "k8s.node_count" WARN "$READY/$NODE_COUNT Ready"
  fi
fi

# ----------------------------------------------------------------------
# 2. GPU 자원
# ----------------------------------------------------------------------
section "2. GPU resources"

if [ "$NODE_COUNT" -gt 0 ] 2>/dev/null; then
  TOTAL_GPU=0
  while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    GPU=$(kubectl get node "$NAME" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
    GPU=${GPU:-0}
    TOTAL_GPU=$((TOTAL_GPU + GPU))
    if [ "$GPU" -gt 0 ]; then
      MODEL=$(kubectl get node "$NAME" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.product}' 2>/dev/null || true)
      record "gpu.node.$NAME" PASS "$GPU GPU ${MODEL:-unknown_model}"
    else
      record "gpu.node.$NAME" WARN "0 GPU exposed"
    fi
  done <<< "$NODES"
  if [ "$TOTAL_GPU" -ge 16 ]; then
    record "gpu.total" PASS "$TOTAL_GPU GPU (≥ 16 expected)"
  elif [ "$TOTAL_GPU" -ge 1 ]; then
    record "gpu.total" WARN "$TOTAL_GPU GPU (< 16, multi-node 실험 일부 제한)"
  else
    record "gpu.total" FAIL "0 GPU"
  fi
fi

# ----------------------------------------------------------------------
# 3. device-plugin + Multus + CNI
# ----------------------------------------------------------------------
section "3. Device plugin / CNI"

DP_PODS=$(kubectl get pods -A -l 'app.kubernetes.io/name=nvidia-device-plugin' --no-headers 2>/dev/null \
       || kubectl get pods -A 2>/dev/null | grep -E 'nvidia-device-plugin')
if [ -n "$DP_PODS" ]; then
  record "device_plugin" PASS "$(echo "$DP_PODS" | wc -l | tr -d ' ') pod(s) found"
else
  record "device_plugin" FAIL "nvidia-device-plugin pod 미검출"
fi

MULTUS=$(kubectl get pods -A 2>/dev/null | grep -c multus || true)
if [ "$MULTUS" -gt 0 ]; then
  record "multus" PASS "$MULTUS pod"
else
  record "multus" WARN "Multus 미검출 — RDMA/SR-IOV 설정에 필요할 수 있음"
fi

# ----------------------------------------------------------------------
# 4. peer repo + helmfile
# ----------------------------------------------------------------------
section "4. peer repo & helmfile"

if [ -d "$PEER_REPO/.git" ]; then
  PEER_SHA=$(git -C "$PEER_REPO" rev-parse --short HEAD 2>/dev/null)
  record "peer_repo" PASS "$PEER_REPO @ $PEER_SHA"
elif [ -d "$PEER_REPO" ]; then
  record "peer_repo" WARN "$PEER_REPO 존재하나 git 저장소 아님"
else
  record "peer_repo" FAIL "$PEER_REPO 없음 (PEER_REPO env 로 경로 지정 가능)"
fi

if command -v helmfile >/dev/null 2>&1; then
  record "helmfile" PASS "$(helmfile --version 2>/dev/null | head -1)"
else
  record "helmfile" FAIL "helmfile 미설치"
fi
if command -v helm >/dev/null 2>&1; then
  record "helm" PASS "$(helm version --short 2>/dev/null)"
else
  record "helm" FAIL "helm 미설치"
fi

# ----------------------------------------------------------------------
# 5. ghcr.io 도달
# ----------------------------------------------------------------------
section "5. registry reachability"

if curl -sf --max-time 8 "https://ghcr.io/" >/dev/null 2>&1 || curl -sI --max-time 8 "https://ghcr.io/" >/dev/null 2>&1; then
  record "ghcr.io" PASS "reachable"
else
  record "ghcr.io" WARN "도달 실패 — registry mirror 필요"
fi

# in-cluster 이미지 캐시 (containerd or cri-dockerd)
if command -v crictl >/dev/null 2>&1; then
  CACHED=$(sudo -n crictl images 2>/dev/null | grep -E 'llm-d|vllm' | wc -l | tr -d ' ' || echo 0)
  if [ "$CACHED" -gt 0 ]; then
    record "image_cache" PASS "$CACHED llm-d/vllm 이미지 캐시됨"
  else
    record "image_cache" WARN "llm-d/vllm 이미지 캐시 없음 — 첫 trial 시 풀 시간 큼"
  fi
fi

# ----------------------------------------------------------------------
# 6. RDMA fabric
# ----------------------------------------------------------------------
section "6. RDMA fabric"

if command -v ibstat >/dev/null 2>&1; then
  IB_PORTS=$(ibstat 2>/dev/null | grep -c "State: Active" || true)
  if [ "$IB_PORTS" -gt 0 ]; then
    record "ib.active_ports" PASS "$IB_PORTS active InfiniBand port(s)"
  else
    record "ib.active_ports" WARN "0 active IB port — RoCE 일 가능성"
  fi
elif ls /sys/class/infiniband/ 2>/dev/null | grep -q .; then
  record "ib.devices" PASS "$(ls /sys/class/infiniband/ | head -3 | tr '\n' ' ')"
else
  record "rdma" WARN "ibstat / /sys/class/infiniband 미존재 — RDMA 미사용 가능성"
fi

if [ -e /dev/infiniband/uverbs0 ] || [ -e /dev/infiniband/rdma_cm ]; then
  record "rdma.uverbs" PASS "RDMA verb device present"
else
  record "rdma.uverbs" WARN "RDMA verb device 없음 — TCP fallback 만 가능"
fi

# ----------------------------------------------------------------------
# 7. (옵션) 인터노드 fabric 테스트
# ----------------------------------------------------------------------
if ! $SKIP_FABRIC; then
  section "7. inter-node fabric (skip with --skip-fabric)"
  if [ "$NODE_COUNT" -ge 2 ]; then
    record "fabric_test" WARN "권장: nccl-tests DaemonSet + iperf3 server/client 측정 (별도 manifest 추가 예정)"
  else
    record "fabric_test" WARN "노드 < 2 — 인터노드 테스트 스킵"
  fi
fi

# ----------------------------------------------------------------------
# 결과 요약
# ----------------------------------------------------------------------
TOTAL=$((PASS + WARN + FAIL))

if $JSON_MODE; then
  printf '{"pass":%d,"warn":%d,"fail":%d,"total":%d,"results":{' "$PASS" "$WARN" "$FAIL" "$TOTAL"
  first=true
  for k in "${!RESULTS[@]}"; do
    $first || printf ','
    first=false
    val="${RESULTS[$k]}"
    status="${val%%|*}"
    msg="${val#*|}"
    printf '"%s":{"status":"%s","msg":%s}' "$k" "$status" "$(printf '%s' "$msg" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  done
  printf '}}\n'
else
  printf '\n=== Summary ===\n'
  printf '  PASS: %d   WARN: %d   FAIL: %d   (total %d)\n' "$PASS" "$WARN" "$FAIL" "$TOTAL"
fi

if [ "$FAIL" -gt 0 ]; then
  exit 2
elif [ "$WARN" -gt 0 ]; then
  exit 1
fi
exit 0
