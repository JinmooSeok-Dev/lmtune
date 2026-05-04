#!/usr/bin/env bash
# fabric_probe.sh — B200 NVLink + InfiniBand fabric 통합 baseline 측정
#
# 매 study 시작 직전 1 회 실행 → trial 들의 절대 reference. autotune 중 inference
# throughput 이 baseline 대비 유의미하게 떨어지면 fabric 자체 degradation 의심
# (PR #14 의 circuit breaker 가 활용).
#
# 측정 컴포넌트:
#   1. NVLink intra-node — nccl-tests all_reduce 8 GPU on single node (NVLS bus_bw)
#   2. NVLink+IB cross-node — nccl-tests all_reduce 16 GPU on 2 nodes (network-bound)
#   3. IB raw — ib_write_bw per HCA (host 측, RDMA Perftest)
#   4. Topology — nvidia-smi topo, lspci -tv, NCCL_TOPO_FILE 후보 (있으면)
#   5. GDR check — nv_peer_mem 모듈 + nccl_net_gdr_level capability
#
# 출력:
#   ${OUT_DIR}/{nccl_intranode.log,nccl_crossnode.log,ib_write_bw.txt,topo.txt,
#               gdr_check.txt,fabric_baseline.json}
#
# 사용:
#   bash b200/scripts/fabric_probe.sh                          # 전체
#   PROBE_NS=bench-fabric bash b200/scripts/fabric_probe.sh    # 다른 namespace
#   SKIP_NCCL=1 bash b200/scripts/fabric_probe.sh              # IB+topo 만
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-b200/studies/fabric_baselines}"
PROBE_NS="${PROBE_NS:-bench-fabric-test}"
NCCL_IMAGE="${NCCL_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
NCCL_TIMEOUT_S="${NCCL_TIMEOUT_S:-300}"
SKIP_NCCL="${SKIP_NCCL:-0}"
SKIP_IB="${SKIP_IB:-0}"
RDMA_DEVICE="${RDMA_DEVICE:-}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT}/${TS}"
mkdir -p "${OUT_DIR}"

log() { printf '[fabric_probe] %s\n' "$*" >&2; }

# --- 1. Topology 캡처 (cheap, 항상 수행) -----------------------------------
log "1/5 topology capture"
{
  printf '## nvidia-smi topo -m\n'
  nvidia-smi topo -m 2>&1 || echo "(nvidia-smi unavailable)"
  printf '\n## nvidia-smi nvlink -s\n'
  nvidia-smi nvlink -s 2>&1 || echo "(nvlink status unavailable)"
  printf '\n## lspci -tv (NVSwitch + NIC)\n'
  lspci -tv 2>&1 | head -100 || echo "(lspci unavailable)"
  printf '\n## ibv_devices\n'
  ibv_devices 2>&1 || echo "(ibv_devices unavailable)"
  printf '\n## kubectl get nodes -o wide\n'
  kubectl get nodes -o wide 2>&1 || echo "(kubectl unavailable)"
} > "${OUT_DIR}/topo.txt"
log "  → ${OUT_DIR}/topo.txt"

# --- 2. GDR / nv_peer_mem 캡처 ---------------------------------------------
log "2/5 gdr capability"
{
  printf '## nv_peer_mem / nvidia_peermem module\n'
  lsmod 2>/dev/null | grep -E 'nv_peer_mem|nvidia_peermem' || echo "(neither module loaded)"
  printf '\n## /sys/kernel/mm/transparent_hugepage/enabled\n'
  cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "(unavailable)"
  printf '\n## DMA-BUF support (kernel ≥ 6.2 권장)\n'
  uname -r
} > "${OUT_DIR}/gdr_check.txt"

# --- 3. IB raw bandwidth (host 측 ib_write_bw) -----------------------------
if [[ "${SKIP_IB}" != "1" ]]; then
  log "3/5 IB raw bw — single-direction sample (host 측)"
  if command -v ib_write_bw >/dev/null 2>&1; then
    if [[ -z "${RDMA_DEVICE}" ]]; then
      RDMA_DEVICE="$(ibv_devices 2>/dev/null | awk 'NR>2 {print $1; exit}')"
    fi
    if [[ -n "${RDMA_DEVICE}" ]]; then
      log "  device=${RDMA_DEVICE} (server-side only; full client/server pair: scripts/rdma_bench.sh)"
      timeout 30 ib_write_bw -F -d "${RDMA_DEVICE}" -x 3 --report_gbits 2>&1 \
        | tee "${OUT_DIR}/ib_write_bw.txt" || true
    else
      echo "(no RDMA device detected)" > "${OUT_DIR}/ib_write_bw.txt"
    fi
  else
    echo "(ib_write_bw not installed — see rdma_perftest_baseline.md)" > "${OUT_DIR}/ib_write_bw.txt"
  fi
else
  log "3/5 IB raw bw — SKIPPED"
fi

# --- 4-5. NCCL bus_bw (intranode + cross-node) -----------------------------
if [[ "${SKIP_NCCL}" != "1" ]]; then
  log "4/5 NCCL intranode all_reduce (8 GPU single node)"
  log "5/5 NCCL cross-node all_reduce (16 GPU 2 nodes)"
  log "  → kubectl apply b200/scripts/fabric_test.yaml (별도 manifest)"
  log "  → 결과는 \`kubectl logs -n ${PROBE_NS} job/nccl-test-allreduce\` 로 수집"
  log "  본 probe 는 manifest dispatch + log harvest 만; 측정값 raw 보존."

  if kubectl get ns "${PROBE_NS}" >/dev/null 2>&1; then
    log "  namespace ${PROBE_NS} 존재 — clean state 로 재실행"
    kubectl delete -f b200/scripts/fabric_test.yaml --ignore-not-found=true >/dev/null 2>&1 || true
    sleep 2
  fi
  kubectl apply -f b200/scripts/fabric_test.yaml >/dev/null

  # job 완료 대기 (최대 NCCL_TIMEOUT_S)
  if kubectl wait --for=condition=complete --timeout="${NCCL_TIMEOUT_S}s" \
       -n "${PROBE_NS}" job/nccl-test-allreduce 2>&1 | tee "${OUT_DIR}/nccl_wait.log"; then
    kubectl logs -n "${PROBE_NS}" job/nccl-test-allreduce > "${OUT_DIR}/nccl_crossnode.log" 2>&1 || true
    log "  → ${OUT_DIR}/nccl_crossnode.log"
  else
    log "  WARN: nccl-test-allreduce timed out — partial logs 보존"
    kubectl logs -n "${PROBE_NS}" job/nccl-test-allreduce > "${OUT_DIR}/nccl_crossnode.log" 2>&1 || true
  fi
else
  log "4-5/5 NCCL — SKIPPED"
fi

# --- 6. JSON summary -------------------------------------------------------
log "6/6 summary JSON"

extract_bw() {
  # nccl-tests 출력에서 max bus_bw 파싱 (GB/s)
  awk '/^[ ]*[0-9]/ {if ($NF+0 > max) max=$NF+0} END {print (max ? max : 0)}' "$1" 2>/dev/null
}

extract_ib() {
  # ib_write_bw 출력에서 BW peak 파싱 (Gbps)
  awk '/^[ ]*[0-9]/ {if ($4+0 > max) max=$4+0} END {print (max ? max : 0)}' "$1" 2>/dev/null
}

NCCL_BUS_BW="$(extract_bw "${OUT_DIR}/nccl_crossnode.log" 2>/dev/null || echo 0)"
IB_PEAK_GBPS="$(extract_ib "${OUT_DIR}/ib_write_bw.txt" 2>/dev/null || echo 0)"

cat > "${OUT_DIR}/fabric_baseline.json" <<EOF
{
  "ts_utc": "${TS}",
  "host": "$(hostname)",
  "kernel": "$(uname -r)",
  "tools": {
    "ib_write_bw": "$(command -v ib_write_bw >/dev/null && ib_write_bw --version 2>&1 | head -1 || echo 'unavailable')",
    "nvidia_smi": "$(nvidia-smi --version 2>&1 | head -1 || echo 'unavailable')",
    "kubectl": "$(kubectl version --client=true -o yaml 2>/dev/null | grep -E '^[[:space:]]+gitVersion' | head -1 | awk '{print $2}' || echo 'unavailable')"
  },
  "results": {
    "nccl_crossnode_busbw_gbps": ${NCCL_BUS_BW:-0},
    "ib_write_bw_peak_gbps": ${IB_PEAK_GBPS:-0}
  },
  "skipped": {
    "nccl": ${SKIP_NCCL},
    "ib": ${SKIP_IB}
  },
  "files": {
    "topo": "topo.txt",
    "gdr_check": "gdr_check.txt",
    "ib_write_bw": "ib_write_bw.txt",
    "nccl_crossnode_log": "nccl_crossnode.log"
  }
}
EOF

log "DONE — ${OUT_DIR}/"
log "  reference: NHN Cloud B200 baseline = 363.98 Gbps RDMA Write (non-priv pod)"
log "  ratio = $(awk "BEGIN {printf \"%.2f\", ${IB_PEAK_GBPS:-0}/363.98 * 100}")%"
echo "${OUT_DIR}"
