#!/usr/bin/env bash
# System snapshot — Phase B6 hook.
#
# 각 trial 직전·직후 호스트 상태(PCIe·IOMMU·NUMA·CPU·NCCL env·hugepages·
# RDMA link)를 캡처한다. 결과는 study 의 system_snapshot/<trial_id>.json
# 으로 저장되어 study-level axis 와 결과 metric 의 상관 분석에 쓰인다.
#
# 사용:
#   bash b200/scripts/system_snapshot.sh <trial_id> [out_dir]
#
# out_dir 미지정 시 b200/studies/_snapshots/ 으로.
set -euo pipefail

TRIAL_ID="${1:-snapshot-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${2:-b200/studies/_snapshots}"
mkdir -p "${OUT_DIR}"
OUT_FILE="${OUT_DIR}/${TRIAL_ID}.json"

cap() { eval "$1" 2>/dev/null || echo "" ; }

# stdout 으로 jq 가능한 JSON 1줄
python3 - "${OUT_FILE}" <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1])

def run(cmd, *, multiline=False):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        return r.stdout.strip() if not multiline else r.stdout
    except Exception:
        return None

def read_file(p):
    try:
        return Path(p).read_text().strip()
    except Exception:
        return None

snap = {
    "trial_id": Path(out).stem,
    "timestamp": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    "kernel": {
        "uname_r": run(["uname", "-r"]),
        "cmdline": read_file("/proc/cmdline"),
        "numa_balancing": read_file("/proc/sys/kernel/numa_balancing"),
        "transparent_hugepage": read_file("/sys/kernel/mm/transparent_hugepage/enabled"),
        "hugepages_2mi": read_file("/proc/sys/vm/nr_hugepages"),
        "hugepages_1gi_total": read_file("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"),
    },
    "cpu": {
        "model": (run(["lscpu"]) or "").split("Model name:")[-1].split("\n")[0].strip() if run(["lscpu"]) else None,
        "online": run(["nproc"]),
        "smt": read_file("/sys/devices/system/cpu/smt/control"),
        "governor": read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "isolated": read_file("/sys/devices/system/cpu/isolated"),
    },
    "numa": {
        "nodes": run(["lscpu", "-p=NODE"], multiline=True),
        "topo": run(["lstopo", "--no-io", "-p"], multiline=True),
    },
    "pcie": {
        "topo_short": run(["lspci", "-tv"], multiline=True),
        "iommu_groups": run(["bash", "-c", "ls -1 /sys/kernel/iommu_groups 2>/dev/null | wc -l"]),
    },
    "gpu": {
        "nvidia_smi_topo": run(["nvidia-smi", "topo", "-m"], multiline=True),
        "nvidia_smi_query": run([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,pstate,clocks.sm,clocks.mem,power.management",
            "--format=csv,noheader",
        ], multiline=True),
    },
    "rdma": {
        "ibv_devices": run(["ibv_devices"], multiline=True),
        "ibstat": run(["ibstat"], multiline=True),
        "perf_query": run(["bash", "-c", "ls -1 /sys/class/infiniband 2>/dev/null"], multiline=True),
    },
    "nccl_env": {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_")
    },
    "k8s": {
        "node": os.environ.get("K8S_NODE_NAME") or run(["bash", "-c", "kubectl get nodes -o name 2>/dev/null | head -1"]),
        "kubelet_version": run(["bash", "-c", "kubelet --version 2>/dev/null"]),
    },
}

out.write_text(json.dumps(snap, indent=2))
sys.stdout.write(f"snapshot -> {out}\n")
PY
