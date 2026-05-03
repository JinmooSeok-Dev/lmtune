# Low-level System — Autotune Axis Catalog (B-II 입력, B6)

> Phase B6 의 search space (`b200/search-spaces/b6_lowlevel.yaml`) 카탈로그.
>
> **2026-05-03 확장**: 17 → **70+ axis**. 4 sub-section 으로 재편:
>
> | Section | axis 수 | 범위 |
> |:---|:---|:---|
> | B6.1 Host | 15 | PCIe (4) + IOMMU (1) + NUMA (3) + Memory (3) + CPU (4) |
> | B6.2 Interconnect | 24 | InfiniBand (8) + SHARP (2) + HCA/NIC (4) + NVLink/NVSwitch (3) + NCCL (10) + Switch NOS (3) |
> | B6.3 GPUDirect | 17 | GDR (5) + GDS (6) + GPU P2P (3) + DCB (2) + Other (1) |
> | B6.4 KV Transport | 28 | NIXL (6) + UCX (8) + LMCache (7) + Mooncake (3) + 추가 NCCL (4) |
>
> 본 프로젝트의 **차별화 핵심** — 어느 OSS LLM autotune 도 이 계층은 1st-class 로 다루지 않음.
>
> **갱신 주기**: 커널·드라이버·NCCL 메이저 버전 업데이트 시. 직전 점검: 2026-05-03

---

## 0. 분류 — 5 그룹 + scope

| 그룹 | axis 수 | scope (변경 비용) |
|:---|:---|:---|
| 1. NCCL / RDMA env | 5 | **per-trial** (pod env, 즉시 변경) |
| 2. Pod-level (NUMA/hugepages/RDMA PF) | 3 | **per-trial** (podSpec, kubelet 만 재기동) |
| 3. Host kernel param | 6 | **study-level** (host reboot 또는 kubelet 재기동) |
| 4. CPU 상태 | 2 | **study-level** (cpufreq/sysfs, reboot 가능 시 BIOS) |
| 5. Switch / Cable | 1 | **switch-NOS-level** (별도 운영 권한) |

→ 17 axis. study 시작 시 system_snapshot 으로 활성 값 기록, per-trial axis 만 trial 마다 mutation.

---

## 1. NCCL / RDMA env (per-trial, pod env 변경)

### 1.1 `nccl_p2p_level`

- **type**: categorical
- **values**: `[NVL, NODE, SYS]`
- **default**: 자동 (NCCL 토폴로지 detection)
- **의미**: GPU 간 P2P 통신 levelLimit. NVL=NVLink only, NODE=intra-node 모든 transport, SYS=cross-node 까지 허용
- **active_if**: `capability: gpu_p2p`
- **측정 대상**: `nccl_topology_hint=auto/rdma/tcp` 와 결합해 NCCL all-reduce time 비교
- **B200 권장**: 단일 노드 = NVL, 양노드 = SYS
- **출처**: [NCCL User Guide §3 Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### 1.2 `nccl_ib_sl` (Service Level)

- **type**: categorical
- **values**: `[0, 3]`
- **default**: 0
- **의미**: InfiniBand QoS Service Level. switch 의 trafficClass 별 우선순위 매핑
- **active_if**: `fabric: rdma`
- **측정 대상**: 다른 namespace 와 RDMA 공유 시 fairness/wait time
- **출처**: [NCCL Env Variables — NCCL_IB_SL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-sl)

### 1.3 `nccl_ib_qps_per_connection`

- **type**: categorical
- **values**: `[1, 2, 4, 8]`
- **default**: 1
- **의미**: 한 NCCL connection 당 QP 개수. 큰 값 → multi-QP 으로 line rate 도달 빠름, 자원 소비 ↑
- **active_if**: `fabric: rdma`
- **측정 대상**: ib_write_bw 의 sweep 결과 (`-q 1` vs `-q 8`) 와 NCCL all-reduce bw 의 정합성
- **B200 권장**: `b200/scripts/rdma_bench.sh QP_COUNT=2` 가 default — autotune 시 1·4 도 시도
- **출처**: [NCCL Env Variables — NCCL_IB_QPS_PER_CONNECTION](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### 1.4 `nccl_net_gdr_level` (GPU Direct RDMA Level)

- **type**: categorical
- **values**: `[LOC, PIX, PXB, PHB, SYS]`
- **default**: PIX
- **의미**: GDR 활성화 PCIe 거리 한계. LOC=같은 PCIe switch only, SYS=NUMA cross 까지 허용
- **active_if**: 모든 RDMA 환경
- **측정 대상**: GDR off vs on 의 TTFT/throughput 차이
- **B200 권장**: PIX 또는 PXB. SYS 는 NUMA 위반 가능성 검증 필요
- **출처**: [NCCL GPUDirect RDMA Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

### 1.5 `nccl_topology_hint`

- **type**: categorical
- **values**: `[auto, tcp, rdma]`
- **default**: auto
- **의미**: 인터노드 transport 강제 선택. tcp=RDMA fallback 회피로 baseline 대조군 제공
- **active_if**: 항상 활성 (양노드 환경에서만 유의)
- **측정 대상**: rdma vs tcp throughput 차이 — RDMA fabric 가용성 회귀 감지
- **출처**: [NCCL Env Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

---

## 2. Pod-level (per-trial, podSpec 변경)

### 2.1 `cpu_pinning_strategy`

- **type**: categorical
- **values**: `[none, numa-aware, single-numa-node]`
- **default**: none
- **의미**: pod 의 CPU 할당 정책. numa-aware = NUMA 노드 내 CPU 만, single-numa-node = 한 NUMA 노드에 CPU+memory+GPU 모두
- **active_if**: kubelet `cpu-manager-policy=static` + `topology-manager-policy=single-numa-node` 사전 설정 필요
- **측정 대상**: NUMA cross 가 RDMA / NCCL 에 미치는 영향 정량
- **B200 권장**: single-numa-node (고성능). cfregly Ch3 추천
- **출처**: [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/), [cfregly Ch3](https://github.com/cfregly/ai-performance-engineering/tree/main/code/ch03)

### 2.2 `hugepages_1gi`

- **type**: categorical
- **values**: `[0, 16, 32]`
- **default**: 0
- **의미**: pod 에 할당할 1 GiB hugepage 개수. KV cache, NCCL buffer 의 TLB miss 감소
- **active_if**: `host_capability: hugepages_1gi` (host kernel cmdline 에 `hugepagesz=1G hugepages=N` 사전 등록)
- **측정 대상**: hugepages on/off 의 TTFT 분포
- **B200 권장**: 16 (= 16 GiB), 큰 모델 + KV cache 에 효과
- **출처**: [vLLM K8s deployment](https://docs.vllm.ai/en/stable/deployment/k8s/), [Linux hugetlbpage](https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt)

### 2.3 `rdma_pf_per_pod`

- **type**: categorical
- **values**: `[1, 2, 4]`
- **default**: 1
- **의미**: 한 pod 에 attach 할 RDMA PF (Physical Function) 개수. 많을수록 인터노드 대역 ↑
- **active_if**: `fabric: rdma` + Multus + SR-IOV CNI 설정
- **측정 대상**: rdma_bench 의 multi-PF sweep 결과와 NCCL bw 정합
- **B200 권장**: 사용자 이력서의 NHN Cloud B200 363.98 Gbps 는 1-PF 측정. multi-PF 검증 필요
- **출처**: [Multus / SR-IOV CNI](https://github.com/k8snetworkplumbingwg/sriov-network-operator), 본 repo `~/.claude/rules/coding-yaml-k8s.md` Multus 섹션

---

## 3. Host kernel param (study-level, host reboot 또는 kubelet 재기동)

> **변경 비용**: host reboot 또는 kubelet 재기동 (수 분 ~ 수십 분). 따라서 한 study 시작 시점에 한 번 setup → 모든 trial 가 같은 값 공유. system_snapshot 이 활성 값 기록.

### 3.1 `pcie_aspm`

- **type**: categorical
- **values**: `[default, performance, powersave, off]`
- **default**: default (보통 powersave)
- **의미**: PCIe Active State Power Management. powersave 면 idle 시 link 가 L1 까지 들어가 latency spike
- **측정 도구**: `lspci -vvv | grep ASPM`, `cat /sys/module/pcie_aspm/parameters/policy`
- **변경 방법**: `pcie_aspm.policy=performance` kernel cmdline
- **B200 권장**: `performance` (LLM 서빙은 항상 link 활성, 절전 불필요)
- **출처**: [Linux PCIe ASPM](https://www.kernel.org/doc/Documentation/PCI/pcieaer-howto.txt), cfregly Ch3

### 3.2 `pcie_acs_override`

- **type**: categorical
- **values**: `[disabled, downstream, multifunction]`
- **default**: disabled
- **의미**: PCIe Access Control Services override. SR-IOV 의 P2P 차단을 우회하여 GPUDirect 활용 폭 ↑
- **측정 도구**: `lspci -vv | grep ACS`, `dmesg | grep ACS`
- **변경 방법**: `pcie_acs_override=downstream,multifunction` kernel cmdline (주의: 보안 격리 약화)
- **B200 권장**: `downstream` (SR-IOV + GPUDirect P2P 활성), 단 vendor 가이드 확인 필요
- **출처**: cfregly Ch3, [PCIe ACS](https://lwn.net/Articles/603544/)

### 3.3 `iommu_passthrough`

- **type**: categorical
- **values**: `[on, off, strict]`
- **default**: off (deferred mode)
- **의미**: IOMMU page table 의 cache invalidation 정책. on=PT mode, strict=invalidation 즉시
- **측정 도구**: `dmesg | grep "iommu="`, `cat /proc/cmdline`
- **변경 방법**: `iommu=pt` 또는 `iommu.passthrough=1` kernel cmdline
- **B200 권장**: `on` (PT mode) — DMA 성능 ↑, 격리 약화 trade-off
- **출처**: [Linux IOMMU](https://www.kernel.org/doc/Documentation/x86/iommu.txt), cfregly Ch3

### 3.4 `numa_balancing`

- **type**: categorical
- **values**: `[0, 1]`
- **default**: 1
- **의미**: 자동 NUMA 페이지 마이그레이션. 잘못된 NUMA 매핑을 자동 보정
- **측정 도구**: `cat /proc/sys/kernel/numa_balancing`
- **변경 방법**: `echo 0 > /proc/sys/kernel/numa_balancing` (즉시) 또는 `numa_balancing=disable` cmdline
- **B200 권장**: 0 (off) — kubelet topology-manager 가 이미 NUMA 매핑 보장하므로 dynamic balancing 은 latency jitter 원인
- **출처**: [Linux NUMA Balancing](https://www.kernel.org/doc/Documentation/sysctl/kernel.txt), cfregly Ch3

### 3.5 `transparent_hugepages`

- **type**: categorical
- **values**: `[always, madvise, never]`
- **default**: madvise
- **의미**: THP 정책. always=모든 anonymous mem, madvise=명시 영역만, never=비활성
- **측정 도구**: `cat /sys/kernel/mm/transparent_hugepage/enabled`
- **변경 방법**: `echo never > /sys/kernel/mm/transparent_hugepage/enabled` (즉시)
- **B200 권장**: `madvise` 또는 `never` — always 는 khugepaged compaction 으로 latency spike
- **출처**: [Linux THP](https://www.kernel.org/doc/Documentation/vm/transhuge.txt), cfregly Ch3

### 3.6 `hugepages_total` (host)

- **type**: categorical (string)
- **values**: `["0", "16Gi", "32Gi", "64Gi"]`
- **default**: 0
- **의미**: host 에 사전 할당된 1GB hugepage 총량. pod 의 `hugepages_1gi` 의 ceiling
- **측정 도구**: `cat /proc/meminfo | grep Huge`
- **변경 방법**: `hugepagesz=1G hugepages=N` kernel cmdline (reboot)
- **B200 권장**: 32Gi (= 32×1G page) for 16-GPU node

---

## 4. CPU 상태 (study-level)

### 4.1 `cpu_governor`

- **type**: categorical
- **values**: `[performance, ondemand, powersave]`
- **default**: ondemand 또는 schedutil
- **의미**: cpufreq governor. performance=항상 최대 frequency
- **측정 도구**: `cpupower frequency-info`, `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- **변경 방법**: `cpupower frequency-set -g performance` 또는 systemd unit
- **B200 권장**: `performance` (LLM 서빙은 CPU 가 항상 활동, 절전 불필요)
- **출처**: [Linux cpufreq](https://www.kernel.org/doc/Documentation/cpu-freq/), cfregly Ch3

### 4.2 `cstates_max`

- **type**: categorical
- **values**: `[c0, c1, default]`
- **default**: default (보통 c6 까지)
- **의미**: CPU C-state 한계. c0=idle 상태 진입 안 함, latency 안정 trade-off 전력
- **측정 도구**: `cpupower idle-info`, BIOS C-state setting
- **변경 방법**: `intel_idle.max_cstate=0` 또는 `processor.max_cstate=0` cmdline
- **B200 권장**: c1 — c0 는 과도, default 는 jitter
- **출처**: [Linux C-states](https://www.kernel.org/doc/html/latest/admin-guide/pm/intel_idle.html), cfregly Ch3

### 4.3 `smt` (Hyper-Threading)

- **type**: categorical
- **values**: `[on, off]`
- **default**: on
- **의미**: SMT (HT) 활성화. off → physical core 만 사용, latency 안정도 ↑ throughput 약간 ↓
- **측정 도구**: `lscpu | grep Thread`, `cat /sys/devices/system/cpu/smt/control`
- **변경 방법**: `echo off > /sys/devices/system/cpu/smt/control` (즉시) 또는 `nosmt` cmdline
- **B200 권장**: 첫 study 에서 on/off 둘 다 측정 — vLLM 의 CPU bound 작업이 적어 보통 on 이 우위
- **출처**: cfregly Ch3

---

## 5. Switch / Cable (switch-NOS scope)

### 5.1 `aec_cable_link_training`

- **type**: categorical
- **values**: `[auto, fixed-3.5dB, fixed-4.5dB]`
- **default**: auto
- **의미**: 400 GbE / NDR 의 AEC (Active Electrical Cable) link training. fixed-3.5dB / fixed-4.5dB 는 케이블 별 최적치
- **측정 도구**: switch NOS CLI (Sonic / Cumulus 등)
- **변경 방법**: switch CLI `interface ... aec-mode <value>`
- **B200 권장**: 사용자 이력서의 363.98 Gbps RDMA Write 은 fixed-4.5dB 등 최적값으로 추정. study 별 sweep 으로 정량
- **출처**: [Sonic AEC tuning](https://github.com/sonic-net/sonic-mgmt), 사용자 이력서 v5

---

## 6. axis 별 측정 결합

각 axis 변경 시 system_snapshot.sh 가 자동 캡처하는 데이터:

| axis | snapshot 의 캡처 위치 |
|:---|:---|
| nccl_* | `nccl_env` (env vars 모두) + trial pod log 의 NCCL_DEBUG=INFO 출력 |
| pcie_aspm/acs/iommu | `pcie.topo_short`, `pcie.iommu_groups`, `kernel.cmdline` |
| numa | `numa.nodes`, `numa.topo` (lstopo) |
| hugepages | `kernel.transparent_hugepage`, `kernel.hugepages_*` |
| cpu | `cpu.governor`, `cpu.smt`, `cpu.online` |
| rdma | `rdma.ibv_devices`, `rdma.ibstat`, host-level `b200/scripts/rdma_bench.sh` 결과 |

→ ANALYSIS.md 의 §3 "원인 분석" 에서 system_snapshot 데이터를 인용해 axis 효과를 메커니즘 수준에서 설명.

---

## 7. autotune 진행 시 주의

### 7.1 변경 비용을 axis 영향력 분석에 가중치로 반영

study-level axis (PCIe/IOMMU/CPU governor) 는 reboot 비용이 trial 비용을 압도. ANOVA pruner 가 이런 axis 의 효과를 발견해도, 실제 운영에서는 영향이 큰 경우만 변경.

### 7.2 PCIe ACS 변경 시 보안 격리 영향

`pcie_acs_override=downstream` 은 SR-IOV 의 P2P 격리를 약화. multi-tenant 환경에서는 검토 필요. plan 의 Non-Goal "multi-tenant 페어니스" 와 직접 연관.

### 7.3 study 시작 시 차이 격리

study A 와 study B 의 system_snapshot 이 다르면, 같은 axis 의 결과 비교가 무의미. **study 간 system_snapshot diff** 를 ANALYSIS.md 의 §1 측정 컨텍스트에 명시.

### 7.4 cfregly Ch3 / Ch4 와의 매핑

cfregly book Ch3 (OS/Docker/K8s Tuning) + Ch4 (Distributed Networking) 의 권장 setting 이 본 카탈로그의 study-level default 값과 직접 대응. 검증 시 expectations 비교 가능:
- Ch3 권장 = `pcie_aspm=performance, iommu=pt, transparent_hugepages=madvise, cpu_governor=performance, numa_balancing=0`
- Ch4 권장 NCCL = `nccl_p2p_level=NVL/SYS auto, nccl_net_gdr_level=PIX, nccl_ib_sl=0`

---

## 8. References

- [NCCL User Guide §3 Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL GPUDirect RDMA Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [Kubernetes Topology Manager](https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/)
- [Linux PCIe ASPM](https://www.kernel.org/doc/Documentation/PCI/pcieaer-howto.txt)
- [Linux IOMMU](https://www.kernel.org/doc/Documentation/x86/iommu.txt)
- [Linux NUMA Balancing](https://www.kernel.org/doc/Documentation/sysctl/kernel.txt)
- [Linux THP](https://www.kernel.org/doc/Documentation/vm/transhuge.txt)
- [Linux cpufreq](https://www.kernel.org/doc/Documentation/cpu-freq/)
- [Linux C-states](https://www.kernel.org/doc/html/latest/admin-guide/pm/intel_idle.html)
- [Multus / SR-IOV CNI](https://github.com/k8snetworkplumbingwg/sriov-network-operator)
- [cfregly/ai-performance-engineering Ch3](https://github.com/cfregly/ai-performance-engineering/tree/main/code/ch03) — OS/Docker/K8s Tuning
- [cfregly/ai-performance-engineering Ch4](https://github.com/cfregly/ai-performance-engineering/tree/main/code/ch04) — Distributed Networking
- 본 repo: `b200/scripts/system_snapshot.sh`, `b200/scripts/rdma_bench.sh`
- 본 repo: `b200/docs/rdma_perftest_baseline.md`
- 사용자 이력서 v5 — NHN Cloud B200 363.98 Gbps RDMA Write
