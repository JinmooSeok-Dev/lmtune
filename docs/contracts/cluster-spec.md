# ClusterSpec — `apiVersion: ariadne/cluster/v1alpha1`

> Master = [ariadne](file:///home/jinmoo/os/qemu/ariadne). lmtune 은 mirror (직접 re-export) + Provider 호스팅 + **DeviceCatalog 보조**.

본 contract 의 책임 = "**우리는 어떤 호스트·디바이스·인터커넥트로 구성된 클러스터에서 측정·튜닝하는가**" 를 단일 객체로 표현. 결과물 (BenchmarkResult) 의 `cluster_id` reference 가 항상 ClusterSpec 1개를 가리킴 → 모든 측정값이 컨텍스트 join 가능.

## 책임 분리

| 영역 | 누가 | 어디서 |
|:---|:---|:---|
| Schema 정의 (Pydantic 모델, host/device/interconnect 추상) | **ariadne** | `ariadne-core/ariadne/model/cluster.py` (A1 PR 에서 신규) |
| Schema bump | **ariadne** (master 변경 시 lmtune mirror 도 같은 PR) | (그쪽 repo) |
| Single-host topology 발견 (PCIe/NUMA/IOMMU/CPU/Memory) | **ariadne** | `ariadne-core/ariadne/collector/{pcie,numa,iommu,cpu,memory}.py` (이미 존재) |
| Multi-host snapshot 집계 (SSH/Ansible inventory → host 별 caller) | **ariadne** | A1 PR (multi-host fan-out + aggregator) |
| GPU/NPU compute 능력 발견 (`nvidia-smi`, `rbln-smi`, `rocm-smi`) | **ariadne** | A2 PR (vendor SDK 통합) |
| **Device Catalog (벤더 spec 정본 yaml)** | **lmtune** | `src/lmtune/devices/catalog/*.yaml` |
| ClusterSpec → Catalog 보강 (peak FLOPS / mem BW 추정 fill) | **lmtune** | `src/lmtune/cluster/enrich.py` (CS PR) |
| 측정 micro-bench 결과 archive | **lmtune** | `device_perf_samples` 테이블 + `lmtune device-bench` (R0 이후) |

→ **drift 0**: lmtune 의 `ClusterSpec` = `from ariadne.model.cluster import ClusterSpec` 직접 re-export. `[cluster]` extra 가 ariadne 를 dep 으로 가져옴.

## ClusterProvider ABC

```python
# src/lmtune/cluster/providers/base.py
class ClusterProvider(ABC):
    """ClusterSpec 을 어디서 가져오든 동일 인터페이스."""

    @abstractmethod
    def provide(self, *, refresh: bool = False) -> ClusterSpec: ...

    def fingerprint(self) -> str:
        """Cache key — 같은 입력은 같은 fingerprint."""
        ...
```

### 구현체 — CS PR 에서 2개

| Provider | 입력 | 동작 | 의존 |
|:---|:---|:---|:---|
| `LiteralClusterProvider(yaml_path)` | yaml 파일 경로 | yaml read → Pydantic validation → ClusterSpec | 없음 |
| `AriadneClusterProvider(inventory, options)` | hosts inventory file 또는 `single-host` | ariadne `cluster_capture()` 호출 → 호스트별 collector → 집계 → ClusterSpec | `ariadne` (`[cluster]` extra) |

### 미래 확장 (entry_points 로 외부 패키지 등록)

| Provider | 의도 |
|:---|:---|
| `KubernetesClusterProvider` | k8s `Node` API + DaemonSet 으로 ariadne collector 자동 fan-out |
| `PrometheusClusterProvider` | DCGM exporter 메트릭에서 device 인벤토리 reverse-engineer |
| 사내 CMDB Provider | 운영 환경별 |

## 3-tier Device 정보 모델 (사용자 질문 답)

GPU/NPU 의 자세한 성능 정보는 **3 tier** 로 분리·합성. ariadne 가 발견 가능한 부분은 ariadne, 벤더 datasheet 가 정본인 부분은 lmtune catalog, 측정값은 lmtune ArtifactStore.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Tier 1 — DeviceCatalog (정적, 벤더 spec)        ← lmtune git-managed   │
│   src/lmtune/devices/catalog/{b200,h100,h200,mi300x,gb200,rbln-*}.yaml │
│   key: pci_id (vendor:device)                                           │
│   필드: peak_flops_dict, mem_bw, mem_capacity, nvlink_bw, pcie_gen      │
│   sources: [datasheet URL + accessed date] 필수                         │
└─────────────────────────────────────────────────────────────────────────┘
                ▲
                │ link by pci_id
                │
┌─────────────────────────────────────────────────────────────────────────┐
│ Tier 2 — Cluster Snapshot (동적, host 측정)     ← ariadne master       │
│   per host: PCIDevice (bdf, vendor:device, link_speed/width, IOMMU)    │
│            + NUMA/CPU/Memory (이미 보유)                                │
│            + GPUDeviceInfo (nvidia-smi: serial, driver, NVLink topo)   │
│            + NPUDeviceInfo (rbln-smi: SDK ver, fw, link)               │
│   inter-host: fabric (IB/RoCE 발견, ariadne A1 multi-host PR)           │
└─────────────────────────────────────────────────────────────────────────┘
                ▲
                │ host+bdf+sample_id 로 join
                │
┌─────────────────────────────────────────────────────────────────────────┐
│ Tier 3 — Performance Probes (실측 micro-bench) ← lmtune ArtifactStore  │
│   `lmtune device-bench` 명령 (R0 이후 신설):                            │
│     • nccl-tests (all-reduce/all-gather/broadcast bus_bw)              │
│     • mem_bw (cudaMemcpy D2D peak)                                     │
│     • flops (cuBLAS GEMM, dtype 별)                                    │
│     • gpu_direct (NIC↔GPU GDR bandwidth)                               │
│   적재: device_perf_samples 테이블 (ClusterSpec.fingerprint 참조)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tier 1 — DeviceCatalog YAML 예시

```yaml
# src/lmtune/devices/catalog/b200.yaml
apiVersion: lmtune/device/v1alpha1
kind: DeviceModel
metadata:
  name: nvidia-b200
  vendor: nvidia
  arch: blackwell-sm100
match:                          # ariadne 의 PCIDevice 와 link 키
  pci_id: "10de:2901"           # primary
  pci_id_aliases: []            # 동일 SKU 의 다른 device id
compute:
  peak_flops_tflops:
    fp64_dense: 30
    fp32_tf32_dense: 1100
    fp16_bf16_dense: 2200
    fp8_dense: 4500
    fp4_dense: 9000             # B200 native, MoE 추론 핵심
  sm_count: 144
memory:
  type: hbm3e
  capacity_gb: 192
  bandwidth_tbps: 8.0
  l2_cache_mb: 60
interconnect:
  nvlink:
    gen: 5
    per_gpu_gbps: 1800          # bidirectional aggregate
    nvswitch_supported: true
  pcie:
    gen: 6
    lanes: 16
    max_payload_bytes: 512
power:
  tdp_watts: 1000
sources:
  - title: "NVIDIA Blackwell Architecture Whitepaper"
    url: "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief"
    accessed: "2026-04"
  - title: "NVIDIA B200 datasheet"
    url: "https://..."
    accessed: "2026-04"
notes: |
  FP4 native 는 Blackwell 의 핵심 차별점. MoE 모델 (DeepSeek-V3, Llama-4 등)
  추론에서 throughput 9 PFLOPS 도달 가능 [출처 정확 측정 필요].
```

### Tier 1 — 8개 정본 DeviceModel (CS PR 산출)

| 파일 | pci_id | 주 사용 phase |
|:---|:---|:---|
| `nvidia-b200.yaml` | 10de:2901 | B200 16-GPU, llm-d wide-EP |
| `nvidia-h200.yaml` | 10de:2335 | 비교군 |
| `nvidia-h100-sxm5.yaml` | 10de:2330 | 비교군 |
| `nvidia-h100-pcie.yaml` | 10de:2331 | 비교군 |
| `nvidia-a100-sxm4.yaml` | 10de:20b0 | legacy 재현 |
| `nvidia-gb200-superchip.yaml` | 10de:2941 | InferenceX baseline |
| `amd-mi300x.yaml` | 1002:74a1 | multi-vendor |
| `rbln-atom.yaml` | (Rebellions) | NPU 비교 |

각 yaml 의 `sources[].url` + `accessed` 필수. `[추정]` / `[측정 미확정]` 태깅 도입.

### Tier 2 — ariadne 확장 사항 (A2 PR)

A2 (CS 의존 PR) 가 ariadne 본체에 추가:

```python
# ariadne-core/ariadne/model/types.py — 신규
class GPUDeviceInfo(BaseModel):
    bdf: str                                   # PCIDevice 와 join 키
    serial: str = ""
    uuid: str = ""
    driver_version: str = ""                   # nvidia-smi
    cuda_compute_capability: str = ""          # e.g., "10.0" for Blackwell
    nvlink_links: list[NVLinkInfo] = []        # nvidia-smi nvlink -s
    persistence_mode: bool = False
    ecc_mode: bool = True
    mig_mode: str = ""

class NVLinkInfo(BaseModel):
    link_id: int
    peer_bdf: str = ""                         # link 의 peer GPU
    bandwidth_gbps: float = 0.0
    state: str = ""                            # active/inactive

class NPUDeviceInfo(BaseModel):
    bdf: str
    vendor: str                                # rebellions/intel/...
    sdk_version: str = ""
    firmware_version: str = ""
    interconnect: str = ""                     # PCIe/UCIe/etc
```

ariadne `collector/gpu.py`, `collector/npu.py` 신규.

### Tier 3 — Performance Probes (lmtune `device-bench`)

R0 (BenchmarkResult contract) 머지 이후 신규 명령. ClusterSpec 을 input 으로 받아 호스트별 micro-bench 실행 → ArtifactStore 에 적재.

```bash
# 기본: 발견된 모든 GPU 에서 mem_bw + flops + nccl all-reduce
lmtune device-bench --cluster-spec cluster.yaml

# 선택적
lmtune device-bench --cluster-spec cluster.yaml \
    --probes mem_bw,flops,nccl_all_reduce,gpu_direct \
    --dtypes fp16,bf16,fp8 \
    --hosts host1,host2 \
    --output device_perf.parquet

# trial 결과와 join
SELECT t.trial_id, t.score, p.bus_bw_gbps, p.peak_tflops_fp16
FROM trials t
JOIN device_perf_samples p ON p.cluster_fingerprint = t.cluster_fingerprint
WHERE t.study_id = '...';
```

DuckDB schema (R0 PR 의 일부):

```sql
CREATE TABLE IF NOT EXISTS device_perf_samples (
    sample_id          TEXT PRIMARY KEY,
    cluster_fingerprint TEXT NOT NULL,         -- ClusterSpec.fingerprint()
    hostname           TEXT NOT NULL,
    bdf                TEXT,                   -- ariadne 의 PCIDevice 와 join
    probe_kind         TEXT NOT NULL,          -- mem_bw|flops|nccl_*|gpu_direct
    dtype              TEXT,                   -- fp16|bf16|fp8|fp4
    measurement_unit   TEXT NOT NULL,          -- gbps|tflops|us|...
    measurement_value  DOUBLE NOT NULL,
    config_json        JSON,                   -- probe-specific params
    sw_versions        JSON,                   -- driver/cuda/nccl/sdk
    measured_at        TIMESTAMP NOT NULL,
    notes              TEXT
);
CREATE INDEX idx_dperf_cluster ON device_perf_samples (cluster_fingerprint);
CREATE INDEX idx_dperf_host_probe ON device_perf_samples (hostname, probe_kind);
```

## CLI 표면

```bash
# 직접 yaml (BYO) — LiteralClusterProvider
lmtune run --cluster-spec cluster.yaml -p profile.yaml -e endpoint.yaml

# ariadne 호출 (Discover) — AriadneClusterProvider, single-host 또는 inventory
lmtune run --cluster-discover -p profile.yaml -e endpoint.yaml
lmtune run --cluster-discover --inventory hosts.txt ...

# yaml 만 만들고 끝 (단독 명령)
lmtune cluster discover --inventory hosts.txt --out cluster.yaml
lmtune cluster discover --single-host --out cluster.yaml

# 검증
lmtune cluster validate cluster.yaml

# Catalog 조회
lmtune devices catalog list                          # 모든 모델
lmtune devices catalog show nvidia-b200              # 단일 spec yaml dump
lmtune devices catalog match --pci-id 10de:2901      # 발견된 device 매칭

# Schema dump
lmtune contracts dump --kind cluster --out cluster.schema.json
lmtune contracts dump --kind device-model --out device-model.schema.json

# 실측 (R0 이후)
lmtune device-bench --cluster-spec cluster.yaml --probes mem_bw,flops,nccl_all_reduce
```

## Sidecar — multi-host inventory 형식

```yaml
# hosts.txt (inventory)
- name: b200-1
  ssh: jinmoo@b200-1.local
  ariadne_user: root             # collector 는 sysfs/proc 권한 필요
- name: b200-2
  ssh: jinmoo@b200-2.local
  ariadne_user: root
fabric:
  - kind: rdma                    # IB/RoCE
    detect_via: nccl-tests        # ariadne A1 의 inter-host probe
```

→ ariadne A1 PR 이 본 inventory 를 read 해서 host 별 collector 호출 + 결과 집계.

## Cache 정책

`~/.lmtune/cache/cluster/<fingerprint>.yaml`, TTL **24h** (host 토폴로지는 workload 보다 더 안정적). `--refresh` 또는 `--cluster-ttl 0` 로 우회.

## Acceptance criteria (CS PR 본체에서 검증)

1. `lmtune cluster validate <yaml>` 가 ariadne master 의 Pydantic 모델로 검증 (drift 0)
2. `LiteralClusterProvider` ABC 구현 + `AriadneClusterProvider` 의 lazy import (ariadne 미설치 환경에서 ImportError fail-fast + 친절한 메시지)
3. `lmtune devices catalog list` / `show` / `match` 동작
4. 8개 DeviceModel yaml 정본 + sources URL + accessed date 모두 보유
5. `enrich.py` 가 ariadne snapshot 의 `PCIDevice.vendor:device_id` → catalog `peak_flops` derive 검증
6. fingerprint 안정성: 같은 호스트 토폴로지 → 같은 fingerprint
7. cache TTL 24h 동작 + `--refresh` 우회
8. 단위 테스트 ≥ 12 케이스 (ABC, builder, fingerprint, catalog match, enrich, ImportError path, e2e with fixture snapshot)

## Non-goals (CS PR)

- Tier 3 의 `device-bench` 실행 코드 — R0 + 신규 OD PR 의 일부
- ariadne A1 (multi-host) / A2 (GPU/NPU info) 본체 변경 — 별도 ariadne PR
- DCGM exporter 통합 — entry_points 로 후속 contributor 가 추가
- Live monitoring (Grafana 라이브 대시보드) — output dashboard PR 소관

## 의존 그래프 — CS PR 진입 전제

```
ariadne#A1 (multi-host snapshot, ClusterSpec schema 신규) ─┐
   사용자 외부 트랙으로 ariadne 측 진행 중 (요청 완료)      │
                                                          ├─→ lmtune#CS 
ariadne#A2 (GPU/NPU info, GPUDeviceInfo / NPUDeviceInfo) ─┘    (본 PR)
   사용자 외부 트랙으로 ariadne 측 진행 중                    │
                                                              ▼
                                                          (R0, OD, OUT 후속)
```

**진행 정책**:
- 본 doc PR = ariadne 작업과 **병렬 진행 가능** (self-contained). schema 형태가 결정되기 전 **lmtune 이 요구하는 contract 명세** 로 ariadne 측에 전달 (양 repo 정렬 가이드 역할).
- DeviceCatalog (Tier 1) yaml 8 종은 ariadne 와 무관 — lmtune 단독으로 생성·검증 가능.
- CS **코드 PR** (provider 구현체) 은 ariadne A1 의 ClusterSpec Pydantic 모델 export 시점에 진입.
- A1 가 schema 를 다른 형태로 결정하면 본 doc 의 "Tier 2 ariadne 확장 사항" 섹션이 master 에 맞춰 갱신. lmtune 측 코드 변경 없음 (re-export 만 동작).

## Trade-offs / 결정 기록

| 결정 | 대안 | 선택 사유 |
|:---|:---|:---|
| Catalog 를 lmtune 안에 둠 | 별도 repo (`lm-devices`) | 8개 yaml 은 lmtune 의존성 그래프의 leaf, 외부 분리 시 PR 갯수 ↑↑. 충분히 커지면 추출 가능 |
| ariadne 가 GPU/NPU info master | lmtune 이 nvidia-smi 직접 호출 | ariadne 의 책임 = "단일 호스트 hardware 발견". GPU info 는 그 자연스러운 확장. lmtune 이 양쪽 호출 시 책임 중복 |
| 측정값은 lmtune ArtifactStore | ariadne 가 측정까지 | ariadne 는 "정적 발견" 만 책임. 측정·튜닝·archive 는 lmtune 역할. 한 측정값 = 한 ClusterSpec ref + N 개 trial ref 가능 |
| Tier 1 yaml 에 `sources` 필수 | optional | 벤더 spec drift / 잘못 인용 사고 방지. CLAUDE.md Accuracy 규칙 |
| pci_id 매칭에 `aliases` 허용 | primary 만 | 같은 SKU 의 stepping 차이 흡수 (예: B200 변형) |
