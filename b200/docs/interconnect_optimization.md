# B200 Interconnect Optimization — NVLink5 + IB NDR 활용 전략

> 본 문서는 사용자 질문 "B200 환경의 NVLink 와 InfiniBand 를 잘 이해하고 활용해서 성능을 높일 수 있는 최적화 및 설정을 어떻게 찾을까?" 에 대한 **본 프로젝트의 답**을 영속화한다. 즉 — 어떤 측정·axis·analysis 가 누적되면 사용자가 답을 들고 다음 인프라로 갈 수 있는가.

## 1. B200 환경의 fabric 구조 (NHN Cloud 기준)

```
┌─ Node 1 (B200×8) ─────────────────┐    ┌─ Node 2 (B200×8) ─────────────────┐
│  GPU0─GPU7 ─ NVSwitch5 ─ 1.8 TB/s │    │  GPU0─GPU7 ─ NVSwitch5 ─ 1.8 TB/s │
│       │                            │    │       │                            │
│  HCA0 (mlx5_0) ─ ConnectX-7 NDR ──┼────┼── HCA0 (mlx5_0) ─ ConnectX-7 NDR  │
│  HCA1 (mlx5_1) ─ ConnectX-7 NDR ──┼────┼── HCA1 (mlx5_1) ─ ConnectX-7 NDR  │
│  ...                               │    │  ...                               │
└────────────────────────────────────┘    └────────────────────────────────────┘
```

| Layer | 매체 | peak | 본 프로젝트가 측정·튜닝 |
|:---|:---|:---:|:---:|
| Intra-node GPU↔GPU | NVLink5 + NVSwitch5 | 1.8 TB/s | ✅ NCCL bus_bw |
| Cross-node GPU↔GPU | NIC ↔ NIC + GDR | 363+ Gbps/HCA × N HCA | ✅ ib_write_bw + NCCL bus_bw |
| In-network reduction | NVLS (NVSwitch SHARP) / IB SHARP | ~2× 효과 | ✅ axis (`nccl_nvls_enable`, `sharp_enable`) |
| Storage ↔ GPU | GPUDirect Storage (NVMe) | 디스크 의존 | (B6.3, 후속) |

NHN Cloud B200 의 **non-priv pod RDMA Write 363.98 Gbps** 가 우리의 **fabric 정합성 reference** — fabric_probe.sh 가 매 study 직전 이 수치 ±20% 이내인지 검증한다.

## 2. application 성능에 fabric 이 꽂히는 3 경로

LLM serving 의 어느 path 가 fabric 을 어떻게 소비하는가 — 이게 안 보이면 axis 정렬이 헛수고:

| serving path | fabric 트래픽 | autotune 시 가장 큰 axis |
|:---|:---|:---|
| **TP within node** (TP=8 single node) | NVLink intra-node all-reduce 매 forward | `nccl_nvls_enable`, NVLS chunk size — NVSwitch SHARP 활용 |
| **TP cross-node** (TP=16 two nodes) | NVLink + IB all-reduce, every layer | `nccl_algo`, `nccl_net_gdr_read`, `nccl_ib_hca`, rail-aligned |
| **EP cross-node** (MoE wide-EP) | All-to-all expert routing, every MoE layer | `nccl_algo` (CollNet), `ucx_tls`, `ep_strategy` |
| **DP across nodes** (replica per node) | Gradient/state sync — inference 에선 미발생 | (training 이슈, 본 프로젝트 외) |
| **P/D disaggregation** | Prefill→Decode KV cache transfer | NIXL `nixl_transport`, `nixl_chunk_size_mb`, UCX |
| **Tiered prefix cache** | LMCache CPU/Disk/Mooncake transfer | LMCache `local_cpu_size_gb`, `remote_url` |

→ **본 프로젝트는 위 6 경로 각각에 별도 search-space 를 두고 macro × env profile 매핑** ([§ Autoresearch Architecture](../../async-cooking-cat-plan/#-autoresearch-architecture--macro--env-profile--micro)).

## 3. 본 프로젝트가 fabric 을 다루는 3 layer

### Layer 1 — fabric **baseline** 캡처

매 study 직전 1회 수행. trial 들의 절대 reference.

```bash
bash b200/scripts/fabric_probe.sh
# → b200/studies/fabric_baselines/<ts>/fabric_baseline.json
```

캡처 항목:
- NVLink intra-node bus_bw (8 GPU all-reduce, NVSwitch5 + NVLS)
- IB cross-node bus_bw (16 GPU all-reduce, NVLink + NIC + GDR)
- IB raw (`ib_write_bw` per HCA, NHN reference 363 Gbps)
- Topology (`nvidia-smi topo -m`, `lspci -tv`, `ibv_devices`)
- GDR capability (`nv_peer_mem` 모듈, `nvidia_peermem`, kernel ≥ 6.2 DMA-BUF)

**활용**: study 끝나면 ANALYSIS.md 의 § 3 (원인 분석) 에서 "trial 의 inference throughput 이 fabric peak 의 X% 였다" 같은 정합성 진술 가능.

### Layer 2 — env axis sweep (현재 PR 의 핵심)

`b200/search-spaces/b6_interconnect_tier1.yaml` — 5 axis 좁게 시작:

| axis | values | 의의 |
|:---|:---|:---|
| `nccl_algo` | Tree / Ring / NVLS | NVSwitch SHARP 활용 여부 |
| `nccl_proto` | Simple / LL128 | message size sweet spot |
| `nccl_buffsize` | 4M / 8M / 16M | cross-node large message 효율 |
| `nccl_nvls_enable` | 0 / 1 | NVLink5 in-network reduction |
| `nccl_net_gdr_read` | 0 / 1 | GDR zero-copy read |

→ 5 axis = 3 × 2 × 3 × 2 × 2 = **72 가능 조합**, TPE 20-30 trial 로 best-known 도달 가능. **70+ axis 동시 sweep 의 함정** (search blow-up + ANOVA noise) 회피.

### Layer 3 — analysis & freeze

study 종료 후:

```bash
lmtune search prune <study_id> --apply
```

→ ANOVA + RandomForest importance + bound-tighten 자동 분석:
- 효과 없는 axis (예: `nccl_proto` 가 LL128/Simple 차이 < 1%) → freeze 권고
- 영향 큰 axis (예: `nccl_nvls_enable=1` 이 +12% throughput) → next-study 의 default 로 승격
- continuous axis (없음 — 본 tier 는 categorical only) → 범위 축소

**결과 산출물**: `b200/studies/<study>/interconnect_analysis.md` (자동 생성) — axis 별 영향력 + recommended pinned values + 다음 tier 후보.

## 4. 다음 단계 (tier-2, tier-3)

본 PR 후 점진적 확장 순서:

| tier | 추가 axis | 트리거 |
|:---|:---|:---|
| tier-2 | `nccl_p2p_level`, `nccl_min_nchannels`, `nccl_cross_nic`, `nccl_ib_timeout`, `nccl_graph_register` | tier-1 결과로 algo/proto 가 freeze 된 후 |
| tier-3 (IB 자체) | `ib_mtu`, `ib_sl`, `ib_traffic_class`, `ib_gid_index`, `sharp_enable` | RoCE vs IB 환경 분리 study |
| tier-4 (HCA / rail) | `hca_count_per_gpu`, `nccl_ib_hca` rail-aligned, `nccl_socket_ifname` | 두 노드 모두 multi-HCA 활용 가능할 때 |
| tier-5 (KV transport) | NIXL/UCX/LMCache axis (B6.4) | P/D well-lit-path 활성 study |

각 tier 는 직전 tier 의 freeze 결과를 inheritance — search-space 가 누적적으로 좁아진다.

## 5. 사용 절차 (gpt-oss-120b sweep 종료 후 즉시)

```bash
# 1. fabric baseline 캡처
bash b200/scripts/fabric_probe.sh
cat b200/studies/fabric_baselines/<latest>/fabric_baseline.json
#    nccl_crossnode_busbw_gbps 가 NHN reference 280 GB/s 의 80% 이상인지 확인

# 2. interconnect tier-1 sweep (gpt-oss-120b 위에서 — 같은 모델, fabric env 만 변경)
lmtune search start --strategy tpe \
  --space b200/search-spaces/b6_interconnect_tier1.yaml \
  --endpoint b200/endpoints/b200_gpt-oss-120b.yaml \
  -p configs/profiles/autotune/{short,medium,long}.yaml \
  --backend k8s-job --workers 1 --max-trials 20 \
  --name B6-interconnect-tier1-gptoss120b

# 3. 분석
lmtune search prune <study_id>          # JSON 권고
lmtune dashboard build --out b200/dashboards
```

20 trial × ~22.5 분 = **~7.5 시간**. 결과의 axis importance ranking 이 tier-2 의 axis 후보를 결정.

## 6. 현재 한계 + 후속 작업

- **fabric_probe.sh 의 nccl-tests image** = `nvcr.io/nvidia/pytorch:25.01-py3` 가정. NCCL 2.23 (NVLS 포함) 미만이면 `nccl_nvls_enable=1` 효과 측정 불가 — image digest 박을 때 검증 필요.
- **B6.3 GPUDirect Storage axis** — 본 PR 범위 외. 모델 weight loading + KV spill-over 시 가치 있음.
- **B6.4 KV transport (NIXL/UCX/LMCache/Mooncake)** — P/D path 활성 study 에서 별도 search-space.
- **Sub-benchmark validation gate** — 매 trial 직전 30s nccl-tests 로 fabric 정합성 빠르게 확인 → degraded 시 trial skip. 본 PR 다음에 도입 (PR #14 의 circuit breaker 와 결합).

## References

- NCCL 2.23 release notes: NVLS (NVSwitch SHARP) 정식 지원
- [NVIDIA InfiniBand Tuning Guide](https://docs.nvidia.com/networking/) — `mlnx_qos`, ECN/PFC
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- 본 repo `b200/docs/rdma_perftest_baseline.md` — NHN B200 363 Gbps 재현 절차
- 본 repo `b200/docs/lowlevel_axis_catalog.md` — host-side (PCIe/IOMMU/NUMA) axis catalog
- `vllm-config-puzzle` simulator (`parallel/comm-overhead.ts`) — TP/PP/EP overhead 공식 정본
