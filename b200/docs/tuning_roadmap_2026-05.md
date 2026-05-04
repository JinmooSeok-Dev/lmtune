# B200 16-GPU NVLink/NVSwitch + InfiniBand RDMA — Tuning Roadmap (2026-05)

> 본 환경은 **2 노드 × 8 B200 = 16 GPU + NVLink/NVSwitch (intra-node) + InfiniBand
> RDMA (inter-node, 363+ Gbps verified)** 로, 한 곳에서 LLM 추론 스택 전 layer
> (application + system + interconnect + GPUDirect) 를 1st-class 로 튜닝 가능한
> 매우 드문 setup 이다. 본 문서는 이 환경에서 **반드시 검증·튜닝해봐야 할 항목**
> 의 우선순위 매트릭스. 각 항목은 측정 가능한 가설 + 측정 도구 + 예상 결과.

> 마지막 검토: 2026-05-04 / 검토 주기: phase 종료 시 / 소유자: B200 lmtune track
> 관련 문서: `model_catalog_2026-05.md`, `well_lit_paths_catalog.md` (예정),
> `vllm_axis_catalog.md` (예정), `lowlevel_axis_catalog.md` (예정)

## 0. 본 환경의 4가지 차별점

본 환경에서만 의미 있는 (또는 가장 깊이 있게 측정 가능한) 튜닝 영역:

1. **B200 sm_100 native MXFP4 / nvfp4** — Hopper sm_90 와 다른 새 dtype path. 모델별 native vs 변환 성능 차이.
2. **NVLink/NVSwitch in-network reduction (NVLS, SHARP)** — 8-GPU intra-node 의 NCCL all-reduce 대역폭 한계 측정 + SHARP/NVLS 효과 정량.
3. **InfiniBand RDMA cross-node** — TP=16 (cross-node), wide-EP, P/D disaggregation 시 NIXL/UCX/NCCL-IB stack 영향 측정.
4. **두 replica 의 토폴로지 자유도** — TP × DP × PP × EP 조합이 단일 노드/16-GPU 에서 가능한 매우 드문 setup. node_split_strategy 자체가 axis.

## 1. 튜닝 layer 분류 + 누가 다루나

| Layer | 설명 | 1차 도구 | 적용 위치 | 본 프로젝트 phase |
|:--|:--|:--|:--|:--|
| L1 Application (vLLM engine_args) | scheduling, batching, prefix cache, speculative decoding, compilation | vllm CLI flags | helmfile values | B2 |
| L2 Parallelism (분산화 전략) | TP/PP/DP/EP/PCP/DCP, well-lit-path, node split | helmfile values + chart | helmfile chart | B3, B4 |
| L3 Engine compile (CUDA graph, kernel) | torch.compile, cudagraph, FlashInfer, DeepGEMM | env var + compilation_config | container env | B2 + B6.3 |
| L4 KV transport (NIXL/UCX/LMCache/Mooncake) | P/D 간 KV 송수신, prefix cache 분산 | env var | container env | B6.4 |
| L5 Interconnect (NCCL, IB, NVLink, SHARP) | collective lib, fabric tuning | env var + switch CLI | container env + host | B6.2 |
| L6 GPUDirect (GDR/GDS/P2P) | NIC ↔ GPU memory, NVMe ↔ GPU | env var + module | container env + kernel | B6.3 |
| L7 Host (PCIe/IOMMU/NUMA/CPU) | low-level OS/BIOS | sysfs + kernel cmdline | host (reboot) | B6.1 |

→ B0 ~ B5 까지의 search space 가 위 7 layer 를 통합한 multi-level autotune.

## 2. 우선순위 매트릭스 — Tier 1 (반드시 검증)

본 환경에서 측정 안 하면 다른 환경에서 재현 불가능한 항목.

### 2.1 NCCL collective × NVLink/NVSwitch (Tier 1) ⭐⭐⭐

**가설**: B200 sm_100 + NVSwitch 의 NVLS (NVLink SHARP) in-network reduction 이
8-GPU all-reduce 대역폭을 1.5~2.5× 향상 (Hopper 측정치 대비).

| axis | 값 | 측정 |
|:--|:--|:--|
| `NCCL_NVLS_ENABLE` | {0, 1} | bus_bw GB/s, all-reduce latency |
| `NCCL_ALGO` | {Tree, Ring, NVLS, CollNet} | 알고리즘별 효과 (NVLS 가 sm_100 에서 default 가 됐는지) |
| `NCCL_PROTO` | {Simple, LL, LL128} | small msg vs big msg 효과 |
| `nvlink_p2p` | {auto, force-on, force-off} | P2P 비활성 시 PCIe fallback |

측정: `nccl-tests` (`all_reduce_perf -b 8 -e 16G -f 2 -g 8`), bus_bw GB/s + latency × 메시지 크기.

**예상 결과**:
- B200 NVLink Gen5 (1.8 TB/s 양방향) → all-reduce bus_bw 700-900 GB/s 가능
- NVLS=1 활성 시 8-GPU 에서 small msg latency ~30-40% 감소 (in-network reduction)
- 본 측정은 vllm autotune 에 직접 적용보다는 **NCCL 권장 default 결정** 의 정량 근거

### 2.2 InfiniBand RDMA × cross-node TP/EP (Tier 1) ⭐⭐⭐

**가설**: 363 Gbps RDMA 가 TP=16 (cross-node) 시 NCCL all-reduce 의 병목이 됨.
GDR + rail-aligned HCA + adaptive routing 활성 시 80% 활용 도달, 비활성 시 40% 이하.

| axis | 값 | 측정 |
|:--|:--|:--|
| `NCCL_NET_GDR_LEVEL` | {LOC, PIX, PXB, PHB, NODE, SYS} | GDR scope 별 bus_bw |
| `NCCL_NET_GDR_READ` | {0, 1} | GDR read 활성 영향 |
| `NCCL_NET_GDR_C2C` | {0, 1} | B200 신규 C2C path |
| `NCCL_IB_HCA` | rail-aligned (`mlx5_0:1,mlx5_1:1`) vs default | HCA 매핑 영향 |
| `NCCL_CROSS_NIC` | {0, 1} | multi-rail 활용 |
| `ib_adaptive_routing` | {on, off} (switch CLI) | switch 단 라우팅 |
| `NCCL_IB_SL` | {0, 3} | QoS service level |
| `sharp_enable` | {on, off} | SHARP in-network reduction (cross-node) |

측정: `ib_write_bw` 단발 (363 Gbps 재현 baseline) + `nccl-tests all_reduce_perf` 8-node × 2-node 토폴로지.

**예상 결과**:
- GDR=SYS + rail-aligned + AR=on → bus_bw 280-320 Gbps (RDMA 80% 활용)
- 비활성 → 100-150 Gbps (40% 이하)
- 본 측정은 (a) NHN B200 RDMA 363 Gbps 의 LLM 워크로드 활용도 정량 입증, (b) cross-node TP/EP 의사결정의 근거

### 2.3 B200 native MXFP4 / nvfp4 (Tier 1) ⭐⭐⭐

**가설**: B200 sm_100 의 MXFP4 native path 가 FP8 대비 2× throughput, 절반 메모리.
nvfp4 는 정확도 손실 없이 비슷한 효과. 모델별 (gpt-oss native vs DSV3 변환) 차이 큼.

| axis | 값 | 활성 모델 |
|:--|:--|:--|
| `dtype × kv_cache_dtype` | {bf16/auto, bf16/fp8, fp8/fp8, mxfp4/auto, nvfp4/nvfp4} | 모델별 native dtype |
| `quantization` | {none, fp8, fp8_dynamic, nvfp4, marlin} | 비-native dtype 시 quant pathway |

측정: 같은 모델 (예: DSV3 671B) × 다른 dtype × 같은 워크로드 (`autotune-medium`) → throughput / TTFT / quality probe (MMLU subset).

**예상 결과** (DSV3 671B, B200 16-GPU):
| dtype | throughput | TTFT p99 | 메모리/GPU | quality (MMLU subset) |
|:--|:--:|:--:|:--:|:--:|
| bf16 | 1.0× | baseline | 168GB ✗ (OOM 위험) | ref |
| FP8 e4m3 | 1.7-2.0× | -10~20% | 84GB ✓ | -0.5~1% |
| MXFP4 (변환) | 2.5-3.0× | -25~30% | 42GB ✓✓ | -1~2% |
| nvfp4 native | 2.5-3.0× | (FP8 동급) | 42GB | ≈ FP8 |

→ B200 의 핵심 가치 입증. 모델별 권장 dtype + quality 손실 표 → b200/docs/dtype_recommendations.md (예정).

## 3. Tier 2 — 본 환경에서 더 깊이 측정 가능

### 3.1 Speculative Decoding × MTP × Eagle3 (Tier 2) ⭐⭐

**가설**: B200 sm_100 + FP8 + 큰 모델 (≥200B) 에서 EAGLE3 / MTP 가 throughput
1.5~3× 향상. GLM-4.7 의 MTP native, DSV3 의 MTP, 일반 모델의 EAGLE3 비교.

| axis | 값 | 활성 모델 |
|:--|:--|:--|
| `speculative_method` | {none, eagle, eagle3, mtp, draft_model, ngram} | 모든 모델 (ngram 은 catch-all) |
| `speculative_num_speculative_tokens` | {1, 2, 4, 5, 8} | spec 활성 시 |
| `speculative_disable_by_batch_size` | {0, 16, 32, 64} | high-batch 시 spec 자동 off |

측정: 같은 모델 × 다른 spec method × 같은 워크로드, throughput + acceptance rate.

**예상**:
- GLM-4.7 MTP native: 2.5-3× throughput (B200 + FP8 + KV cache 충분)
- DSV3 MTP: 1.5-2.0×
- ngram: 1.1-1.3× (cheap fallback)

### 3.2 Wide-EP (DeepSeek-V3, Kimi K2, Qwen3-Coder) (Tier 2) ⭐⭐

**가설**: MoE 모델에서 wide-EP (DP=8 × EP=8) 가 standard EP (TP=8 with EP=1) 대비
expert 메모리 효율 + decode throughput 1.5~2× 향상. 단 cross-node All-to-All 비용
때문에 RDMA 대역폭이 충분해야 효과.

| axis | 값 |
|:--|:--|
| `well_lit_path` | {inference-scheduling, wide-ep-lws} |
| `tp × dp × ep` | {(8,2,1), (4,4,2), (2,8,4), (1,8,8)} |
| `ep_strategy` | {standard, wide} |
| `enable_dbo` | {true, false} (Decode Batch Overlap, MoE 전용) |
| `enable_eplb` | {true, false} (Expert Load Balancing) |
| `all2all_backend` | {allgather_reducescatter, deepep_low_latency, deepep_high_throughput, nixl_ep} |

측정: DSV3 / Kimi K2 / Qwen3-Coder 각각 × 4 토폴로지 × `autotune-medium` (코드 워크로드).

**예상**:
- Kimi K2 (1T) + wide-EP (DP=8 EP=8) → expert 메모리 1/8 → 더 큰 batch 가능 → throughput ↑↑
- All-to-All 이 RDMA 363 Gbps 거의 활용 (NCCL_NVLS + 노드 간 EP)

### 3.3 P/D Disaggregation + NIXL/UCX (Tier 2) ⭐⭐

**가설**: prefill compute-bound + decode memory-bound 의 분리가 throughput 1.3~1.8×.
NIXL 정상화 후 UCX backend 와 NCCL backend 비교.

> **현재 NIXL 미정상** — 이 영역은 NIXL 안정화 후 진입.

| axis | 값 |
|:--|:--|
| `well_lit_path` | pd-disaggregation |
| `prefill_decode_ratio` | {1:1, 1:2, 2:1, 1:4} |
| `nixl_transport` | {tcp, rdma, ucx, gpudirect} |
| `ucx_tls` | {default, rc_x+cuda_ipc, dc_x+cuda_copy} |
| `nixl_chunk_size_mb` | {1, 4, 16, 64} |

### 3.4 Tiered Prefix Cache (LMCache + Mooncake) (Tier 2) ⭐

**가설**: multi-turn 워크로드 (autotune-multiturn-agent) 에서 prefix cache 의 HBM →
CPU → NVMe 계층화가 cache hit rate 30→70% 상승, throughput 2× 가능.

| axis | 값 |
|:--|:--|
| `well_lit_path` | tiered-prefix-cache |
| `enable_prefix_caching` | true |
| `lmcache_local_cpu_size_gb` | {0, 16, 64, 256} |
| `lmcache_remote_url` | {none, mooncake://...} |
| `lmcache_compress` | {none, lz4, zstd} |
| `lmcache_pinned_memory` | {on, off} |

측정 워크로드: `autotune-multiturn-agent` 의 5-turn 세션 시뮬에서 turn N 의 TTFT 가
turn 1 대비 얼마나 감소하는지 (prefix hit rate × 효과).

## 4. Tier 3 — 시스템 저수준 (사용자 커리어 차별화)

### 4.1 PCIe / IOMMU / NUMA × LLM 추론 (Tier 3) ⭐

이력서 차별화 포인트. 일반 OSS autotune 에서는 다루지 않는 영역.

| axis | 값 | 적용 |
|:--|:--|:--|
| `pcie_aspm` | {default, performance, off} | kernel cmdline (reboot) |
| `pcie_acs_override` | {disabled, downstream} | kernel cmdline |
| `pcie_max_payload` | {auto, 256B, 512B} | sysfs |
| `iommu_passthrough` | {on, off} | kernel cmdline `iommu=pt` |
| `numa_balancing` | {0, 1} | sysctl |
| `cpu_pinning_strategy` | {none, numa-aware, single-numa-node, rail-aligned} | k8s topology manager |
| `transparent_hugepages` | {always, madvise, never} | sysfs |
| `hugepages_1gi` | {0, 16, 32} | kernel cmdline |
| `cpu_governor` | {performance, ondemand} | sysfs |
| `smt` | {on, off} | kernel cmdline `nosmt` |

측정: 같은 axis 외 모든 게 동일한 baseline 위에 1개씩 toggle, throughput 차이 통계 검정.

**예상**:
- iommu_passthrough=on (vs strict): cross-node TP/EP 에서 4-8% throughput ↑
- transparent_hugepages=madvise (vs always): 2-4% latency 감소 (page fault 회피)
- cpu_pinning rail-aligned: NUMA 미스매치 시 5-10% throughput 손해

### 4.2 InfiniBand fabric 깊은 튜닝 (Tier 3) ⭐⭐

| axis | 값 | 적용 |
|:--|:--|:--|
| `ib_mtu` | {2048, 4096} | port config |
| `ib_qp_type` | {RC, UC, UD} | application |
| `ib_traffic_class` (DSCP) | {0, 32, 64, 96, 128} | NCCL_IB_TC |
| `ib_pkey` | partitioning | switch + IB |
| `aec_cable_link_training` | {auto, fixed} | switch CLI |
| `pfc_enable` | {on, off} (RoCE 시 필수) | switch + HCA |
| `dcqcn_enable` | (RoCE) | switch |
| `mlxconfig` MR caching | {on, off} | mlxconfig |

측정: `ib_write_bw` + `nccl-tests` cross-node × axis 별 토글.

### 4.3 SHARP / NVLS in-network reduction (Tier 3) ⭐⭐

| axis | 값 |
|:--|:--|
| `sharp_enable` | {on, off} (cross-node) |
| `sharp_group_size_thresh` | {2, 4, 8, 16} |
| `nvlink_sharp` (NVLS, intra-node) | {on, off} |
| `nccl_collnet_enable` | {0, 1} |

측정: nccl-tests 의 group_size × 메시지 크기 매트릭스.

## 5. Tier 4 — 응용 / 모델 비교

### 5.1 Cross-model 동일 환경 비교 (Tier 4) ⭐⭐

같은 B200 16-GPU + 같은 워크로드 + 같은 dtype 정책에서 모델 7-10종 동시 측정 →
"이 환경에서 X 워크로드는 어느 모델이 최적인가" 결정 트리.

대상 모델 (model_catalog 의 1순위 + 검증 완료):
- Kimi K2-Instruct (1T)
- DeepSeek-V3.2 (671B)
- DeepSeek-R1-0528 (671B)
- Qwen3-Coder-480B
- GLM-4.7-355B (MTP)
- Llama-4-Maverick-400B
- gpt-oss-120b (현재)
- Llama-3.1-70B (dense baseline)
- Qwen2.5-72B (dense baseline)

워크로드:
- `autotune-short` / medium / long (단발)
- `autotune-multiturn-agent` (multi-turn ⭐)
- BurstGPT replay (향후)

산출: 모델 × 워크로드 × dtype 의 throughput / TTFT / cost-per-token / quality 매트릭스.

### 5.2 Multi-turn × Prefix-cache 효과 (Tier 4) ⭐⭐

aiperf user_centric mode 의 강점. **단발 측정에서는 절대 보이지 않는** 현상:

| 측정 | 가설 |
|:--|:--|
| Turn N 의 TTFT vs Turn 1 | prefix cache hit 시 70-90% 감소 |
| Conversation length 별 KV cache pressure | 30 turn 이상부터 KV evict 발생 → TTFT 회귀 |
| 동시 user 수 × prefix cache hit rate | 16 user 이상에서 cache thrashing |
| LMCache offload 효과 | HBM 부족 시 CPU/NVMe spill 의 hit rate / TTFT |

→ 본 환경에서만 가능한 측정 (16-GPU 면 multi-turn 동시 user 부하 의미 있음).

### 5.3 Long-context 특화 (Tier 4) ⭐

| 워크로드 | 입력 길이 | 측정 가설 |
|:--|:--|:--|
| short | 256 | TTFT 영향 미미 |
| medium | 1024 | 중간 |
| long | 3072 | KV cache 압박 시작 |
| **ultra-long (예정)** | 32K | KV cache eviction, paged-attn 효율 |
| **needle-in-haystack (예정)** | 128K (DSV3/K2 native) | 정확도 + throughput 동시 |

axes:
- `max_model_len` : {8192, 16384, 32768, 65536} (이미 추가됨)
- `block_size` : {16, 32, 64, 128} (KV block 크기)
- `enable_chunked_prefill` × `chunked_prefill_size` : {512, 1024, 2048, 4096}

## 6. 우선순위 — 6 phase 진행

| Phase | Tier | 핵심 측정 | 기간 | 산출 |
|:--:|:--:|:--|:--:|:--|
| **B0** ✅ | - | 환경 probe + smoke | (완료) | b200_environment.md |
| **B2 stage 1** 🟢 진행 | 1 | gpt-oss-120b 16-GPU baseline + engine_args importance | 1주 | study + ANALYSIS.md |
| **B2 stage 2** | 1 | Tier 1: NVLink/NVSwitch (NCCL/NVLS), MXFP4/FP8/nvfp4 | 1주 | NCCL bandwidth report + dtype matrix |
| **B3** | 1, 2 | Tier 1: RDMA × cross-node TP/EP, Tier 2: parallelism axis | 2주 | Pareto front (single/dual node) |
| **B4** | 2 | well-lit-path × 모델 매트릭스, P/D + LMCache | 2주 | path 의사결정 트리 |
| **B5** | 3, 4 | Tier 3: low-level + Tier 4: cross-model | 3-4주 | RECIPES.md 영구 catalog |
| **B6** | 3 | 시스템 저수준 (PCIe/IOMMU/NUMA) | 2주 | 이력서 차별화 |
| **continuous** | - | autoresearch loop | 지속 | 새 모델/기능 자동 흡수 |

## 7. 측정 원칙 (모든 phase 공통)

1. **3-run median + CV gate** — 단일 측정 금지. CV ≥ 0.10 시 5-run 재측정. CV ≥ 0.10 지속 시 reject.
2. **하나만 변경** — 1 trial 당 axis 1-3개만. ANOVA / Sobol 로 분리 가능한 디자인.
3. **Cold-start 경계** — 첫 trial 의 TTFT 는 60s SLO 까지 허용 (모델 weight 로딩). warm 측정만 비교.
4. **system_snapshot.json** — 매 trial 직전 ariadne (B6 활성 시) 또는 lstopo+lspci 캡처. 결과와 cross-reference.
5. **DuckDB → ANALYSIS.md** — 매 study 종료 시 `b200/studies/<id>/ANALYSIS.md` 5섹션 (컨텍스트/결과/원인/의의/후속) 작성.
6. **Reproducible recipe** — winning config 은 `b200/results/<id>/winner/{apply.sh, values-overlay.yaml}` 자동 생성.

## 8. 본 환경에서 만들어낼 수 있는 outputs

- **블로그 / MLSys 워크숍 submission 후보**:
  - "B200 NVLink/NVSwitch + InfiniBand RDMA 한 setup 에서 wide-EP / P-D / FP8 vs MXFP4 의 throughput 영향 정량"
  - "PCIe ACS / IOMMU pt / NUMA pinning 차이가 LLM 서빙 throughput 에 미치는 영향 — 일반 OSS autotune 미답 영역"
  - "vLLM 0.17.1 의 MTP / EAGLE3 / draft-model 비교 — 본 환경의 spec decoding 효과 매트릭스"

- **vLLM / llm-d upstream contribution 후보**:
  - vllm 의 NVLS default 권장값 변경 PR (B200 sm_100)
  - llm-d 의 well-lit-path × 모델 권장 매트릭스 (model_catalog 자체)
  - aiperf 0.7.0 호환성 패치 (PR #5 이미 완료)

- **사용자 차별화 reference**:
  - GitHub README 의 "B200 16-GPU + RDMA + NVLink/NVSwitch 풀스택 튜닝 정본"
  - 이력서 의 "Multi-vendor (B200 / NPU) autotune 도구 + low-level 시스템 튜닝 통합"

## 9. 알려진 risk / 장애물

| Risk | 영향 | mitigation |
|:--|:--|:--|
| NIXL 미정상 | P/D path 검증 불가 | UCX backend 시도 → 안 되면 NCCL kv-connector 로 대체 |
| EPP / vllm metric mismatch | gateway routing 503 | svc-direct port-forward 우회 (현재 적용) + EPP value 옵션 정리 |
| PCIe/IOMMU 변경 = host reboot | 짧은 down-time | Tier 3 axis 는 야간/배치 모드로만 |
| 모델 변경 = helm uninstall+install | revision churn | helm chart 의 selector 명시 + Recreate strategy |
| 사용자 시간 제약 | 풀-스케일 study 6-9주 | per-tier merge gate, 사용자 결정 시점에 멈춤 가능 |

## 10. 본 문서의 위치

본 환경에서 검증할 항목의 **mother list**. phase 진행 시 본 문서를 참조해 search-space
yaml 을 도출하고, study 종료 시 본 문서의 "예상 결과" 와 실측 비교 → 본 문서 업데이트.
즉 본 문서는 **living roadmap** 으로 매 phase 종료 시 갱신.

작업 단위:
- `b200/search-spaces/b{N}_*.yaml` ↔ § 2-5 의 axis 카탈로그
- `b200/studies/<id>/ANALYSIS.md` ↔ § 2-5 의 가설/예상 검증
- `b200/results/<id>/winner/` ↔ § 8 의 산출
- `b200/RECIPES.md` (예정) ↔ B5 종료 시 모든 winning config 통합
