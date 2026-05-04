# Phase B — 2× B200 (16-GPU) llm-d Autotuning Platform

> 본 디렉토리는 NVIDIA B200 16-GPU k3s 클러스터 위에서 llm-d 기반 autotuning 플랫폼을 운용하기 위한 모든 자산을 담는다. core 코드(`src/bench/**`) 는 거의 건드리지 않고 phase 별로 자산을 누적한다. 전체 plan 은 `(internal dev plan, not in repo)` 의 "Phase B" 섹션 참조.

## 환경 가정

| 항목 | 기본값 | 비고 |
|:---|:---|:---|
| 노드 | 2 × B200 (8 GPU/node, 16 GPU 합) | 컨테이너로 노출, k3s 멤버 |
| HBM | ~180 GB / GPU (~3 TB 합) | HBM3e |
| Compute capability | sm_100 (Blackwell) | FP4 native |
| 인터노드 fabric | RDMA (InfiniBand 또는 RoCE) | `B0` probe 가 검증 |
| K8s | k3s | nvidia-device-plugin + Multus(또는 SR-IOV CNI) 필요 |
| 컨테이너 런타임 | containerd 권장 | cri-dockerd 도 가능 |
| peer repo | `$HOME/ml_ai/agentic/llm-distributed-inference` (override via `PEER_REPO` env) (default) | B0 에서 경로 확인·수정 가능 |

> **사용자 환경 보정**: 위와 다른 부분은 `docs/b200_environment.md` 의 실측치로 갱신하면 모든 helmfile / search-space / endpoint 가 그 값을 참조한다.

## 진입점

> **B200 클러스터에서 바로 시작**: [`b200/QUICKSTART.md`](QUICKSTART.md) — 3 well-lit-path (inference-scheduling / pd-disaggregation / wide-ep-lws) 를 ~30분에 끝까지 돌려보는 복붙 가능한 명령 시퀀스.

```bash
# 1. 환경 진단 (B0 — 최초 1회 + 정기 점검)
#    실행 위치별 모드:
#      B200-1 컨테이너 안 (k3s control plane + RDMA HW 가 직접 접근 가능):
bash b200/scripts/probe.sh --mode host
#      회사 개발 PC (kubeconfig 만 있고 RDMA HW · peer repo 는 부재):
bash b200/scripts/probe.sh --mode client

# 1a. (B200 컨테이너 안에서 한 번) helmfile + peer repo 설치
#     probe --mode host 의 helmfile / peer_repo FAIL 두 개를 sudo 없이 한 번에 해결
PEER_REPO_URL=<peer repo git URL> bash b200/scripts/setup_host.sh

# 1b. (클라이언트 PC 또는 컨테이너 어디서든 한 번, 클러스터 전역 1회)
#     gateway-provider prereq — Gateway API + GAIE CRDs + agentgateway controller 설치
#     llm-d 정본 가이드의 prereq/gateway-provider 단계를 wrap한 스크립트.
bash b200/scripts/setup_gateway_provider.sh agentgateway

# 1c. (사전 조건) 클러스터에 RuntimeClass 'nvidia' 가 등록되어 있어야 한다.
#     NHN B200 처럼 default runtime=runc 인 multi-RuntimeClass 환경에서 GPU pod 가
#     libcuda 를 받게 하는 standard 패턴. helmfile post-renderer 가 모든 Deployment
#     의 spec.template.spec.runtimeClassName 을 'nvidia' 로 자동 주입하므로
#     클러스터/k3s default runtime 변경은 불필요.
kubectl get runtimeclass nvidia      # 'nvidia' 가 보이면 OK

# 2. smoke run (B0 마지막 단계)
lmtune search start \
  --strategy random \
  --space b200/search-spaces/b0_smoke.yaml \
  --backend k8s-job --workers 2 --max-trials 4 \
  --study-prefix B0-smoke

# 3. baseline 카탈로그 (B1) — QUICKSTART.md §8 참조
lmtune search start \
  --strategy nsga2 \
  --space b200/search-spaces/b1_baselines.yaml \
  --backend k8s-job --workers 4 --max-trials 40

# (이후 phase 별 명령어는 각 phase doc 참조)
```

## 디렉토리 구조

```
b200/
├── README.md                          ← 이 파일
├── scripts/
│   ├── probe.sh                       ← 클러스터·fabric·이미지 캐시 진단 (B0)
│   ├── loop.sh                        ← continuous autotuning loop (B5)
│   └── regression_check.py            ← baseline 회귀 알림 (B5)
├── endpoints/                         ← well-lit path × 모델 별 endpoint YAML
│   ├── b200_smoke.yaml                ← B0 smoke endpoint (Llama-3.1-8B 단일 GPU)
│   └── …
├── helmfile/                          ← peer repo phase{1..4} 를 16-GPU 에 맞게 fork
│   ├── base/
│   │   └── values-b200-common.yaml.gotmpl   ← runtimeClassName, securityContext, /dev/shm, topology
│   ├── inference-scheduling/          ← well-lit path #1 (peer phase1/2 base)
│   ├── wide-ep-lws/                   ← well-lit path #2 (peer phase3/4)
│   ├── tiered-prefix-cache/           ← well-lit path #3
│   ├── precise-prefix-cache/          ← well-lit path #4
│   ├── pd-disaggregation/             ← well-lit path #5 (peer phase2/4 pd)
│   ├── predicted-latency-scheduling/  ← well-lit path #6 (실험적)
│   └── workload-autoscaling/          ← well-lit path #7 (peer phase4 Config D)
├── search-spaces/                     ← phase 별 axis 카탈로그
│   ├── b0_smoke.yaml                  ← 단일 GPU smoke (B0)
│   ├── b1_baselines.yaml              ← 7 path × 모델 baseline (B1)
│   ├── b2_vllm_engine.yaml            ← vLLM 최적화 axis (B2)
│   ├── b3_parallelism.yaml            ← TP/PP/DP/EP + topology axis (B3)
│   ├── b4_welllit_paths.yaml          ← well-lit path 자체를 axis 로 (B4)
│   ├── b5_combined_pareto.yaml        ← 통합 Pareto search (B5)
│   ├── b6_lowlevel.yaml               ← PCIe/IOMMU/NUMA/NCCL/RDMA axis (B6)
│   └── b7_multistack.yaml             ← engine × serving stack axis (B7)
├── profiles/                          ← B200 워크로드 preset
│   ├── ultra_long.yaml                ← 32K context (대모델)
│   ├── coding_agent_burstgpt.yaml     ← BurstGPT 재현
│   ├── llmd_official_*.yaml           ← llm-d-benchmark profile 직변환 (B7)
│   ├── mlperf_*.yaml                  ← MLPerf Inference v5.x scenario (B7)
│   ├── mooncake_replay.yaml           ← Mooncake trace (B7)
│   └── azure_llm_*.yaml               ← AzureLLMTraces (B7)
├── studies/                           ← phase 별 study export (DuckDB)
│   └── <phase>_<study_id>/
├── results/                           ← 공개 가능한 winning config + 리포트 (B8)
└── docs/
    ├── b200_environment.md            ← 클러스터·드라이버·fabric 스냅샷
    ├── well_lit_paths_catalog.md      ← 7 path × axis 매핑 (B1 산출)
    ├── vllm_axis_catalog.md           ← vLLM axis 카탈로그 (B2 산출)
    ├── parallelism_combinations.md    ← topology × parallelism (B3 산출)
    ├── path_decision_tree.md          ← workload→path 의사결정 (B4 산출)
    ├── continuous_loop.md             ← B5 운용 가이드
    ├── regression_alerts.md           ← B5 회귀 로그
    ├── lowlevel_axis_catalog.md       ← PCIe/IOMMU/NUMA/NCCL axis 가이드 (B6)
    ├── rdma_perftest_baseline.md      ← ib_write_bw / ib_read_bw 363 Gbps 재현 절차 (B6)
    ├── portability_guide.md           ← 다른 GPU/NPU 환경 재사용 가이드 (B7)
    ├── upstream_pr_candidates.md      ← llm-d / vLLM upstream 기여 후보 (B8)
    └── blog/                          ← 외부 공개 블로그 초고 (B8)
```

## Phase 진행 상황

| Phase | 상태 | 산출물 |
|:---|:---|:---|
| B0 — 클러스터 온보딩 | 🚧 in-progress | scripts/probe.sh, helmfile/base, endpoints/b200_smoke.yaml, search-spaces/b0_smoke.yaml |
| B1 — well-lit path baseline | ⏳ pending | helmfile/{7 paths}/, search-spaces/b1_baselines.yaml, studies/B1_*, docs/well_lit_paths_catalog.md |
| B2 — vLLM 최신 axis | ⏳ pending | search-spaces/b2_vllm_engine.yaml, docs/vllm_axis_catalog.md |
| B3 — 병렬 분산 axis | ⏳ pending | search-spaces/b3_parallelism.yaml, src/bench/deploy/llmd_k8s.py 보강 |
| B4 — well-lit path as axis | ⏳ pending | search-spaces/b4_welllit_paths.yaml, docs/path_decision_tree.md |
| B5 — Continuous loop | ⏳ pending | scripts/loop.sh, scripts/regression_check.py, docs/continuous_loop.md |
| B6 — Low-level system axis | ⏳ pending | search-spaces/b6_lowlevel.yaml, scripts/{system_snapshot.sh,rdma_bench.sh}, docs/lowlevel_axis_catalog.md, docs/rdma_perftest_baseline.md |
| B7 — Multi-engine + Multi-stack | ⏳ pending | src/bench/runners/{sglang,trt_llm,nim}.py, src/bench/deploy/{kserve,ray_serve,triton,nim_adapter}.py, search-spaces/b7_multistack.yaml, docs/portability_guide.md |
| B8 — Public packaging | ⏳ pending | results/RECIPES.md, docs/blog/, docs/conference_submission_skeleton.md, docs/upstream_pr_candidates.md |

## 사용자 실행 흐름

본 디렉토리는 사용자가 B200 클러스터에서 **직접 실행**하는 스크립트와 declarative manifest 위주다. 이 호스트(작성용 머신)는 클러스터에 직접 접근하지 않는다.

1. 작성용 호스트(여기): 본 디렉토리에 자산을 쓰고 git commit
2. B200 호스트: `git pull` 후 `bash b200/scripts/probe.sh` 같은 명령 실행
3. 실행 결과(JSON / log) 를 다시 작성용 호스트로 전달 → 다음 cycle 의 axis / overlay 갱신

## 참고 문서

- **실험 계획 (phase 별 모델·workload·axis·합격 기준 카탈로그)**: [`docs/experiment_plan.md`](docs/experiment_plan.md) ← B0 통과 후 다음 결정 시 참조
- 전체 plan: `(internal dev plan, not in repo)` (Phase B 섹션)
- 도구 스택 (Optuna 4.8 / SALib 1.5 / BoTorch 0.9.5): `docs/search_tooling_2026-04.md`
- llm-d well-lit paths: <https://github.com/llm-d/llm-d/tree/main/guides>
- peer helmfile templates: `$HOME/ml_ai/agentic/llm-distributed-inference` (override via `PEER_REPO` env)
