# Phase B — 2× B200 (16-GPU) llm-d Autotuning Platform

> 본 디렉토리는 NVIDIA B200 16-GPU k3s 클러스터 위에서 llm-d 기반 autotuning 플랫폼을 운용하기 위한 모든 자산을 담는다. core 코드(`src/bench/**`) 는 거의 건드리지 않고 phase 별로 자산을 누적한다. 전체 plan 은 `(internal dev plan, not in repo)` 의 "Phase B" 섹션 참조.

> **북극성 — 사용 시나리오 (project north star)**: [`docs/usage_scenarios.md`](docs/usage_scenarios.md). 모든 코드/문서/도구는 본 시나리오를 만족시키는 형태로만 추가된다. 9 시나리오 (S1 setup / S2 cold launch / S3 reset / S4 re-launch / S5 model swap / S6 path change / S7 halt resume / S8 analyze / S9 governance) 와 진입점 매핑은 그 문서가 정본.

> **결함 catalog**: [`docs/regressions.md`](docs/regressions.md) — 사용자 시간을 한 번이라도 빼앗은 결함은 `R<n>` entry 로 영속화 (PR 게이트 의무, CLAUDE.md 의 § PR 게이트 참조).

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

## Operations — 다음 실험을 위한 환경 정비

> **vLLM 의 본성**: 거의 모든 axis (`max_num_seqs`, `kv_cache_dtype`, `enable_prefix_caching`, `gpu_memory_utilization`, parallelism) 가 engine 재시작을 강제. 즉 **"처음 시작"과 "설정 변경 후 재실행"은 본질적으로 같은 비용** (config change = pod restart = weight reload). 우리 운영 도구는 두 시나리오를 같은 진입점으로 통합한다.

### 1st-class 진입점 — `ops/launch.sh`

endpoint YAML 한 줄을 받아서 처음/재실행/모델 swap 무관하게 동일하게 동작 (idempotent):

```bash
bash b200/scripts/ops/launch.sh b200/endpoints/b200_gpt-oss-120b.yaml
# 또는 well-lit-path 명시:
bash b200/scripts/ops/launch.sh b200/endpoints/b200_gpt-oss-120b.yaml infsch
```

`launch.sh` 가 자동으로 처리:

| step | 역할 |
|:---|:---|
| 1 | endpoint YAML 파싱 → 의도한 model 추출 |
| 2 | model → values 파일 매핑 → `B200_MODEL_VALUES` 자동 export |
| 3 | cluster + namespace 검증 |
| 4 | helm release 3종 검증 + 현 vLLM 의 model id 와 endpoint 일치 비교 → 불일치 시 helmfile apply 자동 |
| 5 | decode Deployment Available 대기 |
| 6 | stale port-forward 정리 + 재시도 wrapper 데몬 |
| 7 | `/v1/models` 200 polling |
| 8 | 응답 model id 와 endpoint 의 model 최종 일치 검증 |

종료 코드 0 = launcher (`lmtune search start`) 진입 가능. 사용자 손작업 0.

### 보조 진입점

```bash
bash b200/scripts/ops/status.sh  infsch              # 현재 상태 한 화면
bash b200/scripts/ops/reset.sh                        # soft: port-forward 정리
bash b200/scripts/ops/reset.sh   infsch --pods        # decode pod rolling restart
bash b200/scripts/ops/reset.sh   infsch --hard        # helmfile destroy (prompt)

# launch.sh 의 step 3-7 만 (모델 검증 없이) — 이미 떠있는 환경 보전 시
bash b200/scripts/ops/prepare.sh infsch
bash b200/scripts/ops/prepare.sh infsch --apply       # release 미설치 시 helmfile apply
```

`<rn>` 은 well-lit-path 식별자: `infsch` (inference-scheduling) / `pd` (pd-disaggregation) / `wideep` (wide-ep-lws).

### 새 모델 추가 — values 매핑 한 줄

`b200/scripts/util/env.sh::values_for_model` 의 case 에 한 줄 추가:

```bash
case "$model" in
  openai/gpt-oss-120b)            echo "values-gpt-oss-120b.yaml.gotmpl" ;;
  meta-llama/Llama-3.1-8B*)       echo "values-llama-3.1-8b-smoke.yaml.gotmpl" ;;
  Qwen/Qwen3-235B*)               echo "values-qwen3-235b-tp2-dp4.yaml.gotmpl" ;;
  # ↑ 새 모델 추가
esac
```

`b200/helmfile/inference-scheduling/values-<...>.yaml.gotmpl` 에 그 모델용 chart values 가 있다면 endpoint YAML 의 `model:` 만 바꾸면 launch.sh 가 알아서 swap.

### Why gateway, not decode

lmtune endpoint YAML 의 `url: http://127.0.0.1:8011/v1` 은 **`infra-<rn>-inference-gateway` Service 의 :80** 으로 port-forward 됨. **decode service 직접 forward 는 우리 시나리오에 없다** — gateway 우회 시 llm-d 의 InferencePool/EPP smart routing 이 측정에서 빠져 autotune 결과가 운영 환경과 어긋난다 (이러면 llm-d 를 쓸 이유가 없다).

### Helper utilities (다른 스크립트에서 source 해서 사용)

```bash
source b200/scripts/util/pf.sh    # pf::list, pf::stop_all, pf::stop_local, pf::start, pf::probe, pf::status
source b200/scripts/util/helm.sh  # helmd::list, helmd::releases_check, helmd::diff, helmd::apply, helmd::wait_decode_ready, helmd::destroy
source b200/scripts/util/env.sh   # bench_env::require_model_values, bench_env::cluster_check, ...
```

상세 (path 별 release 매핑, 트러블슈팅, helmfile rolling 시 끊김 처리 원리): [`docs/port_forward_runbook.md`](docs/port_forward_runbook.md).

## 디렉토리 구조

```
b200/
├── README.md                          ← 이 파일
├── scripts/
│   ├── probe.sh                       ← 클러스터·fabric·이미지 캐시 진단 (B0)
│   ├── fabric_probe.sh                ← NVLink + IB baseline 통합 측정 (B6.2 매 study 직전)
│   ├── rdma_bench.sh                  ← host-level RDMA Perftest (B6.2 raw fabric)
│   ├── system_snapshot.sh             ← trial 직전 PCIe/IOMMU/NUMA capture (B6.1)
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
│   ├── b6_lowlevel.yaml               ← PCIe/IOMMU/NUMA host-level axis (B6.1)
│   ├── b6_interconnect_tier1.yaml     ← NVLink + IB env axis 1차 5종 (B6.2)
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

- **Endpoint 노출 (gateway port-forward 운영 가이드)**: [`docs/port_forward_runbook.md`](docs/port_forward_runbook.md) ← lmtune 실행 직전 필수 단계
- **실험 계획 (phase 별 모델·workload·axis·합격 기준 카탈로그)**: [`docs/experiment_plan.md`](docs/experiment_plan.md) ← B0 통과 후 다음 결정 시 참조
- 전체 plan: `(internal dev plan, not in repo)` (Phase B 섹션)
- 도구 스택 (Optuna 4.8 / SALib 1.5 / BoTorch 0.9.5): `docs/search_tooling_2026-04.md`
- llm-d well-lit paths: <https://github.com/llm-d/llm-d/tree/main/guides>
- peer helmfile templates: `$HOME/ml_ai/agentic/llm-distributed-inference` (override via `PEER_REPO` env)
