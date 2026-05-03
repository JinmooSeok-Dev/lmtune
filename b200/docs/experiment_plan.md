# B200 16-GPU 실험 계획 (Phase B0~B8)

> **본 문서의 위치**: B200 클러스터에서 **무엇을 / 어떤 모델로 / 어떤 axis 로 / 어떤 합격 기준으로** 실험할지의 phase 별 카탈로그.
> Quickstart 명령은 [`b200/QUICKSTART.md`](../QUICKSTART.md), B0 진단 절차는 [`B0_runbook.md`](B0_runbook.md), axis 카탈로그는 [`vllm_axis_catalog.md`](vllm_axis_catalog.md) / [`lowlevel_axis_catalog.md`](lowlevel_axis_catalog.md) 를 참조.
> 전체 plan 은 `/home/jinmoo/.claude/plans/async-cooking-cat.md` § Phase B.
>
> **소유자**: jinmoo · **최종 검토**: 2026-05-04 · **검토 주기**: phase 종료 시마다

## 1. Executive Summary

NHN Cloud B200 16-GPU 클러스터 (2 노드 × 8 GPU, 약 3 TB HBM3e, RDMA) 위에서 **vLLM + llm-d** 의 autotuning 플랫폼을 누적식으로 구축한다. **목적은 단일 페이퍼-grade 결과가 아니라 continuous autotuning 자체** — 새 vLLM/llm-d 기능이 release 될 때마다 search-space 의 axis 한 줄로 흡수되고, B-track 이 끝나면 외부 GitHub 사용자가 README 만 보고 자기 환경에서 B0 smoke 를 재현할 수 있는 reference 코드가 남는다.

**4 트랙으로 재정렬** (plan 2026-04-29 결정):

| Track | Phase | Goal | Wall-clock |
|:---|:---|:---|:---|
| **A — autoresearch 통합** (env 독립) | S6 | LLM 도메인 지식 + Optuna 통계 효율 결합 | 1-2 주 |
| **B-I — Foundation** | B0 → B1 (3 path) → B3 | 16-GPU 위에서 platform 가동 + parallelism axis 활성 | 5-7 주 |
| **B-II — Depth** | B2 + B6 | vLLM 신기능 흡수 + low-level system axis (커리어 차별화) | 4-5 주 |
| **B-III — Continuous** | B5 + B4 evolution | autotune loop 가 새 기능을 axis 1줄로 흡수 | 지속 (ambient) |
| **B-IV — Outreach** (선택) | B7 + B8 | 외부 공개 가능 형태로 다듬기 | 4-6 주 |

**총 합** B-IV 포함 11-14 주, 미포함 7-9 주. **현 위치: B0 (in-progress)**.

## 2. Hardware / Software 가정 (실측치는 [`b200_environment.md`](b200_environment.md) 정본)

| 항목 | 가정 | 실측 갱신 |
|:---|:---|:---|
| GPU | 2 × B200 노드 × 8 GPU = 16 sm_100 | probe.sh PASS 시 |
| HBM | ~180 GB/GPU (~3 TB 합) | nvidia-smi |
| Fabric | RDMA (InfiniBand 또는 RoCE) | rdma_bench.sh |
| K8s | k3s, default runtime=runc + RuntimeClass `nvidia` | `kubectl get runtimeclass nvidia` |
| Container 이미지 | `ghcr.io/llm-d/llm-d-cuda:v0.7.0-rc.2` (B0 commit `e90fb16` 시점) | **v0.6.0 은 B200 sm_100 Triton JIT 실패 → v0.7.0-rc.2 로 bump** |
| llm-d-modelservice chart | v0.4.12 | post-renderer 로 `runtimeClassName=nvidia` 주입 |
| Gateway | agentgateway (default istio 는 k3s 에서 Telemetry CRD 부재로 fail) | `setup_gateway_provider.sh` |

> **알려진 함정 (B0 단계에서 해결됨)**: ① chart 가 `runtimeClassName` 미노출 → helm post-renderer 로 모든 Deployment 에 inject (`b200/helmfile/_postrender/postrender.sh`, 테스트 `tests/deploy/test_postrender_runtime_class.py`). ② istio Telemetry CRD 부재 → `gateway.provider: agentgateway` 명시. ③ v0.6.0 이미지의 Triton JIT 실패 → v0.7.0-rc.2 로 image bump (검증 중).

## 3. Phase 별 실험 카탈로그

### 3.1 Phase B0 — 클러스터 온보딩 + smoke (현재 진행)

| 항목 | 내용 |
|:---|:---|
| Goal | 2-node 16-GPU k3s 위에서 `lmtune search` + K8sJobBackend 가 동작함을 증명. **이전 작업이 B200 에서 그대로 재활용 가능함** 의 정의. |
| 모델 | Llama-3.1-8B-Instruct (single GPU, smoke 전용) |
| Workload | `configs/profiles/autotune/short.yaml` (256 in / 128 out, 30 req) |
| Search axis | [`b0_smoke.yaml`](../search-spaces/b0_smoke.yaml) — `max_num_seqs ∈ {32, 64, 128}` × `enable_prefix_caching ∈ {true, false}` (4 trial random) |
| 명령 | [`B0_runbook.md`](B0_runbook.md) §1-9 또는 [`QUICKSTART.md`](../QUICKSTART.md) §2-4 |
| Acceptance | (1) probe.sh PASS, (2) helmfile dry-run + apply 성공, (3) `curl /v1/models` 응답, (4) `lmtune run` SLO 통과 (TTFT p99 ≤ 500ms), (5) `lmtune search` 4 trial 완주 + DuckDB 적재, (6) `pytest` 96+ PASS |
| 산출물 | `b200/studies/B0_smoke/{probe.txt, run_smoke.log, search.log}`, [`b200_environment.md`](b200_environment.md) 갱신 |
| 다음 phase trigger | 위 6개 항목 모두 PASS + phase commit `B0: B200 cluster onboarding probe results + smoke runs` |

### 3.2 Phase B1 — well-lit path baseline 카탈로그 [Track B-I, scope 축소: 7→3]

| 항목 | 내용 |
|:---|:---|
| Goal | llm-d 가 권장하는 **3 핵심 well-lit path** 를 B200 16-GPU 환경에서 재현 가능한 baseline 으로 확보 (나머지 4 path 는 B-III 에서 점진 추가). **이후 모든 autotuning 의 출발점**. |
| 모델 매트릭스 | (아래 §4 참조) — 핵심 베이스라인 5종: Qwen3-235B / Llama-3.1-70B / Qwen2.5-72B / Mixtral-8x22B / DeepSeek-V3 |
| Workload | `short` + `medium` + `long` 3종 (256/128, 1024/256, 4096/512) |
| Search axis | [`b1_baselines.yaml`](../search-spaces/b1_baselines.yaml) + [`b1_pd_ratio.yaml`](../search-spaces/b1_pd_ratio.yaml) — well_lit_path × model × dtype × parallelism_basic + P/D ratio 9 조합 (`prefill+decode ≤ 4`) |
| Strategy | NSGA-II multi-objective (throughput↑ / TTFT p99↓ / cost↓), 40 trial × 3 path × 5 모델 ≈ 14+ 의미 있는 baseline trial (path-model 호환 필터 후) |
| 명령 |  ```bash<br>NS=b200-pd helmfile -f b200/helmfile/pd-disaggregation/helmfile.yaml.gotmpl --selector role=base apply<br>lmtune search start --strategy nsga2 --space b200/search-spaces/b1_pd_ratio.yaml \\<br>  --endpoint b200/endpoints/b200_pd_llama70b.yaml \\<br>  -p configs/profiles/autotune/{short,medium,long}.yaml \\<br>  --backend k8s-job --workers 4 --max-trials 40 \\<br>  --name B1-pd-ratio<br>``` |
| Acceptance | (1) 3 path × 최소 2 모델 = 6+ baseline 이 archive, (2) 각 path 별 throughput / TTFT p99 / e2e p99 가 SLO 통과, (3) `lmtune search status --pareto` 가 path 별 Pareto front 출력, (4) `b200/docs/well_lit_paths_catalog.md` 신규 — path 별 "어떤 axis 가 미는 핵심인지" 1-page 요약 |
| 산출물 | `b200/studies/B1_*/` (DuckDB export per study), `b200/docs/well_lit_paths_catalog.md`, `b200/results/B1_*/winner/` (top-1 의 apply.sh + values-overlay.yaml) |
| 다음 phase trigger | 6+ baseline winning config 가 `b200/results/RECIPES.md` 에 entry 로 등록 (B0 의 후속). B-II 의 B2 (vLLM 신기능) 또는 B-I 의 B3 (parallelism) 로 분기 가능. |

### 3.3 Phase B2 — vLLM 최신 최적화를 axis 로 흡수 [Track B-II]

| 항목 | 내용 |
|:---|:---|
| Goal | vLLM 0.7+ 도입 axis (`enable_chunked_prefill`, `enable_async_scheduling`, `enable_dbo`, `enable_eplb`, `kv_cache_dtype=fp8/fp4`, `compilation_config`, `enable_speculative_decoding` 등) 를 search space 에 등록. **각 axis 의 효과를 B1 baseline 위에 얹어 검증**. |
| 모델 | B1 의 best 5 config 위에 axis 1개씩 추가 (의존성: dbo/eplb 는 MoE only, fp4 는 Blackwell only) |
| Workload | B1 과 동일 short/medium/long. 추가로 `b200/profiles/coding_agent_burstgpt.yaml` (B5 와 공유) |
| Search axis | [`b2_vllm_engine.yaml`](../search-spaces/b2_vllm_engine.yaml) — 14+ axis. `active_if` 게이팅 (model_family / hw_arch). 자세한 카탈로그는 [`vllm_axis_catalog.md`](vllm_axis_catalog.md) |
| Strategy | TPE 40 trial × workload 3개. 종료 시 `lmtune search prune <study>` 로 ANOVA + RandomForest importance + Sobol total-order 산출 → 효과 없는 axis 자동 freeze |
| Acceptance | (1) B1 best 대비 score ≥ 5% 개선 (또는 변동 ≤ 1% 안정성 입증), (2) `lmtune search prune` 이 axis별 freeze/drop/shrink 권고 JSON 출력, (3) Sobol total-order index ranking 산출 |
| 산출물 | `b200/studies/B2_*/`, `b200/docs/vllm_axis_catalog.md` 갱신 (axis 별 영향력·게이팅·출처) |
| Risks | `kv_cache_dtype=fp4` 가 v0.7.0-rc.2 에서 동작 안 할 수 있음 → first-trial probe 실패 시 study 단위 freeze (S2 pruner 재사용) |

### 3.4 Phase B3 — Parallelism axis 확장 (10-axis vllm-config-puzzle 1:1 port) [Track B-I]

| 항목 | 내용 |
|:---|:---|
| Goal | 기존 `tp/pp/dp/ep` 4 axis → **vllm-config-puzzle simulator 정본의 10 axis** 로 확장: `tp/pp/dp/ep/ep_strategy/pcp/dcp/sp/intraNode_type/crossNode_type` + 10 feasibility constraint (NPU count, attention head divisibility, GPU mem fit, DCP-TP divisibility 등). **사용자 명시 (2026-05-03): "vllm puzzle 의 simulator 구현을 보면 굉장히 다양한 병렬 분산 구성에 대한 내용이 존재해 이를 꼭 반영해줘"** |
| 모델 | Qwen3-235B (TP=2 × DP=4 vs TP=4 × DP=2 vs TP=8 × DP=1), DeepSeek-V3 (DP=8 EP wide vs standard), MiniMax-M1 (Lightning Attention + PCP) |
| Workload | `long` (4K/512) + `ultra_long` (32K context) — PCP/DCP 가 의미 있는 영역 |
| Search axis | [`b3_parallelism.yaml`](../search-spaces/b3_parallelism.yaml) — 10-axis + 10 validation constraint + MoE/MLA conditional axis |
| Strategy | NSGA-II multi-obj (throughput / TTFT / cost) 60 trial × workload 2개. feasibility filter 가 infeasible 80% 자동 prune |
| Acceptance | (1) Pareto front 가 single/dual node trade-off 명시, (2) `b200/docs/parallelism_combinations.md` 신규 — 토폴로지별 valid 조합표, (3) `src/lmtune/search/feasibility.py` 의 10 룰 unit test 모두 PASS, (4) PCP/DCP 활성 trial 이 long context 에서 single-GPU 불가 모델을 fit |
| 산출물 | `b200/studies/B3_*/`, `src/lmtune/search/feasibility.py` (vllm-config-puzzle `validation.ts` 1:1 port), `src/lmtune/search/surrogate_analytical.py` (TTFT/ITL/TPS 공식) |

### 3.5 Phase B4 — well-lit path 자체를 axis 로 [Track B-III, B1 점진 확장과 결합]

| 항목 | 내용 |
|:---|:---|
| Goal | B1 의 3 path → 7 path 로 단계적 확장 + path 자체를 categorical axis (`well_lit_path ∈ {inf-sch, wide-ep, tiered, precise, pd, pred-lat, wva}`). 각 trial 이 path 를 동적 선택 → helmfile root 자동 스위치. |
| 모델 | B1 모델 카탈로그 + GLM-4.7-355B (MTP) |
| Search axis | [`b4_welllit_paths.yaml`](../search-spaces/b4_welllit_paths.yaml) — well_lit_path × dtype × parallelism × workload, `active_if` 로 호환 axis 자동 게이팅 (예: `enable_dbo` 는 wide-ep 에서만) |
| 명령 | `LLMDK8sAdapter.apply()` 가 `well_lit_path` 인자로 helmfile 디렉토리 디스패치. 사용자는 axis 만 켜면 자동 라우팅. |
| Acceptance | (1) workload 3개 × 모델 3개 = 9 조합 각각에서 7 path 중 best 결정, (2) `b200/docs/path_decision_tree.md` 신규 — "이 모델·워크로드 조합에서는 path X 가 최적" 의사결정 트리 |
| 산출물 | `b200/studies/B4_*/`, `b200/docs/path_decision_tree.md` |

### 3.6 Phase B5 — Continuous autotuning loop (ambient, B1 직후 시작) [Track B-III]

| 항목 | 내용 |
|:---|:---|
| Goal | autoresearch 루프를 `b200/` 위에서 백그라운드 상시 실행. **새 vLLM/llm-d 기능 release → axis 1줄 추가 → 자동 재실행 → 결과 archive → git auto-commit 사이클** 정착. |
| 트리거 | (a) `b200/perf-changelog.yaml` (InferenceX-app 호환 schema) 에 new entry, (b) ScheduleWakeup / cron daily, (c) 새 vLLM 릴리스 watch |
| 명령 | `b200/scripts/loop.sh` (신규, B5 산출물). `lmtune search resume` 으로 archive DB 를 warm-start 소스로 재활용 |
| Acceptance | (1) 7일 무인 운영 + 누적 trial ≥ 200, (2) 회귀 알림 동작 (직전 baseline 대비 score 5%↓ 시 `b200/docs/regression_alerts.md` 갱신), (3) axis catalog 1회 자동 prune 권고 적용, (4) git log 에 `B5: <date> <study_id> top-3 configs ↑X% over baseline` commit 자동 |
| 산출물 | `b200/scripts/{loop,regression_check}.{sh,py}`, `b200/docs/continuous_loop.md`, `b200/perf-changelog.yaml` (append-only) |

### 3.7 Phase B6 — Low-level system axis (커리어 차별화) [Track B-II]

| 항목 | 내용 |
|:---|:---|
| Goal | 사용자 이력서 강점 (BIOS / Kernel Params / **PCIe·IOMMU·SR-IOV·NUMA · InfiniBand·NVLink·NVSwitch · GPUDirect RDMA·GDS · NIXL·UCX·LMCache·Mooncake**) 을 **autotuning axis 로 노출**. autotune 결과가 application 하이퍼파라미터가 아니라 **시스템 레이어까지 가는 종단 튜닝 데이터**. |
| 4 sub-section | **B6.1 Host** (PCIe ASPM/ACS/Payload, IOMMU pt, NUMA, hugepages, CPU governor, SMT) · **B6.2 Interconnect** (IB MTU/SL/QP/AR/SHARP, NVLink P2P, NVSwitch, NCCL `algo/proto/p2p_level/buffsize/topo_file/tuner_plugin/ib_hca/cross_nic`) · **B6.3 GPUDirect** (GDR `nccl_net_gdr_level/read/c2c`, GDS `cufile`, P2P, DCB PFC/DSCP) · **B6.4 KV transport** (NIXL transport/chunk/streams, UCX `tls/rndv/rc_qp`, LMCache `local_cpu_size/remote_url/eviction`, Mooncake) |
| Search axis | [`b6_lowlevel.yaml`](../search-spaces/b6_lowlevel.yaml) — **70+ axis**. 자세한 카탈로그 + 측정 도구 + 출처 매핑은 [`lowlevel_axis_catalog.md`](lowlevel_axis_catalog.md) |
| Strategy | TPE + ANOVA + RandomForest importance + Sobol total-order 가 70+ axis → 효과 큰 ~10 axis 자동 축소. 매 trial 직전 `b200/scripts/system_snapshot.sh` 가 PCIe/IOMMU/NUMA/NCCL 상태 capture → `b200/studies/<study>/system_snapshots/<trial>.json` (output F) |
| Acceptance | (1) B1 best 대비 system axis 만 추가했을 때 score ≥ 3% 개선 또는 변동 ≤ 1% 안정성, (2) **RDMA Perftest 363+ Gbps 재현** ([`rdma_perftest_baseline.md`](rdma_perftest_baseline.md), NHN Cloud B200 reference), (3) Sobol total-order ranking 자동 출력, (4) `system_snapshot.json` 이 모든 trial 에 archive |
| 산출물 | `b200/scripts/{system_snapshot.sh, rdma_bench.sh}`, `b200/docs/lowlevel_axis_catalog.md`, `b200/docs/rdma_perftest_baseline.md`, `src/lmtune/runners/system_capture.py` (trial pre-hook) |

### 3.8 Phase B7 — Multi-engine + Multi-stack [Track B-IV, scope 축소: SGLang + TRT-LLM 만]

| 항목 | 내용 |
|:---|:---|
| Goal | 본 프로젝트를 vLLM + llm-d 에 가두지 않고 **전체 LLM inference & serving stack 의 generic autotune 도구**로 일반화. **사용자 명시: "전체 LLM inference & serving stack 에서 활용될 수 있는 코드"** |
| 추상 | 신규 `EngineBackend` ABC (`src/lmtune/runners/base.py`) + `ServingStack` ABC (`src/lmtune/deploy/base.py`). 기존 vLLM runner / LLMDK8sAdapter = 첫 구현체 |
| Scope (축소) | **vLLM + SGLang + TensorRT-LLM** 3 engine 만. NIM/KServe/Ray Serve/Triton/Mooncake 어댑터는 deferral |
| 외부 벤치마크 통합 (축소) | `b200/profiles/burstgpt_replay.yaml` (BurstGPT trace) + `b200/profiles/mlperf_*.yaml` (MLPerf v5.x scenario) 만. ServeGen/Mooncake/Azure trace 는 deferral |
| Search axis | [`b7_multistack.yaml`](../search-spaces/b7_multistack.yaml) — `engine_backend × serving_stack × model × workload` 4D, `active_if` 게이팅 |
| Acceptance | (1) `engine=sglang × stack=llm-d` 1 trial 완주, (2) `engine=trt-llm × stack=raw-vllm-style` 1 trial 완주, (3) `engine=vllm × workload=mlperf-server` 표준 reporting, (4) `engine=vllm × workload=burstgpt-replay` 1K conv/min |
| 산출물 | `src/lmtune/runners/{sglang,trt_llm}.py`, `b200/profiles/{burstgpt_replay,mlperf_*}.yaml`, `b200/docs/portability_guide.md` |

### 3.9 Phase B8 — Public packaging [Track B-IV]

| 항목 | 내용 |
|:---|:---|
| Goal | B0~B7 을 **외부 GitHub 사용자가 README 만 보고 자기 환경에서 B0 smoke 재현** 가능한 형태로. 사용자 커리어 산출물 + Rebellions 외부 노출 가능 reference |
| 산출물 | `b200/RECIPES.md` (모델·HW·workload 별 winning recipe), `b200/docs/blog/` (블로그 초고 2-3편), `b200/docs/conference_submission_skeleton.md` (MLSys / ASPLOS / OSDI workshop figure·table), `b200/docs/upstream_pr_candidates.md` (vLLM/llm-d default 추천값 변경 정당화 데이터) |
| Acceptance | (1) 외부 GitHub 사용자가 b200/README.md 만으로 B0 smoke 까지 재현, (2) 블로그 1편이 사내·외부 검토 통과, (3) 모든 winning config 가 archive DB 에서 재추출 (재현성 입증) |

## 4. 모델 Roster

| 모델 | 크기 | 특이사항 | 등장 phase | helmfile values |
|:---|:---|:---|:---|:---|
| Llama-3.1-8B-Instruct | 8B dense bf16 | smoke (1 GPU) | B0 | `inference-scheduling/values-llama-3.1-8b-smoke.yaml` |
| Llama-3.1-70B-Instruct | 70B dense | TP=4, P/D 정본 | B1 (P/D), B4 | `pd-disaggregation/values-llama-3.1-70b-tp4-pd.yaml` |
| Qwen2.5-72B-Instruct | 72B dense | NHN reference: TP=8/PP=2 64-conc 2,224 tok/s | B1, B2, B6 | `inference-scheduling/values-qwen2.5-72b.yaml` (TBD B1) |
| Qwen3-235B-A22B | 235B dense | B200 16-GPU sweet spot (TP=2 × DP=4) | B1, B3, B4 | `inference-scheduling/values-qwen3-235b-tp2-dp4.yaml` (TBD B1) |
| Mixtral-8x22B-Instruct | 141B MoE | wide-EP 정본 (DP=2 EP=8) | B1 (wide-EP), B4 | `wide-ep-lws/values-mixtral-8x22b-dp2-ep8.yaml` |
| DeepSeek-V3 / V3.2 | 671B MoE | wide-EP standard vs DP=8 EP, MLA | B1, B3, B4 | TBD (B-I) |
| Qwen3-Coder-480B | 480B MoE | DP=8 EP single-node 후보 | B-IV (B1) | TBD |
| GLM-4.7-355B | 355B + MTP | speculative decoding axis 활성 | B4, B5 | TBD |
| MiniMax-M1 | Lightning Attention | PCP/DCP axis 의미 | B3 | TBD |

> **B0 단계 = Llama-3.1-8B 만**. 다른 모델은 B1 진입 시 helmfile values 추가.

## 5. Workload Roster

| Slug | input/output | 용도 | 등장 phase |
|:---|:---|:---|:---|
| `autotune-short` | 256 / 128 | chat-like, smoke 기본값 | 모든 phase |
| `autotune-medium` | 1024 / 256 | 코딩 에이전트 1턴 | 모든 phase |
| `autotune-long` | 4096 / 512 | 긴 컨텍스트 | B1 이후 |
| `ultra_long` (TBD) | 32K context | MoE / Lightning Attention 검증 | B3 |
| `coding_agent_burstgpt` (TBD) | trace replay | BurstGPT diurnal | B5 |
| `mlperf_*` (TBD) | MLPerf v5.x scenario | Server / Offline | B7 |

> 본 프로젝트가 직접 implement 하는 workload 는 lmtune E1~E6 의 SyntheticWorkload + DatasetWorkload + TraceWorkload union. trace replay 는 `src/lmtune/workloads/traces.py` 에서 처리 (B5/B7 산출).

## 6. User Contract — 실험을 정의하는 4 layer YAML

```
[Macro axes] (~10 dim — vllm-config-puzzle 정본)
  parallelism : tp, pp, dp, ep, ep_strategy, pcp, dcp
  network     : intraNode_type, crossNode_type
  model       : model_id (registry 가 numHeads/MoE/MLA/layers 자동 expand)
  path        : well_lit_path
  dtype       : engine_dtype, kv_cache_dtype
       ↓
[Feasibility Filter] (10 validation constraint, vllm-config-puzzle validation.ts 1:1)
       ↓
[Env Profile Template] (자동 binding, 사용자 입력 X)
  90% env_locked   : NCCL_IB_HCA, NCCL_NET_GDR_LEVEL, UCX_TLS, ...
  10% env_tunable  : 3-5 micro fine-tune knob
       ↓
[Micro axes] (profile 별 3-5 dim, sampler 가 fine-tune)
       ↓
[Surrogate Predictor] (Idea 7) — analytical formula (B3) + black-box residual (XGBoost)
  confidence ≥ 0.9 + score ≪ best → skip 실측 (negative example archive)
```

**한 줄 요청 예시**:

```bash
lmtune search start \
  --endpoint b200/endpoints/qwen3-235b-pd.yaml \
  --space b200/search-spaces/b3_parallelism.yaml \
  --objectives "throughput_tok.avg:short:maximize,ttft.p99:short:minimize" \
  --slo ttft_p99=500ms,e2e_p99=30s \
  --strategy nsga2 --backend k8s-job --workers 4 --max-trials 40 \
  --name B3-qwen3-235b-pareto
```

## 7. Risk Register

| Risk | Phase | Mitigation |
|:---|:---|:---|
| llm-d-cuda v0.6.0 + B200 sm_100 Triton JIT 실패 | B0 | v0.7.0-rc.2 로 image bump (`e90fb16`). enforce-eager 같은 성능 우회는 baseline 오염 → 안 씀 |
| chart 가 `runtimeClassName` 미노출 | B0 | helm post-renderer 가 모든 Deployment 에 inject (`b200/helmfile/_postrender/postrender.sh`) |
| `gateway.provider=istio` default → k3s Telemetry CRD 부재로 fail | B0 | helmfile 에 `provider: agentgateway` 명시 + `setup_gateway_provider.sh` 로 prereq 자동 |
| B200 image 풀 대형 (수십 GB) | B0~B1 | image registry 캐시 (peer repo 빌드 결과 또는 in-cluster registry) |
| B200 sm_100 + NCCL ≥ 2.23 호환 이슈 | B0 | image digest 고정, B0 fabric probe 가 NCCL ≥ 2.23 + CUDA ≥ 12.6 검증 |
| `kv_cache_dtype=fp4` Blackwell 미지원 가능성 | B2 | first-trial probe 실패 시 study 단위 freeze (S2 pruner 재사용) |
| 결과 누적으로 DuckDB 비대 | B5 | weekly archive job: 30일 이전 trial 은 parquet 으로 이관 |
| RDMA fabric 변경/장애 | B-track 전반 | B0 probe 를 nightly cron 으로. baseline 보다 5% 이상 저하 시 cross-node trial 자동 중단 |
| 70+ low-level axis 가 search space blow-up | B6 | TPE + ANOVA + RandomForest importance + Sobol 가 ~10 으로 자동 축소 |
| autoresearch loop 가 GPU 점유 충돌 | B5 | `gpu_lease.py` (S3 산출물) 확장 — loop 가 다른 사용자 작업 감지 시 자동 일시정지 |

## 8. Tracking Sheet (사용자가 직접 채움)

| Phase | 상태 | 시작일 | 완료일 | study_id | top-1 score | 비고 |
|:---|:---|:---|:---|:---|:---|:---|
| B0 — smoke | 🚧 in-progress | 2026-05-03 | — | — | — | v0.7.0-rc.2 image 검증 중 |
| B1 — well-lit baseline (3 path) | ⏳ pending | — | — | — | — | — |
| B2 — vLLM 신기능 axis | ⏳ pending | — | — | — | — | — |
| B3 — parallelism 10-axis | ⏳ pending | — | — | — | — | — |
| B4 — well-lit path as axis | ⏳ pending | — | — | — | — | — |
| B5 — continuous loop | ⏳ pending | — | — | — | — | — |
| B6 — low-level axis | ⏳ pending | — | — | — | — | — |
| B7 — multi-engine | ⏳ pending | — | — | — | — | — |
| B8 — public packaging | ⏳ pending | — | — | — | — | — |

## 9. 외부 Reference

- llm-d well-lit paths: <https://github.com/llm-d/llm-d/tree/main/guides>
- vllm-config-puzzle simulator (B3 정본): `/home/jinmoo/new-idea/vllm-config-puzzle/src/engine/llm-dist-sim/`
- peer helmfile templates: `/home/jinmoo/ml_ai/agentic/llm-distributed-inference`
- InferenceX (B5/B8 dashboard schema reference): <https://inferencex.com>, <https://github.com/SemiAnalysisAI/InferenceX>
- ariadne (B6 system topology, [lowlevel] extra): topology 수집·시뮬
- MLPerf Inference v5.x B200 submissions: <https://github.com/mlcommons/inference>
- llm-d-benchmark: <https://github.com/llm-d/llm-d-benchmark>
- cfregly book companion (B6/B7 reference): <https://github.com/cfregly/ai-performance-engineering>

## 10. 본 문서 갱신 절차

- 각 phase 종료 시 §3 의 해당 phase 의 "산출물" 경로가 실제로 생겼는지 확인 후 §8 tracking sheet 갱신
- 새로운 risk 발견 시 §7 에 1행 append (한번 발생한 함정도 보존 — B0 의 v0.6.0 → v0.7.0-rc.2 같은 사례)
- 모델 추가 시 §4 에 1행 append + helmfile values 경로 명시
- workload 추가 시 §5 에 1행 append + 등장 phase 명시
- phase 별 commit message 의 첫 줄을 §3 의 phase 제목과 동기화 (예: `B1: well-lit path baseline 7-path catalog`)
