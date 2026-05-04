# lmtune Architecture — 5 Layer Reference

> **목적**: 본 프로젝트가 "vLLM + llm-d 가 주어진 환경에서 어떻게 구성돼야 하는지 계속 탐색·튜닝하는 시스템" 으로 발전하기 위한 **architectural north star**. 이후 모든 PR 은 본 문서의 layer 와 extension point 를 reference 로 작성된다.
>
> 2026-05-05 사용자 요청 (5-layer breakdown) 의 영속화. plan 의 § Autoresearch Architecture (Macro × Profile × Micro) 와 InferenceX/vllm-config-puzzle 참조 매핑 위에 더 명시적인 책임 분리.

## TL;DR — 5 layer × 1 책임

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1  Inputs        — 무엇을 튜닝하는가                          │
│   (Model × vLLM features × Parallelism × well-lit-path × 조합)      │
└────────────┬────────────────────────────────────────────────────────┘
             │ "이 axis 조합이 실험할 가치 있나?"
┌────────────▼────────────────────────────────────────────────────────┐
│ Layer 2  Environment   — 무엇이 그 튜닝을 제약하는가                │
│   (HW capability × interconnect × virtualization × OS/firmware)     │
└────────────┬────────────────────────────────────────────────────────┘
             │ "환경이 axis 조합을 허용하나?"
┌────────────▼────────────────────────────────────────────────────────┐
│ Layer 3  Measurement   — 무엇을 측정·저장·시각화하는가              │
│   (Runners × Metric Registry × Storage × Analysis × Visualization)  │
└────────────┬────────────────────────────────────────────────────────┘
             │ "관측치를 학습 가능한 형태로"
┌────────────▼────────────────────────────────────────────────────────┐
│ Layer 4  Controller    — 다음에 무엇을 시도할지 결정                │
│   (Sampler × Pruner × Feasibility × Warm-start × Importance)        │
└────────────┬────────────────────────────────────────────────────────┘
             │ "이 params 로 가서 측정하라"
┌────────────▼────────────────────────────────────────────────────────┐
│ Layer 5  Launcher      — 결정을 vLLM/llm-d 에 적용·실행             │
│   (DeploymentAdapter × EngineBackend × ServingStack × Health)       │
└─────────────────────────────────────────────────────────────────────┘
```

각 layer 는 **1 ABC + N plug-in** 패턴을 따른다. 새 모델·기능·하드웨어·도구 추가 시 plug-in 1개만 작성하면 다른 layer 영향 없음.

---

## Layer 1 — Inputs (튜닝 대상)

### 책임
사용자가 declarative YAML 로 "**무엇을 튜닝하고 싶은지**" 표현. SearchSpace × Endpoint × Profile × Model 4 source.

### 현재 자산

| 산출물 | 위치 | 역할 |
|:---|:---|:---|
| **SearchSpace** | `src/lmtune/search/space.py` + `b200/search-spaces/*.yaml` | axis 카탈로그 (categorical/int/float/bool + active_if conditional + feasibility_constraints) |
| **Endpoint** | `src/lmtune/endpoints.py` + `b200/endpoints/*.yaml` | URL + model + deployment (engine_args + parallelism) + adapter |
| **Profile** | `src/lmtune/profiles.py` + `configs/profiles/autotune/*.yaml` | workload + arrival pattern + SLO + analysis 지시 |
| **Model Registry** | `src/lmtune/models/registry.py` | 모델 메타 (params/layers/heads/MoE/MLA/dtype). vllm-config-puzzle `models.ts` 1:1 port |
| **Profile binder** | `src/lmtune/search/profile_binder.py` + `configs/autoresearch/env_profiles/` | macro tuple → env_locked + env_tunable 자동 binding |

### Extension points

| 추가하고 싶은 것 | 어디에 1줄 추가 |
|:---|:---|
| 새 vLLM axis | `b200/search-spaces/b{N}_*.yaml` 에 `axis: {type, values}` |
| 새 vLLM 버전 | 신규 search-space yaml `b2_vllm_engine_v0XYZ.yaml` (deprecated flag 제거) |
| 새 모델 | `src/lmtune/models/registry.py::_RAW` 에 한 줄 + endpoint yaml 1개 |
| 새 well-lit-path | `b200/helmfile/{path}/` + `b4_welllit_paths.yaml` 의 categorical values |
| 새 workload | `configs/profiles/research/*.yaml` (E6 preset 패턴) |

### Gap (미구현)

- ❌ **vLLM version-specific axis whitelist 검증** — `b2_vllm_engine_v0171.yaml` 이 axis 명시했지만 vllm 0.18 로 image 바뀌면 reject. → SearchSpace 의 `vllm_version: ">=0.17,<0.18"` 필드 + pre-flight check 필요
- ❌ **Model-specific axis constraint** — gpt-oss-120b 의 `kv_cache_dtype ∈ {fp8, fp8_e4m3}` 가 b2 yaml 에 hardcode 됨. 모델 바뀌면 다시 yaml 수정. → registry 의 model entry 가 `axis_constraints: {kv_cache_dtype: [fp8, fp8_e4m3]}` 들고 있어야 함

---

## Layer 2 — Environment (제약 조건)

### 책임
"이 axis 조합이 **현재 HW/인프라에서 실행 가능한가**" 의 사실 데이터. 사용자가 직접 입력 X — system probe 가 자동 캡처.

### 현재 자산

| 산출물 | 위치 | 역할 |
|:---|:---|:---|
| `Environment` dataclass | `src/lmtune/search/feasibility.py` | total_npus / npus_per_server / vram / intra_node_type / cross_node_type. 3 factory (b200_dual_node / b200_single_node / local_single_gpu) |
| **Probes** | `b200/scripts/{probe,fabric_probe,rdma_bench,system_snapshot}.sh` | 클러스터·fabric·PCIe/IOMMU/NUMA capture |
| **Capture hook** (B6) | `src/lmtune/runners/system_capture.py` (스텁 — 미구현) | 매 trial 직전 system_snapshot.json 적재 |

### Capability 카탈로그

본 layer 가 모델링해야 할 environment dimension (사용자 명시 + plan § B6):

```yaml
processing_unit:
  gpu_arch: [sm_75, sm_80, sm_90, sm_100, sm_120]   # B200 = sm_100
  gpu_vram_gb: 192                                    # HBM3e
  gpu_count_per_node: 8
  firmware: { driver_ver, cuda_ver, nccl_ver }
interconnect:
  intra_node:
    type: [nvlink, nvlink5, pcie4, pcie5, ucie, xgmi]
    nvswitch_present: true
    nvls_capable: true                                 # NCCL 2.23+ + NVSwitch5
  cross_node:
    type: [ib, ib_ndr, roce_v1, roce_v2, ethernet, none]
    bandwidth_gbps_peak: 400
    rail_aligned: true                                 # rail per HCA
  fabric_extras:
    sharp_capable: true                                # IB SHARP
    gdr_capable: true                                  # nv_peer_mem 또는 nvidia_peermem
    gds_capable: true                                  # cuFile / NVMe direct
virtualization:
  iommu_mode: [pt, strict]
  sriov_enabled: true
  vfio_passthrough: false
  hugepages_1gi: 16
os:
  cpu_governor: performance
  numa_balancing: 0
  transparent_hugepages: madvise
```

### Extension points

| 추가하고 싶은 것 | 어디 |
|:---|:---|
| 새 HW (예: H200, GB200, MI355X) | `Environment.factory()` 에 classmethod 1개 + capability detection 룰 |
| 새 fabric type (예: NVLink C2C) | `intra_node_type` enum + comm_overhead 모델 |
| 새 OS-level axis (예: io_scheduler) | `b6_lowlevel.yaml` 의 categorical |

### Gap

- ❌ **자동 detection** — 현재 `Environment` 는 hand-coded factory. `probe.sh` 결과를 읽어 `Environment.from_probe(json_path)` 가 자동 빌드해야 함
- ❌ **Capability gating in feasibility** — 예: `nccl_nvls_enable=1` 이 sm_90 노드에서 noop 인 케이스 자동 freeze 미구현
- ❌ **Virtualization axis 미모델링** — IOMMU/SR-IOV/VFIO 가 환경 제약으로 안 들어옴

---

## Layer 1 ↔ 2 결합 — Dependency Graph (직교 cartesian 회피)

> **사용자 통찰 (2026-05-05)**: "input 이랑 environment 는 서로 의존관계 및 영향도가 꽤 크니, 단순히 매트릭스로 관리하면 낭비가 조금 심해질 수도 있을 것 같아!"
>
> 정확합니다. 70+ axis × 10 environment dim 을 cartesian 으로 sample 하면 **>95% combo 가 invalid**. 본 프로젝트는 그 dependency 를 **4-stage 압축**으로 처리하고 sampler 가 valid region 만 본다.

### 4 Stage Reduction Pipeline

```
Stage 0 (raw)          : 70 axis × 10 env dim ≈ 5^70 × 5^10 ≈ 10^56 조합
                                ↓
Stage 1 (capability)   : env capability gate     — Layer 2 가 axis 자체를 끔
                                ↓
Stage 2 (model)        : model axis constraint   — Layer 1 model 이 values 부분집합만 허용
                                ↓
Stage 3 (active_if)    : conditional axis        — parent axis 가 child 활성화 결정
                                ↓
Stage 4 (cross-axis)   : feasibility_constraints — sample 후 expression 검증
                                ↓
Effective: ~12 axis    : sampler 가 보는 실제 차원 (모델·환경·path tuple 별로 다름)
```

각 stage 의 책임 + 위치 + 예시:

| Stage | 입력 | 출력 | 책임 위치 | 예시 |
|:---:|:---|:---|:---|:---|
| **1** | `Environment` | active axis set | `feasibility.py::_capability_filter()` (TODO) | `nccl_nvls_enable` axis 자체가 사라짐 (env.intra_node ≠ nvlink-with-NVSwitch5 또는 NCCL < 2.23) |
| **2** | `Model` (registry) | axis values 부분집합 | `models/registry.py::axis_constraints` (TODO) | gpt-oss-120b 의 `kv_cache_dtype` values 가 `[fp8, fp8_e4m3]` 로 자동 축소 |
| **3** | parent axis 값 | child axis 비활성 | `space.py::active_if` (✅ 구현됨) | `prefix_caching_hash_algo` 가 `enable_prefix_caching=true` 일 때만 활성 |
| **4** | sampled params dict | feasibility bool | `feasibility.py::evaluate()` (✅ 구현됨, hook 미장착) | `tp * pp * dp <= total_npus` |

### Stage 1+2 가 미구현 (사용자 질문의 핵심)

현재는 **Stage 3 + Stage 4 만** 동작:
- Stage 3 (`active_if`) — search-space yaml 안에서만, env/model 모름
- Stage 4 (`feasibility_constraints`) — 모듈 있지만 Study.ask() hook 미장착

이게 reactive crash whack-a-mole 의 근원. Stage 1+2 를 추가하면:
- 새 모델 추가 = registry yaml 1줄 (`axis_constraints` 필드)
- 새 환경 = Environment 객체 1개 (`from_probe` 자동 빌드)
- 새 axis 추가해도 호환 안 되는 모델·환경에선 자동으로 search-space 에서 제거

### Dependency 표현 — 데이터 모델

```yaml
# src/lmtune/models/{gpt-oss-120b}.yaml (TODO Sprint 1 PR-B)
model_id: openai/gpt-oss-120b
quantization: mxfp4-native
total_params_b: 120
# ... existing simulator fields ...
axis_constraints:
  kv_cache_dtype: [fp8, fp8_e4m3]      # attention.py:408 hardcoded assert
  cpu_offload_gb: [0]                  # vllm#18298 — V1 engine 비호환
  dtype: [auto, bfloat16]              # MXFP4 native
capability_requirements:                # Layer 2 dependency
  fabric_min:
    intra_node: nvlink                 # 단일 노드 TP=8 가능
  vllm_min: "0.10.0"                   # MXFP4 지원 시점
```

```yaml
# b200/search-spaces/b6_interconnect_tier1.yaml — capability gate 추가
axes:
  nccl_nvls_enable:
    type: categorical
    values: ["0", "1"]
    capability_required:                # Stage 1 — env 가 강제로 활성/비활성
      intra_node: nvlink-nvswitch5
      nccl_min: "2.23"
```

→ Sampler 는 axis 의 active set 만 본다. 매번 새 (model, env, path) tuple 진입 시 4 stage 가 재계산 → effective axis ~12.

### 효과 정량

| 측정 | Stage 1+2 미적용 (현재) | Stage 1+2 적용 (Sprint 1 후) | 압축 |
|:---|:---:|:---:|:---:|
| 30 trial 중 infeasible 비율 | ~67% (gpt-oss-120b sweep 기준) | ~5% (cross-axis constraint 만) | 13× |
| sampler 가 학습할 dim | 70+ flat | (model, env) 별 12-15 | 5× 압축 |
| TPE 수렴 속도 | 30 trial → mediocre | 30 trial → near-optimal | ~3× |

---

## Layer 3 — Measurement (관측·저장·분석·시각화)

### 책임
trial 실행 결과를 **표준화된 schema** 로 저장하고 (도구 차이 흡수), 분석·시각화·detector 가 그 위에서 도구 무관하게 동작.

### 현재 자산

| 산출물 | 위치 | 역할 |
|:---|:---|:---|
| **Runners** (4종) | `src/lmtune/runners/{aiperf,guidellm,vllm_bench,raw_openai}.py` | benchmark 도구 subprocess + 결과 정규화 |
| **Storage** | `src/lmtune/storage/{schema.sql,duckdb_store.py,writer_queue.py}` | DuckDB single-writer + parquet export |
| **Metric Registry** | `src/lmtune/analysis/registry.py` | MetricDef 카탈로그 (unit/direction/category/aggs) |
| **Derived metrics** | `src/lmtune/analysis/derived.py` | tokens_per_usd, eutb, prefix_hit_rate, variance_cv … |
| **Aggregate / Distribution / NWay** | `src/lmtune/analysis/{aggregate,distributions,nway,prom_analysis}.py` | DataFrame 반환 |
| **Plot Registry** | `src/lmtune/visualization/plots/__init__.py` (decorator) | `@register_plot("ttft_vs_turn")` |
| **Sinks** | `src/lmtune/visualization/sinks.py` | md / html / csv / parquet / jupyter |
| **Templates** | `src/lmtune/visualization/templates/*.j2` | Jinja2 — run/variance/nway/preset report |
| **Dashboard** | `src/lmtune/visualization/dashboard/*` + `cli_dashboard.py` | InferenceX-app 호환 정적 HTML |

### Schema 표준 (도구 무관 정규화)

```
runs       — run_id (ULID), profile_yaml, endpoint_meta, tool_versions, started_at
metrics    — run_id, metric, label, value           # ttft.p99 / e2e.p50 / throughput.avg
requests   — run_id, request_id, ttft_ms, e2e_ms, input_tokens, output_tokens,
             cached_tokens, thinking_tokens, tool_call_count, phase, role, energy, cost
sessions   — run_id, session_id, task_id, total_tokens, turn_count, success, cost
trial*     — trial_id, study_id, params, status, score (S1+)
```

→ aiperf JSON, guidellm CSV, vllm bench JSON 모두 **같은 컬럼**으로 적재. 분석 도구는 source 도구 모름.

### Extension points

| 추가하고 싶은 것 | 어디 |
|:---|:---|
| 새 benchmark 도구 (예: bench-llm) | `src/lmtune/runners/{name}.py` — RunnerBase 구현 + parser |
| 새 metric | `analysis/registry.py` 의 카탈로그 + (optional) `derived.py` 의 formula |
| 새 plot | `@register_plot("name")` decorator |
| 새 sink (예: Slack post) | `sinks.py` 에 함수 추가 |
| 새 dashboard view | `templates/{view}.html.j2` + Jinja2 데이터 spec |

### Gap

- ❌ **InferenceX-app schema 호환 dump** — plan 에 정의됐지만 실제 JSON dump 미구현
- ❌ **Grafana JSON dashboard** (output G' in plan) — 미구현
- ❌ **Trajectory events 테이블** (E1 의 optional) — 미구현. ReAct 류 agent path-level 분석 필요 시 합류

---

## Layer 4 — Controller (탐색 결정)

### 책임
관측 데이터 + feedback 으로 search space 를 **점진적으로 축소** + 다음 trial 의 params 결정. **본 프로젝트의 차별화 핵심**.

### 현재 자산

| 산출물 | 위치 | 역할 |
|:---|:---|:---|
| **Sampler** (8종) | `src/lmtune/search/samplers/{grid,random,lhc,tpe,cma_es,nsga2,ucb_bandit,*_native}.py` | Optuna 위임 + 자체 native (수학 가시화) |
| **Pruner** | `search/pruners/{successive_halving,hyperband}.py` | Optuna 위임 |
| **Feasibility** | `src/lmtune/search/feasibility.py` | Constraint AST eval + Environment + Model. **핵심 gating** |
| **Profile binder** | `search/profile_binder.py` | macro → env_locked + env_tunable |
| **Warm-start** | `search/warmstart.py` | archive DB → Optuna `enqueue_trial()` |
| **LLM prior** | `search/llm_prior.py` | hand-curated `axis_priors.yaml` reader (LLM-free) |
| **Cost-aware sampler** | `search/sampler_cost_aware.py` | tier 별 비용 가중 |
| **Analysis** | `search/analysis/{anova,importance,bound_tighten}.py` | freeze/drop/shrink 권고 |
| **Objective** | `search/objective.py` + `objective_pareto.py` | bench_score.py 래핑 + N=3 + CV gate, multi-obj |
| **Failure handler** | `src/lmtune/orchestrate/failure_handler.py` | crash 분류 + circuit breaker (PR #14/#15/#17) |

### Decision flow (정상 trial)

```
1. study.ask()
     → sampler.suggest(active_axes)   ← active_if 로 conditional axis 게이팅
     → params dict
2. is_feasible(params, env, model, constraints)?     ← Layer 2 + Layer 1 결합
     │ no  → study.tell(PRUNED, error="infeasible: c5_ep_divisible")  → 다음 ask
     │ yes
     ▼
3. profile_binder.bind(params, path) → env_locked + env_tunable
4. backend.submit(trial)             ← Layer 5
5. (poll) → TrialResult
6. classify_outcome(status, error, notes)             ← failure_handler
   breaker.record(outcome); halt? → break loop
7. study.tell(trial, result)         ← Optuna + DuckDB
8. (study 종료 시) prune.run()
   → ANOVA + RandomForest importance + bound_tighten
   → 권고 JSON + (옵션) search-space yaml 자동 수정
```

### Extension points

| 추가하고 싶은 것 | 어디 |
|:---|:---|
| 새 sampler | `samplers/{name}.py` + `make_sampler()` 분기 1줄 |
| 새 pruner | `pruners/{name}.py` + `make_pruner()` |
| 새 constraint family | `feasibility.py` 의 _surrogate_namespace 확장 또는 새 root namespace |
| 새 importance metric | `analysis/{name}.py` + `bench search prune` 의 reporter 등록 |

### Gap

- ❌ **Study.ask() 안에 is_feasible() hook 미장착** — 모듈은 있지만 ask() 가 호출 안 함. **다음 PR 의 핵심**
- ❌ **Pre-flight validation gate** — `lmtune search start` 진입 시 (search-space × endpoint × model × env) 충돌 검증 미구현
- ❌ **outcome → search-space 자동 rewrite** — INFEASIBLE 누적되면 `axis values` 에서 자동 제거 (반자동 with `--apply`)

### Pluggability — Controller 갈아끼우기 (사용자 질문 2026-05-05)

> "Controller 부분은 꼭 내 구현을 쓰지 않더라도, 다른 LLM API 를 부른다거나 AI agent API 를 부를 수도 있었으면 좋겠어!"

**현재 상태 솔직 평가**:
- ❌ **In-process plug-in 불가** — `Study` 가 `optuna.create_study()` 직접 호출, sampler 는 Optuna `BaseSampler` 만 받음
- ✅ **CLI seam 유일 가능** — `lmtune search ask/tell` (S6, autoresearch.sh 가 활용)
- ⚠️ 외부 controller (LLM/agent) 가 in-process 통합 불가 — 별도 프로세스로 CLI 호출만

**제안 — Controller ABC 도입 (Sprint 1 후속 PR)**:

```python
# src/lmtune/search/controller.py (TODO)
class Controller(ABC):
    """단일 책임: (active_axes, history) → next params dict.

    Study 는 persistence + breaker + profile_binder 를 담당하고,
    'next params' 결정만 Controller 에 위임 — Optuna 에 락인 해제.
    """
    @abstractmethod
    def ask(self, active_axes: list[Axis], history: list[Trial]) -> dict[str, Any]: ...

    @abstractmethod
    def tell(self, params: dict, score: float | None,
             status: str, metadata: dict | None = None) -> None: ...

    @property
    def name(self) -> str: ...
```

**3 reference 구현으로 진짜 pluggability 입증**:

| 구현체 | 위치 | 용도 |
|:---|:---|:---|
| `OptunaController` | `controller/optuna.py` | 현재 sampler 8종 (TPE/CMA-ES/NSGA-II/...) 그대로 wrap. 기본값 |
| `RandomController` | `controller/random.py` | Optuna 의존성 0, pure Python. 검증/baseline 용 |
| `HTTPController` | `controller/http.py` | URL 에 POST, **외부 LLM/agent API 통합 base** |

**HTTPController 가 가능하게 하는 시나리오**:

```bash
# 시나리오 A: 외부 Anthropic Claude controller
$ python my_claude_controller.py --listen :8080 &
$ lmtune search start ... --controller http --controller-url http://localhost:8080
# my_claude_controller.py 가 Anthropic SDK 로 axis 추론, 우리는 측정만

# 시나리오 B: 외부 OpenAI/Gemini/local LLM controller
$ python my_gpt_controller.py ... &
$ lmtune search start ... --controller http --controller-url ...

# 시나리오 C: 자체 RL/MAB agent
$ python my_rl_controller.py ... &  # 내부 state 자체 관리
$ lmtune search start ... --controller http ...

# 시나리오 D: autoresearch (현재 S6 CLI seam 유지) + in-process LLM controller (신규)
$ lmtune search start ... --controller http --controller-url $AUTORESEARCH_URL
```

**HTTP API 계약** (controller 서비스 작성자용):

```http
POST /ask
Content-Type: application/json
{
  "study_id": "st-...",
  "active_axes": [{"name": "max_num_seqs", "kind": "categorical", "values": [...]}],
  "history": [{"params": {...}, "score": 142.5, "status": "completed"}, ...]
}
→ 200 OK
{"params": {"max_num_seqs": 64, ...}}

POST /tell
{
  "study_id": "st-...",
  "params": {...}, "score": 142.5, "status": "completed",
  "metadata": {"duration_s": 1340, "trial_id": "tr-..."}
}
→ 204 No Content
```

이 계약만 충족하면 어떤 언어·프레임워크의 controller 도 plug-in 됨 — Python (Anthropic SDK), TypeScript (LangGraph), Go (자체 RL), Rust 등.

### Layer 4 Pluggability Roadmap

| PR | 책임 | 분량 |
|:---:|:---|:---:|
| 1 | `Controller` ABC + `OptunaController` (기존 로직 wrap) — 기본 동작 동일 | 1일 |
| 2 | `RandomController` + tests — 진짜 plug-in 작동 입증 | 0.5일 |
| 3 | `HTTPController` + 계약 문서 + `examples/controllers/` reference 구현 | 1일 |
| 4 | CLI flag `--controller {optuna,random,http} --controller-url` | 0.5일 |

→ 3일 안에 user 의 "다른 LLM/agent API 부르기" 시나리오 충족.

---

## Layer 5 — Launcher (적용·실행)

### 책임
sampler 가 정한 params 를 vLLM/llm-d 에 **실제로 적용**해서 endpoint 를 ready 상태로 만들고, 안 되면 빠르게 fail.

### 현재 자산

| 산출물 | 위치 | 역할 |
|:---|:---|:---|
| **DeploymentAdapter** ABC | `src/lmtune/deploy/base.py` | `apply(spec) → HealthReport`, `teardown()` |
| **LocalVLLMAdapter** | `deploy/local_vllm.py` | `vllm_restart.sh` 래핑 (단일 GPU) |
| **LLMDK8sAdapter** | `deploy/llmd_k8s.py` | helmfile values overlay → `helmfile apply` → wait_rollout_smart |
| **Rollout watcher** | `deploy/rollout_watcher.py` | pod status 5s polling + classify_crash (60-120s fast-fail) |
| **Health probe** | `deploy/health.py` | `/v1/models` + 1-token warmup |
| **Backend** ABC | `src/lmtune/orchestrate/backend.py` | `submit/poll/cancel` |
| **K8sJobBackend** | `orchestrate/backend_k8s.py` | trial = K8s Job |
| **ProcessPoolBackend** | `orchestrate/backend_process_pool.py` | 로컬 dev |
| **GPU lease** | `orchestrate/gpu_lease.py` | 동시 사용 lock |

### Adapter ↔ Backend 직교

```
                   K8sJobBackend     ProcessPoolBackend
LocalVLLMAdapter   (드물게 사용)        ✓                 ← S1~S2 inline
LLMDK8sAdapter         ✓               (X — k8s 안에서 K8s X)
SGLangAdapter*         ✓               ✓                 ← B7 미래
TritonAdapter*         ✓               ✓                 ← B7 미래
```

→ Adapter 는 **무엇을 띄우는가**, Backend 는 **어디서 trial 들이 평행 실행되는가**.

### Extension points

| 추가하고 싶은 것 | 어디 |
|:---|:---|
| 새 serving 스택 (예: KServe/Ray Serve/Triton/NIM) | `deploy/{name}.py` — DeploymentAdapter 구현 |
| 새 engine backend (예: SGLang/TRT-LLM) | (B7) `runners/base.py::EngineBackend` ABC 추가 + 구현체 |
| 새 trial backend (예: SSH cluster) | `orchestrate/backend_{name}.py` — TrialBackend 구현 |
| 새 crash 패턴 | `rollout_watcher.CRASH_PATTERNS` 의 카테고리에 regex 추가 |

### Gap

- ❌ **EngineBackend 추상 미존재** — runners 가 vllm 명시 호출. SGLang/TRT-LLM 추가 시 runners/* 전부 분기
- ❌ **Image digest 고정 메커니즘 미구현** — peer repo helmfile 가 tag 만 박으면 매 study 마다 다른 binary
- ❌ **SchedulingStrategy=Recreate 강제 검증** — Deployment.spec.strategy 검증 없이 RollingUpdate 시 16/16 GPU stuck

---

## 5 layer × 현재 가용성 매트릭스

| Layer | ABC/Schema | 구현 자산 | Integration | 사용자 contract |
|:---|:---:|:---:|:---:|:---:|
| 1. Inputs | ✅ pydantic v1alpha1 | 80% (registry, search-space catalog 누적 중) | 60% (registry 가 axis 와 결합 안 됨) | ✅ YAML declarative |
| 2. Environment | ✅ Environment dataclass | 60% (probe 스크립트, factory 3종) | 30% (probe → Environment 자동 변환 X) | ⚠️ 수동 factory |
| 3. Measurement | ✅ DuckDB schema | 90% (4 runner, registry, plots, sinks, dashboard) | 80% (도구별 정규화) | ✅ DuckDB + Parquet + HTML |
| 4. Controller | ✅ Sampler/Pruner ABC | 85% (8 sampler, 2 pruner, feasibility, prune) | 50% (feasibility 가 ask() 에 hook 안 됨) | ✅ `search start/status/prune` |
| 5. Launcher | ✅ DeploymentAdapter / TrialBackend ABC | 70% (2 adapter, 2 backend, rollout_watcher) | 70% | ✅ `--adapter` / `--backend` flag |

---

## 다음 PR 로드맵 (탄탄한 계획)

사용자 요청 "변경하기 쉽게 잘 구조화" → 위 Gap 들을 **layering 무너뜨리지 않고** 메우는 PR 시퀀스.

### Sprint 1 — Layer 1 ↔ 2 ↔ 4 결합 (whack-a-mole 종식)

| PR | 책임 | 산출 | 효과 |
|:---:|:---|:---|:---|
| **#A** | Study.ask() 의 feasibility hook | `Study.ask()` → is_feasible() 호출, infeasible 시 자동 PRUNED + tell(error) | infeasible config 의 helmfile apply 0 — 22.5분 → 0초 |
| **#B** | Model registry × axis constraint 결합 | registry entry 에 `axis_constraints: {kv_cache_dtype: [fp8, ...]}` 필드. SearchSpace.active_axes(model=X) 가 모델 constraint 자동 적용 | gpt-oss-120b 의 kv_cache_dtype/cpu_offload_gb 가 hardcode 빠지고 registry 1줄로 |
| **#C** | Pre-flight validation gate | `lmtune search start` 진입 시 (space × endpoint × model × env) 충돌 검증 + 명확한 에러 | study 시작 전 incompat 발견 (11h 낭비 0) |

**예상 ROI**: gpt-oss-120b 다음 sweep 부터 reactive crash 분류 (PR #12, #15, #17) 가 **백업 안전망**으로 격하. 새 모델 추가 시 registry yaml 1개로 끝.

### Sprint 2 — Layer 2 자동화

| PR | 책임 | 산출 |
|:---:|:---|:---|
| **#D** | `Environment.from_probe()` | `b200/scripts/probe.sh` JSON → Environment 자동 빌드 |
| **#E** | Capability detection in feasibility | `intra_node_type=nvlink + nccl_nvls_enable=1` 정합성 검증 (sm_90 미만이면 freeze) |
| **#F** | system_capture trial hook | trial 직전 system_snapshot.json 적재 → DuckDB `system_snapshots` 테이블 |

### Sprint 3 — Layer 5 generalization (B7 prefigure)

| PR | 책임 | 산출 |
|:---:|:---|:---|
| **#G** | `EngineBackend` ABC | `runners/base.py` 에 추가, vLLM 첫 구현체 |
| **#H** | SGLangAdapter (drop-in) | `deploy/sglang.py` (LLMDK8sAdapter 패턴 포팅) |
| **#I** | Image digest pinning | LLMDK8sAdapter 의 values overlay 가 study 시작 시점의 digest 박음 |

### Sprint 4 — Layer 4 Controller Pluggability (사용자 요청)

| PR | 책임 | 산출 |
|:---:|:---|:---|
| **#J** | `Controller` ABC + `OptunaController` | `src/lmtune/search/controller/{base,optuna}.py` — Study 가 ABC 에 위임 |
| **#K** | `RandomController` (Optuna-free baseline) | plug-in 작동 입증, 단위 테스트 |
| **#L** | `HTTPController` + 계약 문서 + reference controller | 외부 LLM/agent API 통합 base — `examples/controllers/{claude,random}.py` |
| **#M** | CLI flag `--controller {optuna,random,http} --controller-url` | UX |

---

## Plug-in extension 체크리스트

새 기능 추가 시 본 체크리스트 따라 작성:

```
[ ] Layer 결정 (1/2/3/4/5)
[ ] 기존 ABC 어느 것을 구현하는가? (DeploymentAdapter / RunnerBase / Sampler / EngineBackend / etc)
[ ] 다른 layer 의 코드를 수정해야 하는가? → 만약 Yes 면 ABC 가 부족한 신호. ABC 먼저 보강
[ ] YAML 1개로 추가 가능한가? (예: 새 모델 = registry 한 줄, 새 axis = search-space 한 줄)
[ ] 단위 테스트 1개 (해당 layer 의 tests/{layer}/test_*.py)
[ ] 사용자 문서 업데이트 (README / 본 architecture.md / b200/docs/*)
```

→ "다른 layer 수정 필요" = ABC 가 부족한 신호. PR 분리 + ABC 먼저 보강.

---

## References

- 본 repo plan: `/home/jinmoo/.claude/plans/async-cooking-cat.md` § Autoresearch Architecture (Macro × Profile × Micro)
- vllm-config-puzzle simulator (validation/memory/perf 공식 정본): `/home/jinmoo/new-idea/vllm-config-puzzle/src/engine/llm-dist-sim/`
- InferenceX (continuous inference benchmark reference): https://github.com/SemiAnalysisAI/InferenceX
- llm-d well-lit-paths peer repo: `/home/jinmoo/ml_ai/agentic/llm-distributed-inference`
- 본 repo `docs/autotune_loop.md` — autotune cycle sequence diagram
- 본 repo `b200/docs/interconnect_optimization.md` — Layer 2 fabric 측면
- 본 repo `b200/docs/lowlevel_axis_catalog.md` — Layer 2 host-side axis
