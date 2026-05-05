# lmtune — LLM 추론 자동 튜닝 + 벤치마크 플랫폼

vLLM / llm-d / SGLang / TensorRT-LLM 으로 서빙되는 LLM 엔드포인트를 **자기 인프라에서** 자동 튜닝(autotune)하고, 결과를 자기 인프라에 다시 적용 가능한 **self-contained recipe** 로 산출합니다. autoresearch 통합으로 LLM 가이드 모드를 옵션 제공하나, 모든 핵심 경로는 **LLM 콜 0회로 동작**하는 것이 1st-class.

> 처음이라면 [§ 핵심 개념 — Study / Trial / Run](#핵심-개념--study--trial--run) 부터 읽으세요. 본 프로젝트의 작업 단위·실행 순서·자주 쓰는 명령이 거기 있습니다.

> **Phase W (Walkthrough MVP)** — 5분 안에 자기 환경에서 한 사이클(start → mid → visualize → final → cross-compare) 을 돌려보세요. 자세한 단계는 [§ Walkthrough](#walkthrough-5-steps).

## 핵심 개념 — Study / Trial / Run

본 프로젝트는 3 단계 hierarchy 로 작업을 조직합니다. 처음 사용자가 가장 헷갈리는 부분이라 먼저 박아둡니다.

```
Study (autotune 세션)            ← 사용자 1회 호출 = 1 study
  │  └ study_id = st-XXXX
  │  └ 산출: best params + Pareto + ANALYSIS.md + dashboard
  │
  └─ Trial (params 후보 1개)     ← sampler 가 결정한 1 config 시도
       │  └ trial_id = tr-XXXX
       │  └ 1 trial = 1 helmfile redeploy + N runs (재현성 게이트)
       │
       └─ Run (bench 도구 1회 측정)   ← aiperf/guidellm/vllm-bench 단발 실행
            └ run_id = ULID
            └ raw artifact = data/raw/<run_id>/
```

| 개념 | 정의 | 식별자 | 단위 시간 (B200 gpt-oss-120b 기준) |
|:---|:---|:---|:---:|
| **Study** | autotune 세션 1회 — `lmtune search start` 가 시작 | `st-` prefix | ~수 시간 (max-trials × trial 시간) |
| **Trial** | sampler 가 결정한 params 1조합의 시도 | `tr-` prefix | ~22.5 분 (helmfile + 측정) |
| **Run** | 단일 bench 도구 실행 — 1 (workload, repeat) | ULID | ~3-5 분 |

**1 trial = N runs** 인 이유: 재현성 게이트 (`bench_score.py` 가 N=3, CV ≥ 0.10 시 N=5 로 확장). 같은 (config, workload) 를 여러 번 측정해서 noise 격리.

### 실행 순서 — `lmtune search start` 호출 시

```
[t=0] 사용자: lmtune search start --strategy tpe --max-trials 30 ...
       │
       ▼
[t=0+ε]
   1. SearchSpace YAML + Endpoint YAML + Profile YAML × 3 + Model registry 로딩
   2. studies 테이블에 study row 1개 INSERT (study_id 발급)
   3. Optuna sampler (TPE) + DuckDB writer queue + GPU lease 초기화
   4. (옵션) Phase S6 외부 LLM controller 사용 시 — HTTP controller 와 handshake
       │
       ▼
[loop: trial = 1, 2, 3, ...]
   5. Controller.ask(active_axes) → params dict (예: max_num_seqs=128, ...)
   6. trials 테이블에 trial row INSERT (status='pending')
   7. (Sprint 1 후) is_feasible(params, env, model) 검증 → 실패 시 PRUNED + 다음 trial
       │
       ▼ (feasible 이면)
   8. DeploymentAdapter.apply(endpoint_path, params)
       ├─ LocalVLLMAdapter: scripts/vllm_restart.sh 로 vllm 재기동
       └─ LLMDK8sAdapter: helmfile values overlay 생성 → helmfile apply → wait_rollout_smart
            ↓ (rollout crash 시 — fast-fail 60-120s)
            classify_crash → tell(infeasible/oom/transient/...) → 다음 trial
       │
       ▼ (rollout 성공)
   9. probe(/v1/models) + warmup 1-token
       │
       ▼
   10. 3 workloads × N repeats:
        for workload in [short, medium, long]:
          for r in 1..N:
            Runner (aiperf/guidellm/vllm-bench).run(profile, endpoint)
              → runs row INSERT, raw artifact 저장
              → metrics 적재 (ttft.p50, e2e.p99, throughput.avg, ...)
        → CV 계산, ≥ 0.10 이면 N=5 확장 재측정
       │
       ▼
   11. composite score = throughput × penalty(ttft_p99)  (SLO 통과 시에만 > 0)
   12. trial row UPDATE (status='completed', score=X)
   13. Controller.tell(params, score, status)        ← sampler 학습
   14. CircuitBreaker.record(outcome) — 5 연속 infra 실패 시 study halt
       │
       ▼
[loop end: trial == max_trials 또는 budget-hours 도달 또는 halt]
   15. studies row UPDATE (status='completed' 또는 'halted', finished_at=NOW())
   16. (옵션) lmtune dashboard build → 정적 HTML 갱신
   17. (옵션) lmtune search export <study_id> --winner top-1 → winner/apply.sh 생성
```

### 자주 쓰는 명령

| 명령 | 무엇을 보여주나 |
|:---|:---|
| `lmtune ls` | **runs** (개별 측정) 목록 |
| `lmtune search ls` | **studies** (autotune 세션) 목록 — `study_id` 여기서 확인 |
| `lmtune search status <study_id>` | study 진행률 + top-K trial + (있으면) Pareto |
| `lmtune search trace <study_id>` | score over trial sequence — sampler 학습 곡선 |
| `lmtune search prune <study_id>` | ANOVA + RandomForest importance + bound-tighten 권고 |
| `lmtune search export <study_id> --winner top-1` | winner config + apply.sh 생성 |
| `lmtune search ask <study_id>` | (Phase S6) 외부 LLM agent 가 next params 받기 |
| `lmtune search tell <study_id> --trial X --metrics-json ...` | (Phase S6) 외부 agent 가 측정 결과 보고 |
| `lmtune dashboard build` | 정적 HTML 대시보드 갱신 (model × HW × engine 매트릭스 + Pareto) |
| `lmtune compare <run_id> <run_id>` | 2-way **run** 비교 (개별 측정) |
| `lmtune nway <run_ids...>` | N-way **run** 비교 |
| `lmtune variance <profile_slug>` | 같은 profile 의 N 회 반복 측정 분산 (CV/IQR/MAD) |

### Sample (raw 측정) ≠ Run

- **Sample** = 1 request 의 metric (ttft, e2e, …) — `requests` 테이블 row 1개
- **Run** = 1 bench 도구 실행 — 수십~수백 sample 의 집합 (`runs` 테이블 row 1개)
- 분석 도구는 sample 단위 percentile/CDF/histogram 까지 내려간다 (`lmtune analysis distributions`)

### 어디에 뭐가 저장되는가

| 데이터 | 위치 | 누가 씀 |
|:---|:---|:---|
| Study/Trial/Run/Metric meta | `data/db/lmtune.duckdb` | DuckDBWriterQueue (single-writer) |
| Raw bench artifact (aiperf JSON 등) | `data/raw/<run_id>/` | Runner |
| Trial parquet export | `b200/studies/<study_id>/{trials,metrics}.parquet` | `lmtune search export` |
| Winner config recipe | `b200/results/<study_id>/winner/{params.json, apply.sh, values-overlay.yaml, README.md}` | `lmtune search export --winner top-1` |
| Plot 아티팩트 | `b200/studies/<study_id>/plots/` | visualization |
| 정적 HTML dashboard | `b200/dashboards/{index,study/<id>,compare}.html` + `data/*.json` | `lmtune dashboard build` |
| Fabric baseline | `b200/studies/fabric_baselines/<ts>/fabric_baseline.json` | `b200/scripts/fabric_probe.sh` |
| ANALYSIS.md | `b200/studies/<study_id>/ANALYSIS.md` | 사람 + auto-mode |

용어 / 흐름 더 깊은 도식 = [`docs/architecture.md`](docs/architecture.md) (5-layer breakdown), [`docs/autotune_loop.md`](docs/autotune_loop.md) (4관점 sequence diagram).

## 두 트랙 — autotuner 입장에서는 같은 코드 경로

autotuner 는 endpoint YAML 의 `adapter` 필드 (`local-vllm` / `llmd-k8s`) 만 봅니다. 그 너머의 인프라(로컬 GPU / minikube / k3s 2-node B200 / vanilla k8s / OpenShift)는 endpoint YAML 의 `url`·`deployment` 블록과 cluster 의 helmfile 로 추상화됩니다.

| 트랙 | adapter | 사용 환경 (예) | endpoint YAML 예 |
|:---|:---|:---|:---|
| **Local-vLLM** | `local-vllm` | 단일 호스트 + GPU. `scripts/vllm_restart.sh` 가 trial 마다 vllm 서버를 재기동 | `configs/endpoints/local_vllm_autotune.yaml` |
| **K8s/llm-d** | `llmd-k8s` | 어떤 K8s 든 — minikube · k3s 멀티노드 (B200) · vanilla k8s · OpenShift. `LLMDK8sAdapter` 가 helmfile 로 trial 마다 values overlay 적용 + rollout | `configs/endpoints/minikube_pd_qwen25.yaml` (sample) |

→ B200 에서 실제로 돌릴 때는 endpoint YAML 의 `url`·`namespace`·`helmfile_path` 만 그 환경에 맞게 갈아끼우면 됩니다. autotuner 코드 변경 0.

## Prerequisites

### 공통 (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,search,distributed,runners]"
```

### Local-vLLM 트랙

| 항목 | 확인 |
|:---|:---|
| GPU + CUDA driver | `nvidia-smi` 동작 |
| vLLM 설치 | `python -c "import vllm"` |
| 모델 캐시 | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct` 등 |
| `scripts/vllm_restart.sh` | repo 안에 executable 로 존재 |

### K8s/llm-d 트랙 (minikube · k3s · vanilla k8s · OpenShift 공통)

| 항목 | 확인 명령 |
|:---|:---|
| `kubectl` 가 cluster 를 가리킴 | `kubectl get nodes` 가 Ready 노드 반환 |
| GPU device plugin (nvidia / cdi) | `kubectl get nodes -o jsonpath='{..allocatable.nvidia\.com/gpu}'` 가 0 보다 큰 값 |
| Gateway API CRD | `kubectl get crd \| grep gateway.networking.k8s.io` 가 `gateways`, `httproutes` 포함 |
| llm-d helmfile peer repo | 기본 `$HOME/ml_ai/agentic/llm-distributed-inference` (override via `PEER_REPO` env) (env `HELMFILE_ROOT` 로 override) |
| llm-d 배포 | peer repo 에서 `helmfile -f phase1/helmfile.yaml.gotmpl apply` 1회 (트랙 신규 진입 시) |
| (옵션) SR-IOV / RDMA — B200·B-track | Multus + `SriovNetworkNodePolicy` 등록 + `rdma/hca` resource 노출 |
| (옵션) OpenShift | `restricted-v2` SCC 호환 helmfile. peer repo 의 `phase1` 은 OCP 호환 작성 |

→ **현재 환경**(이 repo 를 받은 호스트)에서 K8s 트랙을 처음 시도하려면: minikube 가 가장 가벼움. 실제 B200 진입 시엔 같은 명령에 endpoint YAML 의 `url`·`namespace`·`helmfile_path` 만 B200 cluster 에 맞춰주세요.

## Quick Start — Local-vLLM 5단계 walkthrough

가장 가벼운 코스. RTX-class GPU 1장이면 ~30 분에 한 사이클.

```bash
# (1) 자동 튜닝 시작
lmtune search start \
  --endpoint configs/endpoints/local_vllm_autotune.yaml \
  --space b200/search-spaces/w_local_minimal.yaml \
  --profile configs/profiles/autotune/short.yaml \
  --strategy tpe --max-trials 10 --name w-local-mvp \
  --adapter local-vllm

# (2) 진행 상황 (별도 터미널)
lmtune search status <study_id>

# (3) 정적 HTML 대시보드
lmtune dashboard build --out b200/dashboards
xdg-open b200/dashboards/index.html

# (4) winner export — 자기 환경에 한 줄로 재배포
lmtune search export <study_id> --winner top-1 --out b200/results/<study_id>/winner
bash b200/results/<study_id>/winner/apply.sh --dry-run
cat b200/results/<study_id>/winner/README.md

# (5) cross-study compare
xdg-open b200/dashboards/compare.html
```

## Quick Start — K8s/llm-d 트랙

위 Prerequisites 의 K8s 항목이 모두 ✅ 인 상태에서:

```bash
# 1회 — peer repo 의 helmfile 로 llm-d 배포
cd $HELMFILE_ROOT      # 기본 ~/ml_ai/agentic/llm-distributed-inference
helmfile -f phase1/helmfile.yaml.gotmpl apply   # 또는 사용자 환경의 phase{1..4}
cd -

# 2. autotune (B200·minikube·OpenShift 등 동일 명령)
lmtune search start \
  --endpoint configs/endpoints/<your-llmd-endpoint>.yaml \
  --space b200/search-spaces/w_minikube_minimal.yaml \
  --profile configs/profiles/autotune/short.yaml \
  --strategy tpe --max-trials 6 --adapter llmd-k8s

# 3~5. 위 Local 트랙 (3)~(5) 동일
lmtune dashboard build --out b200/dashboards
lmtune search export <study_id> --winner top-1 --out b200/results/<study_id>/winner
```

`b200/search-spaces/w_minikube_minimal.yaml` 은 트랙 진입 검증용 최소 3-axis. 실제 B-track 으로 가면 `b200/search-spaces/{b1_baselines,b3_parallelism,b6_lowlevel}.yaml` 등으로 axis 를 확장합니다.

## LLM Dependency Policy

본 repo 는 **LLM-free 가 1st-class**. 사용자가 받아 실행하는 핵심 경로(autotune, dashboard, winner export, B-track) 는 **LLM 콜 0회로 동작**해야 한다.

| 모드 | LLM 콜 | 명령 | 의존성 |
|:---|:---:|:---|:---|
| **Headless** (1st-class, 기본) | 0회 | `lmtune search start --strategy tpe --max-trials 40` — Optuna TPE/NSGA-II/CMA-ES + hand-curated `axis_priors.yaml` | `pip install lmtune[search,distributed]` |
| **LLM-guided** (optional) | macro 추론 1 / 분석 1-2 | `autoresearch.sh` 가 Claude/GPT/local LLM 호출 → `lmtune search ask/tell` piping | `pip install lmtune[search,distributed,agent]` (anthropic SDK 추가) |

**의존성 격리 원칙**: `src/lmtune/search/llm_prior.py` (정적 YAML reader, LLM-free) 와 `src/lmtune/tuner/llm_oracle.py` (optional `[agent]` extra, dynamic import) 분리. ImportError 발생 시 명확한 에러 메시지 (`install with lmtune[agent]`). LLM-guided 모드의 산출물(갱신된 `axis_priors.yaml`) 은 정적 YAML 이라 git 검토 + headless 재현 가능.

## PLUG — Backend / Sampler 추가하기

본 프로젝트의 **모든 layer 가 ABC + 구현체** 원칙 (REFACTOR-PLAN 핵심 원칙 #2) 을 외부 사용자가 1 PR 로 확장할 수 있도록 PLUG 패턴을 정착시켰다. 두 추상 모두 같은 형식:

| 추상 | ABC | 첫 빌트인 | 두 번째 빌트인 | PLUG stub | 합류 시 |
|:---|:---|:---|:---|:---|:---|
| **Storage backend** | `lmtune.storage.store.ArtifactStore` | `DuckDBArtifactStore` | `LocalArtifactStore` (jsonl) | `PostgresArtifactStore` (`pip install lmtune[postgres]`) | `lmtune storage migrate --src-kind postgres --src postgres://...` 자동 동작 |
| **Tuner sampler** | `lmtune.tuner.Sampler` | `OptunaSamplerAdapter` (TPE/NSGA-II/CMA-ES 6종) | `Native{Random,LHC,TPE}` | `LLMOracleSampler` (`pip install lmtune[agent]`) | `lmtune.tuner.factory.make_sampler('llm_oracle', space)` 즉시 동작 |

새 backend 추가 시 변경되는 곳: ABC 구현체 1 파일 + factory 매핑 1줄. 외부 사용자 입장에선 `--src-kind X` / `--strategy X` 만 추가하면 즉시 사용 가능. 자세한 step-by-step 은 `docs/architecture/REFACTOR-PLAN.md` PLUG 섹션과 `tests/storage/test_postgres_store_stub.py` / `tests/tuner/test_llm_oracle_stub.py` 의 acceptance 케이스 참조.

## Storage 변환 — DuckDB ↔ Local jsonl

\`\`\`bash
# 운영 DB 를 git 친화 jsonl 디렉토리로 export
lmtune storage migrate \
  --src-kind duckdb --src data/db/lmtune.duckdb \
  --dst-kind local  --dst data/archive/2026-05-06/

# 외부 archive (jsonl) 를 다시 DuckDB 로 import
lmtune storage migrate \
  --src-kind local  --src data/archive/2026-05-06/ \
  --dst-kind duckdb --dst data/db/imported.duckdb

# 등록된 backend 목록 (PLUG 합류 시 자동 노출)
lmtune storage list-backends

# BenchmarkResult result.json → records 디렉토리 (raw_dir 외부에서 받은 산출 archive)
lmtune contracts records-from-result <result.json> --out <records-dir>
\`\`\`

위 4 명령은 모두 `ArtifactStore.put` / `query` 만 사용하므로 신규 backend (`postgres` 등) 가 합류해도 수정 0.

## User Contract — Inputs & Outputs

### Inputs — 사용자가 제공해야 하는 것

| # | 입력 | 형식 | 빈도 | 위치 |
|:--|:---|:---|:---|:---|
| 0 | 인프라 컨텍스트 | kubeconfig + (옵션) ariadne snapshot | 환경 1회 | 시스템 |
| 1 | 무엇을 서빙 | `endpoints/*.yaml` | 모델 추가 시 | `configs/endpoints/`, `b200/endpoints/` |
| 2 | 어떤 워크로드 | `profiles/*.yaml` 또는 trace | 워크로드 정의 시 | `configs/profiles/autotune/`, `b200/profiles/` |
| 3 | 어디까지 탐색 | `search-spaces/*.yaml` (3 layer 토글) | 실험 정의 시 | `b200/search-spaces/` |
| 4 | 무엇을 최적화 | objective + SLO | 실험 정의 시 | CLI flag |
| 5 | 얼마나 탐색 | budget + strategy + workers | 실험 호출 시 | CLI flag |
| 6 | 어디에 적용 | adapter | 환경 1회 | `endpoint.adapter` (`local-vllm` / `llmd-k8s`) |

### Outputs — autotuner 가 사용자에게 전달

| # | 출력 | 위치 | 활용 |
|:--|:---|:---|:---|
| **A** | Winning config recipe (self-contained) | `b200/results/<study>/winner/{params.json, values-overlay.yaml, apply.sh, README.md}` | `bash apply.sh --dry-run` 한 줄로 재배포 |
| B | Pareto front | `b200/studies/<study>/pareto.{json,html}` | multi-obj trade-off |
| C | ANALYSIS.md | `b200/studies/<study>/ANALYSIS.md` | 컨텍스트/결과/원인/의의/후속 |
| D | Raw 데이터 | `data/db/lmtune.duckdb` + `studies/*/trials.parquet` | warm-start 소스 |
| E | Plot 아티팩트 | `studies/<study>/plots/` | 보고서 인용 |
| F | system_snapshot.json (B6) | `studies/<study>/system_snapshots/` | PCIe/IOMMU/NUMA/RDMA 토폴로지 |
| **G** | InferenceX-style 정적 HTML 대시보드 | `b200/dashboards/index.html` + `studies/<id>.html` + `compare.html` + `data/index.json` | git push 만으로 공유 |
| G' | Grafana 대시보드 JSON | `b200/dashboards/grafana_dashboard.json` (B-IV) | 라이브 모니터링 |
| **H** | Self-contained apply 스크립트 | (A 와 동일) | 외부 사용자 git pull → `bash apply.sh` 한 줄 |

## Walkthrough (5 Steps)

5단계 사용자 시나리오 — Phase W 의 검증 시나리오 그대로:

1. **시작** — `lmtune search start ...` 한 줄로 autotune 개시.
2. **중간 결과** — 별도 터미널에서 `watch -n 10 lmtune search status <study_id>` 또는 dashboard 라이브 갱신으로 진행 확인.
3. **시각화** — study 종료 후 `lmtune dashboard build --out b200/dashboards` → 정적 HTML 열어서 모델×HW 매트릭스 + Pareto + axis importance 카드 확인.
4. **최종 결과** — `lmtune search export <study_id> --winner top-1` → `winner/apply.sh`, `values-overlay.yaml`, `README.md` 자동 생성. `bash apply.sh --dry-run` 으로 valid 한 helmfile/vllm 명령 확인 가능.
5. **다른 인프라 비교** — dashboard 의 `compare.html` 에서 같은 모델·워크로드를 다른 cluster (로컬 RTX / minikube / B200 k3s / vanilla k8s / OpenShift) 에서 돌린 study 를 동일 매트릭스 카드로 비교.

### 두 트랙의 추상 — 한 줄 도식

```
SearchSpace YAML                       endpoint YAML (adapter 만 다름)
       │                                       │
       └──────► lmtune search start ◄───────────┤
                       │                       │
                       ▼                       ▼
            ┌─────────────────┐    ┌──────────────────────────────┐
            │ adapter:        │    │ adapter:                     │
            │ local-vllm      │    │ llmd-k8s                     │
            │                 │    │                              │
            │ vllm_restart.sh │    │ helmfile state-values-file   │
            │ 단일 host GPU    │    │ overlay → kubectl rollout    │
            └─────────────────┘    └──────────────────────────────┘
                                              │
                                              ▼
                       어떤 K8s 든: minikube / k3s 2-node B200 /
                       vanilla k8s / OpenShift (kubeconfig 만 갈아끼움)
```

## Architecture (1-page)

3 layer 압축: **Macro × Env Profile × Micro** — 70+ flat axis 를 ~12 effective axis 로 4×10⁴⁰× 압축.

```
[Macro axes] (~10 dim — § vllm-config-puzzle Simulator 정본)
   parallelism : tp, pp, dp, ep, ep_strategy, pcp, dcp     ← simulator 7 dim
   network     : intraNode_type, crossNode_type            ← simulator 2 dim
   model       : model_id (registry 가 numHeads/MoE/MLA 자동 expand)
   path        : well_lit_path                             ← llm-d 7 path
   dtype       : engine_dtype, kv_cache_dtype
       ↓
[Feasibility Filter] (10 validation 제약, vllm-config-puzzle validation.ts 1:1)
   sampling 후 is_feasible(params, env) → infeasible 80% 자동 prune
       ↓
[Env Profile Template] (자동 binding, 사용자 입력 X)
   (macro tuple) → configs/autoresearch/env_profiles/<auto-selected>.yaml
   90% env_locked   : NCCL_IB_HCA, UCX_TLS, NCCL_NVLS_ENABLE — sampler 가 안 만짐
   10% env_tunable  : 3-5 의미 있는 fine-tune knob 만 노출
       ↓
[Surrogate Predictor]
   white-box (TTFT/ITL/TPS 공식) + black-box residual (XGBoost on archive) hybrid
   confidence ≥ 0.9 + score ≪ best → skip 실측 (negative example archive)
```

핵심 출처: vllm-config-puzzle simulator (TP/PP/DP/EP-std/EP-wide/PCP/DCP/SP, MoE/MLA 모델 조건, 10 validation 제약, performance formulas) 가 본 plan 의 Python port reference. autotune loop 의 4 관점 sequence diagram 은 [`docs/autotune_loop.md`](docs/autotune_loop.md) 참조.

> **상세 architecture 도식 (5 layer × extension points × gap)** = [`docs/architecture.md`](docs/architecture.md). 새 모델·기능·도구 추가 시 어느 layer 의 어느 plug-in 에 1줄 넣으면 되는지 1:1 매핑.

## External 자산 참조

| 외부 repo | 정체 | 결합 패턴 | 본 프로젝트의 어디 |
|:---|:---|:---|:---|
| [`qemu/ariadne`](https://github.com/qemu/ariadne) | Python 토폴로지 수집·시뮬 (sysfs/procfs/lspci → NetworkX → DES, FastAPI) | `[lowlevel]` extra 로 dynamic import + `system_capture.py` 에서 `build_topology()` 호출 + JSON snapshot | B6 + `src/bench/runners/system_capture.py` |
| [`vllm-config-puzzle`](https://github.com/vllm-project/vllm-config-puzzle) | TypeScript/React 퍼즐 + simulator | **§ vllm-config-puzzle Simulator 섹션의 1:1 Python port reference** (`src/bench/search/{feasibility,surrogate_analytical}.py`, `src/bench/models/registry.py`) | B3 정본 |
| [`agentic/llm-distributed-inference`](https://github.com/agentic/llm-distributed-inference) | helmfile + Kustomize phase{1..4} (peer repo, 7 well-lit-path 정본) | 그대로 helmfile apply (`LLMDK8sAdapter`) | B3 보강, B-IV 까지 |
| [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) | continuous inference benchmarking platform | `perf-changelog.yaml` schema + dashboard schema + AGENTS.md 패턴 | Phase W (대시보드), B5 (changelog) |

설치 (선택적):
```bash
pip install -e ".[lowlevel]"   # ariadne 활성 (B6 시스템 토폴로지)
pip install -e ".[agent]"      # LLM-guided 모드 (LLMOracleSampler — anthropic SDK)
pip install -e ".[postgres]"   # PostgresArtifactStore (multi-writer / 분산 환경)
```

## Subcommand 요약

```text
lmtune run         단일 워크로드 측정 → DuckDB
lmtune sweep       여러 profile 순차 측정
lmtune repeat      같은 profile N 번 반복 (variance)
lmtune variance    최근 N run 의 μ/σ/CV/IQR
lmtune nway        N 개 run 비교 매트릭스
lmtune export      requests 테이블 → CSV/Parquet/JSON
lmtune report      Markdown 리포트 + 플롯
lmtune compare     2 run 직접 비교
lmtune detect      SLO/IQR/회귀 이상탐지
lmtune ls / show   run 조회
lmtune search      자동 튜닝 (start/status/ask/tell/export)
lmtune orchestrate DeploymentAdapter 직접 호출 (수동 배포)
lmtune dashboard   정적 HTML 대시보드 (build/serve)
```

## 디렉토리

| 경로 | 역할 |
|:-----|:-----|
| `configs/profiles/` | 워크로드 정의 YAML |
| `configs/endpoints/` | Endpoint 접속 정보 (API key 는 env 변수 참조) |
| `configs/autoresearch/env_profiles/` | Macro × Env Profile binder 의 8 profile YAML |
| `b200/search-spaces/` | B200 / W-Local / W-Minikube 의 SearchSpace YAML |
| `b200/helmfile/` | peer repo `agentic/llm-distributed-inference` 의 phase{1..4} 를 16-GPU 로 fork·adapt |
| `b200/dashboards/` | 정적 HTML 대시보드 산출 (Output G) |
| `b200/results/<study>/winner/` | self-contained recipe (Output A/H) |
| `b200/perf-changelog.yaml` | 외부 PR 머지 → baseline 영향 시계열 (B5 watch) |
| `src/lmtune/search/` | SearchSpace, Objective, profile_binder, llm_prior (정적 YAML reader) |
| `src/lmtune/tuner/` | Sampler / Pruner ABC + Optuna adapter + Native + LLMOracleSampler stub |
| `src/lmtune/storage/store/` | ArtifactStore ABC + DuckDB / Local / InMemory / Postgres stub |
| `src/lmtune/orchestrate/` | TrialBackend (k8s_job / process_pool), Driver |
| `src/lmtune/deploy/` | DeploymentAdapter (LocalVLLM, LLMDK8s) |
| `src/lmtune/visualization/dashboard/` | Jinja2 + Tailwind CDN 정적 dashboard 빌더 |
| `data/db/lmtune.duckdb` | 결과 누적 저장소 |
| `data/archive/` | 과거 study archive (read-only) |

## 상태

Phase W (Walkthrough MVP) 코드 진입 완료. B0~B8 (B200 16-GPU) 진입 대기. 전체 로드맵은 `(internal dev plan, not in repo)` 참조.

## Acknowledgements

- **Anthropic Claude Code** — 본 repo 의 코드/문서/플랜은 사용자(`@jinmoo`) + Claude Code (Opus 4.7 1M context) 의 공동 작업.
- **Peer repos**: [`agentic/llm-distributed-inference`](https://github.com/agentic/llm-distributed-inference) (helmfile 정본), [`qemu/ariadne`](https://github.com/qemu/ariadne) (PCIe/RDMA 토폴로지), [`vllm-config-puzzle`](https://github.com/vllm-project/vllm-config-puzzle) (parallelism simulator).
- **External references**: [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) (perf-changelog/dashboard schema, AGENTS.md), [`llm-d/llm-d`](https://github.com/llm-d/llm-d) (well-lit path 정본), [`vllm-project/vllm`](https://github.com/vllm-project/vllm), [`sgl-project/sglang`](https://github.com/sgl-project/sglang), [`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM).

## License

본 repo 의 코드는 작성자 + Anthropic 공동 저작. 외부 reference 의 라이선스는 각 repo 를 따릅니다.
