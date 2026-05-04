# B200 — 사용 시나리오 (project north star)

| 메타 | 값 |
|:---|:---|
| 최종 검토일 | 2026-05-05 |
| 검토 주기 | PR 머지 시마다 (시나리오 변경 시 본 문서 갱신 의무 — CLAUDE.md 의 § PR 게이트) |
| 소유자 | 본 프로젝트 운영자 |
| 상태 | active — 본 문서가 모든 코드/문서/도구의 북극성 |

## 1. Executive Summary

본 프로젝트는 vLLM/llm-d 기반 LLM 서빙의 (모델 × 워크로드 × HW × engine 옵션 × parallelism × low-level) 결합 공간에서 **사용자 자기 클러스터의 best config 를 자동 탐색** 하고, 즉시 재배포 가능한 형태 (`apply.sh`) 로 결과를 전달한다.

운영자는 **3 입력** (endpoint YAML / search-space YAML / profiles 3종) 을 작성하고 **3 단계** (LAUNCH → SEARCH → EXPORT) 를 실행하면 **3 출력** (winner recipe / Pareto + ANALYSIS / DuckDB archive) 을 받는다. 각 단계는 단일 명령으로 끝난다 (`ops/launch.sh`, `lmtune search start`, `lmtune search export`). 시나리오가 명세된 흐름에서 어긋나면 그건 결함 신호 (사용자 시간 소비) 이고 `b200/docs/regressions.md` 의 catalog 에 영속화된다.

## 2. Problem Definition

### 2.1 누구의 문제

- **Who** — vLLM/llm-d 기반 자기 클러스터를 운용하는 LLM 서빙 운영자 (사용자: NHN Cloud B200 16-GPU k3s)
- **What** — 매 실험 cycle 의 환경 정비 (helmfile apply, port-forward, 환경변수, decode pod ready 검증, 모델 일치 확인) 가 손작업으로 분산되어 1회 재실행에 1시간+ 소요
- **When/Where** — 첫 cycle 이후 모든 재실행, 모델 swap, axis 변경 시 매번 발생
- **Impact** — PR #24~#26 의 chain failure (2026-05-05) 에서 사용자 누적 4시간+ 좌절. lmtune search 본 작업은 5 trial halt 로 시작도 못 함
- **Urgency** — 다음 phase (B2 sweep 30 trial × 3 repeats × 3 workloads = 270 measurements) 진입 전 운영 자동화 필수. 운영 결함이 측정 결함을 가린다

### 2.2 결함 분류 (관측된 5종, 영속화: regressions.md R1~R5)

| ID | 결함 | 사용자 영향 |
|:---|:---|:---|
| R1 | chart Deployment 라벨 부재로 `wait deploy -l role=decode` 매칭 실패 | step 3 fail → port-forward 미작동 → 5 trial halt |
| R2 | `set -u` + 한 줄 다중 `local` 선언 unbound | prepare.sh 진입 즉시 실패 |
| R3 | RollingUpdate strategy 가 GPU 16-pool 환경에서 deadlock | 새 pod Pending 영원, 기존 pod 안 죽음 |
| R4 | `--backend k8s-job` hard-gated (Phase S4) | 사용자 명령 거부 |
| R5 | endpoint url ↔ port-forward ↔ model 의 strict ordering | 한 단계 누락 시 chain 폭발 |

→ 이 시나리오 정의는 위 결함이 **다시 사용자 손에 닿지 않도록** 도구 진입점으로 재구조화한 산물.

## 3. User Scenarios (lifecycle 분해)

각 시나리오는 functional 명세 (Actor / Precondition / Flow / Exception / Postcondition) + 영속화 코드 위치로 정의. 시나리오 간 전이는 § 4. Design 의 진입점 매핑 표 참조.

### S1 — 환경 준비 (Setup, 1회)

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 |
| 빈도 | 환경 1회 (k3s 클러스터 셋업 / 노드 추가 시) |
| Precondition | k3s 가 deploy 됨, 노드에 nvidia driver + CUDA + RDMA HCA 펌웨어 |
| Flow | (1) `bash b200/scripts/probe.sh --mode host` (2) `bash b200/scripts/setup_host.sh` (3) `bash b200/scripts/setup_gateway_provider.sh agentgateway` (4) `kubectl get runtimeclass nvidia` 검증 |
| Exception | RDMA 미동작 → `b200/docs/rdma_perftest_baseline.md`. nvidia-device-plugin 미설치 → host setup 단계 |
| Postcondition | RuntimeClass nvidia, peer-repo cache, agentgateway controller 모두 ready |
| 영속화 | `b200/scripts/{probe.sh, setup_host.sh, setup_gateway_provider.sh}` |

### S2 — 처음 실행 (Cold Launch)

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 |
| 빈도 | endpoint YAML 신규 작성 후 첫 실험 |
| Precondition | S1 완료. endpoint YAML / search-space / profiles 작성됨 |
| Flow | (1) `bash b200/scripts/ops/launch.sh <endpoint.yaml>` 한 줄 — `launch.sh` 가 8 step 자동 (endpoint 파싱 → values 매핑 → cluster check → release 검증 → helmfile apply → wait → pf → probe → model 일치) (2) `lmtune search start --adapter llmd-k8s ...` |
| Exception | release 미설치 → launch.sh 가 자동 install. namespace 없음 → 즉시 명확 에러. helmfile apply rc!=0 → adapter 가 다음 trial 에서 재시도 |
| Postcondition | study DB row + studies/<study> + winner export 가능 상태 |
| 영속화 | `b200/scripts/ops/launch.sh`, `src/lmtune/cli_search.py` |

### S3 — 환경 초기화 (Reset)

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (직전 cycle 의 잔재로 다음 cycle 출발 시) |
| 빈도 | trial halt / 모델 swap / port-forward 충돌 발생 시 |
| Precondition | 직전 study 종료 또는 halt |
| Flow (수준별) | (a) **soft** `bash b200/scripts/ops/reset.sh` — port-forward 만 정리 (b) **pods** `bash b200/scripts/ops/reset.sh infsch --pods` — decode pod rolling restart, release/weight 캐시 유지 (c) **hard** `bash b200/scripts/ops/reset.sh infsch --hard` — `helmfile destroy` (확인 prompt) |
| Exception | 살아있는 lmtune 프로세스 있음 → 사용자가 먼저 kill (reset 은 서비스 종료 X) |
| Postcondition | (a) port-forward 0 + listener 0 (b) decode pod 새로 떠 ready (c) release 3종 모두 uninstall |
| 영속화 | `b200/scripts/ops/reset.sh` |

### S4 — 재실행 (Re-launch, 동일 endpoint)

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (axis 만 변경 / repeat / warmstart 활용 시) |
| 빈도 | 매 실험 (사용자 가장 자주) |
| Precondition | S2 또는 S3 후 — release 가 살아있을 수도, 부분 dead 일 수도, decode pod 가 다른 모델일 수도 |
| Flow | (1) `bash b200/scripts/ops/launch.sh <endpoint.yaml>` — **S2 와 동일 명령**. launch.sh 가 idempotent 으로 (a) release 살아있음 + 모델 일치 → fast path skip apply (b) release 없거나 모델 mismatch → 자동 helmfile apply (2) `lmtune search start ... --warmstart-db data/db/lmtune.duckdb` |
| Exception | helmfile rolling 중 port-forward 끊김 → 재시도 wrapper 가 자동 재연결 (`pf::start`). final model id mismatch → exit 3, 사용자에게 hard reset 권장 |
| Postcondition | S2 와 동일 — 단 archive 의 직전 trial 들이 warm-start seed 로 활용됨 |
| 영속화 | `b200/scripts/ops/launch.sh` (idempotent + 모델 검증) |

> **핵심 통찰** — vLLM 본성상 config change = engine restart = weight reload 라서 S2 와 S4 의 비용은 본질적으로 같다. 따라서 **두 시나리오의 진입점도 같다 (`ops/launch.sh`)**. 분리된 두 명령으로 노출하지 않는다 — 분리하면 사용자가 "처음 vs 재실행" 구분에 불필요한 인지 비용을 지불한다.

### S5 — 모델 swap

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (gpt-oss-120b → llama / qwen) |
| 빈도 | 새 모델 평가 시 |
| Precondition | 신규 모델용 endpoint YAML 작성됨. 해당 모델의 `values-<model>.yaml.gotmpl` 이 helmfile 디렉토리에 존재. `util/env.sh::values_for_model` 의 case 에 매핑 등록됨 |
| Flow | (1) `bash b200/scripts/ops/launch.sh b200/endpoints/b200_<new-model>.yaml` — launch.sh 가 model id mismatch 자동 감지 → helmfile apply (해당 model 의 values 파일로) → weight 재로딩 5–15분 → `/v1/models` 가 새 model id 응답할 때까지 polling (2) `lmtune search start --endpoint <new-endpoint.yaml> ...` |
| Exception | 매핑 미등록 → `[env] no values mapping for model='...'` 즉시 에러 (helmfile apply 안 함). final mismatch → exit 3 |
| Postcondition | 새 모델로 vLLM 응답 + 이전 모델은 삭제 (Recreate strategy 가 보장) |
| 영속화 | `util/env.sh::values_for_model` + `pf::current_model` mismatch 감지 + `LLMDK8sAdapter` |

### S6 — well-lit-path 변경

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (inference-scheduling → P/D disaggregation 또는 wide-EP) |
| 빈도 | 새 path 평가 시 |
| Precondition | 해당 path 의 `b200/helmfile/<path>/` 와 endpoint YAML 작성됨 |
| Flow | (1) S3-c (hard reset) 로 직전 path uninstall — namespace 분리됨 (b200-infsch / b200-pd / b200-wideep) (2) `bash b200/scripts/ops/launch.sh <endpoint.yaml> <rn>` — `<rn>` = `pd` / `wideep` (3) `lmtune search start ...` |
| Exception | 두 path 의 release 가 동시에 떠있어 GPU 충돌 → S3-c 로 한쪽 먼저 정리 |
| Postcondition | 해당 path 의 release 3종만 떠있고 다른 path 는 미존재 |
| 영속화 | `helmd::file_for` (path → helmfile 매핑) + `LLMDK8sAdapter::well_lit_path` 디스패치 |

### S7 — Halt 후 재시도

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (5-trial circuit breaker halt 발생 시) |
| 빈도 | infeasible axis 조합 / cluster 일시 fail 시 |
| Precondition | `lmtune search start` 가 5 consecutive failure 로 halt — `study` row 가 DB 에 partial 로 남음 |
| Flow | (1) `bash b200/scripts/ops/status.sh infsch` — 환경 한 화면 (2) 결함 종류에 따라 S3 의 (a)/(b)/(c) 중 선택 (3) `bash b200/scripts/ops/launch.sh <endpoint.yaml>` (4) `lmtune search resume <study_id>` 로 이어서 또는 새 `--name` 으로 시작 |
| Exception | DB lock 충돌 → `lmtune search ls --plain` 으로 study 상태 확인 후 처리 |
| Postcondition | 새 trial 이 halt 이전 지점부터 진행 또는 별도 study |
| 영속화 | `failure_handler.py` + `cli_search.py::cmd_resume` + `ops/{status,reset}.sh` |

### S8 — 결과 분석 + 비교

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (study 종료 후) |
| 빈도 | 매 study 종료 후 |
| Precondition | study 가 `completed` 상태. trial_metrics 가 충분히 채워짐 |
| Flow | (1) `lmtune search status <study_id>` — top-N + Pareto JSON (2) `lmtune nway <a> <b> [...]` — 다수 study 비교 (3) `lmtune dashboard build` — 정적 HTML (4) `lmtune search export <study_id> --winner top-1` — apply.sh 생성 |
| Exception | warm-start DB 가 다른 모델 archive 와 같이 묶여있어 잡음 → `--warmstart-top-k` 줄임 또는 archive 분리 |
| Postcondition | 출력 A/B/C 가 `b200/results/<study>/` 에 생성됨 |
| 영속화 | `cli_search.py::{cmd_status, cmd_export}` + `cli_dashboard.py` + `objective_pareto.py` |

### S9 — 결함 발견 + 영속화 (governance)

| 항목 | 값 |
|:---|:---|
| Actor | 운영자 (사용자 시간 소비 결함 발견 시) |
| 빈도 | 결함 1건마다 |
| Precondition | 한 번이라도 사용자 손작업이 chat / 산문 가이드에 의존했음 |
| Flow | (1) 결함 즉시 fix 코드 작성 (2) `b200/docs/regressions.md` 에 `R<n>` entry 신설 (3) `b200/scripts/tests/test_*.sh` 회귀 테스트 ≥ 1 (4) `bash b200/scripts/tests/run_all.sh` 통과 (5) PR 한 번에 (코드 + catalog + test) 묶음 |
| Exception | 회귀 테스트가 fake-kubectl 한계로 못 잡는 결함 (R3 의 GPU deadlock 같은) → catalog entry + chart/post-renderer 패치로 영속화. CLAUDE.md § PR 게이트 의 한계 명시 그대로 |
| Postcondition | 동일 결함 재발생 시 catalog 가 1차 진단처. 본 시나리오 자체가 갱신될 수도 (예: § 비-목표 추가) |
| 영속화 | `CLAUDE.md` § PR 게이트 + `b200/docs/regressions.md` + `b200/scripts/tests/` |

## 4. Design — 시나리오 → 진입점 매핑

### 4.1 단계별 단일 진입점

| 단계 | 명령 | 책임 |
|:---|:---|:---|
| LAUNCH | `bash b200/scripts/ops/launch.sh <endpoint.yaml> [<rn>]` | 환경 준비 (S2/S4/S5/S6 모두 동일) |
| SEARCH | `lmtune search start --endpoint ... --space ... --adapter llmd-k8s ...` | 본 실험 (axis sweep + composite score) |
| EXPORT | `lmtune search export <study_id> --winner top-1` | 결과 회수 (apply.sh + values-overlay) |
| RESET | `bash b200/scripts/ops/reset.sh [<rn>] [--pods\|--hard]` | 잔재 정리 (S3) |
| STATUS | `bash b200/scripts/ops/status.sh [<rn>]` | 진단 한 화면 (S7) |
| RESUME | `lmtune search resume <study_id>` | halt 후 이어서 (S7) |

운영자가 직접 부르는 명령은 위 6 종으로 한정. helmfile / kubectl / curl 직접 호출은 결함 신호.

### 4.2 ops/launch.sh 의 8 step (S2/S4/S5/S6 공통)

```
1. endpoint YAML 파싱 → 의도한 model 추출
2. model → values 파일 매핑 → B200_MODEL_VALUES 자동 export
3. cluster + namespace 검증
4. helm release 3종 + 현 vLLM 모델 일치 비교 → 불일치 시 helmfile apply 자동
5. decode Deployment Available 대기
6. stale port-forward 정리 + 재시도 wrapper 데몬
7. /v1/models 200 polling
8. 응답 model id 와 endpoint 의 model 최종 일치 검증
```

종료 코드 0 = SEARCH 진입 가능 / 2 = 사용자 조치 필요 / 3 = redeploy 후에도 mismatch.

## 5. Architecture — 3 입력 / 3 출력 / 도구 layer

### 5.1 입력 (사용자 작성)

| # | 입력 | 형식 | 위치 |
|:--|:---|:---|:---|
| 1 | endpoint YAML | `apiVersion: lmtune/v1alpha1` | `b200/endpoints/*.yaml` |
| 2 | search-space YAML | `apiVersion: lmtune/search/v1alpha1` | `b200/search-spaces/*.yaml` |
| 3 | profiles (≥3) | `apiVersion: lmtune/v1alpha1` | `configs/profiles/autotune/*.yaml` |

### 5.2 출력 (autotuner 가 전달)

| # | 출력 | 위치 |
|:--|:---|:---|
| A | winner recipe (apply.sh + values-overlay + README) | `b200/results/<study>/winner/` |
| B | Pareto top-N + ANALYSIS.md | `b200/studies/<study>/` |
| C | DuckDB archive | `data/db/lmtune.duckdb` |

### 5.3 도구 Layer

```
┌─────────────────────────────────────────────────┐
│  ops/  (사용자 진입점)                            │
│   launch.sh   reset.sh   status.sh              │
└─────────────────────────────────────────────────┘
        ↓ source
┌─────────────────────────────────────────────────┐
│  util/  (재사용 함수 라이브러리)                   │
│   pf.sh   helm.sh   env.sh                      │
└─────────────────────────────────────────────────┘
        ↓ subprocess / kubectl / helmfile
┌─────────────────────────────────────────────────┐
│  helmfile + post-renderer + chart               │
│  (k8s manifest 생성 + RuntimeClass + Recreate)   │
└─────────────────────────────────────────────────┘
        ↓ /v1/models HTTP
┌─────────────────────────────────────────────────┐
│  lmtune  (Python — 본 실험)                       │
│   cli_search   LLMDK8sAdapter   Study + Sampler │
└─────────────────────────────────────────────────┘
```

## 6. 시나리오 ↔ 코드 매핑 (영속화 위치)

| 시나리오 / 출력 | 코드 위치 |
|:---|:---|
| 입력 1 endpoint YAML | `src/lmtune/endpoints.py` + `b200/endpoints/*.yaml` |
| 입력 2 search-space YAML | `src/lmtune/search/space.py` + `b200/search-spaces/*.yaml` |
| 입력 3 profiles | `src/lmtune/profiles.py` + `configs/profiles/autotune/*.yaml` |
| S1 setup | `b200/scripts/{probe.sh, setup_host.sh, setup_gateway_provider.sh}` |
| S2 / S4 / S5 / S6 LAUNCH | `b200/scripts/ops/launch.sh` + `b200/scripts/util/{pf,helm,env}.sh` |
| S3 reset | `b200/scripts/ops/reset.sh` |
| S7 status / resume | `b200/scripts/ops/status.sh` + `src/lmtune/cli_search.py::cmd_resume` + `src/lmtune/orchestrate/failure_handler.py` |
| S8 EXPORT / 분석 | `src/lmtune/search/export_winner.py` + `src/lmtune/cli_dashboard.py` + `src/lmtune/search/objective_pareto.py` |
| S9 governance | `CLAUDE.md` § PR 게이트 + `b200/docs/regressions.md` + `b200/scripts/tests/` |
| 출력 A winner | `src/lmtune/search/export_winner.py` |
| 출력 B Pareto + ANALYSIS | `src/lmtune/search/objective_pareto.py` + `b200/docs/ANALYSIS_template.md` |
| 출력 C DuckDB archive | `src/lmtune/storage/duckdb_store.py` |

## 7. 비-목표 (코드가 약속하지 않는 것)

- 매 axis 변경 시 weight reload 회피 — vLLM 본성, 받아들임 (출력 A 의 apply.sh 만이 production overhead 회피)
- helmfile / kubectl 직접 호출 — `ops/` 가 추상. 사용자가 직접 부르는 건 결함 신호 → S9 영속화
- 동시 multi-study (single-writer DuckDB + single GPU pool 전제)
- vLLM 서버를 사용자가 별도 띄움 — `ops/launch.sh` 책임
- 모델 학습 / finetuning
- 페이퍼-grade 일회 측정 (별도 harness 필요)

## 8. 시나리오의 진화 (governance)

본 시나리오 자체가 사용자 통찰 / 결함 영속화로 변경된다. chat 의 산문이 아니라 **본 문서가 갱신되어야 시나리오 변경**.

- 사용자가 새 시나리오 변형 요청 → 본 § 3 에 S<n> entry 추가 + § 6 매핑 갱신
- 결함 발견 시 → `regressions.md` entry + § 7 비-목표 또는 § 6 매핑 변경 여부 결정
- chat 합의된 시나리오 결정은 **코드/매핑 표/회귀 테스트 갱신까지** 완료되어야 PR 머지 (CLAUDE.md § PR 게이트)

## 9. References

- `CLAUDE.md` (repo root) — § PR 게이트 (운영 결함 영속화 의무)
- `b200/docs/regressions.md` — 결함 catalog R1~R5
- `b200/docs/port_forward_runbook.md` — gateway port-forward 운영
- `b200/docs/B0_runbook.md` — S1 환경 준비 절차
- `b200/scripts/tests/run_all.sh` — 회귀 테스트 진입점
- `~/.claude/rules/documentation.md` — 본 문서의 top-down 구조 원칙
