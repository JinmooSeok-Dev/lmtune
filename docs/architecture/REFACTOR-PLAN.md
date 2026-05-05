# lmtune Architecture — Refactor Plan

> 본 문서는 lmtune 의 **목표 아키텍처** 와 거기에 도달하는 **단계별 PR sequence** 를 영속화한다.
> 단일 진실: 매 PR description 은 본 문서의 단계 하나를 인용해야 한다.

## 핵심 원칙 5

1. **외부 master schema 존중** — workload spec 은 [lm-workloads](https://github.com/.../workloads) 가 master, hardware/network/infra spec 은 [ariadne](https://github.com/.../ariadne) 가 master. lmtune 은 mirror + consumer.
2. **모든 layer 가 ABC + 구현체** — Sampler/Pruner/Backend/Launcher/WorkloadStream/BenchmarkRunner/WorkloadStore/ArtifactStore. 새 구현체 추가 = 1 PR.
3. **Provider 단일 abstraction** — BYO yaml 과 외부 도구 호출 (Discover) 모두 `WorkloadProvider`/`ClusterProvider` 의 한 구현체. 분기 0.
4. **Storage 가 single shared hub** — DuckDB/Postgres/ClickHouse 같은 backend 가 ABC 뒤로. Tuner/Output/Analysis/사용자 CLI 모두 readonly query.
5. **leaf-up refactoring** — 의존성 그래프의 upstream leaf 부터 수정. 변경 영향 격리.

## 의존성 그래프 (lmtune 내부)

```
L0  Input (leaf — 외부 master 만 의존)
    ├─ workload/providers/        ← lmtune#WS
    └─ cluster/providers/         ← lmtune#CS
                  │
L1  Storage (leaf-near — contracts subset 만 의존)
    └─ storage/                    ← lmtune#SS
                  │
L2  Compute layers
    ├─ tuner/                      ← lmtune#S1
    ├─ deploy/ (Launcher)
    ├─ workload/streams/
    └─ runners/                    ← lmtune#R0 (BenchmarkResult contract)
                  │
L3  Coordination
    └─ orchestrate/                ← lmtune#OD (driver/backend 분리)
                  │
L4  Output (Storage readonly)
    └─ output/                     ← lmtune#OUT
                  │
L5  CLI (모두 통합)
    └─ cli/
```

## PR Sequence (총 10 PR, leaf-up)

| # | PR | layer | 의존 | size | 설명 |
|:--|:---|:---:|:---|:---|:---|
| 1 | `lmtune#WS` | L0 | 외부 lm-workloads | 중 | Workload Spec contract + LiteralProvider + LMWorkloadsProvider |
| 2 | `ariadne#A1` (병렬) | (외부) | — | 중 | ariadne 본체 multi-host snapshot MVP |
| 3 | `lmtune#CS` | L0 | A1 | 중 | Cluster Spec contract + LiteralProvider + AriadneProvider |
| 4 | `lmtune#SS-rec` | (contracts subset) | — | 소 | RecordSpec + QuerySpec Pydantic + JSON Schema |
| 5 | `lmtune#SS` | L1 | SS-rec | 중-대 | WorkloadStore + ArtifactStore ABC, DuckDBStore + InMemoryStore + LocalArtifactStore |
| 6 | `lmtune#R0` | (contracts) | SS-rec | 중 | result_spec 정형화, 4 runner 가 BenchmarkResult emit |
| 7 | `lmtune#S1` | L2 | SS + R0 | 중 | search/ → tuner/ (Sampler/Pruner ABC) |
| 8 | `lmtune#OD` | L3 | S1 | 소-중 | Orchestrator → Driver/Backend 분리 |
| 9 | `lmtune#OUT` | L4 | SS + R0 | 대 | output/{winner, dashboard, report} |
| 10 | `lmtune#PLUG` (옵션) | leaf | S1 + R0 + SS | 소 | LLMOracleSampler / PostgresStore stub — plug-in 패턴 시연 |

**Critical path**: WS → SS-rec → SS → R0 → S1 → OD → OUT (8 PR, ~8주). A1+CS 는 별도 라인 병렬.

## 4 Component (단순화된 view)

| Component | 역할 (한 줄) | ABC | 디렉토리 |
|:---|:---|:---|:---|
| **Tuner** | "다음에 뭘 실험할까" — 알고리즘 | Sampler, Pruner | `src/lmtune/tuner/` |
| **Driver** | "1 trial 의 routing 책임" — main loop | (없음, 단일 클래스) | `src/lmtune/orchestrate/driver.py` |
| **Backend** | "trial 1개 실행" | TrialBackend | `src/lmtune/orchestrate/backend.py` |
| **Launcher** | "endpoint 띄우고 내림" | DeploymentAdapter | `src/lmtune/deploy/` |
| **Benchmark+Anal** | "endpoint 측정 + 집계" | WorkloadStream, BenchmarkRunner | `src/lmtune/workload/`, `src/lmtune/runners/` |
| **Storage** | "single shared resource" | WorkloadStore, ArtifactStore | `src/lmtune/storage/` |
| **Output** | "winner / dashboard / report" | (Visualizer) | `src/lmtune/output/` |

의존 방향: Driver → Tuner / Backend (둘 다 호출). Tuner ↔ Backend 직접 의존 없음. Storage 는 hub — 모두가 read/write.

## Tuning loop — 데이터 흐름 (요약)

```
Inner loop (1 trial):
   Driver.ask()      → Tuner.Sampler  (in-memory archive 보고 params 결정)
   Driver.submit()   → Backend        (K8sJob/ProcessPool 으로 trial 1개 실행)
       Backend → Launcher.apply()    (helmfile redeploy / vllm restart)
       Backend → BenchmarkRunner.run() (workload 부하 + raw 측정)
       BenchmarkResult → Storage (writer_queue, single writer thread)
   Driver.tell(score) → Tuner.Sampler (가장 최근 1건만 — archive 는 Storage 가 owner)

Outer loop (시각화/분석):
   Output/Visualizer ─readonly─→ Storage (DuckDB/Postgres/...)
   Analysis           ─readonly─→ Storage
```

**Storage 가 영속 archive owner**. Tuner 의 in-memory archive 는 sampler refit 용 cache 일 뿐. 시각화는 Storage 만 봄, Tuner 거치지 않음.

## Contract 6종

| # | Contract | apiVersion | Master | lmtune 의 역할 |
|:--|:---|:---|:---|:---|
| 1 | WorkloadSpec | `workloads/v1alpha1` | **lm-workloads** | mirror (re-export) + Provider |
| 2 | ClusterSpec | `ariadne/cluster/v1alpha1` | **ariadne** | mirror (re-export) + Provider |
| 3 | EndpointSpec | `lmtune/endpoint/v1alpha1` | lmtune | own |
| 4 | ProfileSpec | `lmtune/profile/v1alpha1` | lmtune | own |
| 5 | SearchSpace | `lmtune/search/v1alpha1` | lmtune | own |
| 6 | BenchmarkResult | `lmtune/result/v1alpha1` | lmtune | own (controller input) |

**RecordSpec/QuerySpec** (Storage 전용, Pydantic 만) 도 contracts/ 안에서 SS-rec PR 에서 추가.

## 변경 거버넌스

- workloads/ariadne master 가 v 올리면 → lmtune mirror 는 같은 PR 에서 동기. drift 감지는 CI 에서 schema diff 검사.
- lmtune own contract 는 단독 관리. 추가 only, drop 은 v 분리.
- 본 문서 변경은 PR description 에 변경 사유 명시 + CHANGELOG entry.

## CHANGELOG

- 2026-05-06: 초안. WS / A1 / CS / SS-rec / SS / R0 / S1 / OD / OUT / PLUG 10 PR sequence 확정.
- 2026-05-06 (afternoon): leaf-up 진행. 머지된 PR 누적:
  - **#37 WS** — WorkloadSpec contract + LiteralProvider + LMWorkloadsProvider
  - **#38 CS docs** — ClusterSpec contract design + GPU/NPU 3-tier 모델 (docs only; ariadne 본체 코드 의존)
  - **#39 SS-rec** — RecordSpec + QuerySpec Pydantic + lmtune contracts CLI (`dump-schema`, `validate-record`)
  - **#43 R0** — BenchmarkResult contract + `to_records()` + `validate-result` CLI
  - **#44 R0-rt** — RunArtifact → BenchmarkResult 변환 helper (`runners/result_emit.py`)
  - **#45 SS** — ArtifactStore ABC + InMemoryArtifactStore + DuckDBArtifactStore
  - **#46 S1** — Tuner Sampler/Pruner ABC + OptunaAdapter (`src/lmtune/tuner/`)
  - **#47 S1-native** — NativeRandom/LHC/TPE 가 tuner.Sampler 구현체로 어댑트
  - **#48 R0-bridge** — `bench run` 이 raw_dir/<run_id>/result.json 으로 BenchmarkResult 덤프 (production wire-up)
  - **#49 S1-factory** — `tuner.factory.make_sampler` 통합 dispatch
  - **#50 R0-roundtrip** — DuckDB PK-less 테이블 INSERT 분리 + BenchmarkResult round-trip 통합 검증
  - **#51 SS-tz** — DuckDBArtifactStore datetime UTC round-trip (tz-aware 일관성)
  - **#53 SS+Local** — LocalArtifactStore (JSONL per kind, primary_key dedup) — 3-way (Mem/Local/Duck) query 동등성 검증
  - **#54 SS-cli** — `bench run` 이 raw_dir/<run_id>/records/ 에 LocalArtifactStore mirror 적재 (result.json + records 양립)
  - **#55 SS-cli2** — `lmtune contracts records-from-result` — BenchmarkResult JSON → records/<kind>.jsonl 변환 도구 (archive/migration)
  - **#56 SS-migrate** — `lmtune storage migrate` — local↔duckdb backend 무관 일괄 복사 (`list-backends` 포함, PLUG 진입로)
  - **#58 PLUG** — PostgresArtifactStore stub: ArtifactStore ABC plug-in 패턴 실증 (psycopg optional, ImportError → typer.BadParameter 변환, _BACKENDS 자동 합류)
  - **#59 PLUG** — LLMOracleSampler stub: Sampler ABC plug-in 패턴 실증 (anthropic optional, `tuner.factory.make_sampler('llm_oracle')` dispatch)
  - **#60 PLUG-deps** — `[postgres]` / `[agent]` extras 등록 + drift test (ImportError 메시지의 install command 가 pyproject 와 항상 일치)
  - **#62 docs(README)** — PLUG/Storage 섹션 추가 + `pip install lmtune[...]` 정합화 (rename drift 차단)
  - **#63 SS-info** — `lmtune storage info` — record kind 별 count 보고 (--json 으로 monitoring loop 친화)
  - **#64 cli-version** — `lmtune --version` / `-V` flag + drift test (`__version__` ↔ pyproject)
  - **#65 docs(arch)** — `PLUG_PATTERN.md` 5단계 + 체크리스트 (외부 기여자가 새 backend/sampler 추가하는 step-by-step recipe)
  - **#66 SS-validate** — `lmtune storage validate` — record schema validity 검증 (외부 archive 신뢰성 + CI drift 가드)
  - **#68 SS-diff** — `lmtune storage diff` — 두 store 의 record 차이 (`only_left` / `only_right` / `mismatched`) 보고
  - 부수: **#41 fix(ci)** — vllm_restart venv fallback + ruff format 일괄 적용 (post-rename)
  - 부수: **#52 docs** — REFACTOR-PLAN CHANGELOG 13 PR 누적 정리
  - 부수: **#57 docs** — REFACTOR-PLAN CHANGELOG SS-leaf 4종 누적
  - 부수: **#61 docs** — REFACTOR-PLAN CHANGELOG PLUG 3종 (#58-#60) 누적
  - 부수: **#67 docs** — README Storage 운영 도구 4종 + CHANGELOG #65/#66 누적
  - 부수: **#69 docs** — README storage diff 명령 노출 + CHANGELOG #67/#68 누적
  - **#70 S1-pruner-bridge** — `tuner.factory.make_pruner` ABC 어댑터 결합 + `_OPTUNA_PRUNER_KINDS` drift 가드 (SH / Hyperband 가 Pruner ABC 위에서 통일 dispatch)
  - **#72 docs(arch)** — PLUG_PATTERN.md 에 Pruner axis 항목 + 'Pruner 의 경우' 5단계 + 세 reference impl 섹션 (#70/#71)
  - **#73 PLUG-NativeMedianPruner** — Pruner axis 의 첫 native impl. `_NATIVE_PRUNER_KINDS = {"median_native"}` 신규 + `tuner.factory.make_pruner` 분기 — Optuna 위임 없이 stdlib statistics 만으로 cross-trial median 기반 prune
  - **#74 docs(README)** — PLUG 표 두 추상 → 세 추상 (Pruner row 추가) + Native MedianPruner 노출
  - **#75 PLUG-NativePercentilePruner** — Pruner axis 의 두 번째 native impl. `_NATIVE_PRUNER_KINDS` 에 `percentile_native` 합류 — 임의 percentile (0.5 = median 동치, 수학적 검증 포함). PLUG 패턴이 한 axis 안에서 2회 시연됨
  - **#76 cli-tuner-list** — `lmtune tuner list-{samplers,pruners}` 신규. PLUG 화이트리스트 (`_NATIVE_STRATEGIES`, `_LLM_STRATEGIES`, `_NATIVE_PRUNER_KINDS`, `_OPTUNA_PRUNER_KINDS`) 를 단일 진실원으로 노출 — `lmtune storage list-backends` 와 동등한 PLUG 가시성
  - **#77 cli-tuner-describe** — `lmtune tuner describe <kind>` 신규. `inspect.signature` 로 native + llm 클래스의 hyperparameter introspect, Optuna 빌트인은 reference URL fallback. 외부 기여자가 새 PLUG 추가 시 자동으로 describe 가능
- **Storage 운영 도구 5종 완비** (`migrate` / `info` / `validate` / `diff` / `list-backends`) — 모두 ArtifactStore ABC 만 사용. backend 추가 시 코드 수정 0.
- **Tuner 메타 도구 3종** (`lmtune tuner list-samplers` / `list-pruners` / `describe <kind>`) — PLUG 합류 즉시 자동 노출 + introspect (drift 가드 테스트 포함).
- **Pruner axis PLUG 합류** — Optuna (SH/Hyperband) + Native (Median + Percentile) 네 빌트인. ASHA / 외부 SDK pruner 추가 시 1+1 줄 변경.
- 미진입: OD (Orchestrator Driver/Backend 분리), OUT (output module). PLUG 패턴은 #58/#59/#60/#73/#75 으로 세 축 (Storage + Sampler + Pruner) 모두 시연. PLUG 추가 절차는 [`PLUG_PATTERN.md`](./PLUG_PATTERN.md) 의 5단계 + 체크리스트 참조.
