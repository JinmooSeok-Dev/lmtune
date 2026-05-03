# Phase S6 — autoresearch ↔ lmtune search ask/tell 라이브 검증 (minikube + RTX)

> 본 문서는 `b200/docs/ANALYSIS_template.md` 의 첫 실제 적용 사례.
> Phase S6 는 B200 host 가 free 하지 않아도 진행 가능한 환경-독립 phase 였으므로,
> minikube 위에 구축된 llm-d P/D Qwen2.5-1.5B endpoint 와 로컬 RTX 5060 Ti 환경에서 라이브 E2E 를 수행했다.

## 1. 측정 컨텍스트

- **HW**: RTX 3060 12GB (decode) + RTX 5060 Ti 16GB (prefill) — 동일 host, GPU0/GPU1 분리
- **K8s**: minikube v1.32 (single-node, RuntimeClass=nvidia, NVIDIA Device Plugin)
- **llm-d 배포**: phase2 P/D Qwen2.5-1.5B helmfile (peer repo `llm-distributed-inference`)
  - prefill pod: `ms-pd-qwen25-llm-d-modelservice-prefill-8cd7ccc84-g8fqw` (8h uptime)
  - decode pod: `ms-pd-qwen25-llm-d-modelservice-decode-7df5dfc545-9lpms`
  - GAIE EPP, infra-gateway 정상 동작
  - NIXL TCP fallback (intra-host RDMA 불필요)
- **vLLM**: 0.5.1 (helmfile-baked engine_args, `enforce_eager=true`, `max_num_seqs=32`, `max_model_len=4096`)
- **시점**: 2026-04-30 00:20 ~ 00:42 KST
- **study_id**: `st-01KQCX8F8A9M620HDX8GGDCMWS`
- **strategy**: TPE (Optuna multivariate, group=true)
- **search space**: `configs/search/spaces/vllm_engine_args_tier1.yaml` (6 활성 axis — tp/pp/dp 는 `active_if: {adapter: llmd-k8s}` 로 비활성)
- **workload**: `autotune-short` (256/128) + `autotune-medium` (1024/256), guidellm runner, REPEATS=2
- **SLO**: `ttft.p99 ≤ 500ms`, `e2e.p99 ≤ 30s`
- **Composite score**: `score = throughput_tok.avg × max(0, 1 − ttft_p99 / 1000)` per-workload, 합산

> **중요한 단서**: minikube llm-d 의 engine_args 는 helmfile values 에 baked 되어 있어, 본 검증의 **trial 별 params 는 실제 vLLM 서버에 적용되지 않았다**. 즉 본 study 는 **autotune 결과의 정량적 비교가 목적이 아니라, ask/tell 메시징 통합과 study 누적/score 계산/SLO 게이트가 정확히 동작하는지를 입증** 하는 것이다 (Track A acceptance 의 1, 2, 3 항목). 진짜 engine_args autotune 은 Track B-I 의 B200 환경에서 시작된다.

## 2. 결과 — 수치 + 시각화

### Top trials (4 cycles)

| seq | trial_id | status | score | dur(s) | params 요약 |
|:---:|:---|:---:|---:|---:|:---|
| 1 | `tr-01KQCX909M…` | **pruned** | 0.00 | 11.1 | (BENCH_BIN PATH 미설정으로 guidellm 미발견 — ask/tell 흐름 자체는 정상) |
| 2 | `tr-01KQCXC6QW…` | completed | 1371.22 | 179.6 | max_num_seqs=256, prefix=false, chunked=false, mem=0.819, len=8192, kv=auto |
| 3 | `tr-01KQCY3MDQ…` | **completed (top-1)** | **1390.94** | 178.5 | max_num_seqs=16, prefix=true, chunked=true, mem=0.861, len=8192, kv=fp8 |
| 4 | `tr-01KQCY98CH…` | completed | 1388.71 | 184.0 | max_num_seqs=16, prefix=false, chunked=true, mem=0.880, len=4096, kv=fp8 |

### Workload-level metrics (top-1 = trial 3)

| workload | throughput_tok.avg | ttft.p99 | e2e.p99 | per-WL score | SLO |
|:---|---:|---:|---:|---:|:---:|
| short  (256/128)  | 796.86 tok/s | 69.87 ms | 1.29 s | 741.18 | ✅ |
| medium (1024/256) | 706.83 tok/s | 80.74 ms | 2.73 s | 649.75 | ✅ |
| **합산** |  |  |  | **1390.94** | ✅ |

### 누적 동작 (status output 발췌)

```
trials: total=4  pruned=1  completed=3
Top-1: tr-01KQCY3MDQ…  score=1390.94
Top-2: tr-01KQCY98CH…  score=1388.71
Top-3: tr-01KQCXC6QW…  score=1371.22
```

`lmtune search status` 가 pruned 1, completed 3 을 정확히 분류, top-K 정렬, params JSON 직렬화까지 일관되게 출력.

### Artifacts

- `trials_with_metrics.parquet` — DuckDB `trials` × `trial_metrics` 조인 export (4.2 KB)

## 3. 원인 분석 — 왜 이 결과가 나왔는가

### 3.1 첫 cycle 의 pruned 원인

trial 1 은 score=0 으로 즉시 pruned. 직접 분석:

- `lmtune_score.py` 가 `subprocess.run(["bench", ...], ...)` 를 호출하지만 `bench` 가 PATH 에 없음 → `FileNotFoundError`
- 즉, **autoresearch.sh 는 venv `bench` 를 BENCH_BIN 으로 export 하지만 그 자식인 lmtune_score.py 의 자식 subprocess 에서는 PATH 가 그대로 시스템 PATH 임**
- `bench` 자체가 venv 에서 호출되어도, 그 내부에서 `subprocess.run(["guidellm", ...])` 가 다시 PATH 의존이라 **guidellm` not found` 로 실패**
- **수정**: cycle 2 부터 `PATH=.venv/bin:$PATH ./autoresearch.sh` 로 prefix 한 결과 즉시 정상 동작
- 본 cycle 은 의도와 다르게 pruned 됐지만, 그것 자체가 **"score=0 → status='pruned'" 분기가 정확히 동작함을 입증** 하는 우발적 회귀 테스트가 됨

→ **action item (적용 완료)**: `autoresearch.sh` 가 venv 발견 시 PATH 에 `.venv/bin` 을 prepend 하도록 수정. 회귀 cycle 5 (`tr-01KQCYKCH77QMJSXJY2ZZN3VTM`) 로 PATH prefix 없이 정상 동작 확인. 동일 PATH 회귀가 `tests/test_s6_ask_tell.py` 의 fixture 에도 있어 `_bench` 가 venv `bench` 를 절대경로로 찾도록 동기 수정. 전체 180 pytest pass.

### 3.2 cycle 2~4 에서 score 가 거의 동일한 이유

위 제약대로 trial 별 params 는 **실제 vLLM 서버에 반영되지 않음**. 따라서:

- short workload: 모두 throughput ≈ 795~797 tok/s, ttft.p99 ≈ 67~72 ms (CV ≈ 0.4%)
- medium workload: throughput 705~707 tok/s, ttft.p99 80~111 ms (CV ≈ 12% — REPEATS=2 의 noise 한계)
- 즉 측정은 **endpoint 의 정상 동작 baseline 을 3회 반복** 했고, score 의 작은 등락 (1371 → 1390 → 1388) 은 **same-config noise** 에 해당

→ **이것이 본 검증에 오히려 valuable**:
- TPE sampler 는 noise 를 score 차이로 받아도 같은 동작 패턴 (top-K 정렬, 누적 ask 의 기록 반영) 을 정확히 수행
- ask 가 발급한 각 trial 의 unique trial_id 가 충돌 없이 DB 에 적재
- **REPEATS=2 + 256/128 짧은 워크로드의 CV 가 ~0.4% 수준** 으로 안정적이어서, B200 phase 진입 시 이 endpoint 에서 ask/tell 통합의 회귀 (regression) smoke 로 즉시 재사용 가능

### 3.3 시스템 시점

- decode pod 는 RTX 3060 GPU0 점유 (7756 MiB), prefill 은 RTX 5060 Ti GPU1 점유 (13971 MiB), 양쪽 GPU 모두 측정 idle 시 0% util — 측정 동안에만 burst
- minikube 단일 노드라 양 GPU 가 동일 host PCIe 위에 있고 NIXL 은 TCP fallback. 본 측정은 **인터노드/인트라노드 RDMA 효과를 보지 않는 케이스** (B200 phase 에서 의미 있는 비교).

## 4. 의의 — 이 측정이 무엇을 입증/반증하는가

본 검증은 plan `async-cooking-cat.md` 의 **Track A — Phase S6** 의 acceptance 항목 3개를 모두 만족한다:

1. **"autoresearch.sh 가 LLM 자체 가설 생성 코드 0 줄, 모든 추천이 `lmtune search ask` 에서 옴"** — 4 cycle 모두 ask 가 발급한 params 만 사용. autoresearch agent 의 자체 추론 분기 0회.
2. **"autoresearch.jsonl 매 entry 가 study_id + trial_id 매핑 키 보유 → DuckDB 와 cross-reference 가능"** — `BENCH_STUDY` + `BENCH_TRIAL` 환경변수 통한 binding 동작. DuckDB `trials` 테이블에 4 행 모두 study_id 외래키와 함께 적재.
3. **"검증 1, 2, 3 PASS"** — RTX 5060 Ti 로컬 cycle (3.1 의 PATH 회귀 케이스 포함) + minikube llm-d cycle (cycle 2~4) + smoke 통합 테스트 (`tests/test_s6_ask_tell.py` 4 PASS, 별도 commit).

### 더 광범위한 의의

- **LLM domain knowledge × Optuna 통계 효율 결합** 의 인터페이스가 바깥 (autoresearch agent) 측 코드 변경 없이 동작함을 입증. 외부 에이전트는 `lmtune search ask --json` + `lmtune search tell --metrics-json` 두 호출만 알면 study 에 참여 가능.
- **headless mode 와의 양립** 도 동일 study 위에서 가능 — `lmtune search start --backend k8s-job --workers N` 으로 background batch trial 을 동시에 띄울 수 있음 (S3 산출물). 본 cycle 은 외부 LLM mode 만 실증했으나, study schema 가 양 모드를 차별 없이 받음을 코드 검수로 확인.
- **Track B-III (Continuous loop) 의 prerequisite** — B5 의 `loop.sh` 는 ask/tell 만 호출하면 되므로, 본 PR 이 그 루프를 위한 가장 얇은 protocol 을 확정.

### 이론·문헌 비교

- openshift-psap/auto-tuning-vllm — Optuna + GuideLLM 동일 스택. 차이: Ray vs (우리) K8sJob/ProcessPool. 본 S6 는 그쪽 stack 의 외부 ask/tell 인터페이스도 동일 호출 흐름으로 수용 가능.
- InferenceMAX (SemiAnalysis) — 일별 자동 회귀, but 외부 lab. 우리 S6+B5 는 self-hosted 이며 **외부 LLM 에이전트 → 통계 sampler** 링크가 명시됨.
- cfregly Ch19 (kernel autotune + RL) — application-level autoresearch ↔ kernel-level RL 의 hybrid 가 본 ask/tell 인터페이스 위에서 가능 (장기 vision).

## 5. 다음 가설 / 후속 실험

1. ~~**`autoresearch.sh` PATH 자동 prepend**~~ — 본 PR 에서 적용 완료 (cycle 5 회귀 검증 + pytest 180 PASS).
2. **B200 phase 진입 시 첫 cycle 회귀 smoke** — 본 study 의 cycle 2 ~ 4 결과를 baseline 으로 두고, B0 통과 후 같은 short+medium workload 로 1 cycle 만 실행. score 가 ±5% 안에 들어오면 ask/tell 통합 회귀 없음.
3. **REPEATS=3 + CV gate** 검증 — 본 cycle 들은 REPEATS=2 였음. 진짜 autotune 시 N=3, CV≥0.10 → N=5 확장 분기를 별도 트리거 (`scripts/lmtune_score.py` 의 fallback 경로).
4. **multi-objective 통합** — 본 study 는 single objective (`total_score`). NSGA-II + multi-obj study 위에서 ask/tell 이 같은 protocol 로 trial 을 받을 수 있는지 (S5 산출물과의 결합) 별도 smoke.
5. **`enable_thinking=false` 검증** — Qwen3 계열 도입 시 profile YAML 의 `extra_body` axis 추가가 ask/tell 에 자연스럽게 흡수되는지 (E1~E2 의 Workload Spec 확장 테스트).

## RECIPES.md 후보 entry — (없음)

본 study 는 통합 검증 목적이라 winning recipe 후보를 만들지 않는다. 첫 실 recipe 는 Track B-I 의 B0 smoke 후 B1 (well-lit path baseline) 에서 등록 예정.

---

**관련 산출물**:
- 코드: `src/bench/cli_search.py::cmd_ask`, `cmd_tell` (Phase S6 신규)
- 스크립트: `autoresearch.sh` (USE_BENCH_SEARCH=1 분기, Phase S6 추가)
- 테스트: `tests/test_s6_ask_tell.py` (4 tests, all passing)
- 문서: `docs/autotune_loop.md` (autotune 사이클 도식, B-I 산출), `b200/docs/ANALYSIS_template.md` (본 ANALYSIS 의 모태)
