# AGENTS.md

> 본 파일은 **LLM 에이전트** (autoresearch / Claude Code / 외부 LLM 코드 어시스턴트) 가 본 repo 의 코드·YAML 을 안전하게 수정하기 위한 가드레일.
> 사람 기여자도 동일 규약을 따른다.
>
> 패턴 출처: [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) `AGENTS.md`. 우리 프로젝트의 컨텍스트(autotuning + 사용자 자기 인프라 적용) 에 맞게 조정.

## Core principles

1. **이 repo 는 LLM-free 가 1st-class.** 사용자가 받아 실행하는 모든 핵심 경로 (autotune, dashboard, winner export, B-track) 는 LLM 콜 0회로 동작해야 한다. LLM 가이드는 optional `[agent]` extra 만.
2. **Schema 는 코드의 source of truth.** Pydantic / dataclass 의 `extra='forbid'` 또는 동등 제약을 유지한다. 에이전트가 새 필드를 도입할 때는 **모델 정의를 먼저 수정** 하고 YAML 을 거기에 맞춘다 — 반대 방향 금지.
3. **YAML 키는 `kebab-case` 또는 `snake_case` 일관성**을 유지한다. 한 파일 안에서 두 컨벤션 혼용 금지. 기존 파일을 수정할 때는 그 파일의 컨벤션을 따른다.
4. **`b200/perf-changelog.yaml` 은 append-only.** 가운데에 entry 삽입 금지. 새 entry 는 항상 파일 끝에 추가한다.
5. **archive 디렉토리 (`data/archive/`) 는 read-only.** 에이전트는 절대 수정·삭제하지 않는다.
6. **`.claude/`, `.git/`, `data/` 는 에이전트가 직접 수정 X.** `data/` 의 DuckDB 는 `bench` CLI 를 통해서만 갱신.
7. **destructive 명령 금지** — `git reset --hard`, `git push --force`, `rm -rf data/`, `kubectl delete namespace ...` 등은 사용자 명시 승인 없이 실행하지 않는다.

## Where agents may modify

| 영역 | 자유도 | 가드레일 |
|:---|:---|:---|
| `src/bench/**/*.py` | ✅ 자유 (테스트 통과 의무) | 새 모듈은 `tests/` 에 단위 테스트 동반 |
| `configs/**/*.yaml` | ✅ 새 파일 추가 | 기존 파일 수정 시 헤더 주석에 변경 사유 1줄 |
| `b200/search-spaces/*.yaml` | ✅ axis 추가 | `cost_tier` 필수, `active_if` 게이팅 명시 |
| `b200/helmfile/**` | ⚠️ 신중 | peer repo `agentic/llm-distributed-inference` 의 commit SHA 를 헤더에 기록 |
| `b200/perf-changelog.yaml` | ✅ append-only | 위 §4 |
| `tests/**` | ✅ 자유 | mock 보다 fixture DuckDB 우선 |
| `docs/`, `README.md` | ✅ 자유 | top-down 구조 (problem→scenario→design→architecture) 유지 |
| `data/`, `data/archive/` | ❌ 금지 | bench CLI 또는 archive 도구를 통해서만 |
| `.claude/`, `.git/` | ❌ 금지 | — |

## YAML 파일 작성 규약

- `apiVersion: bench/<resource>/<version>` 헤더 필수 (예: `lmtune/search/v1alpha1`)
- 파일 상단에 다음 3줄 주석:
  ```yaml
  # 목적: 이 파일이 무엇을 정의하는지 1줄
  # phase: 어느 phase 에서 활용되는지 (예: W-Local, B3, B6.4)
  # 출처: 외부 reference 가 있으면 commit SHA 또는 URL
  ```
- search-space 의 `axes:` 항목은 1 줄 표현 우선:
  ```yaml
  max_num_seqs: {type: categorical, values: [16, 32, 64, 128, 256], cost_tier: 4}
  ```
- 사람이 읽을 수 있게 `effect:` 필드로 axis 의미 1줄 부연 (선택).

## Commit / PR 규약

- 커밋 메시지 prefix: `<phase>: <subject>` (예: `B3: add PCP/DCP axes`, `W: dashboard study detail page`)
- 1 PR = 1 study 또는 1 logical change. autotune 결과 archive 는 별도 commit 으로 분리.
- 자동 생성된 결과물 (`b200/results/<study>/winner/`, `b200/dashboards/`) 은 commit 하되 PR 본문에 1줄 요약 ("top-1 score X, baseline 대비 ±Y%").
- `Co-Authored-By: <agent-name>` 라인 권장 — autoresearch / Claude Code 가 작성한 commit 은 추적 가능하게.

## perf-changelog.yaml 규약 (InferenceX 패턴)

`b200/perf-changelog.yaml` 은 vLLM/llm-d/SGLang/TRT-LLM 의 새 PR 머지가 우리 baseline 에 미친 영향을 시계열로 기록. continuous loop (Phase B5) 가 매 cycle 에 watch.

```yaml
- timestamp: 2026-05-03
  config-keys:
    - "qwen25-1.5b-pd-*"
    - "*-tp1-*"
  description:
    - "vLLM 0.10.2: chunked-prefill default 가 16384 로 상향, TTFT p99 12% 개선"
  pr-link: "https://github.com/vllm-project/vllm/pull/12345"
  evals-only: false
```

- `config-keys` 는 glob. `*` 는 axis 값 와일드카드.
- `description` 은 bullet list. **변경 사항 + 측정된 영향** 을 한 줄씩.
- `pr-link` 는 외부 PR URL (vLLM/llm-d upstream 등).
- `evals-only: true` 는 throughput 측정 변경 없이 quality probe 만 영향받는 경우.
- 가운데 삽입 금지 — 새 entry 는 파일 끝에 append.

## When in doubt

1. 기존 코드의 패턴을 모방한다 (특히 `src/bench/search/`, `src/bench/deploy/`).
2. 모르는 영역은 사용자에게 묻는다 — 추측해서 schema 를 확장하지 않는다.
3. 테스트가 먼저, 코드가 나중. 새 동작은 `tests/` 의 fixture DuckDB 위에서 먼저 실패하는 테스트로 정의.
4. 기존 테스트가 깨지면 먼저 원인을 분석한다 — 그냥 expected 값을 수정해서 맞추지 않는다.
