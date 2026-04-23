# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트

**LLM endpoint 벤치마크 자동화 시스템.** OpenAI-compatible endpoint(vLLM / llm-d / SGLang)나 Anthropic API에 코딩 에이전트 워크로드를 재현하여 TTFT · ITL · TPOT · throughput · goodput 을 측정·저장·분석·시각화하고 이상치를 탐지합니다. 전체 로드맵은 `/home/jinmoo/.claude/plans/async-cooking-cat.md`, 구현 상태는 README.md 참조.

## 개발 명령

```bash
pip install -e ".[dev]"        # editable + dev extras
bench --help                    # Typer CLI entrypoint (src/bench/cli.py)
pytest                          # tests/
ruff check src tests            # lint
ruff format src tests           # format
```

`bench` CLI 서브커맨드: `run` · `sweep` · `report` · `compare` · `detect` · `ls`.

## 아키텍처

데이터 흐름은 고정 파이프라인입니다:

```
Profile YAML + Endpoint YAML
       │
       ▼
Runner (AIPerf / vLLM bench / GuideLLM subprocess) → Endpoint under test
       │                                           │
       ▼                                           ▼
Raw artifact (JSON/CSV, data/raw/)        Collectors (Prom /metrics, request log)
       └──────────────────┬────────────────────────┘
                          ▼
                  DuckDB (data/db/bench.duckdb) + Parquet
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
      Analysis       Visualization      Detector
   (p50/p95/p99,     (matplotlib plots,  (rules → TS → LLM
    goodput, diff)    Markdown report,    단계별 확장)
                      Grafana JSON)
```

- **Endpoint는 HW 비종속.** `configs/endpoints/*.yaml` 은 `{slug, url, model, api_key_env, notes}` 만 담는다. A5000 / H200 / REBEL 같은 HW 구성은 `slug` + 자유 `notes` 로만 태깅되며 자동화 로직에서는 참조하지 않는다.
- **Profile YAML 은 `docs/` v3 문서의 기계화본이다.** 새 profile 을 추가할 때는 해당 파라미터의 출처를 `references: [S-N]` 필드로 기록해서 v3 의 신뢰도 체계를 보존한다.
- **AIPerf 실행 모드 2종 혼용 금지** — `docs/coding_agent_benchmark_plan_v3.md` §2.3 참고. `concurrency` 모드 (1-turn stateless, `--concurrency` + `--request-count`) 와 `user_centric` 모드 (multi-turn, `--num-users` + `--user-centric-rate` + `--conversation-*`) 는 프로파일의 `mode` 필드로 택일한다. Runner 는 모드별 CLI args 조립만 담당한다.
- **AIPerf 파라미터 이름 버전 차이 흡수.** `--conversation-num` vs `--num-conversations` 등은 릴리즈 간 변경 이력이 있어 runner 가 `aiperf profile --help` 탐지 또는 환경변수로 보정하게 구현한다. 직접 YAML 에 CLI flag 를 박지 말 것.
- **Runner 선택 기준**: AIPerf = multi-turn·user-centric 기본값, vLLM bench = v3 문서 포팅 호환, GuideLLM = `rate_type: sweep`·goodput 탐색. Profile 의 `runner` 필드로 택일하며 세 runner 는 동일한 `requests` / `metrics` 스키마로 정규화된다.

## 데이터/결과 관리

- Raw artifact(AIPerf JSON, vLLM bench 출력)는 `data/raw/<run_id>/` 에 보존하고, 가공 메트릭/요청 행만 DuckDB 에 적재한다. `.gitignore` 에 포함되어 있다.
- Run 식별자는 ULID. `runs` 테이블에 `profile_yaml`, `endpoint_meta`, `tool_versions` 스냅샷을 저장해 후속 비교 시 재구성 가능하게 한다.
- 스키마는 `src/bench/storage/schema.sql` 이 정본. 변경 시 마이그레이션이 아닌 신규 테이블/뷰를 추가한다 (초기 단계라 backward compat 유지 대상 아님).

## 확장 포인트

YAML 스키마는 `apiVersion: bench/v1alpha1` 기반으로 다음 축을 비파괴적으로 확장한다.

- **Workload source**: `workload.source: synthetic | dataset | trace` 디스크리미네이터로 Union 분기. 기본 synthetic. dataset/trace 는 스키마만 예약되어 있고 로더는 추후 구현.
- **Endpoint deployment**: `deployment: {engine, parallelism, engine_args}` 블록으로 같은 URL 의 vLLM 서빙 구성 축을 식별·비교. 자동화 로직은 이 필드를 읽지 않고 runs 메타로만 저장된다.
- **Runner overrides**: `runner_overrides: {aiperf: {...}, guidellm: {...}}` 는 pydantic 스키마 수정 없이 도구별 임의 flag 를 CLI 뒤에 pass-through. boolean flag, 값 있는 flag 모두 지원.
- **SLO checks**: 기존 flat 필드(`ttft_p99_ms` 등) 와 병행해 `slo.checks: [{metric, p, op, value, severity, label}]` 로 임의 assertion 표현. Detector 는 `SLOSpec.resolved_checks()` 로 둘을 합쳐 읽는다.

새 runner/데이터셋 로더/이상탐지 규칙을 추가할 때 **코어 pydantic 모델을 건드리지 않고** 이 네 축으로 먼저 시도한다.

## 규약

- API key 는 `api_key_env` 로 환경변수만 참조. 파일에 평문 저장 금지.
- 문서 v3 의 `[S-N]` 은 공식 출처 확인 수치, `(추정)` 은 비출처. 새 참조를 `[S-N]` 으로 승격하려면 웹에서 1차 출처 확인이 필요하다 (v3.4 신뢰도 체계).
- Python indent 는 4 spaces (PEP8). YAML indent 2 spaces. 불필요한 docstring / type 주석은 덧붙이지 않는다.
