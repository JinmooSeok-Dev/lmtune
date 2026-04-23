# bench — Coding Agent 워크로드 기반 LLM Endpoint 벤치마크 자동화

OpenAI-compatible LLM endpoint (vLLM / llm-d / SGLang 등)와 Anthropic API를 대상으로, **코딩 에이전트 워크로드를 재현**하여 추론 성능(TTFT, ITL, TPOT, throughput, goodput)을 측정·저장·분석·시각화하고 이상치를 탐지하는 자동화 시스템입니다.

## 범위

- **입력**: `configs/profiles/*.yaml` (워크로드 정의) + `configs/endpoints/*.yaml` (대상 endpoint)
- **실행**: AIPerf (NVIDIA) / vLLM `benchmark_serving`·`benchmark_serving_multi_turn` / GuideLLM 중 profile 의 `runner` 필드로 선택
- **저장**: DuckDB (`data/db/bench.duckdb`) + Parquet (raw trace)
- **시각화**: Grafana 실시간 + Markdown 리포트
- **이상탐지**: 규칙 기반(SLO/회귀/IQR) → 시계열 → LLM-as-judge (단계별 확장)

Endpoint의 HW(A5000/H200/REBEL 등)는 시스템 로직과 무관합니다. `endpoint.slug` + `notes`로만 태깅합니다.

## 배경 문서

- `docs/coding_agent_benchmark_plan_v3.md` — Profile A/B/C/C-agg/Goodput 실행 계획 (v3)
- `docs/워크로드_데이터셋_가이드_v3.md` — 17개 워크로드 케이스 × 데이터셋 카탈로그 (v3)

Profile YAML은 위 문서의 파라미터를 기계화한 것입니다. 각 profile의 `references: [S-N]` 필드로 원문 출처 추적이 유지됩니다.

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,runners]"        # guidellm 포함
```

외부 runner (profile `runner` 필드로 선택, 필요할 때만 설치):

- `guidellm` — `[runners]` extra 로 pip 설치. rate sweep·goodput 탐색.
  ([vllm-project/guidellm](https://github.com/vllm-project/guidellm))
- `aiperf` — NVIDIA AIPerf, git clone 설치 권장. multi-turn user-centric 모드 기본.
  ([ai-dynamo/aiperf](https://github.com/ai-dynamo/aiperf))
- vLLM repo 의 `benchmarks/benchmark_serving.py`·`benchmark_serving_multi_turn.py`
  → 환경변수 `VLLM_REPO=/path/to/vllm` 로 지정.

## Quickstart — 실제 endpoint 에서 smoke test

Int4 양자화 모델(예: Qwen2.5-Coder-3B/7B/14B GPTQ-Int4, Qwen3-30B-A3B GPTQ-Int4) 이
이미 서빙 중인 경우:

```bash
# 1) runners extra 설치 (GuideLLM)
pip install -e ".[dev,runners]"

# 2) 실제 endpoint 에 맞춰 수정
#    - url, model, (필요 시) api_key_env
vim configs/endpoints/llmd_k8s.yaml

# 3) 명령 조립 확인 (실제 호출 없이)
bench run \
  --profile configs/profiles/smoke_guidellm.yaml \
  --endpoint configs/endpoints/llmd_k8s.yaml \
  --dry-run

# 4) 실제 실행 — 10 요청 × concurrency 2, 수십 초 내 완료
bench run \
  --profile configs/profiles/smoke_guidellm.yaml \
  --endpoint configs/endpoints/llmd_k8s.yaml

# 5) 결과 확인
bench ls -n 5
bench report <run_id>                          # Markdown 리포트 + 플롯
bench detect <run_id> \
  --profile configs/profiles/smoke_guidellm.yaml  # SLO/IQR 검사
```

검증 완료 후 `configs/profiles/stage{1,2,3}_*/` 의 실제 Profile A/B/C 계열로 확장.
Multi-turn 측정에는 AIPerf runner (별도 설치) 가 필요합니다.

## 사용

```bash
# 단일 run
bench run --profile configs/profiles/profile_a_tab.yaml \
          --endpoint configs/endpoints/local_vllm.yaml

# 여러 profile 순차
bench sweep --profile-dir configs/profiles/ \
            --endpoint configs/endpoints/llmd_k8s.yaml

# 리포트
bench report <run_id>
bench compare <run_id_a> <run_id_b>

# 이상탐지
bench detect <run_id>

# 결과 조회
bench ls --endpoint llmd-a5000-tp4ep --profile profile-c-agent --last 10
```

## 디렉토리

| 경로 | 역할 |
|:-----|:-----|
| `configs/profiles/` | 워크로드 정의 YAML |
| `configs/endpoints/` | Endpoint 접속 정보 YAML (API key는 환경변수 참조) |
| `src/bench/runners/` | AIPerf / vLLM bench / GuideLLM 서브프로세스 래퍼 |
| `src/bench/collectors/` | Prometheus `/metrics` + 요청 로그 수집 |
| `src/bench/storage/` | DuckDB schema + I/O |
| `src/bench/analysis/` | 메트릭 계산 + run 비교 |
| `src/bench/visualization/` | matplotlib plots + Markdown 리포트 |
| `src/bench/detectors/` | 이상탐지 (Phase 6a 규칙 기반 시작) |
| `dashboards/grafana/` | Grafana 대시보드 JSON |
| `monitoring/` | Prom + Grafana docker-compose (endpoint 비포함) |
| `data/db/bench.duckdb` | 결과 누적 저장소 |
| `data/raw/` | Runner raw artifact (AIPerf JSON 등) |

## 상태

초기 구현 진행 중. 로드맵은 `/home/jinmoo/.claude/plans/async-cooking-cat.md` 참조.
