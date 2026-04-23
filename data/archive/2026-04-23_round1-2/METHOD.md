# 최적 goodput 구성을 쉽게 찾아내는 방법

## 1. 문제 정의

**Goodput** 은 단순 throughput 이 아니라 **"SLO 를 충족한 요청의 처리량"** 입니다.
llm-d 문서가 말하는 goodput 도 동일 정의: P99 TTFT/e2e, 에러율이 SLO 를 만족한 조건에서의 tok/s.

최적 goodput 탐색은 다음을 만족해야 합니다.
- **SLO-aware**: TTFT, e2e 둘 다 SLO 넘으면 점수 0
- **Reproducible**: N 회 반복 측정 + 분산(CV) 게이트
- **Model/workload-sensitive**: 모델과 워크로드마다 최적 구성이 다름
- **Cheap-first**: 재배포 비용이 작은 축(engine_args, prefix cache on/off) 먼저, 큰 축(모델 교체) 나중
- **Declarative**: "탐색 공간" 을 YAML/JSON 한 파일로 선언

## 2. 본 프로젝트의 4-layer 설계

```
Layer 4  hypotheses_round*.json        (탐색 공간 — 사람 또는 LLM 이 채움)
            ↓
Layer 3  autotune_run.py               (orchestrator — 각 hypothesis 를 실험으로 실행)
            ↓
Layer 2  bench_score.py                (reproducibility gate — N회 반복 + CV + composite score)
            ↓
Layer 1  bench run --json-summary      (단발 측정 — 마지막 줄에 machine-readable JSON)
```

각 layer 는 **독립적으로 수정 가능**합니다.
- 탐색 공간만 바꾸고 싶으면 Layer 4 의 JSON 만 다시 쓰면 됨
- 점수 식을 바꾸고 싶으면 Layer 2 의 `aggregate()` 만 수정
- vLLM 대신 다른 엔진으로 재기동하려면 `vllm_restart.sh` → `sglang_restart.sh` 같은 형태로 교체
- Reproducibility 기준만 바꾸고 싶으면 `--cv-threshold` flag

## 3. Composite Objective (핵심)

```python
penalty = max(0, 1 - ttft_p99 / (2 * ttft_slo_ms))
score   = throughput_tok_avg * penalty
# SLO 한 번이라도 실패 → score = 0
```

해석:
- TTFT 0ms → penalty=1.0, score = 순수 throughput
- TTFT SLO 경계(500ms) → penalty=0.5, score = throughput 의 절반
- TTFT ≥ 2×SLO(1000ms) → penalty=0, score=0
- 3-workload total_score = short + medium + long 의 합

**장점**: Pareto front 탐색 없이 스칼라 최적화로 환원. autoresearch LLM-guided 또는 단순 greedy 로도 동작.
**단점**: SLO ms 값과 penalty 기울기 선택에 민감. 현재는 ttft_slo_ms=500 고정.

## 4. Reproducibility Gate

각 `(config, workload)` 쌍은 **N=3 회 반복 측정**됩니다.
- throughput CV(변동계수) ≥ 0.10 이면 **N=5 로 자동 확장 재측정**
- 그래도 CV ≥ 0.10 이면 실험 `accepted=False` (노이즈 구성으로 분류)

이를 통해 우연히 한 번 빠른 결과가 winner 로 뽑히는 경우를 방지합니다.

## 5. 탐색 공간 선언 예시

`hypotheses_round2.json`:
```json
[
  {
    "label": "qwen3-8b-awq-winner",
    "model": "Qwen/Qwen3-8B-AWQ",
    "engine_args": {
      "enable_prefix_caching": true,
      "enable_chunked_prefill": true,
      "max_num_seqs": 64,
      "gpu_memory_utilization": 0.90,
      "max_model_len": 4096
    }
  },
  ...
]
```

각 entry 가 하나의 **실험 단위**. `autotune_run.py` 가 이를 순회하며
1. endpoint YAML 덮어쓰기 (`model`, `deployment.engine_args`)
2. `vllm_restart.sh` — 기존 vLLM kill, 새 config 로 기동, `/v1/models` ready polling
3. `bench_score.py` × 3 workloads — N회 반복 + composite score
4. `experiments_round2.jsonl` 에 한 줄 append

## 6. 대표 워크로드 3종

| 워크로드 | input / output | concurrency | 대표 시나리오 |
|:---|:---:|:---:|:---|
| `autotune-short`  | 256 / 128   | 8 | 채팅, 짧은 질의응답 |
| `autotune-medium` | 1024 / 256  | 8 | 코딩 에이전트 턴, 일반 RAG |
| `autotune-long`   | 3072 / 512  | 4 | 긴 컨텍스트 QA, 문서 분석 |

SLO 동일: `ttft_p99 ≤ 500ms, e2e_p99 ≤ 30s`.
3개 워크로드의 composite score 를 합산해 `total_score` 산정.

## 7. autoresearch 연동 포인트

**본 실험은 autoresearch plugin 없이도 돌아갑니다** — 미리 채운 hypothesis JSON 이면 충분.
autoresearch 를 붙이면:
- baseline 1개 실행 후 LLM 이 `score` 와 `metrics` 보고 다음 hypothesis 를 제안
- 일부 축은 pruning (예: 첫 3번 실험에서 `enable_prefix_caching=false` 가 모두 열세 → 이후 제외)
- grid 144개 조합 → 20-40개로 수렴 가능

Plugin 설치 후 다음으로 트리거:
```bash
autoresearch optimize --max-experiments 40 \
  --goal "$(cat autoresearch/goal.md)"
```

## 8. llm-d 실배포 환경으로 이식

K8s 클러스터에서 llm-d 로 바꿀 때는 `vllm_restart.sh` 를 교체만 하면 됩니다:

| 역할 | 단일 vLLM | llm-d 전환 시 |
|:---|:---|:---|
| config 적용 | endpoint YAML 덮어쓰기 | helmfile values 덮어쓰기 + `helmfile apply` |
| 재기동 | `pkill vllm; vllm serve …` | `helmfile apply` → 새 Deployment rollout |
| ready 확인 | `curl /v1/models` | `kubectl wait --for=condition=ready` |
| 결과 수집 | stdout JSON | 동일 (endpoint URL 만 cluster IP) |

즉 Layer 1-2-4 는 그대로, Layer 3 의 restart 스크립트만 환경별로 제공합니다.

## 9. 최적 구성을 찾는 실전 워크플로우

1. **Tier 1 (저비용 축) 먼저** — engine_args 4개 축 (max_num_seqs, prefix_caching, chunked_prefill, gpu_mem_util). 모델 고정하고 5-10개 hypothesis 로 탐색.
2. **Tier 1 winner 위에서 모델 축** — Tier 1 에서 뽑힌 best engine_args 를 각 모델에 동일 적용해 모델 density 비교 (bf16 vs AWQ vs FP8).
3. **Workload 별 winner 분리 기록** — short/medium/long 각각 best 가 다를 수 있음. 리포트에 per-workload top 1 명시.
4. **Variance 검증** — 상위 3개 config 는 N=10 으로 재측정해 변동계수 확인.
5. **llm-d 로 이식** — 승자 config 의 engine_args 를 llm-d `ms-phase*/values-*.yaml` 로 복사.

## 10. 산출물

| 파일 | 역할 |
|:---|:---|
| `data/autotune/hypotheses_round*.json` | 탐색 공간 선언 |
| `data/autotune/experiments_round*.jsonl` | 원본 실험 로그 (한 줄 = 한 실험) |
| `data/autotune/report_round*.md` | Top-K 요약 + per-workload winner |
| `data/db/bench.duckdb` | 개별 run 의 raw metrics + requests 적재 |
| `data/raw/<run_id>/` | guidellm 원본 JSON artifact |

**재현**: `git clone` + `pip install -e .` + `python scripts/autotune_run.py --hypotheses <round>.json` 로 끝.
