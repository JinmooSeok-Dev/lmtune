# llm-d 구성 자동탐색 — 최적 Goodput 조합 리포트

**HW**: RTX 5060 Ti 16GB (Blackwell sm_120, CUDA 12.8, PyTorch 2.10)
**Engine**: vLLM 0.19.1 (단일 GPU)
**Benchmark**: guidellm 0.6.0 via `bench` harness
**Rounds**: Round 1 (engine_args 7 configs on Qwen2.5-1.5B) + Round 2 (모델·양자화 8 configs)
**Experiments total**: 15
**Measurement date**: 2026-04-23

---

## 1. Executive Summary

| 워크로드 | 최적 구성 | 모델 | Goodput score | Throughput | TTFT p99 |
|:---|:---|:---|---:|---:|---:|
| **short** (256/128) | `chunked-prefill-on` | Qwen2.5-1.5B-Instruct | **740.9** | 810 tok/s | 86ms |
| **medium** (1024/256) | `qwen3-0.6b-winner` | Qwen3-0.6B | **785.6** | 823 tok/s | 45ms |
| **long** (3072/512) | `qwen3-0.6b-winner` | Qwen3-0.6B | **516.4** | 574 tok/s | 100ms |

**핵심 결론**
1. **16GB 단일 GPU 에서는 1-2B 모델이 goodput 승자** — 8B+ 는 weight 가 아닌 KV cache 가 bottleneck.
2. **AWQ/FP8 quantization 이득은 제한적** — 16GB 에서는 8B-AWQ·14B-AWQ 모두 KV 포화로 SLO fail. 8B-FP8 은 short/medium 에선 동작했으나 long 실패.
3. **최적 engine_args** (전 워크로드 공통): `enable_prefix_caching=true, enable_chunked_prefill=true, max_num_seqs=128, gpu_memory_utilization=0.85`.
4. **모델 선택이 engine_args 보다 영향 큼** — Round 1 의 engine_args sweep 은 ±5% 차이, Round 2 의 모델 sweep 은 ±100% 차이.

---

## 2. 환경 제약과 llm-d Well-Lit Path 매핑

K8s 클러스터(minikube `192.168.49.2:8443`) 도달 불가로 **llm-d 실배포는 수행 못함**. 단일 vLLM 으로 축을 proxy:

| llm-d Well-Lit Path | 단일 vLLM proxy | 이번 실험 |
|:---|:---|:---|
| inference-scheduling | `max_num_seqs` 스윕 | ✅ Round 1 |
| precise-prefix-cache-aware | `enable_prefix_caching` | ✅ Round 1 |
| chunked-prefill | `enable_chunked_prefill` | ✅ Round 1 |
| 모델·양자화 축 (Phase 1-2 `ms-*/values-*.yaml`) | model+quantization | ✅ Round 2 |
| P/D disaggregation | — | ❌ GPU 2개 아키 다름 |
| wide-EP-LWS | — | ❌ 8 GPU 필요 |
| WVA / EPP tiered | — | ❌ K8s 필수 |

---

## 3. 실험 결과 상세

### Round 1 — engine_args sweep (Qwen2.5-1.5B-Instruct 고정)

| rank | label | total_score | short | medium | long | 비고 |
|-----:|:------|-----------:|------:|-------:|-----:|:-----|
| 1 | `chunked-prefill-on` | 1395.0 | 740.9 | 654.0 | — | long 미측정(Round 1 의 long profile 버그) |
| 2 | `gpu-mem-util-0.90` | 1390.5 | 735.5 | 655.0 | — | |
| 3 | `gpu-mem-util-0.80` | 1389.1 | 739.0 | 650.1 | — | |

→ engine_args 7 config 간 차이는 5% 이내. **prefix_caching on + chunked_prefill on 이 기본 권장**.

### Round 2 — 모델·양자화 sweep (engine_args 고정·per-model tuned)

| rank | label | model | total | short | medium | long | SLO | 해석 |
|-----:|:------|:------|-----:|------:|-------:|-----:|:---:|:-----|
| 1 | `qwen25-1.5b-winner` | Qwen2.5-1.5B-Instruct | **1906.4** | 731.6 | 662.1 | 512.7 | ✅ | 전 워크로드 통과 유일 |
| 2 | `qwen3-1.7b-winner` | Qwen3-1.7B | 1685.7 | 717.6 | 598.6 | 369.6 | ✅ | 1.5B 대비 density 이득 없음 |
| 3 | `qwen3-0.6b-winner` | Qwen3-0.6B | 1302.0 | 0 | 785.6 | 516.4 | ❌ | short cold-start fail, 다른 2개는 최고 |
| 4 | `qwen3-4b-fp8-winner` | Qwen3-4B-FP8 | 1085.6 | 612.9 | 472.7 | 0 | ❌ | long 에서 1개 run TTFT 513ms (boundary) |
| 5 | `qwen3-8b-fp8-winner` | Qwen3-8B-FP8 | 842.0 | 491.5 | 350.5 | 0 | ❌ | long KV 공간 부족 |
| 6 | `qwen3-4b-winner` | Qwen3-4B (bf16) | 727.4 | 422.3 | 305.0 | 0 | ❌ | bf16 은 FP8 대비 열세 |
| 7 | `qwen3-8b-awq-winner` | Qwen3-8B-AWQ | 0 | 0 | 0 | 0 | ❌ | KV cache 96% 포화, 전 워크로드 TTFT 초과 |
| 8 | `qwen3-14b-awq-winner` | Qwen3-14B-AWQ | 0 | 0 | 0 | 0 | ❌ | KV 초포화 |

### Per-Workload 승자 (cross-round, per-workload SLO 기준)

각 워크로드의 실제 SLO 통과 구성 중 score 최고:

**Short (256/128, concurrency 8)**
- 🏆 `chunked-prefill-on` on Qwen2.5-1.5B-Instruct — score=740.9, throughput=810 tok/s, TTFT p99=86ms
- 차선: `qwen3-1.7b-winner` — 717.6 (TTFT 더 낮지만 throughput 낮음)

**Medium (1024/256, concurrency 8)**
- 🏆 `qwen3-0.6b-winner` on Qwen3-0.6B — score=785.6, throughput=823 tok/s, TTFT p99=**45ms**
- 차선: `qwen25-1.5b-winner` on Qwen2.5-1.5B — 662.1 (품질 높음)

**Long (3072/512, concurrency 4)**
- 🏆 `qwen3-0.6b-winner` on Qwen3-0.6B — score=516.4, throughput=574 tok/s, TTFT p99=100ms
- 차선: `qwen25-1.5b-winner` — 512.7 (거의 동률, 모델 품질 선택)

**품질 vs 처리량 trade-off**:
- 0.6B: 최고 처리량이지만 generation 품질 제한
- 1.5B-Instruct: 통합 승자이자 품질 균형점
- 1.7B: 파라미터 증가분 대비 이득 거의 없음 → 1.5B 우위
- 4B+: 16GB 에서 KV 여유 부족, 추천 안 함

---

## 4. Goodput 구성을 쉽게 찾는 방법 (method)

전체 방법은 [`METHOD.md`](METHOD.md) 참조. 요약:

### 4-layer 설계

```
Layer 4  hypotheses_round*.json        ← 탐색 공간 선언 (JSON)
            ↓
Layer 3  autotune_run.py               ← orchestrator (endpoint 재기동 + 측정)
            ↓
Layer 2  bench_score.py                ← N회 반복 + CV 게이트 + composite score
            ↓
Layer 1  bench run --json-summary      ← 단발 측정 (JSON 출력)
```

### 한 줄 재현 (llm-d 없이)

```bash
pip install -e ".[dev]"
python scripts/autotune_run.py \
  --endpoint configs/endpoints/local_vllm_autotune.yaml \
  --hypotheses data/autotune/hypotheses_round2.json \
  --log data/autotune/experiments_round2.jsonl
python scripts/autotune_report.py --log data/autotune/experiments_round2.jsonl
python scripts/analyze_per_workload.py \
  --logs data/autotune/experiments.jsonl data/autotune/experiments_round2.jsonl
```

### 탐색 공간 선언 방식

```json
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
}
```

### Composite Objective

```
penalty = max(0, 1 − ttft_p99 / (2 × ttft_slo_ms))
score   = throughput_tok_avg × penalty
# SLO 실패 시 score = 0
```

### Reproducibility Gate

- 각 (config, workload) 쌍당 N=3 반복
- CV(throughput) ≥ 0.10 → 자동 N=5 확장
- 여전히 CV ≥ 0.10 → `accepted: false` 로 분류

### autoresearch 연동 (선택)

hypotheses 를 미리 채우지 않고 LLM 이 결과 보고 다음 hypothesis 를 제안하게 하려면:
```bash
autoresearch optimize --max-experiments 40 --goal "$(cat autoresearch/goal.md)"
```

---

## 5. llm-d 실배포로 이식

K8s 클러스터가 복구되면 `vllm_restart.sh` 만 `helmfile apply` 기반 스크립트로 교체.

| 역할 | 단일 vLLM | llm-d (K8s) |
|:---|:---|:---|
| config 적용 | endpoint YAML 덮어쓰기 | `helmfile apply -f phase1/helmfile.yaml.gotmpl` |
| 재기동 | `pkill vllm; vllm serve …` | Deployment rollout |
| ready 확인 | `curl /v1/models` | `kubectl wait --for=condition=ready` |
| 결과 수집 | stdout JSON 동일 | 동일 (URL 만 cluster IP) |

승자 config 는 `ms-phase*/values-*.yaml` 의 engine_args 에 그대로 반영:

```yaml
# ms-phase1/values-qwen25-1.5b.yaml (신규 생성 예)
modelspec:
  modelArtifactUri: hf://Qwen/Qwen2.5-1.5B-Instruct
  vllmArgs:
    enable-prefix-caching: true
    enable-chunked-prefill: true
    max-num-seqs: 128
    gpu-memory-utilization: 0.85
    max-model-len: 4096
```

---

## 6. 다음 Round 권장사항

현재 실험의 명백한 한계와 Round 3 로 제안:

### (a) Warmup 게이트 추가
Qwen3-0.6B short (cold-start TTFT 1161ms → SLO fail), Qwen3-4B-FP8 long (1개 run 513ms) 은 warmup 만 하면 통과 가능. **bench_score.py 에 `--warmup-runs 1` 옵션 추가** 추천.

### (b) 8B+ 모델을 16GB 에 맞추기
```json
{
  "model": "Qwen/Qwen3-8B-AWQ",
  "engine_args": {"max_num_seqs": 16, "max_model_len": 2048, "gpu_memory_utilization": 0.92}
}
```
KV 공간 확보로 short/medium 통과 가능성. long 은 여전히 컨텍스트 부족.

### (c) Qwen3 의 thinking mode 비활성화
Qwen3 는 thinking tokens 때문에 e2e 가 길어질 수 있음. `extra_body: {"enable_thinking": false}` 를 bench profile 에 추가하면 1.7B/4B 점수 개선 여지.

### (d) llm-d 실배포 재개
K8s 복구 후 phase1 inference-scheduling 에 위 승자 config 로 `ms-phase1/values-qwen25-1.5b.yaml` 생성하여 실제 EPP 라우팅 포함 측정.

---

## 7. 산출물 위치

| 파일 | 내용 |
|:---|:---|
| `data/autotune/hypotheses_round1.json` / `_round2.json` | 탐색 공간 선언 |
| `data/autotune/experiments.jsonl` / `experiments_round2.jsonl` | Round 1/2 원본 실험 로그 |
| `data/autotune/report_round1.md` / `report_round2.md` | Round별 요약 |
| `data/autotune/report_per_workload.md` | Cross-round per-workload 랭킹 |
| `data/autotune/METHOD.md` | 탐색 방법론 상세 |
| `data/autotune/FINAL_REPORT.md` | **본 문서** |
| `data/db/bench.duckdb` | 개별 run 의 raw metrics |
| `data/raw/<run_id>/` | guidellm 원본 artifact |
