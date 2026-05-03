# Autoresearch: vLLM & llm-d endpoint config autotune (performance-first)

## Objective

최적의 **serving goodput** 을 내는 엔드포인트 구성(engine_args, 양자화, KV dtype, 워크로드 파라미터)을 LLM-guided 탐색으로 찾는다.

- **현재 세그먼트 (Segment 0)**: 로컬 vLLM on RTX 5060 Ti 16GB (Qwen2.5-1.5B-Instruct baseline)
- **다음 세그먼트**: K8s 복구 후 llm-d (`configs/endpoints/llmd_k8s.yaml`) — 동일 metric·프로토콜로 이식

탐색 방식은 "cheap-first": 재기동 비용이 작은 축(engine_args flag) 먼저, 큰 축(모델 교체·양자화)은 Segment 1 이후.

## Metrics

**Primary** (higher is better):

- `total_score` — 3 워크로드(short/medium/long) 합산 composite score. 단위 없음(throughput × TTFT penalty).
  - per-workload: `score = throughput_tok_avg × max(0, 1 − ttft_p99/1000)`
  - SLO 1개라도 실패하면 해당 워크로드 `score = 0`

**Secondary** (모니터링용, keep/discard 결정에는 원칙적으로 영향 X):

- `ttft_p99_short`, `ttft_p99_medium`, `ttft_p99_long` (ms)
- `throughput_avg_short`, `throughput_avg_medium`, `throughput_avg_long` (tok/s)
- `e2e_p99_short`, `e2e_p99_medium`, `e2e_p99_long` (ms)
- `slo_pass_all` (bool) — 3 워크로드 모두 SLO 통과 여부

**Hard constraints (SLO, per-workload)**:

- `ttft_p99 ≤ 500ms`
- `e2e_p99 ≤ 30000ms`
- `failure_rate ≤ 1%`

## How to Run

### 운영 모드 1 — Standalone (LLM-guided 가설, 자체 sampler)

```bash
./autoresearch.sh
```

### 운영 모드 2 — `lmtune search` 통합 (Phase S6, 권장)

`lmtune search` 의 통계 sampler (TPE/NSGA-II/CMA-ES) 가 다음 trial 을 추천. autoresearch (LLM 에이전트) 는 가설 생성 대신 ask/tell 호출만 담당:

```bash
# (1) study 만 생성 (실제 trial 0개)
lmtune search start \
  --space configs/search/spaces/vllm_engine_args_tier1.yaml \
  --strategy tpe \
  --max-trials 0 \
  --name autoresearch-segment2 \
  --dry-run
# → study_id=st-XXXX

# (2) 매 cycle:
#     ① lmtune search ask → 다음 params (autoresearch 가 받음)
PARAMS=$(lmtune search ask st-XXXX | jq '.params')
TRIAL_ID=$(lmtune search ask st-XXXX | jq -r '.trial_id')   # 위와 같은 trial 을 재호출하면 새로 발급되므로 한 번만 호출
#     ② autoresearch agent 가 endpoint YAML 의 engine_args 에 PARAMS 적용
#     ③ 측정 + tell:
USE_BENCH_SEARCH=1 BENCH_STUDY=st-XXXX BENCH_TRIAL=$TRIAL_ID ./autoresearch.sh
#        본 분기가 자동으로 `lmtune search tell` 호출 — 결과를 study 에 기록
```

→ LLM 의 도메인 지식 (어떤 axis 가 의미 있는지) + Optuna 의 통계 효율 (이력 기반 효율적 탐색) 결합.

출력 마지막에 machine-readable 라인:

```
METRIC total_score=<N>
METRIC ttft_p99_short=<N> ttft_p99_medium=<N> ttft_p99_long=<N>
METRIC throughput_avg_short=<N> throughput_avg_medium=<N> throughput_avg_long=<N>
METRIC e2e_p99_short=<N> e2e_p99_medium=<N> e2e_p99_long=<N>
METRIC slo_pass_all=<0|1>
```

- `ENDPOINT` 환경변수로 대상 전환: `ENDPOINT=configs/endpoints/llmd_k8s.yaml ./autoresearch.sh`
- 기본값: `configs/endpoints/local_vllm_autotune.yaml`
- 내부 파이프라인: `scripts/vllm_restart.sh` → `scripts/lmtune_score.py × {short, medium, long}` 재현성 게이트(N=3, CV≥0.10 → N=5 로 확장)

## Files in Scope

에이전트가 수정해도 되는 파일(자세한 축 아래 참조):

- `configs/endpoints/local_vllm_autotune.yaml` — **`deployment.engine_args` 블록만**
  - `max_num_seqs` ∈ {16, 32, 64, 128, 256}
  - `enable_prefix_caching` ∈ {true, false}
  - `enable_chunked_prefill` ∈ {true, false}
  - `gpu_memory_utilization` ∈ {0.80, 0.85, 0.88, 0.90, 0.92}
  - `max_model_len` ∈ {2048, 4096, 8192}
  - `kv_cache_dtype` ∈ {auto, fp8} *(Blackwell sm_120 호환 여부 첫 실험에서 확인. 실패 시 auto 로만 고정)*
  - 양자화 모델 교체(옵션, 비쌈): `model` 을 `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-FP8` 중에서만 선택 가능
- `configs/profiles/autotune/{short,medium,long}.yaml` — `synthetic_*`, `concurrency`, `request_count`, `slo` 조정 가능 (단 SLO 값 자체는 위 hard constraint 와 일치)
- `scripts/lmtune_score.py` — `--warmup-runs` 도입 같은 측정 로직 개선(선택)

## Off Limits

**절대 수정 금지**:

- `configs/endpoints/local_vllm_autotune.yaml` 의 `url`, `api_type`, `metrics_url`, `parallelism.*` (모두 TP=DP=1 고정)
- `src/bench/**` — 코어 벤치마크 로직. runner 버그가 의심되면 worklog 에 기록만.
- `scripts/vllm_restart.sh` — `CUDA_VISIBLE_DEVICES=1`, health-check 로직 유지.
- `data/**` — 과거 실험 로그는 보존.
- `.gitignore`, `pyproject.toml` — 새 의존성 필요하면 사용자에게 확인 후.

**`configs/endpoints/llmd_k8s.yaml`** 은 Segment 0 에서는 건드리지 않음. Segment 1 시작 시 re-init.

## Constraints

- **HW**: RTX 5060 Ti 16GB 단일 GPU. TP/PP/DP>1 불가.
- **vLLM**: 0.19.1. `--enforce-eager=false` 유지 (CUDA graph).
- **재현성 게이트**: 각 (config, workload) 쌍 N=3; `CV(throughput_tok) ≥ 0.10` 이면 N=5 로 확장 후에도 넘으면 `accepted=false` 로 discard.
- **한 실험에 1축만 변경** 권장 (A/B 분리, 원인 귀속 가능).
- **실험당 예상 시간**: restart 30~90s + 3 workloads × 3 runs ≈ 3~7분. 최대 40 실험까지 목표.

## What's Been Tried (Segment 0 baseline = Round 2 winner)

(자세한 내용은 `data/autotune/FINAL_REPORT.md`)

**Round 1 (engine_args sweep)**: 7 config × Qwen2.5-1.5B. 1위 `chunked-prefill-on` (`prefix_caching=true, chunked_prefill=true, max_num_seqs=128, gpu_mem=0.85`). **축 간 차이 ±5%**.

**Round 2 (model sweep)**: 8 모델·양자화. 1위 `qwen25-1.5b-winner` total=1906.4 (3 워크로드 합산, SLO 통과). Qwen3-0.6B 는 medium/long 에서 workload 단독 최고(785.6/516.4) 이나 short cold-start 로 실패. 8B+ 는 16GB 에서 KV 포화.

**현재 baseline** (`configs/endpoints/local_vllm_autotune.yaml`):

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
engine_args:
  enable_prefix_caching: true
  enable_chunked_prefill: true
  max_num_seqs: 128
  gpu_memory_utilization: 0.85
  max_model_len: 4096
```
기대 baseline total_score ≈ **1906**.

## What to Explore (아이디어 backlog — `autoresearch.ideas.md` 로 옮겨도 됨)

우선순위 높음:

1. **`kv_cache_dtype=fp8`** — weight 아닌 KV 를 8bit 로 줄여 16GB 에서 long context 여유 확보. Blackwell sm_120 호환 첫 실험에서 확인. (Round 2 에서 배제됐던 가장 유망한 축)
2. **`max_num_seqs` 극값** — 16 (메모리 여유 최대화, 8B+ 실험용) vs 256 (throughput 상한 탐색).
3. **`warmup_runs` 추가** — Qwen3-0.6B short, Qwen3-4B-FP8 long 이 cold-start 때문에 탈락한 이력. `scripts/lmtune_score.py` 에 `--warmup-runs 1` 도입하면 통과 가능성.
4. **`max_model_len=2048`** 로 낮춰 KV 여유 → 8B 계열 재시도 가능.
5. **Qwen3 thinking mode off** — profile YAML 에 `extra_body: {"enable_thinking": false}` 추가하면 1.7B/4B 개선 여지.

우선순위 낮음:

6. `gpu_memory_utilization=0.92` 로 KV 공간 추가 확보 (안정성 리스크).
7. `block_size` 탐색 (vLLM 기본 16).
8. Segment 1 (llm-d) 시작 전 `llmd_k8s.yaml` 의 `url` 도달성 확인. TP=4, 모델·양자화 축이 전혀 다름.

## Segment Plan

- **Segment 0**: vLLM engine_args 단일축 탐색. 20~30 실험 목표. 승자 config 를 `data/autotune/FINAL_REPORT.md` 에 추가.
- **Segment 1 (re-init 시점)**: llm-d on K8s. config header 의 `name` 을 `llmd-k8s-autotune` 으로 교체. `ENDPOINT` 변경. Files in Scope 도 `llmd_k8s.yaml` 로 스위치.

재시작 시: `/autoresearch` (인자 없이) → 이 파일 + `autoresearch.jsonl` 읽고 다음 실험부터 이어감.
