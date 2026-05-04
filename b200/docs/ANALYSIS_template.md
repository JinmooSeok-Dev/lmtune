# `<Study/Recipe Name>` — 분석 (template)

> **본 파일은 template 이다.** 각 study 종료 후 `b200/studies/<study_id>/ANALYSIS.md` 또는 `b200/results/<recipe>/ANALYSIS.md` 로 복사한 뒤 채운다. 채울 때 본 헤더 (이 인용 블록) 는 제거한다.
>
> **목적**: 단순히 수치를 archive 하지 않고 (1) **왜** 이 결과가 나왔는지 메커니즘을 설명하고, (2) 이 측정이 **무엇을 입증/반증**하는지 정리하고, (3) **다음 가설**을 명시한다.
>
> **분량 가이드**: section 3 (원인) 과 section 4 (의의) 가 본문의 절반 이상을 차지해야 한다. 수치 표만 나열한 ANALYSIS.md 는 가치가 없다.

---

# `<Study/Recipe Name>` — 분석

**study_id**: `st-XXXX`
**date**: 2026-MM-DD
**author**: jinmoo
**phase**: B-I / B-II / B-III / B-IV (해당 표기)
**status**: complete | regression | inconclusive

---

## 1. 측정 컨텍스트

### Hardware
- 노드: 2 × B200 (8 GPU/node, 16 GPU 합), HBM ~3 TB
- Compute capability: sm_100 (Blackwell), FP4 native
- 인터노드 fabric: RDMA (InfiniBand / RoCE) — `b200/docs/b200_environment.md` §4 측정값 인용
- NUMA / PCIe topology: `b200/studies/<id>/system_snapshot.json` 인용

### Software
- vLLM: vX.X.X
- llm-d: vX.X (peer repo commit `<sha>`)
- NCCL: 2.X / CUDA: 12.X
- B200 image digest: `ghcr.io/llm-d/llm-d-cuda:vX.X.X@sha256:...`
- bench: commit `<sha>`

### Search space
- file: `b200/search-spaces/<name>.yaml`
- 활성화된 axes (active_if context 적용 후): `<list>`
- pinned axes: `<list>`
- 총 grid 크기 또는 budget: `<N trials | M hours>`

### Workload preset
- short / medium / long 의 (input_tokens, output_tokens, concurrency, num_requests)
- SLO: ttft.p99 ≤ 500ms, e2e.p99 ≤ 30s, fail_rate ≤ 1%
- N=3 + CV gate 적용

### Sampler / Strategy
- `<grid | random | lhc | tpe | cma_es | nsga2>` — 선택 이유 한 줄
- multi-objective 인 경우: objectives = `(metric:workload:direction, ...)`

---

## 2. 결과 — 수치 + 시각화

### Top-K configurations

| rank | trial_id | params (요약) | total_score | ttft.p99 (short) | throughput.avg (short) | e2e.p99 (long) | slo_pass |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |
| ... |  |  |  |  |  |  |  |

### Pareto front (multi-obj 시)

`b200/studies/<id>/pareto.png` 인용. axis: throughput.avg (x) vs ttft.p99 (y).

해석: dominated 영역 vs non-dominated 영역, knee point 위치.

### Sobol total-order index (axis 영향력)

`b200/studies/<id>/sobol.png` 인용.

| axis | total-order index | 해석 |
|:---|:---|:---|
| max_num_seqs | 0.42 | 가장 큰 영향. 큰 값일수록 throughput↑ 동시에 ttft↑ |
| ... |  |  |

### 회귀 비교 (직전 baseline 대비)

| 비교 대상 | metric | baseline | new | 변화 |
|:---|:---|:---|:---|:---|
| B1 winner | throughput.avg (short) | 1234.5 | 1289.2 | +4.4% |
| ... | ... | ... | ... | ... |

회귀 알림 발생 여부: ✅/❌

---

## 3. 원인 분석 — **왜 이 결과가 나왔는가** (핵심 섹션 1/2)

> 이 섹션은 단순한 결과 재진술이 아니다. 메커니즘 설명 + 데이터로 입증.

### 3.1 가장 영향력 큰 axis 의 작용 메커니즘

예: "tp=4 가 tp=8 보다 우위인 이유"

- **이론**: cross-node TP 는 NCCL all-reduce 에 매 layer 마다 RDMA 통신 발생 → tp=8 이 두 노드에 걸치면 latency 증가가 throughput 이득보다 큼
- **데이터로 입증**:
  - `b200/studies/<id>/system_snapshot.json` 의 `nvidia_smi_topo`: tp=4 는 NV4(NVLink 4-way) 인반면 tp=8 은 NV4 + SYS (cross-socket)
  - RDMA Perftest baseline (`b200/studies/<id>/rdma_summary.json`): cross-node bandwidth 363 Gbps vs NVLink intra-node 900 GB/s — 약 2.5× 차이
  - Trial log 의 NCCL_DEBUG: tp=8 trial 에서 all-reduce time 이 forward time 의 18%, tp=4 에서 7%
- **결론**: tp=4 가 16-GPU 16K context 워크로드에서 NCCL 통신 오버헤드를 layer 당 최소화

### 3.2 흥미로운 상호작용 (axis 간 interaction)

예: "enable_chunked_prefill=true × max_num_seqs=256 의 비선형 효과"

- 단독 효과: chunked_prefill on 하나만으로 throughput +3%, max_num_seqs 256 하나만으로 +5%
- 결합 효과: 둘 다 켜면 +12% (예상 +8% 보다 큼)
- **메커니즘 가설**: chunked_prefill 이 prefill batching 을 더 잘게 쪼개 KV cache 의 fragmentation 감소 → max_num_seqs 늘려도 OOM 없이 큰 batch 유지
- **추가 검증 필요**: 다른 모델군(MoE) 에서도 같은 결합 효과 나타나는지

### 3.3 SLO 탈락한 trial 의 패턴

예: "trial #15, #23, #28 이 ttft.p99 SLO 위반 (>500ms)"

- 공통 패턴: gpu_memory_utilization=0.92 + max_num_seqs ≥ 256
- **메커니즘**: KV cache 가 GPU 메모리의 92% 점유 → CUDA graph 의 workspace 부족 → 첫 token 생성 시 graph rebuild 발생 → TTFT 급증
- **검증**: trial #15 의 vLLM log 에서 "Compiling CUDA graph for batch_size=N" 메시지 다수
- **의미**: gpu_memory_utilization 의 안전 상한은 0.90 (90%) 으로 결정. 0.92 는 추후 재검증 시까지 axis 에서 freeze

### 3.4 noise 분류된 trial (CV ≥ 0.10) 의 원인

예: "trial #11 이 CV=0.18 로 reject"

- 원인 후보:
  1. cri-dockerd 가 동시간대 다른 image pull 중이었음 (background dmesg 확인)
  2. 인접 노드의 다른 namespace pod 가 RDMA 대역 점유 (B200 host 의 `b200/scripts/rdma_bench.sh` 동시 실행 안 했는지 확인)
  3. fabric jitter (peer repo 의 known issue)
- **결정**: 이 trial 은 reject 유지. fabric 점유 격리 후 별도 study 에서 재측정.

---

## 4. 의의 — **이 측정이 무엇을 입증/반증하는가** (핵심 섹션 2/2)

> 본 study 의 결과가 외부에서 재인용 가능한 형태로 정리. blog/conference submission 의 figure 후보 도출.

### 4.1 사전 가설과 실측의 일치/불일치

| 가설 (study 시작 전) | 실측 결과 | 일치 여부 |
|:---|:---|:---|
| tp=8 이 16-GPU 환경에서 최대 throughput | tp=4 가 더 높음 | ❌ 반증 — 16-GPU 환경에서는 NCCL 비용이 우세 |
| max_num_seqs 가 가장 영향력 큰 axis | Sobol 결과 max_num_seqs 0.42 (1위) | ✅ |
| ... | ... | ... |

### 4.2 외부 reference 와의 비교

- **cfregly Ch15 (`expectations_b200.json`)**: 본 책의 B200 expectation 은 throughput=1500 tok/s @ Llama-8B. 본 study 의 winner 는 1289 — **86% 수준**. 차이 원인: 본 study 는 16-GPU + cross-node 인 반면 cfregly 는 1×B200 단일 GPU
- **llm-d v0.5 release notes**: B200 decode GPU 당 3,100 tok/s @ 16×16 PD 토폴로지. 본 study 는 inference-scheduling (단일 path) 이므로 직접 비교 부적절. B1 의 P/D path study 결과와 비교 예정 (B-I 후속)
- **InferenceMAX** (vLLM blog 2025-10): vLLM Blackwell 4× Hopper 입증. 본 study 는 Hopper 비교 불포함 (B-IV B7 에서 가능)

### 4.3 본 study 가 만드는 차별 포인트

- **plan 의 differentiation 5점 중**:
  - 1. B6 Low-level axis ✅ 활용 — system_snapshot 데이터로 NCCL/RDMA 메커니즘 설명
  - 4. Sobol + ANOVA pruner ✅ 활용 — axis importance ranking 자동 출력

### 4.4 RECIPES.md 의 새 entry 후보

```yaml
# b200/results/RECIPES.md 에 추가 예정
- name: llama-3.1-8b-inference-scheduling-b200-balanced
  hardware: 2× B200 (16 GPU, RDMA)
  model: meta-llama/Llama-3.1-8B-Instruct
  workload: short/medium/long autotune preset
  winning_config:
    well_lit_path: inference-scheduling
    tensor_parallel_size: 4
    max_num_seqs: 128
    enable_prefix_caching: true
    enable_chunked_prefill: true
    gpu_memory_utilization: 0.85
  results:
    throughput.avg (short): 1289.2 tok/s
    ttft.p99 (short): 145 ms
    slo_pass: true
  source_study: st-XXXX
```

---

## 5. 다음 가설 / 후속 실험

본 study 가 발견한 새 axis 후보 / 미해결 의문점 / 다음 search space 변경 제안.

### 5.1 새 axis 후보

- `kv_cache_dtype=fp4` (B200 native FP4 검증) — 본 study 에선 auto 만 사용
- `block_size` ∈ {16, 32, 64} — vLLM 기본 16, 큰 batch 에서 영향 가능

### 5.2 freezing/축소 권고 (`lmtune search prune`)

- `enable_chunked_prefill`: ANOVA p < 0.001, freeze 권고 → true 로 고정
- `block_size`: importance < 0.05, drop 권고
- `gpu_memory_utilization`: best 0.86 ± σ 0.02 → range 를 [0.84, 0.90] 으로 축소

### 5.3 미해결 의문점

1. tp=2 가 sample 되지 않았는데, tp=2 + dp=2 조합이 tp=4 + dp=1 보다 좋은 시나리오가 있을까? (parallelism axis 결합 study 필요)
2. wide-EP path 에서도 같은 winning config 가 통하는가? (B1 의 path 변경 study 에서 검증)

### 5.4 다음 study 의 search space 변경

```diff
# b200/search-spaces/b1_baselines.yaml diff
- max_num_seqs: {type: categorical, values: [16, 32, 64, 128, 256, 512]}
+ max_num_seqs: {type: categorical, values: [64, 128, 256]}      # 축소

- gpu_memory_utilization: {type: float, low: 0.80, high: 0.92}
+ gpu_memory_utilization: {type: float, low: 0.84, high: 0.90}   # 축소

- enable_chunked_prefill: {type: bool}
+ # frozen at true (ANOVA confirmed)

+ block_size: {type: categorical, values: [16, 32]}              # 신규 axis
+ kv_cache_dtype: {type: categorical, values: [auto, fp8, fp4]}  # 신규 axis
```

---

## 6. References

- 본 study 의 study_id: `st-XXXX`
- 사용한 search space: `b200/search-spaces/<name>.yaml`
- 시각화: `b200/studies/<id>/{pareto,sobol_bar,search_trace}.png`
- 시스템 스냅샷: `b200/studies/<id>/system_snapshot.json`
- 직전 baseline study: `b200/studies/<prev_id>/`
- 비교 reference: cfregly `expectations_b200.json`, llm-d v0.5 release notes
- 관련 plan 섹션: `(internal dev plan, not in repo)` Phase B-X
