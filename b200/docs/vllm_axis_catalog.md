# vLLM Engine Args — Autotune Axis Catalog (B-II 입력)

> Phase B-II 의 search space (`b200/search-spaces/b2_vllm_engine.yaml`) 작성에 쓰이는 axis 카탈로그. axis 별 의미 / 범위 / 권장 값 / active_if 조건 / 출처 링크. 새 vLLM 릴리스마다 본 문서를 갱신하면 axis 1 줄 추가만으로 B5 continuous loop 이 자동 흡수.
>
> **갱신 주기**: vLLM 릴리스마다 (분기). 직전 점검: 2026-04-29 (vLLM 0.7+/2026-Q1 기준)

---

## 0. 분류 — 6 그룹

| 그룹 | axis 수 | 적용 모델군 |
|:---|:---|:---|
| 1. Scheduling / Batching | 5 | 모든 모델 |
| 2. KV Cache | 4 | 모든 모델 (B200 fp4/fp8 native) |
| 3. Prefix Cache | 1 | 멀티턴 / 코드 에이전트 워크로드 |
| 4. MoE Optimizations | 6 | MoE 전용 (DeepSeek, Qwen3-Coder-480B, Mixtral, Llama-4-Maverick) |
| 5. Speculative Decoding | 1 | 모델별 검증 필요 |
| 6. Compilation / CUDA Graph | 4 | 모든 모델, B200 효과 큼 |

총 21 axis 후보. `active_if` 게이팅으로 필요한 것만 활성화.

---

## 1. Scheduling / Batching (5)

### 1.1 `max_num_seqs`

- **type**: int 또는 categorical
- **default**: 모델·HW 의존 (보통 256-512)
- **권장 범위 (B200 16-GPU)**: `[16, 32, 64, 128, 256, 512]`
- **의미**: 한 iteration 에서 동시에 처리할 최대 시퀀스 수. 큰 값 → throughput 높지만 TTFT 악화 가능
- **autotune 효과**: 가장 영향력 큰 axis 중 하나. Sobol total-order index 보통 0.3+ 측정
- **active_if**: 항상 활성
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/), [vLLM Optimization Guide](https://docs.vllm.ai/en/stable/configuration/optimization/)

### 1.2 `max_num_batched_tokens`

- **type**: int
- **default**: max_num_seqs × max_model_len 의 자동 산출
- **권장 범위 (B200 long-context)**: `[2048, 4096, 8192, 16384]`
- **의미**: 한 iteration 의 토큰 총량 상한. chunked_prefill 와 결합 시 prefill chunk 크기 결정
- **active_if**: `enable_chunked_prefill: true` 권장 (off 시 영향 작음)
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 1.3 `enable_chunked_prefill`

- **type**: bool
- **default**: `False` (vLLM 0.7+ 에서 일부 모델 자동 활성화)
- **의미**: 큰 prefill 을 chunk 로 쪼개 decode 와 함께 batch. **Mixed batching** 의 핵심
- **autotune 효과**: long-context 워크로드에서 throughput +10-20%, TTFT +5-15%
- **alert**: MLA 모델 (DeepSeek-V2/V3) 에서 prefix_caching 와 호환성 이슈 (vLLM #14069)
- **active_if**: 항상 활성, MLA 모델은 `kv_cache_dtype=auto` 강제
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/), [vLLM Forums — chunked prefill + prefix caching](https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903)

### 1.4 `async_scheduling`

- **type**: bool
- **default**: `True` (vLLM 2026)
- **의미**: 다음 iteration scheduling 을 현재 iteration 과 overlap. GPU utilization gap 회피
- **autotune 효과**: latency·throughput 모두 약간 개선. off 로 두면 회귀 데이터 확보용
- **active_if**: 항상 활성
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 1.5 `stream_interval`

- **type**: int
- **default**: 1
- **권장 범위**: `[1, 4, 8]`
- **의미**: streaming 시 token batching 간격. 작을수록 즉시 전송
- **autotune 효과**: TTFT/ITL trade-off. 코딩 에이전트 워크로드에선 1 권장
- **active_if**: streaming endpoint
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

---

## 2. KV Cache (4)

### 2.1 `kv_cache_dtype`

- **type**: categorical
- **default**: `auto` (모델 dtype 그대로)
- **B200 권장 범위**: `[auto, fp8_e4m3, fp8_e5m2, nvfp4]`
- **의미**: KV cache 저장 dtype. 작은 dtype → KV 공간 ↓ → 큰 batch / long context 가능
- **autotune 효과**: B200 native FP4 ≥ 50% KV 공간 절감 + Blackwell tensor core 가속. SLO 지키며 throughput +15-30% 가능
- **alert**: MLA 모델 (DeepSeek-V2/V3) 에서 fp8 호환성 이슈 (vLLM #14069). 첫 trial 에서 probe → 실패 시 study-level freeze
- **active_if**: B200 / Hopper 만 (구 GPU 는 auto 고정)
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 2.2 `kv_cache_memory_bytes`

- **type**: categorical (string)
- **default**: `None` (gpu_memory_utilization 으로 간접 결정)
- **권장 값**: `["8G", "16G", "32G", "64G"]` (B200 GPU 당 180GB HBM 기준)
- **의미**: KV cache 의 절대 크기 직접 지정. utilization 보다 정밀
- **autotune 효과**: 큰 batch + long context 시 미세 튜닝
- **active_if**: `gpu_memory_utilization` 과 양자택일
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 2.3 `gpu_memory_utilization`

- **type**: float
- **default**: 0.90
- **권장 범위 (B200)**: `[0.80, 0.92]`
- **의미**: GPU 메모리 중 vLLM 이 사용할 비율. KV cache + workspace 합산
- **autotune 효과**: 큰 값 → KV ↑ throughput ↑, 너무 크면 (0.92+) CUDA graph workspace 부족 → TTFT 악화 가능
- **alert**: 0.92 + max_num_seqs ≥ 256 조합에서 TTFT spike 빈발 (Round 1/2 archive 기록)
- **active_if**: 항상 활성, `kv_cache_memory_bytes` 와 양자택일
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 2.4 `block_size`

- **type**: categorical
- **default**: 16
- **권장 범위**: `[16, 32, 64]`
- **의미**: KV cache 의 block (page) 크기. 큰 block → fragmentation ↓ workspace ↓, 작은 block → cache 활용도 ↑
- **autotune 효과**: 영향 작은 편 (Sobol 0.05 미만 보통). 그러나 long-context 에선 block_size=32 가 유의 차이
- **active_if**: 항상 활성
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

---

## 3. Prefix Cache (1)

### 3.1 `enable_prefix_caching`

- **type**: bool
- **default**: `False` (vLLM 0.7+ 에서 일부 모델 기본 on)
- **의미**: 동일 prefix 의 KV cache 블록 재사용. 코딩 에이전트 / 멀티턴 채팅에서 효과 큼
- **autotune 효과**: hit rate 의존. 코딩 에이전트는 50-80% hit, throughput +30-50%
- **alert**: MLA 모델에서 chunked_prefill off 시 runtime error (vLLM #14069)
- **active_if**: chunked_prefill 와 호환되면 항상 활성
- **출처**: [vLLM Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching/)

---

## 4. MoE Optimizations (6)

> **모두 `active_if: {model_family: moe}` 게이팅**. Dense 모델에선 무의미.

### 4.1 `enable_expert_parallel`

- **type**: bool
- **default**: `False`
- **의미**: MoE expert 를 expert parallelism 으로 분배 (TP 대신). 여러 expert 를 GPU 별로 나누어 메모리 절약
- **autotune 효과**: DeepSeek-V3 (671B), Qwen3-Coder-480B 등 대형 MoE 의 16-GPU 배치 가능 여부 결정
- **active_if**: `model_family: moe`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 4.2 `expert_parallel_size` (`ep`)

- **type**: int
- **default**: 1
- **권장 범위 (16-GPU)**: `[1, 2, 4, 8, 16]`
- **의미**: EP degree. 큰 값 → expert 분산도 ↑ all-to-all 통신 ↑
- **active_if**: `model_family: moe AND enable_expert_parallel: true`

### 4.3 `enable_dbo` (Dual Batch Overlap)

- **type**: bool
- **default**: `False`
- **의미**: 모델 executor 내부에서 두 batch 의 forward 를 overlap. MoE 의 all-to-all 대기 시간 hide
- **autotune 효과**: MoE 에서 throughput +5-10%, dense 에선 무효
- **active_if**: `model_family: moe`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 4.4 `enable_eplb` (Expert Parallelism Load Balancing)

- **type**: bool
- **default**: `False`
- **의미**: MoE expert 의 부하를 동적으로 재분산. hot expert 의 bottleneck 회피
- **autotune 효과**: 비균등 워크로드 (특히 코딩 에이전트) 에서 효과 큼
- **active_if**: `model_family: moe AND enable_expert_parallel: true`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 4.5 `all2all_backend`

- **type**: categorical
- **default**: `allgather_reducescatter`
- **권장 범위**: `[allgather_reducescatter, deepep_low_latency, deepep_high_throughput, nixl_ep, flashinfer_pplx]`
- **의미**: MoE expert 간 all-to-all 통신 backend. RDMA·NVLink 활용 패턴 다름
- **autotune 효과**: B200 + RDMA 조합에서 deepep / nixl_ep 가 standard 대비 +10-20%
- **active_if**: `model_family: moe AND enable_expert_parallel: true`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 4.6 `moe_backend`

- **type**: categorical
- **default**: `auto`
- **권장 범위**: `[auto, triton, deep_gemm, cutlass, flashinfer_cutlass, marlin]`
- **의미**: MoE FFN 의 GEMM kernel backend. cutlass / flashinfer 가 B200 native FP8/FP4 활용
- **autotune 효과**: B200 에서 cutlass 또는 flashinfer 가 throughput +10-30%
- **active_if**: `model_family: moe`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

---

## 5. Speculative Decoding (1)

### 5.1 `speculative_config`

- **type**: dict (JSON)
- **default**: `None`
- **권장 변형**: `[null, {"model": "<draft>", "num_speculative_tokens": 4}, {"method": "eagle"}, {"method": "mtp"}]`
- **의미**: Draft model 또는 EAGLE/MTP 로 미래 N 토큰 예측, accept rate 높으면 throughput 배가
- **autotune 효과**: accept rate 의존. coding agent 워크로드에서 EAGLE 1.5-2× throughput
- **alert**: 모델별 호환성 검증 필요. 첫 trial 에서 probe → 실패 시 freeze
- **active_if**: 모델별 (Llama / Qwen / DeepSeek 별 호환 model 다름)
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

---

## 6. Compilation / CUDA Graph (4)

### 6.1 `compilation_config`

- **type**: dict (JSON)
- **default**: 종합 default
- **권장 변형**: `[default, {"mode": "full_cuda_graph"}, {"mode": "piecewise_cuda_graph"}, {"mode": "no_cuda_graph"}]`
- **의미**: torch.compile + cudagraph capture 의 mode/backend/level. B200 에서 효과 큼
- **autotune 효과**: full_cuda_graph 가 default 대비 throughput +5-15%, 단 startup 시간 ↑
- **active_if**: 항상 활성, GPU 모델 별 (B200 / H100) 권장 다름
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 6.2 `max_cudagraph_capture_size`

- **type**: int
- **default**: `min(max_num_seqs * 2, 512)`
- **권장 범위**: `[256, 512, 1024]`
- **의미**: CUDA graph 로 capture 하는 최대 batch size. 그 이상은 eager 실행
- **autotune 효과**: 큰 max_num_seqs 와 정합성 필요
- **active_if**: `enforce_eager: false`
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 6.3 `cudagraph_num_of_warmups`

- **type**: int
- **default**: 0
- **권장 범위**: `[0, 1, 3]`
- **의미**: graph capture 전 warmup iteration 수. 첫 trial cold-start TTFT 개선
- **autotune 효과**: cold-start sensitive 워크로드에서 TTFT spike 회피 (Round 2 의 Qwen3-0.6B short cold-start 탈락 사례 참고)
- **active_if**: 항상 활성
- **출처**: [vLLM Engine Args](https://docs.vllm.ai/en/latest/configuration/engine_args/)

### 6.4 `enable_flashinfer_autotune`

- **type**: bool
- **default**: `None` (모델 의존 자동)
- **의미**: kernel warmup 동안 FlashInfer 의 attention kernel tactic 자동 선택
- **autotune 효과**: 다양한 ISL/OSL 조합에서 안정적 throughput. 본 axis = vLLM 내부의 mini autotune
- **active_if**: 항상 활성, FlashInfer 사용 모델만
- **출처**: [vLLM blog — InferenceMAX Blackwell](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)

---

## 7. axis 별 active_if 매트릭스

| axis 그룹 | dense (Llama-3.1-8B) | dense (Qwen2.5-72B) | MoE (Qwen3-Coder-480B) | MoE 대형 (DeepSeek-V3.2) |
|:---|:---|:---|:---|:---|
| Scheduling 5 | 모두 활성 | 모두 활성 | 모두 활성 | 모두 활성 |
| KV Cache 4 | 모두 활성 | 모두 활성 | 모두 활성 | MLA 호환 검증 (kv_cache_dtype) |
| Prefix Cache 1 | 활성 | 활성 | 활성 | MLA 호환 시만 |
| MoE 6 | 비활성 | 비활성 | 모두 활성 | 모두 활성 |
| Speculative 1 | EAGLE 가능 | EAGLE 가능 | MTP 가능 | MTP 가능 |
| Compilation 4 | 모두 활성 | 모두 활성 | 모두 활성 | 모두 활성 |

---

## 8. autotune 사전 prune 규칙

새 axis 추가 후 첫 study 의 결과 기반 자동 freeze 권고:

- **ANOVA p > 0.05** → freeze (값 영향 없음)
- **RandomForest importance < 0.05** → drop (모델이 무시)
- **Best ± σ** → continuous axis range 축소

→ 본 문서의 21 axis 가 study 후 보통 5-7 로 좁혀짐.

---

## 9. References

- [vLLM Engine Args (latest)](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [InferenceMAX Blackwell — vLLM Blog](https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html)
- [vLLM #14069 — MLA + chunked_prefill 호환성](https://github.com/vllm-project/vllm/issues/14069)
- [vLLM Forums — chunked prefill + prefix caching](https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903)
- 본 repo: `data/autotune/FINAL_REPORT.md` (Round 1/2 archive — RTX 5060 Ti baseline)
- 본 repo: `b200/docs/lowlevel_axis_catalog.md` (system 축, B6)
