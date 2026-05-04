# B200 16-GPU Model × vLLM × llm-d Configuration Catalog (2026-05)

> 본 catalog 는 B200 16-GPU (2 노드 × 8 GPU, 192GB HBM3e/GPU = ~3.07TB 합) 환경에서
> **vllm 0.17.1 + llm-d 0.4.x + 7 well-lit-paths** 위에 띄울 수 있는 OSS 모델과
> 호환되는 구성 (parallelism, dtype, well-lit-path) 의 정본 매트릭스. 신규 모델이
> 출시될 때마다 본 파일에 한 줄 추가 → endpoint·values·search-space 한 세트가
> 자동 도출되도록 schema 일관성 유지.

> 마지막 검토: 2026-05-04 / 검토 주기: 월 1회 또는 새 모델 출시 시 / 소유자: B200 lmtune track / 상태: living

## 1. 환경 전제

| 항목 | 값 | 출처 |
|:--|:--|:--|
| HW | 2× NHN B200 8-GPU node, B200 sm_100 (Blackwell), 192GB HBM3e/GPU | NHN Cloud spec |
| GPU 합 | 16 (3.07TB HBM3e) | (계산) |
| Inter-node fabric | InfiniBand/RoCE 400+Gbps, RDMA Perftest 363.98 Gbps verified | NHN B200 신규 구축 검증 |
| vllm | 0.17.1 (V1 engine, MTP/Eagle3 support, FP8/MXFP4/nvfp4 native) | image vllm/vllm-openai:v0.17.1 |
| llm-d | 0.4.12 modelservice chart, 7 well-lit-paths | peer repo `agentic/llm-distributed-inference/phase{1..4}` |
| K8s | k3s (single host 2 컨테이너 노드) | NHN B200 |
| 가용 dtype | bf16, fp8 (e4m3/e5m2), MXFP4 (B200 native), nvfp4 (실험) | vllm 0.17.1 release notes |

## 2. Well-lit-path 카탈로그 (llm-d 0.4)

| Path | helmfile | 핵심 axis | B200 적합성 |
|:--|:--|:--|:--|
| inference-scheduling | b200/helmfile/inference-scheduling/ | 표준 EPP, monolithic | ✅ default (현재 gpt-oss-120b 진행) |
| pd-disaggregation | b200/helmfile/pd-disaggregation/ | NIXL kv-connector, prefill/decode replica 비율 | ⚠️ NIXL 미정상 (현재 비활성) |
| wide-ep-lws | b200/helmfile/wide-ep-lws/ | LeaderWorkerSet EP, MoE 전용 | ✅ MoE 모델 (DSV3, Kimi K2, Qwen3-Coder, GLM-4.7) |
| tiered-prefix-cache | b200/helmfile/tiered-prefix-cache/ | LMCache HBM/CPU/NVMe 계층 | △ B6.4 axis 와 페어링 |
| precise-prefix-cache | b200/helmfile/precise-prefix-cache/ | exact-match prefix routing | △ multi-turn 평가 시 효과 |
| predicted-latency-scheduling | b200/helmfile/predicted-latency-scheduling/ | EPP 회귀 모델 | △ 실험적 |
| workload-autoscaling | b200/helmfile/workload-autoscaling/ | KEDA + WVA HPA | △ continuous loop 시 |

## 3. 모델 카탈로그

> dtype 별 GPU 메모리 / replica / GPU = `total_params_b × dtype_bytes / TP` 추정. activation + KV cache 별도 (~10-20% 추가).
>
> 16-GPU 토폴로지: TP=8 × DP=2 (2 replica, 노드별 1) 가 기본. TP=16 (cross-node) 은 RDMA 대역폭 감수.

### 3.1 OSS 1순위 — 최대·최신·평가 우수 (2025-Q3 ~ 2026-Q1)

| 모델 | params (active) | arch | 출시 | HF repo | License | vllm 0.17.1 | dtype × TP=8 메모리/GPU | B200 16-GPU 토폴로지 | 워크로드 정합성 | 우리 status |
|:--|:--:|:--|:--:|:--|:--|:--:|:--:|:--|:--|:--|
| **Kimi K2-Instruct** | 1T (32B) | MoE 384×top8 | 2025-07 | `moonshotai/Kimi-K2-Instruct` | Modified MIT | ✅ (DSV3 fork) | FP8 125 / MXFP4 63 / bf16 250✗ | TP=8 × DP=2 (FP8) | **Agentic / tool-use SOTA**, 1T = OSS 최대 | 🆕 다음 study 후보 |
| **DeepSeek-V3.2** | 671B (37B) | MoE+MLA 256×top8+1shared | 2025-Q4 | `deepseek-ai/DeepSeek-V3.2` | DeepSeek License (commercial OK) | ✅ | FP8 84 / MXFP4 42 / bf16 168 | TP=8 × DP=2 (FP8 권장) | 종합 평가 1위, 코드+수학+추론 | 🆕 |
| **DeepSeek-R1-0528** | 671B (37B) | MoE+MLA (V3 base + R1 RL) | 2025-Q1 | `deepseek-ai/DeepSeek-R1-0528` | DeepSeek License | ✅ | (V3 동일) | (V3 동일) | **추론 SOTA** AIME 91+, codeforces top | 🆕 |
| **Qwen3-Coder-480B-A35B** | 480B (35B) | MoE | 2025 mid | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Apache 2.0 | ✅ | FP8 60 / bf16 120 | TP=8 × DP=2 | **코드 특화** SWE-bench top, BCB 1위 | 🆕 |
| **GLM-4.7-355B-A32B** | 355B (32B) | MoE + **MTP** | 2025-Q4 | `zai-org/GLM-4.7-355B-A32B` 추정 | Apache 2.0 추정 | ✅ (4.6 verified, 4.7 확인 필요) | FP8 44 / bf16 89 | **TP=4 × DP=4** 가능 (4 replica) | tool-use, MTP speculative decoding | 🆕 |
| **GLM-4.7-Flash** | ~70B dense | dense | 2025-Q4 | `zai-org/GLM-4.7-Flash` 추정 | Apache 2.0 추정 | ✅ | bf16 18 / FP8 9 | TP=2 × DP=8 (8 replica) | 빠른 라이트버전, 다중 replica sweep | 🆕 |
| **Llama-4-Maverick-400B** | 400B (17B) | MoE 128×top1 | 2025-Q2 | `meta-llama/Llama-4-Maverick-400B-17B-Instruct` | Llama 4 Community | ✅ | bf16 100 / FP8 50 | TP=8 × DP=2 | 멀티모달 (텍스트만 사용 시 일반) | 🆕 |
| **Llama-4-Scout-109B** | 109B (17B) | MoE 16×top1 | 2025-Q2 | `meta-llama/Llama-4-Scout-109B-17B-Instruct` | Llama 4 Community | ✅ | bf16 27 / FP8 14 | TP=2 × DP=8 | 짧은 latency, 다중 replica | 🆕 |
| **MiniMax-M2** | (확인 필요) | Lightning Attention | 2025-Q4 | `MiniMaxAI/MiniMax-M2` 추정 | (확인) | △ (LA support 확인 필요) | (확인) | (확인) | **장컨텍스트 LA**, 4M+ context 가능 | △ vllm 호환 우선 |
| **Mistral Large 3** | (확인 필요) | dense 추정 | 2025-Q3~Q4 | `mistralai/Mistral-Large-3-Instruct` | Mistral AI Research | △ (확인) | (확인) | (확인) | dense 안정 | △ |

### 3.2 검증 완료 / 진행 중

| 모델 | params (active) | arch | dtype × TP=8 | 토폴로지 | 우리 status |
|:--|:--:|:--|:--:|:--|:--|
| **gpt-oss-120b** | 117B (5.1B) | MoE 128×top4, MXFP4 native | MXFP4 8 | TP=8 × DP=2 (현재 study) | 🟢 진행 중 (PR #10 후 재시작) |
| **gpt-oss-20b** | 20B (3.6B) | MoE 32×top4, MXFP4 | MXFP4 1.3 | TP=1 × DP=16 가능 | ✅ baseline 검증 완료 |
| **Llama-3.1-8B-Instruct** | 8B dense | dense | bf16 2 | smoke 전용 | ✅ verified (b200-aiperf-v1 study) |
| **Qwen2.5-72B-Instruct** | 72B dense | dense | bf16 18 / FP8 9 | TP=4 × DP=4 | △ B1 baseline 후보 |
| **Llama-3.1-70B-Instruct** | 70B dense | dense | bf16 17 / FP8 9 | TP=4 × DP=4 | △ B1 baseline 후보 |

### 3.3 중간 크기 (200-500B 범위)

| 모델 | params (active) | arch | 노트 |
|:--|:--:|:--|:--|
| Qwen3-235B-A22B | 235B (22B) | MoE 128×top8 | 다국어 안정 |
| Mixtral-8x22B | 141B (39B) | MoE 8×top2 | 비교 대조군 |
| DBRX | 132B (36B) | MoE 16×top4 | 비교 대조군 |
| Command-R+ | 104B dense | dense | tool-use 평가 |
| Qwen3-30B-A3B | 30B (3B) | MoE 64×top8 | 빠른 sweep |

### 3.4 Long-context / 특화

| 모델 | 특화 | 노트 |
|:--|:--|:--|
| MiniMax-M1 | Lightning Attention 1M+ | vllm 호환 확인 필요 |
| MiniMax-M2 | LA 후속 | 위와 동일 |
| Yi-2 Large 추정 | (확인) | (확인) |
| Falcon-H1 추정 | hybrid (Mamba+Transformer?) | vllm 호환 확인 필요 |

### 3.5 작은 검증용 (probe / smoke)

- Qwen2.5-1.5B-Instruct (E7 검증)
- Qwen3-0.6B
- Qwen2.5-7B-Instruct
- Phi-4 (14B)
- Gemma-2-27B

## 4. 모델 × Well-lit-path × Topology 권장 매트릭스

| 모델 | inference-scheduling | wide-ep-lws | pd-disaggregation | tiered-prefix-cache |
|:--|:--:|:--:|:--:|:--:|
| Kimi K2 (1T MoE) | TP=8 DP=2 (FP8) | DP=8 EP=8 wide-EP | TP=4 P=1+D=1 | LMCache offload |
| DSV3 (671B MoE) | TP=8 DP=2 (FP8) | DP=8 EP=8 | TP=4 P=2+D=1 | LMCache |
| Qwen3-Coder-480B | TP=8 DP=2 | DP=8 EP=4 | (NIXL OK 후) | △ |
| GLM-4.7-355B | **TP=4 DP=4** | DP=4 EP=4 | TP=4 P=1+D=1 | △ |
| GLM-4.7-Flash 70B | **TP=2 DP=8** (8 replica) | n/a (dense) | TP=2 P=2+D=2 | △ |
| Llama-4-Maverick | TP=8 DP=2 | DP=8 EP=2 | (NIXL OK 후) | △ |
| Llama-4-Scout 109B | **TP=2 DP=8** | DP=4 EP=2 | TP=2 P=2+D=2 | △ |
| gpt-oss-120b (현) | TP=8 DP=2 ✅ | DP=8 EP=8 | (NIXL OK 후) | △ |

## 5. 워크로드 카탈로그

### 5.1 단발 (concurrency mode, guidellm/aiperf)

- `autotune-short` 256/128, conc=8, 30 req — chat-like
- `autotune-medium` 1024/256, conc=8, 20 req — 코딩 에이전트 단발 turn
- `autotune-long` 3072/512, conc=4, 15 req — KV cache 압박

### 5.2 Multi-turn (user_centric mode, aiperf 0.7.0+) ⭐ 신규

- **`autotune-multiturn-agent`** — 8 user × 5-turn 평균 × 30 conv (방금 추가)
- 향후 추가 예정:
  - `agent_phase_breakdown` (research/) — editing/execution 단계별 토큰 분포
  - `tokenomics_sdlc` (research/) — SDLC 단계별 multi-turn 시뮬

### 5.3 Trace replay (TraceWorkload, 향후)

- BurstGPT (Zipf 입력 + bimodal 출력)
- ServeGen
- Mooncake KV trace
- AzureLLMTraces

## 6. dtype 선택 가이드 (B200 sm_100)

| dtype | bytes/param | B200 native | KV cache 호환 | 권장 사용 |
|:--|:--:|:--:|:--:|:--|
| bf16 | 2 | ✅ | auto | 기본, 정합성 ↑ |
| FP8 (e4m3) | 1 | ✅ | fp8_e4m3 | 큰 모델 (≥200B), throughput ↑↑ |
| FP8 (e5m2) | 1 | ✅ | fp8_e5m2 | 동일 |
| MXFP4 | 0.5 | ✅ (sm_100 native) | auto 권장 | gpt-oss native, 다른 모델은 quant 필요 |
| nvfp4 | 0.5 | ✅ (실험) | nvfp4 | probe 후 사용 (B200 sm_100 호환 검증) |

## 7. 다음 단계 — 신규 모델 추가 절차

새 모델 N 을 catalog 에 추가하려면:

1. **registry**: `src/lmtune/models/registry.py` 에 `normalize_model_spec(...)` 한 줄 추가
2. **search-space**: `b200/search-spaces/b1_baselines.yaml` 의 `model_id` enum 에 추가
3. **endpoint**: `b200/endpoints/b200_<slug>.yaml` 신규 (gpt-oss-120b 패턴 복사)
4. **values gotmpl**: `b200/helmfile/<path>/values-<slug>.yaml.gotmpl` 신규
5. **dtype/topology**: 본 catalog 의 § 4 매트릭스 갱신
6. **PR**: feat(b200) commit + b200/docs/model_catalog 업데이트

## 8. 우선순위 로드맵

| 단계 | 모델 | 이유 |
|:--:|:--|:--|
| 0 (현재) | gpt-oss-120b | MXFP4 native, 16-GPU 검증 |
| 1 | **Kimi K2-Instruct** | OSS 최대 1T, agentic 워크로드 (multi-turn profile 과 정합) |
| 2 | **DeepSeek-V3.2** | 종합 평가 1위, MLA |
| 3 | **GLM-4.7-355B-A32B (MTP)** | MTP speculative decoding 효과 측정 |
| 4 | **Qwen3-Coder-480B** | 코드 워크로드 절대 비교 |
| 5 | DeepSeek-R1-0528 | 추론 task 비교군 |
| 6 | Llama-4-Maverick / Scout | Meta latest, 멀티모달 baseline |
| ∞ | continuous loop (B5) | 새 모델 출시 → 1줄 추가 → 자동 흡수 |

## 9. 알려진 호환성 제한 (2026-05 기준)

- **NIXL** 미정상 → P/D disaggregation 비활성. UCX backend 시도 후 활성화 검토.
- **EPP / vllm metric schema mismatch** → `vllm:lora_requests_info` 미존재 시 EPP 가 endpoint unhealthy 마킹 → gateway 503. svc 직접 port-forward 우회 가능.
- **MXFP4 non-native quant** — gpt-oss 외 모델은 별도 quant 단계 필요 (벤더 quant tool 또는 vllm 의 dynamic quant).
- **vllm V1 engine** 에서 deprecated flags: `--num-scheduler-steps`, `--worker-use-ray`, `--weight-dtype`, `--activation-dtype`, `--swap-space-gb`, `--compilation-level/mode/cudagraph-num-of-warmups`, `--enable-kv-cache-compression`, `--kv-cache-offload-target`, `--speculative-method`. b2_vllm_engine_v0171.yaml 가 이를 반영한 verified subset.

## 10. 참고

- vllm release notes: https://github.com/vllm-project/vllm/releases
- llm-d well-lit-paths: https://github.com/llm-d/llm-d
- DeepSeek-V3 paper: https://arxiv.org/abs/2412.19437
- Kimi K2 release: https://github.com/MoonshotAI/Kimi-K2
- Llama 4 release: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- GLM-4.7: https://github.com/zai-org/GLM-4 (확인 필요)
- aiperf 0.7.0: https://github.com/triton-inference-server/aiperf
- 본 프로젝트 plan: `/home/jinmoo/.claude/plans/async-cooking-cat.md`
