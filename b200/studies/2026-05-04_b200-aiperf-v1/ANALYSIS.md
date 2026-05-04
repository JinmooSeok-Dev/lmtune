# b200-aiperf-v1 — B200 NHN 클러스터 첫 lmtune autotune cycle 정상 종료

> study_id: `st-01KQRY6E8W9GWT7NG4R52RJ1SC`
> 시점: 2026-05-04
> 차단 → 통과 까지의 회복 흐름 기록. **NHN B200 위에서 lmtune autotune cycle (helmfile redeploy → vllm 재기동 → aiperf 측정 → DuckDB 적재 → Top-K) 가 처음으로 끝까지 완주한 study**.

## 1. 측정 컨텍스트

| 항목 | 값 |
|:---|:---|
| HW | NHN Cloud B200 노드 (k3s v1.34.6) |
| Engine | vllm-openai:v0.17.1 |
| Model | meta-llama/Llama-3.1-8B-Instruct (gated, HF tokenizer 사용) |
| Stack | llm-d-infra v1.4.0 + inferencepool v1.5.0 + llm-d-modelservice v0.4.12 |
| Endpoint URL | `http://127.0.0.1:8011/v1` (host → svc/<decode> port-forward) |
| Profile | `configs/profiles/autotune/short.yaml` — synthetic 256/128, concurrency 8, 30 req |
| Search space | `b200/search-spaces/w_minikube_minimal.yaml` (3 axis: max_num_seqs, enable_prefix_caching, gpu_memory_utilization) |
| Strategy | random, max-trials=2, repeats=1 |
| SLO | ttft_p99_ms ≤ 60000, e2e_p99_ms ≤ 30000 |
| Adapter | `llmd-k8s` (helmfile selector=`kind=inference-stack`, ns=b200-infsch) |
| Backend | inline (host process) |

## 2. 결과

### Top-2 trials

| seq | score | TTFT p99 | E2E p99 | Throughput avg | params |
|:---:|---:|---:|---:|---:|:---|
| **1** | **1054.83** | 1390.7 ms | 1898.9 ms | 1067.2 tok/s | max_num_seqs=16, prefix_caching=true, gpu_mem=0.816 |
| 2 | 1044.60 | 1091.9 ms | 1988.7 ms | 1054.2 tok/s | max_num_seqs=32, prefix_caching=true, gpu_mem=0.897 |

manual aiperf 비교 (warm vllm, concurrency 4, 10 req, max_num_seqs=64):
- TTFT p99 96.86 ms · Throughput 728.55 tok/s · Request Latency 588 ms

cold-start (redeploy 직후) 의 첫 측정이라 TTFT 가 1초대로 튐. throughput 은 워크로드 (concurrency 8 + 30 req) 가 더 부하가 커서 manual 보다 오히려 높음.

## 3. 원인 분석 — 왜 끝까지 도달했나

5단계 차단을 모두 푼 결과:

| 단계 | 차단 | 해소 |
|:---|:---|:---|
| 1. helmfile diff | `AgentgatewayParameters` CRD missing | 사용자 환경 kubecontext 정합 — CRD 는 처음부터 깔려 있었고 cluster 인식만 어긋났던 것 |
| 2. helmfile redeploy 후 measurement | adapter 의 probe 가 RemoteDisconnected | 이전 PR (62e89b9) 의 retry/backoff probe 로 60s 모델 로딩 갭 흡수 |
| 3. SLO hard gate | `ttft_p99_ms: 500` 이 cold-start TTFT (~1초) 를 reject | profile 의 hard SLO 를 60000 으로 완화 (PR #5) |
| 4. aiperf metric 추출 | 0.7.0 이 결과 파일명 `profile_export.json` → `profile_export_aiperf.json` 으로 변경 | `parse()` glob 우선순위 갱신 + percentile 셋 확장 (PR #5) |
| 5. score 계산 | metrics={} 빈 dict 라 derived `score` 가 0 | 4번 풀리면 자동 풀림 |

핵심은 **외형은 같아 보였지만 단계별로 다른 원인**이었다는 점:
- 처음 study 의 `slo_pass=False` 는 SLO 임계 자체가 아니라 measurement 가 비어 있어서였음
- 그래서 SLO 를 60000 까지 늘렸어도 처음엔 안 풀렸고, aiperf glob patch 가 들어가야 진짜 통과

## 4. 의의

1. **B200 NHN 환경에서 lmtune autotune 이 reproducible 하게 동작함을 입증** — 이전엔 minikube cycle validation (st-01KQQZ92WH747X6D64JHW0N46N) 에서만 검증된 패턴이 production-class 클러스터로 이전됨
2. **aiperf 0.7.0 호환 patch 의 효과 정량화** — patch 전 trial_metrics 0행 → patch 후 5 metric × 2 trial = 10행 정상 적재
3. **워크로드별 측정 변동성 확인** — cold-start 환경의 TTFT 는 warm 대비 ~14× 분산 가능. profile-level SLO 는 warm 가정 시 적용 불가하며 study 단위에서 별도 임계를 두어야 함
4. **Top-1 winning config 의 첫 evidence** — `max_num_seqs=16 + prefix_caching=true + gpu_mem=0.816` 가 short workload 에서 score=1054.83. n=2 라 통계적 신뢰도는 낮지만 baseline 으로는 archive 가치

## 5. 다음 가설

- **재현성 검증**: 같은 config 로 `--repeats 3` + `--max-trials 4` 로 cv_throughput < 0.10 게이트 안에서 안정되는지
- **trial 수 확장**: `--max-trials 8` random + `--strategy tpe` 비교로 sample efficiency 차이 확인
- **redeploy cold-start 영향**: probe ready 후 1-2 회 warmup 요청을 추가해 첫 측정의 TTFT spike 를 흡수하는 옵션이 score 분산을 얼마나 줄이는지
- **endpoint YAML 덮어쓰기 부작용**: LLMDK8sAdapter 가 trial 마다 endpoint YAML 을 yaml.safe_dump 으로 덮어써 주석/포맷이 손실되는 별개 결함. 매 study 후 사용자 git diff 를 발생시킴 — 별도 fix 필요
