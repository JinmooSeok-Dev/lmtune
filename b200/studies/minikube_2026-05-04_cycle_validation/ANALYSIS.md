# minikube llm-d autotune cycle — 분석

**study_id**: `manual-2026-05-04-cycle-validation`
**date**: 2026-05-04
**author**: jinmoo
**phase**: gate (S6 의 미검증 부분 매워짐, B0 진입 게이트)
**status**: complete

---

## 1. 측정 컨텍스트

### Hardware
- 호스트: RTX 3060 (8 GB) GPU0 + RTX 5060 Ti (16 GB) GPU1, 동일 PC
- minikube driver: docker (kvm2 → docker 전환), `--gpus all`, k8s v1.33.1
- 인터노드: 단일 노드 (minikube)

### Software
- minikube v1.36.0 (containerd 1.7.23), nvidia-device-plugin v0.17.2 (addon)
- vLLM image: `vllm/vllm-openai:v0.17.1` (B200 정본 정렬)
- llm-d charts: `llm-d-infra v1.4.0`, `inferencepool v1.5.0`, `llm-d-modelservice v0.4.12`
- gateway provider: `agentgateway v1.0.0`
- 모델: `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB bf16)
- bench commit: `5086eb7` (PR #1 머지 직후) + 본 분석 phase 의 신규 커밋들

### 본 검증의 input
- helmfile-mini.yaml.gotmpl + values-qwen25-1.5b-mini.yaml.gotmpl (templated)
- 두 overlay 파일:
  - `/tmp/overlay-trial-A.yaml`: `{max-num-seqs:64, max-model-len:2048, enable-prefix-caching:true}`
  - `/tmp/overlay-trial-B.yaml`: `{max-num-seqs:16, max-model-len:4096, enable-prefix-caching:false}`

### Workload
- 본 검증은 cycle 무결성만 평가 → 단발 `/v1/completions` 호출 (max_tokens=5~10)
- SLO 측정 / repeats / score 산출은 스코프 외 (다음 단계 lmtune search 로 이어짐)

---

## 2. 결과 — 수치 + 시각화

### 2.1 Trial A (overlay-A)

| 측정 | 값 |
|:---|:---|
| pod 의 vllm container args | `--max-num-seqs 64 --max-model-len 2048 --enable-prefix-caching` (overlay-A 정확히 반영) |
| `/v1/models` `max_model_len` | 2048 |
| `/v1/completions` HTTP | 200 |
| 응답 시간 | 0.093 s (warm cache, 5-token 생성) |
| pod IP | 10.244.0.14 |
| svc `vllm-decode` Endpoints | 자동 갱신 → 10.244.0.14:8000 |

### 2.2 Trial B (overlay-B)

| 측정 | 값 |
|:---|:---|
| pod 의 vllm container args | `--max-num-seqs 16 --max-model-len 4096` (overlay-B 정확히 반영, prefix-caching 없음 = false 처리) |
| `/v1/models` `max_model_len` | 4096 |
| `/v1/completions` HTTP | 200 |
| 응답 시간 | 13.07 s (cold start: 모델 로드 49 s + 첫 generation) |
| pod IP | 10.244.0.15 |
| svc `vllm-decode` Endpoints | 자동 갱신 → 10.244.0.15:8000 |

### 2.3 Cycle 무결성

- helmfile state-values-file overlay 변경 → helmfile apply → 새 ReplicaSet 생성 → 새 pod 기동 → 모델 로드 (~50 s) → vllm `Application startup complete` → 측정 가능. 약 60-80 s/trial 의 transition cost.
- svc `vllm-decode` (별도 manifest, chart 가 안 만들어주는 것) 는 selector `llm-d.ai/role=decode` 라 새 pod 자동 매칭. **즉 svc 기반 endpoint 는 cycle 견딤**.
- 호스트 `kubectl port-forward svc/vllm-decode` 는 새 pod 으로 transparently 따라감 (한 번만 띄우고 재기동 불필요).

### Artifacts

- 본 ANALYSIS.md
- 신규 fix 들: `fix/llmd-mini-templated-values` branch 의 commit 2 개
  - `fb172c5`: templated values gotmpl + image swap + routing.proxy guard
  - `087faa4`: vllm-route-mini.yaml (svc + HTTPRoute, agentgateway 우회)
- 머지된 main commit: `5086eb7` (PR #1 — vllm-route.yaml B200 namespace)

---

## 3. 원인 분석 — 왜 이 결과가 나왔는가

### 3.1 S6 가 빠뜨린 검증 — 본 분석에서 매워진 것

S6 ANALYSIS.md (`b200/studies/minikube_s6_validation/ANALYSIS.md`) 가 명시:

> "trial 별 params 는 실제 vLLM 서버에 반영되지 않았다 ... 진짜 engine_args autotune 은 Track B-I 의 B200 환경에서 시작된다"

S6 는 acceptance scope 를 두 부분으로 쪼개 절반만 PASS:
- ✅ ask/tell messaging 통합
- ❌ params 가 실제로 vllm 에 적용

후자가 미구현인 채로 다음 phase (B0/B-track) 로 넘어감 → B200 진입 시 같은 미구현 burden 이 environment 차이와 합쳐 디버깅 부담 증폭. 본 분석이 그 미구현 분기를 명시적으로 매움.

### 3.2 Params injection 의 정확한 메커니즘 — 무엇이 작동하지 않았고 무엇을 고쳤는가

**구 상태 (S6 시점)**:

```
LLMDK8sAdapter.render_values_overlay()
   ↓ overlay yaml 작성 (vllmArgs dict, release 키 nesting)
   ↓
helmfile apply --state-values-file overlay.yaml
   ↓ helmfile 의 .StateValues 에 적재
   ↓
releases[].values: [values-X.yaml]    ← 정적 yaml. .StateValues 미참조
   ↓
chart values 에 trial params 안 도달
   ↓
vllm container args 가 helmfile baked-in default 만 보유 → trial 무관 동일 측정
```

**신 상태 (본 fix)**:

```
LLMDK8sAdapter.render_values_overlay()    (변경 없음, 기존 코드 그대로)
   ↓
helmfile apply --state-values-file overlay.yaml
   ↓ .StateValues 적재
   ↓
releases[].values: [values-X.yaml.gotmpl]   ← 본 fix: gotmpl 화
   ↓ values gotmpl 이 .StateValues.{release}.vllmArgs 를 읽어 args 동적 직렬화
   ↓
chart 가 받은 decode.containers[0].args 가 trial params 반영
   ↓
새 pod 이 trial params 로 vllm serve
```

**핵심 fix 한 줄**: `values-qwen25-1.5b-mini.yaml` → `values-qwen25-1.5b-mini.yaml.gotmpl` 로 templating 활성. helmfile 이 values gotmpl 을 렌더링할 때 `.StateValues` 가 바인딩되어 들어감. defaults dict 와 mergeOverwrite 로 overlay 미제공 시도 안전 동작 (manual `helmfile apply` 도 그대로 됨).

### 3.3 svc 기반 endpoint 가 cycle 견디는 이유

K8s Service 의 selector → endpoints 매핑은 **pod 이름이 아니라 label 기반**. helmfile 이 redeploy 하면:
1. 새 ReplicaSet 생성 → 새 pod 가 같은 label (`llm-d.ai/role=decode`) 보유
2. 기존 svc `vllm-decode` (selector: `llm-d.ai/role=decode`) 의 endpoints 가 새 pod IP 자동 추가
3. 기존 pod 가 terminating 되면 endpoints 에서 자동 제거
4. `kubectl port-forward svc/vllm-decode` 는 svc 의 현재 endpoints 만 보면 되므로 transparent

→ 결과: cycle 마다 endpoint URL `http://127.0.0.1:9100/v1` 가 **동일하게 살아있고**, 사용자가 매 trial 마다 port-forward 재기동할 필요 없음. 이게 lmtune autotune cycle 의 "stable measurement endpoint" 패턴.

본 fix 가 만든 `vllm-route-mini.yaml` 의 svc 가 이 패턴의 핵심. modelservice chart v0.4.12 는 `routing.proxy.enabled: false` 일 때 decode pod 의 stable svc 를 안 만들어주므로 (chart 한계) 우리가 별도 manifest 로 박는 것. agentgateway provider 가 InferencePool 라우팅 미지원 (B200 + minikube 양쪽 동일 함정) 도 같은 manifest 의 HTTPRoute 가 우회.

### 3.4 GPU 단일 GPU 점유 cycle 의 함정

본 검증 중 **rolling update GPU contention** 발생:
- 새 pod 가 nvidia.com/gpu: 1 요구 → cluster 의 2 GPU 중 하나 선점
- 그러나 기존 pod 가 termination 대기 중 GPU 미반납 → 새 pod scheduling 막힘
- helmfile rollout timeout (120s) 도달 후 사용자 개입 (force delete) 으로 해소

→ B200 (16 GPU/cluster, replica 1) 에서는 영향 없음 (GPU 여유). minikube (2 GPU) 에서만 발생. cycle 의 일반론 결함은 아니지만 lab 환경 cycle script 는 force-cleanup 옵션 권장.

### 3.5 응답 시간 차이 — Warm vs cold

- Trial A 두 번째 generation: 0.093 s (vllm engine warm + 모델 GPU 상주)
- Trial B 첫 generation: 13.07 s = 모델 다운로드 49 s + GPU upload + first-token compile 시간 일부. 이후 요청은 warm path.

이게 N=3 + CV gate 의 존재 이유 — 같은 trial 안에서도 첫 vs 후속 호출 변동이 큼. lmtune 은 warmup 분리 후 측정 윈도우만 평균 내는 패턴.

---

## 4. 의의 — 이 측정이 무엇을 입증/반증하는가

### 4.1 입증된 것

| 명제 | 본 분석 이전 | 본 분석 후 |
|:---|:---|:---|
| K8s autotune cycle 에서 trial params 가 vllm container 까지 도달 | 미증명 (S6 가 명시적으로 미구현) | ✅ Trial A/B 의 args + /v1/models 응답으로 정량 입증 |
| helmfile redeploy 가 새 pod 을 trial params 로 띄움 | 미증명 | ✅ overlay-A → max_seq_len=2048, overlay-B → max_seq_len=4096 |
| svc 기반 endpoint 가 cycle 견딤 | 추측 | ✅ Endpoint IP 자동 갱신 + port-forward survives |
| /v1/completions 실제 generation | B200 에서 한 번도 시도 안 함 | ✅ 200 응답 + 토큰 생성 검증 |
| agentgateway InferencePool 우회 (Service + HTTPRoute) | B200 에서만 만든 우회 | ✅ minikube 도 동일 우회 필요 (cross-env 재발 함정) |

### 4.2 반증된 것

- "S6 PASS 면 K8s autotune cycle ready" — **거짓**. acceptance 가 절반 scope 였고 핵심 부분 (params injection) 미구현.
- "minikube 가 인프라 검증을 충분히 한다" — 부분 거짓. S6 시점 검증 절차 자체에 빈틈.

### 4.3 B200 으로의 일반화

본 fix 들 (`values-qwen25-1.5b-mini.yaml.gotmpl` 패턴, `vllm-route-mini.yaml` 패턴) 은 그대로 B200 의 `values-llama-3.1-8b-smoke.yaml` 과 `vllm-route.yaml` 에 대응. 이미 B200 의 `vllm-route.yaml` 은 main 머지됨 (PR #1). 남은 일:
1. `values-llama-3.1-8b-smoke.yaml` → `values-llama-3.1-8b-smoke.yaml.gotmpl` 로 templating (mini 와 같은 패턴)
2. `helmfile.yaml.gotmpl` 의 `values:` 에서 `.gotmpl` 참조
3. B200 에서 helmfile state-values-file 적용 → vllm container 의 args 가 trial params 받는지 직접 확인

이 3 단계가 B200 에서 autotune cycle 시작 가능 조건의 정량 정의.

### 4.4 본 검증으로 풀린 사용자 차단점

세션 직전 사용자 인용:
> "도대체 무슨 근거로 b200에서 실행가능하다고 한거야"
> "autotune cycle 을 왜 안해;; 우리 프로젝트가 autotune 인데"

근거 없는 "ready" 라벨 → 본 분석으로 ready 의 정량 정의 확립. 이제 "ready" 는 5 검증 항목 PASS 여부로 결정 (§ 4.1 표). 추측 라벨링 종료.

---

## 5. 다음 가설 / 후속 실험

### 5.1 즉시 (current sprint)

1. **lmtune search ↔ LLMDK8sAdapter 통합 실행** (Task #154)
   - `lmtune search start --adapter llmd-k8s --space configs/search/spaces/vllm_engine_args_tier1.yaml --max-trials 4` 로 자동 cycle. 본 검증의 수동 helmfile apply 를 Python 어댑터가 똑같이 하는지.
   - 가설: render_values_overlay 가 만든 overlay 가 위 수동 검증과 같은 형태이고 같은 결과 도출.

2. **B200 적용** (Task: 별도 phase)
   - 위 § 4.3 의 3 단계.
   - `values-llama-3.1-8b-smoke.yaml.gotmpl` PR.
   - B200 호스트에서 동일 검증 (Trial A/B → /v1/completions 200 + 다른 args 박힘).

### 5.2 본 분석이 발견한 검증 게이트 강화 항목 (process change)

- `b200/docs/ANALYSIS_template.md` 에 § 4.1 같은 **명제별 입증/미증 표** 섹션 의무화. acceptance 좁힘 → 다음 phase 가 그 좁힘을 알고 진입.
- minikube validation runbook 에 "params injection 은 단발 generation 까지 검증" 명시. ask/tell 만 검증하는 절반-PASS 금지.

### 5.3 미해결 / 자료 보강 필요

- 본 검증은 `--enforce-eager` 사용 (mini variant 디폴트). torch.compile/CUDA graph 활성화 시 cycle 시간이 늘어날 가능성 (compile cache 미스).
- modelservice chart v0.4.12 의 svc 자동 생성 옵션 — peer repo 의 helmfile 이 다른 path (P/D, wide-EP) 에서는 어떻게 처리하는지 비교 필요.
- 본 검증은 단일 GPU. multi-GPU TP/DP cycle 은 별도 검증 필요.

---

## 6. 참조

- 본 분석 직전 세션의 사용자 인용 8 종 (의미: 라벨링 정밀도 강제)
- S6 ANALYSIS.md (`b200/studies/minikube_s6_validation/ANALYSIS.md`) — 본 분석이 매운 미구현 분기의 출처
- Helmfile docs — values gotmpl 의 `.StateValues` 액세스
- llm-d-modelservice v0.4.12 — `routing.proxy.enabled: false` 가드 한계
- agentgateway v1.0.0 — InferencePool 라우팅 미지원 (cross-env 함정)
