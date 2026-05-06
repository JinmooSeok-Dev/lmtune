# B200 운영 결함 카탈로그 (regressions)

> 운영 중 한 번이라도 사용자 시간을 빼앗은 결함은 본 catalog 에 entry 로 영속화한다. 신규 entry 는 PR 머지 의무 (CLAUDE.md 의 § PR 게이트). 동일 결함 재발생 시 본 문서가 1차 진단처가 된다.
>
> Entry 형식: 고유 ID `R<n>` / 증상 / 진단 / 코드 영속화 위치 / 회귀 테스트.

## R1 — `kubectl wait deploy -l llm-d.ai/role=decode` 가 "no matching resources"

**증상**
```
[helmd::wait] ns=b200-infsch  timeout=20m
error: no matching resources found
[prepare] decode 가 20m 안에 Available 안 됨
NAME                                                  READY   STATUS    RESTARTS   AGE
ms-infsch-llm-d-modelservice-decode-...               1/1     Running   0          24m
```

Pod 는 1/1 Running 인데 Deployment wait 실패.

**진단**
`llm-d-modelservice` chart v0.4.12 가 Deployment **metadata.labels** 에 `llm-d.ai/role=decode` 를 안 붙임. 라벨은 `spec.selector.matchLabels` 와 `spec.template` (= Pod) 에만 존재.

| 명령 | 결과 |
|:---|:---|
| `kubectl wait deploy -l llm-d.ai/role=decode` | no matching resources |
| `kubectl get pods -l llm-d.ai/role=decode` | 매칭 정상 |

**영속화 위치**
- Code: `b200/scripts/util/helm.sh::helmd::wait_decode_ready` — 이름 패턴 `*-llm-d-modelservice-decode` 매칭으로 deployment 들을 직접 발견 후 `wait deploy/<name>` 호출
- Test: `b200/scripts/tests/test_helm_util.sh` — fake-kubectl `FAKE_K8S=has_decode` / `no_decode` 시나리오

## R2 — `set -u` + 한 줄 다중 `local` 선언이 unbound variable 폭주

**증상**
```
$ bash b200/scripts/ops/prepare.sh infsch --apply
── [1] cluster check (rn=infsch, ns=b200-infsch)
util/env.sh: line 50: rn: unbound variable
```

**진단**
bash 의 `local rn="$1" ns="b200-${rn}"` 한 줄 다중 선언은 `set -u` 환경에서 `${rn}` 평가 시점에 rn 이 같은 줄 좌측 변수임에도 local 스코프에 아직 바인드되지 않은 것으로 평가되어 unbound 폭주.

**영속화 위치**
- Code: `b200/scripts/util/env.sh::cluster_check`, `helm.sh::{list, releases_check, wait_decode_ready}` — 두 줄로 분리
- Test: `b200/scripts/tests/test_env_util.sh` (cluster_check 호출), `test_helm_util.sh` (helmd 함수들 호출)
- Test 가 `set -u` 환경에서 동작하도록 모든 unit test 가 명시적으로 `set -u` 적용

## R3 — Deployment RollingUpdate 가 GPU deadlock 유발

**증상**
- 사용자 보고: "이전 ms-decode 는 삭제도 안 되고, 새로운 게 또 생성되서 스케쥴링 되려 함"
- 새 pod `Pending` `Insufficient nvidia.com/gpu`
- 기존 pod 안 죽음 → 영원히 deadlock

**진단**
chart default = `RollingUpdate` (maxSurge=25%, maxUnavailable=25%):
- decode pod 1 개 = TP=8 = **8 GPU 점유**
- replicas=2 → 클러스터 16 GPU 다 점유
- helm upgrade 시 surge 새 pod 8 GPU 못 잡음 → Pending → 기존 pod 안 죽음 → deadlock

GPU 가 클러스터 전체 크기인 LLM 서빙에선 surge 가 의미 없으므로 `Recreate` 가 정합 strategy.

**영속화 위치**
- Code: `b200/helmfile/_postrender/postrender.sh` — 모든 Deployment 의 `spec.strategy` 를 `{type: Recreate}` 로 patch (kustomize replace op)
- Test: `b200/scripts/tests/test_postrender_strategy.sh` — multi-doc YAML stdin → strategy.type=Recreate 보장

## R4 — `--backend k8s-job` 은 hard-gated (Phase S4 미구현)

**증상**
```
$ lmtune search start --backend k8s-job ...
Error: k8s-job backend 은 Phase S4 에서 활성화됩니다
```

**진단**
`src/lmtune/cli_search.py:243` 가 명시적으로 차단. helmfile redeploy 흐름은 `--adapter llmd-k8s` 가 담당하고, k8s-job 은 trial parallelism (workers > 1) 의 별도 axis.

**영속화 위치**
- Code: `src/lmtune/cli_search.py` — error 메시지에 의도 명시
- Doc: 본 catalog + README § Operations 의 lmtune 명령 예시

## R7 — pf::probe budget 5분이 큰 MoE 모델 weight 로딩에 부족

**증상**
사용자가 `bash b200/scripts/ops/launch.sh b200/endpoints/b200_gpt-oss-120b.yaml infsch` 실행 → step 7 의 `/v1/models polling` 에서 hang 처럼 보임. 실제로는 vLLM 이 117B MXFP4 weight 다운로드+로딩 (10–25분) 중.

```
── [launch:infsch] 7. /v1/models 200 polling
[pf::probe] http://127.0.0.1:8011/v1/models
                               ← 5분 후 fail 또는 사용자 Ctrl+C 까지 hang 으로 오해
```

**진단**
- `pf::probe` 의 default budget = 60 attempts × 5s = **5분**
- gpt-oss-120b 117B MoE MXFP4 weight 다운로드 + GPU 로딩 = **10–25분**
- endpoint YAML 의 `rollout_timeout_s: 1500` (= 25분) 와도 정렬 안 됨
- 진행 표시 없어서 사용자에게 hang 처럼 보임 (실은 정상 polling)

**영속화 위치**
- Code: `b200/scripts/util/pf.sh::pf::probe` — default budget 60→360 (5분→30분), 60초마다 진행 표시 stderr 로 출력
- Test: `b200/scripts/tests/test_pf_util.sh` — probe 가 빠른 응답 시 즉시 break (긴 budget 도 OK)

**관련**
사용자가 launch.sh 에서 hang 이라고 판단해 Ctrl+C 후 lmtune 직접 실행하는 우회를 발견 — `LLMDK8sAdapter.apply()` 의 probe budget = `rollout_timeout_s` (25분) 가 충분히 길어 lmtune 쪽은 정상 처리. 즉 lmtune 본체 흐름엔 결함 없음, **launch.sh 의 시간 정렬만 결함**.

## R5 — endpoint url=127.0.0.1:8011 ↔ port-forward / 모델 / strict ordering

**증상**
- helmfile apply → 첫 install 시 `B200_MODEL_VALUES` env 미설정으로 default llama 가 떠버림
- redeploy 시 모델이 갈아끼워졌는지 사용자가 손으로 확인해야
- port-forward 가 helmfile rolling 마다 끊김
- 단계 누락 시 chain 폭발 (port-forward 안 떠 → lmtune 5 trial fail → circuit breaker halt)

**진단**
"처음 시작" 과 "재실행" 이 vLLM 본성상 같은 비용 (config change = engine restart = weight reload) 인데, 운영 도구가 두 시나리오를 분리된 손작업 시퀀스로 노출.

**영속화 위치**
- Code: `b200/scripts/ops/launch.sh` — 8 단계 자동 (endpoint 파싱 → values 매핑 → cluster check → release/모델 검증 → helmfile apply → wait → pf → probe → model id 검증)
- Code: `b200/scripts/util/env.sh::values_for_model` — model id → values 파일 매핑 카탈로그 (신규 모델은 한 줄 추가)
- Code: `b200/scripts/util/pf.sh::current_model` — `/v1/models` 응답에서 model id 추출, mismatch 자동 감지
- Test: `b200/scripts/tests/test_env_util.sh` (값 매핑), `test_pf_util.sh` (model id 추출)

## R8 — TP/EP/DP infeasible 후보가 helmfile redeploy 후에야 reject (3분 낭비)

**증상**
- `lmtune search start --space b200/search-spaces/b3_parallelism.yaml` 가 sampler 에서 `(tp=16, ep=3)` 같은 명백히 infeasible 한 후보를 sample
- helmfile apply (3분) → vLLM startup → engine 이 `Number of attention heads (64) % tensor_parallel_size (16) != 0` 같은 에러로 crash → SLO timeout 으로 reject
- 한 trial 당 3-5분 낭비, study 전체 wallclock 폭발

**진단**
sampler (TPE/Random/NSGA-II) 가 search-space 의 axis 개별 분포만 보고 sample. axis 간 cross-constraint (`model.numAttentionHeads % tp == 0`, `model.numExperts % ep == 0`, `tp <= npus_per_server` 등) 는 axis 정의에 표현 불가 — sample 한 후 별도 evaluator 가 reject 해야 한다.

vllm-config-puzzle simulator (TypeScript) 가 같은 문제를 `validation.ts` 의 10 룰로 풀고 있음. 우리는 그 알고리즘을 1:1 port 해서 본 프로젝트의 search loop 에 wire-up.

**영속화 위치**
- Code: `src/lmtune/search/feasibility.py` — Constraint AST evaluator (whitelist-only, eval 안전). 12 룰 declarative loader.
- Code: `b200/search-spaces/b3_parallelism.yaml::feasibility_constraints` — vllm-config-puzzle/validation.ts:31~162 의 10 룰 + 2 보조 (warning/dp-pair) 1:1
- Code: `src/lmtune/models/registry.py` — gpt-oss-120b/Llama/Qwen/MoE 메타 카탈로그 (constraint 의 `model.*` 참조)
- Code: `src/lmtune/search/study.py::Study.ask()` — sample 후 `_FeasibilityChecker.is_feasible()` 호출, infeasible 시 `optuna.tell(state=PRUNED)` 후 retry (max 30회). helmfile redeploy 0회.
- Code: `src/lmtune/search/space.py::SearchSpace.feasibility_constraints` — YAML 의 `feasibility_constraints` 블록을 SearchSpace 에 carry, `to_yaml()` 에서도 round-trip
- Test: `tests/search/test_feasibility.py` — 12 룰 + gpt-oss-120b 5 시나리오 (TP=8/DP=2 feasible, TP=16 reject, TP=3 reject by heads%TP, EP=3 reject by experts%EP, wide-EP DP=16 feasible, PP=2 cross_node=none reject)
- Test: `tests/search/test_study.py::test_study_feasibility_skips_infeasible_candidates` — Study.run() 이 infeasible 후보를 helmfile 호출 없이 prune 하는지 검증
- Test: `tests/search/test_study.py::test_study_feasibility_disabled_when_no_environment` — context 에 environment 없으면 checker 미설치 (안전한 default)

**활성 조건**
`StudyConfig.context['environment']` 에 `Environment` 객체 (b200_dual_node / b200_single_node / local_single_gpu) 를 명시 주입해야 활성. `model_id` 도 같이 넣으면 model.* 참조도 평가. 둘 중 하나라도 없으면 checker 미설치 — 모든 candidate 가 그대로 실행 (회귀 안전).

---

## R9 — EPP v1.5.0 의 `lora_requests_info` system-default 가 vLLM (LoRA 비활성) backend 를 unhealthy 처리 → "no valid backends"

**증상**
```
$ curl -s http://127.0.0.1:8011/v1/models
no valid backends
```
- decode pod 1/1 Running, 직접 `curl http://localhost:8000/v1/models` (pod 내부) 200 정상
- pod label 도 InferencePool selector 와 정합 (`llm-d.ai/inferenceServing=true`)
- InferencePool status `Accepted` + `ResolvedRefs` 다 True
- EPP 로그에 반복 에러:
  ```
  extract failed   extractor: core-metrics-extractor
  error: metric family "vllm:lora_requests_info" not found
  ```

**진단**
- EPP image `registry.k8s.io/gateway-api-inference-extension/epp:v1.5.0` 가 `system defaults` 로 `metrics-data-source` + `core-metrics-extractor` plugin 을 강제 주입.
- ConfigMap 의 `default-plugins.yaml` 에서 두 plugin 을 빼도 EPP 가 다시 채워 넣음 (configloader.go:107 의 `Instantiated all plugins and applied system defaults`).
- `core-metrics-extractor` 는 `vllm:lora_requests_info` metric 을 require — vLLM 이 `--enable-lora` 없이 시작하면 export 안 됨 → metric extract 실패가 datalayer 의 `logErrorTransition` 으로 endpoint 를 unhealthy 처리 → backend 0개 → gateway 가 "no valid backends" 응답.
- `--lora-info-metric=` 빈 값 override 도 v1.5 에서 deprecated (`flag "lora-info-metric" is deprecated and cannot be used; configure metrics via engineConfigs in EndpointPickerConfig instead`) — chart values 에 `engineConfigs` schema 노출 안 됨.
- 동일 chart v1.5.0 의 EPP image 만 v1.4.0 으로 downgrade 하면 metric 못 찾아도 warning 만, backend healthy 유지. minikube + chart v1.5.0 + EPP v1.4.0 정상 시작 검증.

**영속화 위치**
- Config: `b200/helmfile/inference-scheduling/values-gaie.yaml` — `inferenceExtension.image.tag: v1.4.0` 추가
- Config: `b200/helmfile/wide-ep-lws/values-gaie.yaml` — 동일
- Config: `b200/helmfile/pd-disaggregation/values-gaie.yaml` — 동일
- 사용자가 helmfile reapply 시 EPP image 가 자동 v1.4.0 으로 재배포됨

**즉시 적용 (이미 떠있는 cluster, helmfile reapply 전)**
```bash
kubectl set image deployment/gaie-infsch-epp -n b200-infsch \
  epp=registry.k8s.io/gateway-api-inference-extension/epp:v1.4.0
kubectl rollout status deployment/gaie-infsch-epp -n b200-infsch
```

**향후 v1.5+ 채택 시 필요 작업**
- 새 schema (`engineConfigs in EndpointPickerConfig`) 로 lora 등 optional metric 비활성 표현 검증
- chart values 의 `pluginsCustomConfig` 에 engineConfigs YAML 작성
- `system defaults` 로 추가되는 plugin 을 어떻게 disable 할 수 있는지 chart 내부 동작 재확인

---

## 신규 결함 entry 추가 절차

1. 결함 발견 (사용자 보고 / 운영 중 발생)
2. 즉시 fix 코드 작성
3. 본 catalog 에 R<n> entry — 증상 + 진단 + 영속화 위치 + 회귀 테스트
4. `b200/scripts/tests/test_*.sh` 에 회귀 테스트 1건 이상 추가
5. `bash b200/scripts/tests/run_all.sh` 통과
6. PR 한 번에 (코드 fix + catalog entry + test) 묶음
7. 머지 후 본 catalog 가 다음 동일 패턴의 1차 진단처

PR 본문에 회귀 테스트만 적고 코드는 안 박는 패턴 (PR #24~#26 의 결함) 은 본 catalog 가 강제로 차단한다.
