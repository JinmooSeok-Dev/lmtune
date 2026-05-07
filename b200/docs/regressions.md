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

3 path 모두에 두 단계 fix 가 필요 (PR #95 의 image-only fix 가 불완전 — chart configmap 의 v1.5 schema 가 v1.4 EPP 의 모르는 plugin type `core-metrics-extractor` 을 포함해 EPP 시작 자체가 실패):

- Config: `b200/helmfile/{inference-scheduling,wide-ep-lws,pd-disaggregation}/values-gaie.yaml` — 모두 동일 schema:
  - `inferenceExtension.image.tag: v1.4.0` (image downgrade)
  - `inferenceExtension.pluginsConfigFile: custom-plugins.yaml` (EPP args `--config-file` redirect)
  - `inferenceExtension.pluginsCustomConfig.custom-plugins.yaml: |` (v1.4 호환 plugin 만 — queue-scorer / kv-cache-utilization-scorer / prefix-cache-scorer)
- mini helmfile (`helmfile-mini.yaml.gotmpl`) 도 동일 `values-gaie.yaml` 사용 — 자동 반영

**검증 (minikube + chart v1.5.0 + EPP v1.4)**
- chart 가 configmap 에 `default-plugins.yaml` (v1.5 schema, EPP 미사용) + `custom-plugins.yaml` (v1.4 호환, EPP 사용) 둘 다 emit
- EPP args 는 `--config-file=/config/custom-plugins.yaml` 로 우리 override 만 가리킴
- `core-metrics-extractor` 부재 → lora metric polling 안 함 → backend healthy

**즉시 적용 (이미 떠있는 cluster, helmfile reapply 전)**
```bash
# image v1.4 + configmap 에서 v1.4 미지원 plugin 둘 제거 (core-metrics-extractor + metrics-data-source)
kubectl set image deployment/gaie-infsch-epp -n b200-infsch \
  epp=registry.k8s.io/gateway-api-inference-extension/epp:v1.4.0
kubectl get cm gaie-infsch-epp -n b200-infsch -o yaml \
  | sed '/- type: core-metrics-extractor/d; /- type: metrics-data-source/,/insecureSkipVerify: true/d' \
  | kubectl apply -f -
kubectl rollout restart deployment/gaie-infsch-epp -n b200-infsch
kubectl rollout status deployment/gaie-infsch-epp -n b200-infsch
```

**향후 v1.5+ 채택 시 필요 작업**
- 새 schema (`engineConfigs in EndpointPickerConfig`) 로 lora 등 optional metric 비활성 표현 검증
- chart values 의 `pluginsCustomConfig` 에 engineConfigs YAML 작성 + `pluginsConfigFile` 로 redirect
- `system defaults` 로 추가되는 plugin 을 어떻게 disable 할 수 있는지 chart 내부 동작 재확인

---

## R10 — httproute.yaml `backendRef.group` 이 deprecated `x-k8s.io` → "no valid backends" / "route not found"

**증상**
```
$ curl -s http://127.0.0.1:8011/v1/models
no valid backends
# 또는
route not found
```
- pod 4개 (gateway / EPP / decode×2) 다 1/1 Running
- decode pod 직접 curl 시 vllm 200 정상
- EPP 가 v1.4 + custom-plugins.yaml 로 정상 시작 (R9 fix 적용)
- httproute 적용됨 (`kubectl get httproute` 보임)
- 그러나 gateway 가 backend 거절

**진단**
httproute status 의 `ResolvedRefs` 가 False:
```yaml
- conditions:
  - reason: InvalidKind
    status: "False"
    type: ResolvedRefs
    message: 'referencing unsupported backendRef: group "inference.networking.x-k8s.io" kind "InferencePool"'
```

InferencePool API group 이 chart v1.5.0 에서 정식 승격됨 — `inference.networking.x-k8s.io/v1alpha2` (deprecated) → `inference.networking.k8s.io/v1`. EPP runner.go 의 `--pool-group` default 가 `inference.networking.k8s.io` 인 것으로도 확인 가능. agentgateway 는 deprecated x-k8s 를 unsupported 로 reject.

**영속화 위치**
- Config: `b200/helmfile/{inference-scheduling,wide-ep-lws,pd-disaggregation}/httproute.yaml` — `backendRefs[0].group: inference.networking.k8s.io` 로 수정
- 검증: minikube 에서 httproute apply 후 `kubectl get httproute -n <ns> -o yaml | grep -A5 ResolvedRefs` 가 `status: "True"` 인지

**즉시 적용 (이미 떠있는 cluster, helmfile reapply 전)**
```bash
kubectl get httproute b200-infsch-smoke -n b200-infsch -o yaml \
  | sed 's|inference.networking.x-k8s.io|inference.networking.k8s.io|' \
  | kubectl apply -f -
# 잠시 후
kubectl get httproute b200-infsch-smoke -n b200-infsch -o yaml | grep -A5 ResolvedRefs
```

---

## R11 — simulator-only axis 가 vllm CLI args 로 emit 되어 trial 전부 reject

**증상**
study 시작 후 모든 trial 이 vllm crash:
```
WARNING ... Found duplicate keys --tensor-parallel-size
vllm: error: unrecognized arguments: --cross-node-type roce --intra-node-type pcie --node-split-strategy dual-node-pp2-tp8
```
search-space 에 `cross_node_type` 등 axis 가 없어도 발생 — 특히 `--warmstart-db` 옵션 사용 시 (옛 b3_parallelism study 의 trial params 가 enqueue 됨).

**진단**
- `src/lmtune/deploy/base.py::merge_params_into_endpoint` 의 fallback 분기 (line 89-90):
  ```python
  else:
      engine_args[k] = v   # "Unknown keys are written under deployment.engine_args
                           # (vLLM passes unknowns through as CLI flags)"  ← 잘못된 가정
  ```
- 코멘트가 "vllm passes unknowns through" 라 가정했지만 vllm 은 unrecognized arg 로 reject. simulator metadata (cross_node_type / intra_node_type / node_split_strategy / pcp / dcp / ep_strategy / sequence_parallel) 가 그대로 vllm CLI 로 흘러감.
- warmstart-db 가 옛 study 의 trial.params 그대로 enqueue → 새 search-space 에 없는 axis 도 trial.params 에 carry → engine_args 로 emit.

**영속화 위치**
- Code: `src/lmtune/deploy/base.py::_SIMULATOR_ONLY_KEYS` — 7 axis 명시 set
- Code: `merge_params_into_endpoint` 가 simulator-only key 를 silently skip (engine_args / parallelism / replicas 어느 쪽으로도 안 감)
- Test: `tests/deploy/test_base.py::test_merge_simulator_only_axes_dropped` — 7 simulator key 모두 어디에도 안 나타남
- Test: `tests/deploy/test_base.py::test_merge_simulator_only_warmstart_replay` — warmstart 시나리오 (10 키 enqueue 중 5 simulator drop, 5 정상 emit)

**즉시 우회 (study 진행 중)**
- `--warmstart-db ... --warmstart-top-k 8` 옵션을 빼고 study 재시작 (옛 study 의 dirty params 차단)
- 또는 PR #98 (R11 fix) 머지 후 git pull — warmstart 사용 가능

**향후 (chart wiring 검증 시 axis 합류)**
PCP/DCP/ep_strategy 가 chart values gotmpl 의 vllmArgs 경로에 정상 emit 되는 것 검증 후 본 set 에서 제거 + `_PARALLELISM_KEYS` 또는 `_ENGINE_ARG_KEYS` 로 합류.

---

## R12 — `merge_params_into_endpoint` 의 in-place YAML mutation 이 endpoint 영구 오염

**증상**
study 한 번 돌리면 `git status b200/endpoints/<endpoint>.yaml` 에 변경 발생. trial params 가 그대로 endpoint YAML 에 박혀 다음 study 가 그것을 reuse → vllm reject (R11 의 root cause). 사용자가 endpoint 깨끗 상태에서 시작했다 가정해도 `git checkout HEAD -- <endpoint>` 로 매번 원복해야 함.

**진단**
- `src/lmtune/deploy/base.py::merge_params_into_endpoint` 가 endpoint YAML 을 in-place 로 mutate + write back (line 137 `p.write_text(...)`)
- `LLMDK8sAdapter.apply()` 는 그 결과 dict 만 사용 (helmfile state-values 로 emit) — file write 가 불필요
- 그러나 모든 trial 마다 file 변경 → study 종료 시 endpoint YAML 이 마지막 trial 의 params 로 영구 오염
- 다음 study 가 그 endpoint 를 reuse → 새 trial.params 와 stale params 가 merge → simulator axis (cross_node_type 등) 가 vllm 에 emit (R11)
- `LocalVLLMAdapter` 만이 진짜로 file 이 필요 (`scripts/vllm_restart.sh` 가 endpoint YAML 을 읽음)

**영속화 위치**
- Code: `src/lmtune/deploy/base.py::merge_params_to_dict` — pure function, file write 안 함, dict 만 반환
- Code: `src/lmtune/deploy/base.py::merge_params_into_endpoint` — 기존 함수 유지 (LocalVLLMAdapter 호환), `merge_params_to_dict + write_text` 의 wrapper 로 단순화
- Code: `src/lmtune/deploy/llmd_k8s.py::apply` — `merge_params_to_dict` 호출로 변경 (file 안 건드림)
- Test: `tests/deploy/test_base.py::test_merge_params_to_dict_does_not_write_file` — 호출 후 파일이 byte-identical 한지 검증
- Test: `tests/deploy/test_base.py::test_merge_params_into_endpoint_still_writes_for_local_vllm` — backward compat (LocalVLLMAdapter) 보장
- E2E: minikube 에서 lmtune search start 1 trial 실행 후 `diff` 가 0 차이, 그러면서 pod args 는 정상 emit 검증

---

## R13 — gpt-oss-120b wide-EP 토폴로지에서 `dtype: bfloat16` 강제로 OOM (KV profiling 단계)

**증상**
- B200 wide-ep-lws path (`b4-gpt-oss-120b-wide-ep`) 의 첫 trial 에서 vllm pod 가 KV profiling 단계에서 worker subprocess 사망
- 로그: `RuntimeError: Worker failed with error ''` (빈 따옴표) — DP0~DP3 EngineCore 모두 같은 패턴
- Stack trace 는 `vllm/v1/engine/core.py:120 _initialize_kv_caches → vllm/v1/engine/core.py:252 model_executor.determine_available_memory()` 에서 끊김
- b3 inference-scheduling (TP=8 × DP=2 packed) 는 동일 dtype 으로 정상 동작

**진단**
- 빈 RuntimeError = subprocess 가 traceback 남길 시간 없이 SIGKILL 당함 → OOM Killer 가 worker process 직살. CUDA exception 이면 stack trace 가 채워짐
- gpt-oss-120b 는 publish 자체가 **MXFP4 quantized native** (~63 GB). endpoint 의 `dtype: bfloat16` 강제가 vllm 으로 하여금 dequantize → ~234 GB 로 부풀림 (4×)
- TP=2 × DP=4 wide-EP 토폴로지에서 각 GPU 가 117 GB weight 점유 + KV profiling 시도 → 192GB B200 의 0.9 한계 (172 GB) 를 일시적으로 over-shoot
- b3 (TP=8 packed) 가 OK 였던 이유: weight 가 8 GPU 에 spread → ~29 GB/GPU
- 본 catalog 의 R11/R12 와 무관 (simulator-only key 와 endpoint mutation 모두 OK 였음)

**영속화 위치**
- Code: `b200/endpoints/b200_gpt-oss-120b-wide-ep.yaml` — `engine_args.dtype` 줄 제거. 주석으로 R13 reference + bf16 강제 시 OOM 위험 명시
- Code: `b200/helmfile/wide-ep-lws/values-gpt-oss-120b-wide-ep.yaml.gotmpl` — `$defaults` dict 에서 dtype 제거. 주석으로 R13 reference. → vllm 가 모델 config 의 `quantization_config` 자동 감지 (MXFP4 native 사용)
- 검증: helmfile template render 결과 args 에 `--dtype` emit 안 됨 (이전: `--dtype bfloat16` emit). vllm 자동 감지로 native MXFP4 사용

**향후 (다른 native-quant 모델 추가 시 동일 패턴 차단)**
DSV3 (FP8 native), Kimi K2 (FP8 native), Llama-4 (bf16 native) 등 추가 시 dtype 결정 룰:
- 모델이 native quantized 형식 (FP8/MXFP4/nvfp4) 으로 publish → **dtype 미설정** (vllm 자동 감지)
- 모델이 bf16/fp16 publish → 명시적 dtype 가능
- model_catalog (`b200/docs/model_catalog_2026-05.md`) 의 dtype 컬럼이 source of truth

---

## R14 — wide-EP values gotmpl 의 args duplicate emission + invalid `nixl_ep` choice

**증상**
- `b4-gpt-oss-120b-wide-ep` study 진행 중 vllm pod 가 두 가지로 죽음:
  1. (FATAL) `vllm serve: error: argument --all2all-backend: invalid choice: 'nixl_ep'`
  2. (WARNING) `Found duplicate keys --max-model-len, --enable-expert-parallel, --gpu-memory-utilization, --kv-cache-dtype, --max-num-seqs`
- 두 이슈 합쳐 첫 trial 부터 모든 trial 이 score=0 (vllm CLI parse 실패 → pod CrashLoopBackOff)

**진단**
- **R14a (invalid choice)**: 내가 search-space 에 넣은 `nixl_ep` 가 vllm 0.17.1 의 `--all2all-backend` valid choice 가 아님. 실제 vllm 의 valid 목록 = `{allgather_reducescatter, deepep_high_throughput, deepep_low_latency, flashinfer_all2allv, mori, naive, pplx}`. `nixl_ep` 는 vllm release notes 의 KV transport (NIXL) 와 혼동한 결과 — KV cache 분산용 transport 와 expert all-to-all backend 는 별개.
- **R14b (duplicate emission)**: `src/lmtune/deploy/llmd_k8s.py::render_values_overlay` 가 `vllmArgs` 를 **kebab-case** 로 emit (line 128-129: `vllm_args[k.replace("_", "-")] = v`) + `ep:true` 시 `enable-expert-parallel: true` 자동 추가 (line 139-140). 그런데 내 `values-gpt-oss-120b-wide-ep.yaml.gotmpl` 의 `$defaults` dict 가 **snake_case** (`max_num_seqs` 등) 였고 + 하드코드 `- "--enable-expert-parallel"` 라인 까지 있었음. mergeOverwrite 가 snake/kebab 를 다른 키로 보존 → range 가 둘 다 emit → 같은 logical flag 가 두 번. inference-scheduling 의 ORIGINAL 패턴 (kebab defaults + replace 없음 + 하드코드 expert flag 없음) 을 잘못 변경한 것.

**영속화 위치**
- Code: `b200/search-spaces/b4_gpt_oss_120b_wide_ep.yaml` — `all2all_backend.values` 에서 `nixl_ep` 제거, `flashinfer_all2allv + pplx` 추가 (vllm 0.17.1 의 7 valid choice 중 의미 있는 5개). 주석으로 R14a reference + valid 목록 명시
- Code: `b200/helmfile/wide-ep-lws/values-gpt-oss-120b-wide-ep.yaml.gotmpl`:
  - `$defaults` dict 키를 kebab-case 로 통일 (`max-num-seqs` 등) → adapter 가 emit 하는 vllmArgs (kebab) 와 mergeOverwrite 가 정확히 collapse
  - 하드코드 `- "--enable-expert-parallel"` 라인 제거 — adapter 가 `parallelism.ep:true` 시 vllmArgs 로 자동 inject
  - `replace "_" "-"` 제거 — defaults/vllmArgs 둘 다 kebab 이라 불필요
  - 주석으로 R14a/R14b reference
- Test: 본 결함은 search-space YAML 의 invalid value + values gotmpl 의 mergeOverwrite 동작 결합이라 shell-only 테스트로 caught 안 됨. 대신:
  - `tests/deploy/` 의 render_values_overlay 동작 검증 테스트가 kebab-case emit 을 보장
  - `b200/helmfile/wide-ep-lws/values-gpt-oss-120b-wide-ep.yaml.gotmpl` 의 주석 — 다음 사용자가 새 wide-EP values gotmpl 추가 시 kebab defaults + 하드코드 flag 없이 작성하도록 가이드
  - 본 catalog entry — search-space 추가 시 axis values 가 vllm 실제 CLI choice 인지 검증하는 1차 진단처

**향후 (새 axis 추가 시 동일 패턴 차단)**
- search-space 에 categorical axis 값 추가 시: `vllm serve --help | grep -A 5 <flag>` 으로 valid choice 사전 확인 의무
- 새 모델용 values gotmpl 작성 시: inference-scheduling 의 패턴 그대로 복사 (kebab defaults + range 만, 하드코드 flag 추가 금지). chart 가 자동 inject 하는 axis 식별 후 그것만 명시 emit 회피.

---

## R15 — search-space 의 `ep` axis 가 무력 → enable_eplb 와 결합 시 vllm pydantic 거부

**증상**
- PR #101 (R13/R14 fix) 머지 후 b4 wide-EP study 재시작 시 일부 trial 의 vllm pod 가 다음 에러로 즉사:
  ```
  pydantic_core._pydantic_core.ValidationError: 1 validation error for ParallelConfig
    Value error, enable_expert_parallel must be True to use EPLB.
  ```
- vllm CLI args 보면 `--enable-eplb` 는 있는데 `--enable-expert-parallel` 이 빠져있음

**진단**
- 내가 만든 b4 search-space:
  ```yaml
  ep:
    type: bool
    values: [true]   # ← bool sampler 가 무시
  ```
- `type: bool` 은 sampler 가 무조건 true/false 둘 다 sample. `values: [true]` 제약은 categorical type 에서만 동작
- 결과: 50% 의 trial 이 `ep=False` 로 sample → adapter (`render_values_overlay`) 가 `if parallelism.get("ep")` 조건 통과 못 해서 `enable-expert-parallel: true` 를 vllmArgs 에 추가 안 함
- 그런데 `enable_eplb` axis 는 별개로 sample 되어 50% true → 그 중 절반이 `(ep=False, enable_eplb=True)` infeasible 조합 → vllm pydantic ValidationError
- 같은 패턴이 `enable_dbo` (DBO = Dual Batch Overlap, MoE-only) 에도 발생 가능 — 본 fix 로 자동 해결 (ep 항상 true)

**영속화 위치**
- Code: `b200/search-spaces/b4_gpt_oss_120b_wide_ep.yaml` — `ep` axis 자체 삭제 (wide-EP study 의 정의상 항상 true 라 axis 가 아님). 주석으로 R15 reference + 제거 사유 명시
- Code: `b200/helmfile/wide-ep-lws/values-gpt-oss-120b-wide-ep.yaml.gotmpl` — `$defaults` dict 에 `"enable-expert-parallel" true` 추가 → 모든 trial 에 강제 emit. 주석으로 R15 reference
- 검증: helmfile template render (state-values 시뮬레이션) — `--enable-expert-parallel` 가 vllmArgs 에 ep 가 없어도 항상 emit

**향후 (다른 study 에서 fixed-on flag 패턴 차단)**
- bool axis 에 단일 값 강제하고 싶을 때: `type: categorical, values: [true]` 사용 (bool 이 아니라). categorical sampler 는 values 제약 준수
- 또는 더 좋은 방법: search-space 에서 axis 제거 + values gotmpl `$defaults` 에 강제 — fixed parameter 는 axis 가 아니라는 시맨틱 정확
- search-space PR review 룰: `type: bool` + `values:` 같이 적힌 axis 발견 시 reject (모순 시그널)

---

## R16 — wide-EP study 의 DBO + non-deepep all2all_backend infeasible 조합

**증상**
- PR #102 (R15 fix) 머지 후 b4 study 재시작 시 일부 trial 에서 vllm pydantic ValidationError:
  ```
  Microbatching currently only supports the deepep_low_latency and
  deepep_high_throughput all2all backend. allgather_reducescatter is not supported.
  ```

**진단**
- vllm v0.17.1 의 `vllm/config/vllm.py:1128-1134` 가 `enable_dbo=True` 시 `all2all_backend ∈ {deepep_low_latency, deepep_high_throughput}` 을 강제
- 내가 b4 search-space 에 넣은 `all2all_backend` values = `[allgather_reducescatter, deepep_low_latency, deepep_high_throughput, flashinfer_all2allv, pplx]`
- `enable_dbo` 가 별개 axis 로 50% true sample → trial 의 일부가 `(enable_dbo=true, all2all_backend=allgather_reducescatter)` infeasible 조합 → vllm reject
- 같은 패턴이 다른 non-deepep choice (flashinfer / pplx / mori / naive) 에도 적용

**영속화 위치**
- Code: `b200/search-spaces/b4_gpt_oss_120b_wide_ep.yaml` — `all2all_backend.values` 를 `[deepep_low_latency, deepep_high_throughput]` 으로 축소. 주석으로 R16 reference + 다른 backend 의 baseline 비교는 별도 study (b4-baselines) 로 분리 명시
- Docs: `b200/docs/vllm_0.17.1_args_catalog.md` § 2.2 — DBO 호환성 룰 vllm_config.py:1128 source 로 명시

**향후 (다른 wide-EP study 추가 시 동일 패턴 차단)**
- `enable_dbo` axis 가 있는 search-space 의 `all2all_backend` 는 자동으로 deepep 2개로 제한 (validator 예정 — § 본 catalog § 9)
- DBO=false 만 쓰는 baseline 비교 study 는 별개 search-space 로 분리

---

## R17 — vllm CLI flag 카탈로그 미확보 → 비-flag axis + wrong name axis (3 sub-issue)

**증상**
- b4 search-space 의 다음 axis 들이 vllm CLI 에 emit 되어도 의미 없음 / 거부됨:
  - `eplb_window_size` (R17a)
  - `eplb_step_interval` (R17b)
  - `dbo_token_threshold` (R17c)
- 일부 trial 의 vllm 로그에서 `Found duplicate keys --eplb-window-size, --dbo-token-threshold` 또는 silently ignored
- 본 결함은 R14a (nixl_ep invalid choice) 와 같은 root cause: **search-space 작성 시 vllm 0.17.1 의 실제 CLI flag 카탈로그를 확보 안 함**

**진단** — vllm v0.17.1 source 직접 확인

| 내가 만든 axis | 실제 vllm 0.17.1 | 출처 |
|:--|:--|:--|
| **R17a** `eplb_window_size` | **CLI flag 아님**. `EPLBConfig.window_size` 가 `--eplb-config` JSON 안 | parallel.py:54-65, arg_utils:916 |
| **R17b** `eplb_step_interval` | **CLI flag 아님**. `EPLBConfig.step_interval` (동일) | 동일 |
| **R17c** `dbo_token_threshold` | **wrong name**. 실제는 `--dbo-decode-token-threshold` (default 32) 와 `--dbo-prefill-token-threshold` (default 512) **분리** | arg_utils:907-913 |

EPLB 의 window_size/step_interval 는 별개 CLI flag 가 아니라 `--eplb-config` JSON 통째로 전달:
```bash
vllm serve ... --eplb-config '{"window_size": 1000, "step_interval": 3000}'
```

→ search-space axis 로 노출하려면 lmtune adapter 가 JSON 합치는 wiring 추가 필요 (별도 PR)

**영속화 위치**
- Code: `b200/search-spaces/b4_gpt_oss_120b_wide_ep.yaml`:
  - R17a/b: `eplb_window_size`, `eplb_step_interval` axis 제거 (vllm default 1000/3000 사용). 주석으로 reference + JSON wiring 추가 후 활성화 가능 명시
  - R17c: `dbo_token_threshold` → `dbo_decode_token_threshold` 로 rename + values 도 vllm default 32 기반으로 조정 ([16, 32, 64, 128])
- Docs: `b200/docs/vllm_0.17.1_args_catalog.md` 신규 — 모든 vllm 0.17.1 CLI flag + Literal choices + cross-flag 제약 source-verified
- Docs: catalog § 7 — lmtune adapter 의 axis ↔ vllm flag 매핑 표

**향후 (validator 자동화)**
- `b200/scripts/validate_search_space.py` 신규 (별도 PR): search-space YAML 의 모든 axis name + categorical values 를 catalog 와 1:1 자동 검증
- pytest mark `@pytest.mark.search_space_validity` 로 회귀 테스트
- search-space 신규/변경 시 catalog 미확인 PR 거부

---

## R18 — adapter `render_values_overlay` 의 `dp → decode.replicas` hardcode 매핑이 wide-EP 에서 over-provision

**증상**
- PR #103 (R16/R17 fix) 머지 후 사용자가 b4 study 재시작 시 vllm pod 일부가 Pending 상태로 stuck:
  ```
  Warning  FailedScheduling  ...  0/2 nodes are available:
  2 Insufficient cpu, 2 Insufficient nvidia.com/gpu.
  ```
- `kubectl get deploy`: `ms-wideep-llm-d-modelservice-decode 4/8 8 4` — DESIRED=8, READY=4
- `kubectl get rs`: 9 ReplicaSet 누적 (매 trial 마다 새 RS 생성)
- 4 Running pods + 4 Pending pods

**진단**
- `src/lmtune/deploy/llmd_k8s.py::render_values_overlay` line 143-144 (당시):
  ```python
  if "dp" in parallelism:
      decode_overlay["replicas"] = int(parallelism["dp"])
  ```
- 이 매핑은 **inference-scheduling (b3) 패턴 가정**: `dp` axis = independent replica count, 각 replica 가 TP=N 로 자체 모델 인스턴스 운영
- **wide-EP (b4) 의 `dp` 는 의미가 다름**: within-pod data-parallel groups (expert 분산용). chart 의 `decode.parallelism.data` 로 emit 되어야 함
- 현재 매핑으로 wide-EP 에서 trial 이 `dp=8` sample → adapter 가 `decode.replicas=8` 로 chart 에 inject
- chart 가 8 replicas 생성, 각 pod 의 GPU 요청 = `tp × dp_default = 2 × 4 = 8` GPU
- 8 replicas × 8 GPU = **64 GPU 요구** (B200 dual-node 가용 = 16) → 4 pods schedule, 4 Pending

**영속화 위치**
- Code: `src/lmtune/deploy/llmd_k8s.py::render_values_overlay` — 신규 `dp_routing` 인자 추가
  - `dp_routing="replicas"` (default): 기존 동작 — `dp → decode.replicas` (b3 inference-scheduling, backward compat)
  - `dp_routing="data"`: wide-EP — `dp → decode.parallelism.data`. `decode.replicas` 는 set 안 함 (chart values 의 hardcode 가 그대로 = 노드 수)
- Code: `src/lmtune/deploy/llmd_k8s.py::LLMDK8sAdapter.__init__` + `from_endpoint` — `dp_routing` field 추가, endpoint YAML 의 `helmfile_overrides.dp_routing` 에서 읽음
- Code: `src/lmtune/deploy/llmd_k8s.py::LLMDK8sAdapter.apply` — `render_values_overlay` 호출 시 `self._dp_routing` 전달
- Endpoint: `b200/endpoints/b200_gpt-oss-120b-wide-ep.yaml` — `helmfile_overrides.dp_routing: data` 추가. 주석으로 R18 reference + GPU over-provision 사유 명시
- Test: `tests/deploy/test_llmd_overlay.py`:
  - `test_overlay_dp_routing_data_for_wide_ep` — wide-EP 매핑 확인
  - `test_overlay_dp_routing_replicas_default_unchanged` — backward compat (default behavior)
  - `test_adapter_from_endpoint_reads_dp_routing` — endpoint YAML 의 hint 가 adapter 에 전달

**향후 (다른 path 추가 시 동일 패턴 차단)**
- 새 well-lit-path 추가 시 endpoint YAML 의 `helmfile_overrides.dp_routing` 값 결정 룰:
  - `inference-scheduling`: `replicas` (default). `dp` = independent model replica
  - `wide-ep-lws`: `data`. `dp` = within-pod data-parallel
  - `pd-disaggregation`: 별도 검토 필요 (prefill / decode 별 replica count 분리)
- search-space 의 `dp` axis values 는 path-specific 으로 좁혀야:
  - inference-scheduling: dp ∈ [1, 2, 4] (replica 수 — GPU 가용 / TP 의 분모)
  - wide-EP: dp ∈ [2, 4, 8] but tp × dp ≤ npus_per_server 제약 (within-pod GPU 한계)

---

## R19 — vllm 0.17.1 의 wide-EP + DeepEP + non-internal-MK quant 조합 NotImplementedError

**증상**
- PR #104 (R18 fix) 머지 + b200-wideep clean reset 후 첫 trial 의 vllm pod 가 init 단계에서:
  ```
  File "vllm/distributed/device_communicators/all2all.py", line 264, in dispatch_router_logits
      raise NotImplementedError
  NotImplementedError
  ```
- DP0~DP3 EngineCore 모두 같은 stack trace, MoE forward path 의 `default_moe_runner.py:657` 에서
  ```python
  hidden_states, router_logits = get_ep_group().dispatch_router_logits(...)
  ```

**진단** — vllm v0.17.1 source 직접 확인

`default_moe_runner.py:631-633` — naive dispatch path 진입 조건:
```python
do_naive_dispatch_combine = (
    self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
)
```

→ **DP > 1** (wide-EP within-pod data-parallel) **AND** **quant 가 internal-MK 미지원**

`all2all.py:237-282` — DeepEP base 의 미구현 메서드:
```python
class DeepEPAll2AllManagerBase(All2AllManagerBase):
    def dispatch_router_logits(...):
        raise NotImplementedError   # line 264
```

DeepEPHTAll2AllManager (HT) 와 DeepEPLLAll2AllManager (LL) 둘 다 base 의 이 메서드를 **override 안 함**. 즉 vllm 0.17.1 의 DeepEP 구현이 미완성.

**호환성 매트릭스** (vllm 0.17.1):

| backend | dispatch_router_logits | DBO 호환 | line |
|:--|:--:|:--:|:--|
| naive | ✅ (조건부) | ❌ | all2all.py:61 |
| allgather_reducescatter | ✅ | ❌ | all2all.py:144 |
| **deepep_low_latency** | **❌ NotImplementedError** | ✅ | all2all.py:264 |
| **deepep_high_throughput** | **❌ NotImplementedError** | ✅ | all2all.py:264 |
| flashinfer_all2allv | (확인 필요) | ❌ | all2all.py:416 |
| pplx | (확인 필요) | ❌ | all2all.py:520 |
| mori | (ROCm 전용) | ❌ | all2all.py:520 |

**gpt-oss-120b 의 quant**: MXFP4 native. `supports_internal_mk` 가 False (확인됨, 본 study 에서 OOM/error 흐름으로 검증). → 본 모델 + DP>1 + DeepEP 조합은 **vllm 0.17.1 에서 미구현 영역**. 우리 코드 fix 로 해결 불가.

**영속화 위치 (회피만 가능)**
- 본 catalog entry — 다음 wide-EP study 시도 시 1차 진단처
- `b200/docs/vllm_0.17.1_args_catalog.md` § 2.2 — DeepEP 의 dispatch_router_logits 미구현 호환성 룰 source 로 명시
- 의사결정 룰: gpt-oss-120b 같은 **MXFP4/internal-MK 미지원 quant 모델은 wide-EP-LWS path 부적합**. inference-scheduling (b3 검증) 만 사용
- wide-EP study 는 internal-MK 지원 quant 모델 (예: FP8 native — DSV3, Kimi K2 등) 로 별도 진행

**향후 (vllm 0.18+ 출시 시 재검증)**
- vllm release notes 에서 `DeepEPAll2AllManagerBase.dispatch_router_logits` 구현 / supports_internal_mk 확장 여부 추적
- 구현 시 R19 가 resolved 로 표시 + b4 wide-EP gpt-oss-120b study 재진입 가능
- 그 전엔 wide-EP 는 다른 모델로

---

## R20 — adapter 의 helmfile apply 가 helm strategic merge conflict 로 trial 마다 fail

**증상**
- PR #105 (b3-v2) 머지 후 b3-gpt-oss-120b-v2 study 시작 시 5 trial 다 ~3-4 초 만에 crash
- circuit breaker 가 5 consecutive failures 로 study HALT
- ApplyResult.detail 에:
  ```
  Error: UPGRADE FAILED: failed to create patch: The order in patch list:
    [...HF_TOKEN... HF_TOKEN... TP_SIZE...]
     doesn't match $setElementOrder list:
    [VLLM_LOGGING_LEVEL HF_TOKEN DP_SIZE TP_SIZE DP_SIZE_LOCAL HF_TOKEN]
  notes: helmfile apply rc=1
  ```

**진단**
- helm 의 strategic merge patch 가 env list 같은 ordered 필드 patch 시 같은 key (HF_TOKEN) 가 두 번 등장하는 spec 변경을 처리 못 함
- 사용자가 직접 helmfile apply (수동 deploy) 한 release 의 spec 와 lmtune adapter 가 trial 에서 만드는 spec 가 미세히 다름 (modelArtifacts authSecretName + chart auto-injected HF_TOKEN env + values gotmpl env 의 결합 결과)
- 두 번째 apply 시 helm 이 두 spec 의 env list 순서를 비교하다 conflict
- ~4 초 trial duration = helmfile apply 가 chart fetch + render 후 helm upgrade 단계에서 즉시 reject (rollout 까지 못 감)

**영속화 위치**
- Code: `src/lmtune/deploy/llmd_k8s.py::LLMDK8sAdapter.apply` — helmfile apply 명령에 `--args "--force"` 추가. helm 의 --force flag 는 strategic merge 실패 시 release 통째 replace 로 fallback. 매 trial 이 어차피 새 spec 이라 의도와 일치
- 주석으로 R20 reference + safety rationale 명시

**향후 (root cause 의 제거 — 별도 PR)**
- chart 의 modelservice template 이 HF_TOKEN env 를 자동 inject 하는지 확인
- values gotmpl 의 env 블록과 chart auto-inject 가 중복 안 되도록 정리
- 매 trial 의 helmfile apply 가 동일 spec 산출하도록 adapter 의 render_values_overlay 결정성 강화 (동일 params → 동일 spec, 순서 안정)

**워크플로우 변경 권장**
- 사용자가 직접 helmfile apply 로 deploy 후 lmtune search start 하는 패턴 유지 가능 (--force fallback 으로 conflict 자동 처리)
- 단 lmtune 시작 전 사용자 helmfile apply 와 lmtune 의 첫 trial helmfile apply 가 spec 일치하면 더 빠름 (--force 안 거치면 helm upgrade in-place patch 로 더 빠름)

---

## R21 — R20 의 helmfile apply --args "--force" 가 helm-diff plugin 에서 reject (부작용)

**증상**
- PR #106 (R20 fix: helmfile apply 에 --args "--force" 추가) 머지 후 b3-v2 study 재시작 시 5 trial 다 같은 fail
- ApplyResult.detail 에:
  ```
  Error: plugin "diff" exited with error
  notes: helmfile apply rc=1
  ```

**진단**
- helmfile apply 는 내부적으로 helm-diff plugin (변경 비교) → helm upgrade (실제 적용) 흐름
- `--args "--force"` 는 helm-diff 와 helm upgrade **모두에 전달**됨
- helm-diff plugin 이 `--force` flag 를 모름 → "plugin diff exited with error" 로 reject
- 즉 R20 의 fix 가 또 다른 결함 (R21) 만든 것 — trial-and-error 의 비용

**영속화 위치**
- Code: `src/lmtune/deploy/llmd_k8s.py::LLMDK8sAdapter.apply` — `apply` 를 `sync` 로 변경. helmfile sync 는 helm-diff 우회하고 `helm upgrade --install` 직접 호출. `--args "--force"` 가 helm upgrade 에만 전달
- 주석으로 R20 + R21 reference

**Trade-off (sync vs apply)**
- sync: 변경 비교 없이 무조건 upgrade — 변경 없는 trial 도 약간 느림 (~수초). 대신 plugin 의존 없음
- apply: 변경 있을 때만 upgrade — 더 효율적이지만 plugin (helm-diff) 호환성 의존
- 매 trial 의 spec 가 다른 lmtune autotune 에선 어차피 upgrade 매번이라 sync 가 손해 적음

**향후 (root cause 의 제거 — 별도 PR)**
- chart 의 modelservice template 이 HF_TOKEN env 자동 inject + values gotmpl env 블록 의 중복 정리해서 R20 의 strategic merge conflict 자체 제거 → --force 와 sync 둘 다 제거하고 normal apply 복귀 가능
- 또는 lmtune adapter 의 render 결정성 강화 (동일 params → byte-identical state-values overlay)

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
