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
