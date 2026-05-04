# B200 — llm-d Port-Forward Runbook

> lmtune endpoint YAML 의 `url:` 을 만족시키기 위해 host 에서 어디로 port-forward 해야 하는지, 왜 그래야 하는지, 어떻게 trial-rolling 환경에서도 끊기지 않게 유지하는지의 정본.
>
> 적용 대상: `--adapter llmd-k8s` 로 lmtune 을 운영하는 모든 well-lit-path (`inference-scheduling`, `pd-disaggregation`, `wide-ep-lws`).

## 결론 한 줄

**`infra-<rn>-inference-gateway` Service 의 :80 → 로컬 8011 로 port-forward 한다.** decode service 직접 forward 는 금지.

```bash
kubectl -n b200-<rn> port-forward svc/infra-<rn>-inference-gateway 8011:80
```

`<rn>` = release name suffix. `inference-scheduling` = `infsch`, `pd-disaggregation` = `pd`, `wide-ep-lws` = `wideep`.

## 왜 gateway 인가 — decode 직접 forward 가 틀린 이유

llm-d 의 well-lit-path 는 다음 라우팅 흐름을 측정·튜닝하기 위해 설계된 토폴로지다:

```
client (lmtune)
   ↓ HTTP /v1/*
infra-<rn>-inference-gateway   (Service ClusterIP, port 80)
   ↓ Gateway API HTTPRoute
gaie-<rn>  (InferencePool + EPP — Endpoint Picker)
   ↓ scheduler scoring → endpoint 선택
ms-<rn>-llm-d-modelservice-decode-<replica>   (실제 vLLM pod, port 8000)
```

decode service 로 직접 port-forward 하면 다음이 측정에서 빠진다:

1. **EPP smart routing** — KV cache hit-rate scoring, prefix cache aware, in-flight batching 인지 라우팅
2. **HTTPRoute / Gateway 정책** — rate limiting, header rewriting, retry, timeout
3. **InferencePool replicas** axis 의 의미 — `decode.replicas=2` 같은 axis 가 healthcheck 대상이 1개로 줄어 측정 무의미
4. **P/D disaggregation 호환성** — pd-disaggregation path 는 prefill/decode 가 분리되어 client 가 직접 어느 쪽을 부를지 결정 불가. gateway 만이 정답

따라서 autotune 의 모든 axis (`max_num_seqs`, `kv_cache_dtype`, `enable_prefix_caching`, …) 가 **운영 환경 그대로** 측정되어야 하고, 그러려면 gateway 를 거쳐야 한다.

> 과거 B0 runbook 에 "agentgateway 는 InferencePool 미지원이라 decode 직접 forward" 라고 적힌 부분은 부정확. llm-d 0.4+ / agentgateway provider 도 InferencePool 라우팅을 정상 지원한다 (`b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl:51` 의 `provider: agentgateway` + `gatewayClassName: agentgateway` 조합으로 검증됨).

## 사전 조건

3 release 가 모두 deployed 상태여야 한다:

```bash
helm -n b200-<rn> list
# NAME            STATUS
# infra-<rn>      deployed     ← gateway
# gaie-<rn>       deployed     ← InferencePool + EPP
# ms-<rn>         deployed     ← decode Deployment + Service
```

ms-<rn> 가 빠져 있으면 gateway 응답이 `upstream call failed: Connect: Connection refused` 로 떨어진다. 이때는 helmfile apply 먼저:

```bash
helmfile \
  --environment default --selector kind=inference-stack \
  -f b200/helmfile/<path>/helmfile.yaml.gotmpl \
  apply

kubectl -n b200-<rn> wait --for=condition=Available deploy \
  -l llm-d.ai/role=decode --timeout=20m
```

(gpt-oss-120b 같은 큰 모델은 weight 다운로드/로딩에 5–15 분.)

## 정본 절차 — `ops/prepare.sh` 한 줄

helmfile apply 가 trial 마다 돌면서 decode pod 가 rolling update 되어 port-forward 가 잠깐 끊긴다. 무한 재연결 wrapper 가 필요하다 — `b200/scripts/ops/prepare.sh` 가 모두 처리한다.

```bash
# 1) release 가 이미 떠 있을 때 — port-forward 만 띄우고 검증
bash b200/scripts/ops/prepare.sh infsch

# 2) release 빠져 있을 때 — helmfile apply 까지 자동
export B200_MODEL_VALUES=values-gpt-oss-120b.yaml.gotmpl   # 의도한 모델
bash b200/scripts/ops/prepare.sh infsch --apply
```

내부 단계 (`prepare.sh` 가 호출하는 함수들):

| step | 호출 | 역할 |
|:---|:---|:---|
| 1 | `bench_env::cluster_check` | kubectl 닿음 + ns 존재 |
| 2 | `helmd::releases_check` (+ `helmd::apply` if `--apply`) | infra/gaie/ms 3종 검증 |
| 3 | `helmd::wait_decode_ready` | decode Deployment Available |
| 4 | `pf::stop_local 8011` | stale wrapper 정리 |
| 5 | `pf::start <ns> <svc> 8011 80` | 재시도 wrapper 데몬 |
| 6 | `pf::probe 8011 /v1/models` | 200 polling, max 5분 |

이후 lmtune 을 그대로 실행한다. trial 경계에서 `/tmp/pf_8011.log` 에 `[pf] disconnected ... retry 3s` 가 한 번씩 보이고 곧 재연결된다.

### 함수 직접 호출 (다른 스크립트에서 source 시)

```bash
source b200/scripts/util/pf.sh
pf::list                              # 살아있는 port-forward 모두
pf::stop_all                          # 전체 정리 + PID 파일 제거
pf::stop_local 8011                   # 특정 로컬 포트만
pf::start b200-infsch infra-infsch-inference-gateway 8011 80
pf::probe 8011 /v1/models             # 200 polling
pf::status                            # 한 화면
```

## endpoint YAML 명세

`b200/endpoints/*.yaml` 의 `url:` 은 **항상 `http://127.0.0.1:8011/v1`** 로 고정한다 (gateway port-forward 전제). 사용자가 환경마다 바꾸지 않도록.

```yaml
url: http://127.0.0.1:8011/v1   # ← gateway port-forward 전제 (port_forward_runbook.md 참조)
```

> **decode service 로 직접 forward 하지 않는다**. llm-d 의 InferencePool/EPP smart routing 이 빠져 autotune 결과가 운영 환경과 어긋나기 때문이다 — 그럴 거면 llm-d 를 쓸 이유가 없다. 이 항목은 옵션이 아니라 금지 사항이다.

## 트러블슈팅

| 증상 | 원인 | 해결 |
|:---|:---|:---|
| `port-forward` 가 즉시 `exit 1` | service 이름 오타 또는 없음 | `kubectl get svc -n <ns>` 로 정확한 이름 확인 |
| `Connect: Connection refused (os error 111)` | gateway 는 떠 있지만 decode upstream 없음 | `helm list` 에서 `ms-<rn>` 확인 → `helmfile apply` |
| `unable to do port forwarding: socat not found` | 노드에 socat 미설치 | `apt install socat` (k3s host 측) |
| `bind: address already in use` | 이전 port-forward 잔재 | `pkill -f "kubectl.*port-forward"` |
| trial 진행 중 모든 trial fail | rolling update 시 port-forward 가 한 번만 끊기고 재연결 안 됨 | wrapper 스크립트 사용 (위 §정본 절차 step 3) |
| `upstream timeout`, 504 | decode pod ready 인데 vLLM 모델 로딩 중 | 첫 trial 전 `/v1/models` 가 200 받을 때까지 대기 |

## 다른 well-lit-path 의 적용

| path | release suffix | namespace | gateway svc 이름 | local port |
|:---|:---|:---|:---|:---|
| inference-scheduling | `infsch` | `b200-infsch` | `infra-infsch-inference-gateway` | 8011 |
| pd-disaggregation | `pd` | `b200-pd` | `infra-pd-inference-gateway` | 8011 |
| wide-ep-lws | `wideep` | `b200-wideep` | `infra-wideep-inference-gateway` | 8011 |

같은 host 에서 두 path 를 동시에 실험할 일은 없으므로 local port 는 `8011` 로 통일.

## 참고

- helmfile 정의: `b200/helmfile/inference-scheduling/helmfile.yaml.gotmpl:36-92` (3 release 정의)
- gateway provider 설정: `b200/scripts/setup_gateway_provider.sh` (agentgateway 가 본 setup 의 default)
- lmtune 의 endpoint URL probe 위치: `src/lmtune/deploy/llmd_k8s.py:338-355` (`probe_openai_models(url)`)
- 본 runbook 진입점: `b200/README.md` 의 § Operations 섹션
