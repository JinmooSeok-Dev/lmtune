# well-lit path #5 — P/D disaggregation (B200)

Prefill 과 Decode 를 별도 pod 으로 분리해 NIXL v2 (UCX/RDMA) 로 KV cache 전송. B200 fabric 의 RDMA 효과가 가장 직접적으로 드러나는 path.

## 구조

```
┌─────────────────────────────────────────────────┐
│ Gateway (Inference Gateway)                     │
└────────────────┬────────────────────────────────┘
                 │
            EPP routing
                 │
┌────────────────┼────────────────────────────────┐
│  decode pod    │   prefill pod                  │
│  (TP=4)        │   (TP=4)                       │
│  + routing-    │                                │
│    sidecar     │                                │
│  (NIXL v2)     │                                │
│                │                                │
│  ◄─────── KV cache transfer ──────────►         │
│           UCX/RDMA over IB                      │
└──────────────────────────────────────────────────┘
```

## 적용

```bash
# B0 통과 후 → B1 P/D baseline (Llama-3.1-70B TP=4)
NS=b200-pd helmfile -f helmfile.yaml.gotmpl --selector role=base apply

# HTTPRoute (모든 gateway provider 공통)
kubectl apply -f httproute.yaml
```

## Endpoint 진입 — port-forward 는 inference-gateway 에 ❗

llm-d 의 외부 진입은 **반드시 Gateway service 에 port-forward**. ms-* (modelservice) pod / service 에 직접 port-forward 금지 — EPP (External Processing Pipeline / Endpoint Picker), GAIE prefix-cache aware routing, routing-sidecar 의 prefill/decode 선택 로직을 모두 우회해 측정 결과가 무의미해진다.

HTTPRoute (httproute.yaml) 가 가리키는 진입:

```
Gateway: infra-pd-inference-gateway
  └─ HTTPRoute path / → InferencePool gaie-pd:8000
       └─ EPP 선택 → prefill pod (routing-sidecar) → decode pod (NIXL)
```

```bash
# 1. service 이름 / listener port 확인 (gateway controller 별로 다를 수 있음)
kubectl get gateway -n b200-pd
kubectl get svc     -n b200-pd | grep inference-gateway
kubectl get gateway infra-pd-inference-gateway -n b200-pd \
  -o jsonpath='{.spec.listeners[0].port}{"\n"}'   # 보통 80

# 2. auto-restart port-forward (매 trial helmfile apply 시 endpoint 변경 대비)
while true; do
  kubectl port-forward -n b200-pd svc/infra-pd-inference-gateway 8011:80
  sleep 2
done

# 3. 검증
curl -s http://127.0.0.1:8011/v1/models | jq .
```

endpoint YAML 의 `url: http://127.0.0.1:8011/v1` 는 위 port-forward 와 매핑되는 host loopback. lmtune 의 `LLMDK8sAdapter` 가 자동 port-forward 를 띄우지 않으므로 별도 terminal 필요.

## 측정 시 주목

P/D 의 핵심 메트릭은 **inter-pod KV transfer time**. 다음 환경 변수와 로그 분석 필요:

- `UCX_LOG_LEVEL=info` (필요 시 `debug`)
- `UCX_TLS=rc,cuda_copy,cuda_ipc` (B200: RDMA RC + CUDA IPC)
- decode/prefill log 의 NIXL transfer size + latency
- `b200/scripts/rdma_bench.sh` host-level RDMA bandwidth 와 P/D 의 effective transfer rate 비교

## 환경 가정

- `huggingface-token` secret 사전 생성
- B200 fabric RDMA 가용 (`b200/scripts/probe.sh` 통과)
- gateway provider: agentgateway (default) 또는 kgateway

## values 파일

- `values-gaie.yaml` — EPP 최소 설정
- `values-llama-3.1-70b-tp4-pd.yaml` — 모델별 (B1)
- `../base/values-b200-common.yaml.gotmpl` — B200 공통 (runtime, securityContext, /dev/shm)

## 다음 단계 (B-II 이후)

- `values-qwen2.5-72b-tp8-pd.yaml` — TP=8 prefill + TP=8 decode (인터노드)
- `values-deepseek-v3-pd.yaml` — MoE P/D, EP+TP 결합
- system_snapshot 결합한 ANALYSIS — NIXL transfer 가 RDMA bandwidth 의 몇 % 수준에 도달하는지
