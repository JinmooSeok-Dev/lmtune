# well-lit path #1 — inference-scheduling (B200)

기본 EPP(Endpoint-Picker Plugin) 라우팅 패턴. 단일 GPU smoke + 모델별 baseline 의 시작점.

## 적용

```bash
# B0 smoke (Llama-3.1-8B 단일 GPU)
NS=b200-infsch helmfile -f helmfile.yaml.gotmpl --selector role=smoke apply

# HTTPRoute (gatewayClass = kgateway 가 있으면)
kubectl apply -f httproute.yaml
# gatewayClass = agentgateway 면 InferencePool 미지원 → decode pod 로 직접 port-forward
DECODE_POD=$(kubectl get pods -n b200-infsch -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n b200-infsch pod/$DECODE_POD 8011:8000

# bench smoke
lmtune run -p configs/profiles/autotune/short.yaml -e b200/endpoints/b200_smoke.yaml --json-summary
```

## 정리

```bash
helmfile -f helmfile.yaml.gotmpl destroy
kubectl delete ns b200-infsch
```

## 환경 가정

- `huggingface-token` secret 사전 생성:
  ```bash
  kubectl create ns b200-infsch
  kubectl create secret generic huggingface-token \
    -n b200-infsch --from-literal=HF_TOKEN=$HF_TOKEN
  ```
- gateway provider: agentgateway (default) 또는 kgateway. kgateway 권장 (InferencePool 직지원).

## values 파일 구조

- `values-gaie.yaml` — EPP 설정 (gateway-api-inference-extension chart)
- `values-llama-3.1-8b-smoke.yaml` — 모델별 modelservice values
- `../base/values-b200-common.yaml.gotmpl` — B200 공통 (runtime, securityContext, /dev/shm)

## 다음 단계 (B1)

- `values-qwen3-235b-tp2.yaml` (TP=2, 4 replicas)
- `values-llama-3.1-70b-tp4.yaml` (TP=4, 2 replicas)
- `values-qwen2.5-72b-tp4.yaml` (TP=4, 2 replicas)
