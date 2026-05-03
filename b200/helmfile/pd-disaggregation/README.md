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

# HTTPRoute (kgateway 환경)
kubectl apply -f httproute.yaml

# 또는 agentgateway 면 InferencePool 미지원 → decode pod 직접 port-forward
DECODE_POD=$(kubectl get pods -n b200-pd -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n b200-pd pod/$DECODE_POD 8011:8000
```

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
