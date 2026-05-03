# well-lit path #2 — wide-EP-LWS (B200)

MoE 모델의 expert 를 여러 GPU 에 wide 분산. 본 baseline 은 modelservice multinode 패턴으로 단순화 (LWS 없이). 더 큰 MoE (Qwen3-Coder-480B / DeepSeek-V3) 의 LeaderWorkerSet 적용은 Track B-IV (B7) 에서 추가.

## 두 deployment 옵션

| 옵션 | 사용 시점 | 특징 |
|:---|:---|:---|
| **A. modelservice multinode** (본 helmfile) | B-I baseline | 단일 노드 dp+ep, modelservice chart standard. 8 GPU MoE 까지 |
| **B. LeaderWorkerSet (peer repo manifests)** | B-IV 이후 | 16-GPU 인터노드 wide-EP. peer repo `llm-d/guides/wide-ep-lws/manifests/` 패턴 (kustomize) |

## 적용 (옵션 A)

```bash
# B0 통과 후 → B1 wide-EP baseline (Mixtral-8x22B DP=2 EP=8)
NS=b200-wideep helmfile -f helmfile.yaml.gotmpl --selector role=base apply

# HTTPRoute (kgateway 환경)
kubectl apply -f httproute.yaml

# agentgateway 면 decode pod 직접 port-forward
DECODE_POD=$(kubectl get pods -n b200-wideep -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n b200-wideep pod/$DECODE_POD 8011:8000
```

## 측정 시 주목

wide-EP 의 핵심 메트릭은 **all-to-all 통신 비용**:

- vLLM `--all2all-backend` axis: `[allgather_reducescatter, deepep_low_latency, deepep_high_throughput, nixl_ep]`
- B200 fabric 에서 deepep / nixl_ep 가 standard 대비 우위 예상
- B-II 에서 추가 검증: `--enable-dbo` (dual batch overlap), `--enable-eplb` (expert load balance)
- system_snapshot 의 NCCL_NET_GDR_LEVEL · NCCL_IB_QPS 설정과 결합 분석

## 환경 가정

- `huggingface-token` secret 사전 생성 (Mixtral gated 아니지만 Llama 등 다른 모델용)
- B200 단일 노드 8 GPU 가용 (Mixtral-8x22B 는 ~280GB 으로 8×180GB HBM 충분)
- gateway provider: agentgateway (default) 또는 kgateway

## values 파일

- `values-gaie.yaml` — EPP 최소 설정
- `values-mixtral-8x22b-dp2-ep8.yaml` — B1 baseline
- `../base/values-b200-common.yaml.gotmpl` — B200 공통 (runtime, /dev/shm, NCCL env)

## 다음 단계

- **B-II**: 같은 모델 위에서 `enable_dbo` / `enable_eplb` / `all2all_backend` axis 변주
- **B-IV (옵션 B)**: Qwen3-Coder-480B 또는 DeepSeek-V3 의 LWS 기반 인터노드 wide-EP. peer repo `llm-d/guides/wide-ep-lws/manifests/` 의 kustomize manifests 를 b200/manifests/wide-ep-lws/ 로 복사·adapt
